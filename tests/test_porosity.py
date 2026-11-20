"""Routing, safety and direct Zeo++ checks for optional porosity analysis."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read, write
from ase.io.cif import CIFBlock

from coflandscaper import Porosity
from coflandscaper._internal.porosity import _write_p1


@pytest.mark.unit
def test_workflow_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(Porosity.__module__)
    monkeypatch.setattr(module, "find_spec", lambda _name: True)
    monkeypatch.setattr(
        Porosity, "_analyse", lambda _self, _path: {"lcd_angstrom": 8.0}
    )
    root = Path("cof")
    for stage, mode in [(4, "serr"), (4, "incl"), (2, "")]:
        folder = root / f"{stage}_cof_optimization" / mode
        folder.mkdir(parents=True)
        (folder / "sample.cif").touch()

    job = Porosity()
    results = job.run("cof")
    assert list(results["mode"]) == ["serr", "incl"]
    for tag in ("serr", "incl"):
        saved = pd.read_csv(
            root / "5_cof_analysis" / "porosity" / tag / "cof_porosity.csv"
        )
        assert saved["mode"].tolist() == [tag]
    assert job.run("cof", "3d")["mode"].tolist() == ["3d"]
    assert (root / "3_cof_analysis/porosity/cof_porosity.csv").exists()
    inputs = tmp_path / "independent"
    inputs.mkdir()
    for filename in ("b.cif", "a.CIF"):
        (inputs / filename).touch()
    explicit = tmp_path / "results.csv"
    assert job.run(input_path=inputs, output_path=explicit)[
        "structure"
    ].tolist() == ["a", "b"]
    assert pd.read_csv(explicit)["structure"].tolist() == ["a", "b"]


@pytest.mark.unit
def test_optional_backend_and_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        importlib.import_module(Porosity.__module__),
        "find_spec",
        lambda _: None,
    )
    with pytest.raises(ImportError, match=r"cof-landscaper\[porosity\]"):
        Porosity().run(input_path=tmp_path)
    with pytest.raises(ValueError, match="radii"):
        Porosity(probe_radius=float("nan"))
    with pytest.raises(ValueError, match="counts"):
        Porosity(volume_samples=0)


@pytest.mark.integration
@pytest.mark.parametrize("dense", [False, True])
def test_geometric_porosity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dense: bool
) -> None:
    """Compare unfolded cell quantities, native diameters and mass normalisation."""
    zeo = pytest.importorskip("pyzeo")
    monkeypatch.chdir(tmp_path)
    cell = [3, 3, 3, 90, 90, 90] if dense else [9, 10, 12, 85, 94, 110]
    atoms = Atoms(
        "S", scaled_positions=[[0.25, 0.25, 0.25]], cell=cell, pbc=True
    )
    cif = Path("sulfur.cif")
    write(cif, atoms)
    original = cif.read_bytes()
    job = Porosity(
        probe_radius=1.7,
        channel_radius=1.9,
        surface_samples=200,
        volume_samples=4000,
    )
    row = job.run(input_path=cif, save_csv=False).iloc[0]
    assert not Path("porosity").exists()
    assert cif.read_bytes() == original
    network = zeo.AtomNetwork.read_from_CIF(str(cif))
    assert network.atoms[0].radius == pytest.approx(1.80)
    accurate = network.copy()
    zeo.high_accuracy_atomnet(accurate, "DEF")
    kwargs = {
        "high_accuracy": True,
        "high_accuracy_atmnet": accurate,
        "exclude_pockets": True,
    }
    area = zeo.surface_area(network, 1.9, 1.7, 200, **kwargs).decode().split()
    volume = zeo.volume(network, 1.9, 1.7, 4000, **kwargs).decode().split()
    accurate.calculate_free_sphere_parameters("direct.res")
    sa = float(area[area.index("ASA_A^2:") + 1])
    av = float(volume[volume.index("AV_A^3:") + 1])
    density = float(volume[volume.index("Density:") + 1])
    unit_cell = atoms.get_volume()
    per_volume = 1e4 * sa / unit_cell
    # Native text output rounds intermediate quantities to six significant figures.
    np.testing.assert_allclose(
        [
            row["accessible_surface_area_m2_cm3"],
            row["accessible_surface_area_m2_g"],
            row["accessible_pore_volume_cm3_g"],
            row["accessible_void_fraction_percent"],
        ],
        [
            per_volume,
            per_volume / density,
            av / unit_cell / density,
            100 * av / unit_cell,
        ],
        rtol=2e-5,
    )
    np.testing.assert_allclose(
        row[
            [
                "lcd_angstrom",
                "pld_angstrom",
                "largest_included_diameter_along_free_path_angstrom",
            ]
        ].to_numpy(float),
        np.asarray(Path("direct.res").read_text().split()[1:], dtype=float),
    )
    if dense:
        assert av == sa == 0
        assert row["lcd_angstrom"] == pytest.approx(
            3 * np.sqrt(3) - 3.6, abs=1e-5
        )
        assert row["pld_angstrom"] == pytest.approx(
            3 * np.sqrt(2) - 3.6, abs=1e-5
        )
    else:
        assert sa > 0
        assert av > 0


@pytest.mark.integration
def test_invalid_and_unsupported_cifs(tmp_path: Path) -> None:
    pytest.importorskip("pyzeo")
    filename = tmp_path / "invalid.cif"
    filename.write_text("data_empty\n")
    job = Porosity()
    with pytest.raises(ValueError, match="Invalid CIF"):
        job.run(input_path=filename)
    write(
        filename, Atoms("Og", positions=[[0, 0, 0]], cell=[8, 8, 8], pbc=True)
    )
    with pytest.raises(RuntimeError, match=r"(?s)invalid.cif.*radius for Og"):
        job.run(input_path=filename)
    assert not (tmp_path / "porosity").exists()


@pytest.mark.unit
@pytest.mark.parametrize("group", [1, 2])
def test_p1_geometry_preserves_cell_and_expands_sites(
    tmp_path: Path, group: int
) -> None:
    source = tmp_path / "original.cif"
    source.write_text(
        f"data_cell\n_space_group_IT_number {group}\n"
        "_cell_length_a 30.650925698216515\n_cell_length_b 30.666855062719765\n"
        "_cell_length_c 42.66701275707882\n_cell_angle_alpha 89.96041765448992\n"
        "_cell_angle_beta 90.05115025840301\n_cell_angle_gamma 126.79543220653909\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n"
        "_atom_site_fract_z\nS1 0.1234567890123456 0.23 0.42\n"
    )
    atoms = read(source, store_tags=True)
    assert isinstance(atoms, Atoms)
    prepared = tmp_path / "prepared.cif"
    _write_p1(prepared, atoms)
    actual = read(prepared, store_tags=True)
    assert isinstance(actual, Atoms)
    assert [actual.info[tag] for tag in CIFBlock.cell_tags] == [
        atoms.info[tag] for tag in CIFBlock.cell_tags
    ]
    site = np.array([0.1234567890123456, 0.23, 0.42])
    positions = np.column_stack(
        [actual.info[f"_atom_site_fract_{axis}"] for axis in "xyz"]
    )
    if group == 1:
        np.testing.assert_array_equal(positions, [site])
    else:
        np.testing.assert_allclose(
            positions, [site, 1 - site], atol=1e-14, rtol=0
        )
