from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from coflandscaper._internal.helpers import CenterZ, RemoveLayer, SetVacuum


def _write_toy_structure(path: Path) -> None:
    lat = Lattice.from_parameters(10.0, 10.0, 20.0, 90, 90, 90)
    structure = Structure(
        lattice=lat,
        species=["C", "N", "H"],
        coords=[[0.1, 0.1, 0.10], [0.4, 0.3, 0.20], [0.7, 0.7, 0.90]],
        coords_are_cartesian=False,
    )
    CifWriter(structure).write_file(str(path), mode="wt")


def _patch_structure_to(monkeypatch: pytest.MonkeyPatch) -> None:
    def _compat_to(self: Structure, filename: str | Path) -> None:
        CifWriter(self).write_file(str(filename), mode="wt")

    monkeypatch.setattr(Structure, "to", _compat_to)


@pytest.mark.unit
def test_set_vacuum_places_lowest_atom_near_cell_bottom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures SetVacuum shifts atoms so the slab starts at the cell bottom."""
    input_cif = tmp_path / "in.cif"
    output_cif = tmp_path / "out.cif"
    _patch_structure_to(monkeypatch)
    _write_toy_structure(input_cif)

    SetVacuum()._set_vacuum_on_top(
        str(input_cif), str(output_cif), vacuum_top=15.0
    )

    out = Structure.from_file(str(output_cif))
    c_hat = out.lattice.matrix[2] / np.linalg.norm(out.lattice.matrix[2])
    z_proj = np.array([np.dot(site.coords, c_hat) for site in out.sites])

    assert output_cif.exists()
    assert float(np.min(z_proj)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_center_z_moves_slab_midpoint_to_half_fractional_height(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures CenterZ recenters slab atoms around fractional z equal to 0.5."""
    input_cif = tmp_path / "in.cif"
    output_cif = tmp_path / "centered.cif"
    _patch_structure_to(monkeypatch)
    _write_toy_structure(input_cif)

    CenterZ()._center_slab_z(str(input_cif), str(output_cif))

    out = Structure.from_file(str(output_cif))
    fz = np.array([site.frac_coords[2] for site in out.sites], dtype=float)
    z_mid = 0.5 * (float(np.min(fz)) + float(np.max(fz)))

    assert output_cif.exists()
    assert z_mid == pytest.approx(0.5, abs=1e-6)


@pytest.mark.unit
def test_remove_layer_frac_removes_targeted_atoms_and_preserves_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures fractional RemoveLayer removes only atoms within the periodic z tolerance band."""
    input_cif = tmp_path / "in.cif"
    output_cif = tmp_path / "removed.cif"
    _patch_structure_to(monkeypatch)
    _write_toy_structure(input_cif)

    RemoveLayer()._remove_layer(
        str(input_cif),
        str(output_cif),
        z_value=0.10,
        tol=0.05,
        mode="frac",
    )

    out = Structure.from_file(str(output_cif))
    out_fz = sorted(float(site.frac_coords[2]) for site in out.sites)

    assert output_cif.exists()
    assert len(out.sites) == 2
    assert out_fz == pytest.approx([0.2, 0.9], abs=1e-6)


@pytest.mark.unit
def test_remove_layer_raises_for_invalid_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures RemoveLayer rejects unsupported coordinate modes."""
    input_cif = tmp_path / "in.cif"
    output_cif = tmp_path / "removed.cif"
    _patch_structure_to(monkeypatch)
    _write_toy_structure(input_cif)

    with pytest.raises(ValueError, match="mode must be 'frac' or 'cart'"):
        RemoveLayer()._remove_layer(
            str(input_cif),
            str(output_cif),
            z_value=0.10,
            tol=0.05,
            mode="bad",
        )
