import importlib
import inspect
import json
import runpy
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.cell import Cell
from ase.constraints import FixCartesian
from ase.io import read, write

import coflandscaper as cl

calc_mod = importlib.import_module(cl.GeometryOpt.__module__)
SinglePoint = cl.SinglePoint
GeometryOpt = cl.GeometryOpt


class _DummyAtoms:
    def __init__(self, energy: float = 0.0) -> None:
        self._energy = energy
        self.calc = None
        self.written_to: str | None = None

    def get_potential_energy(self) -> float:
        return self._energy

    def write(self, output_path: str) -> None:
        self.written_to = output_path


@pytest.mark.unit
@pytest.mark.parametrize("options", [{}, {"output_csv_dir": "results"}])
def test_single_point_run_routes_both_modes(
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, str],
) -> None:
    """This test ensures the public SinglePoint.run API preserves mode routing."""
    calls: list[tuple[str, str | None]] = []
    backend = EMT()
    monkeypatch.setattr(calc_mod, "_default_mace", lambda: backend)

    def fake_run_folder(
        _self: SinglePoint,
        input_folder: str,
        output_csv_dir: str | None = None,
        **_kwargs: str,
    ) -> None:
        assert _self.calc is backend
        calls.append((input_folder, output_csv_dir))

    monkeypatch.setattr(SinglePoint, "_run_folder", fake_run_folder)
    SinglePoint().run("cof-a", **options)

    assert calls == [
        ("cof-a/2_cof-a_matrix/serr", options.get("output_csv_dir")),
        ("cof-a/2_cof-a_matrix/incl", options.get("output_csv_dir")),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("configured_fmax", "override_fmax", "expected_preopt"),
    [(None, None, 0.03), (0.01, None, 0.01), (0.02, 0.04, 0.04)],
)
def test_optimize_cof_respects_max_steps_and_warns_when_not_converged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_fmax: float | None,
    override_fmax: float | None,
    expected_preopt: float,
) -> None:
    """Forward the selected limits to ASE and report them on nonconvergence."""
    input_cif = tmp_path / "a.cif"
    output_cif = tmp_path / "a_opt.cif"
    input_cif.write_text("data_dummy\n", encoding="utf-8")

    dummy_atoms = _DummyAtoms()

    class _DummyLBFGS:
        last_fmax: float | None = None
        last_steps: int | None = None

        def __init__(self, _ucf) -> None:
            pass

        def run(self, fmax: float, steps: int) -> bool:
            _DummyLBFGS.last_fmax = fmax
            _DummyLBFGS.last_steps = steps
            return False

    monkeypatch.setattr(calc_mod, "read", lambda _p: dummy_atoms)
    monkeypatch.setattr(calc_mod, "FrechetCellFilter", lambda atoms: atoms)
    monkeypatch.setattr(calc_mod, "LBFGS", _DummyLBFGS)

    opt = GeometryOpt(fmax=configured_fmax, max_steps=500, calculator=EMT())

    with pytest.raises(RuntimeError) as failure:
        opt.run_preopt(
            "cof",
            input_path=str(input_cif),
            output_path=str(output_cif),
            fix_z=False,
            fmax=override_fmax,
        )
    message = str(failure.value)
    assert (
        "Single-layer preoptimisation of cof did not converge within 500 steps"
        in message
    )
    assert _DummyLBFGS.last_fmax == expected_preopt
    assert _DummyLBFGS.last_steps == 500
    assert f"fmax={expected_preopt:g} eV/Å" in message
    assert "unconverged intermediate" in message
    assert str(output_cif) in message
    assert dummy_atoms.written_to == str(output_cif)
    assert "strict" not in inspect.signature(opt.run_preopt).parameters

    dummy_atoms.written_to = None
    with pytest.warns(UserWarning, match="did not converge within 500 steps"):
        converged = opt.optimize_cof(str(input_cif), str(output_cif))

    assert converged is False
    assert _DummyLBFGS.last_fmax == (
        0.01 if configured_fmax is None else configured_fmax
    )
    assert _DummyLBFGS.last_steps == 500
    assert dummy_atoms.written_to == str(output_cif)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("options", "head", "cutoff"),
    [
        ({}, "spice_wB97M", None),
        ({"head": "omol"}, "omol", None),
        ({"head": "omat_pbe"}, "omat_pbe", 21.167088422553647),
        ({"head": "matpes_r2scan"}, "matpes_r2scan", 40.0),
        (
            {"head": "omat_pbe", "dispersion": False, "dispersion_cutoff": 9},
            "omat_pbe",
            9,
        ),
    ],
)
def test_default_factory_preserves_head_presets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict,
    head: str,
    cutoff: float | None,
) -> None:
    captured: dict[str, object] = {}

    def fake_mace_mp(**kwargs):
        captured.update(kwargs)
        return EMT()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("mace.calculators.mace_mp", fake_mace_mp)
    cl.SinglePoint(calculator_kwargs=options)
    assert captured["model"] == "mh-1"
    assert captured["default_dtype"] == "float64"
    assert captured["device"] == "cpu"
    assert captured["head"] == head
    assert captured["dispersion"] is options.get(
        "dispersion", cutoff is not None
    )
    assert captured.get("dispersion_cutoff") == cutoff
    if cutoff is not None:
        assert captured["dispersion_xc"] == (
            "pbe" if head == "omat_pbe" else "r2scan"
        )


@pytest.mark.unit
def test_geometry_opt_energy_csv_replaces_only_processed_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_base = tmp_path / "cof-1" / "4_cof-1_optimization"
    incl_folder = output_base / "incl"
    serr_folder = output_base / "serr"
    incl_folder.mkdir(parents=True)
    serr_folder.mkdir(parents=True)

    (incl_folder / "cof-1_a_incl.cif").write_text(
        "data_dummy\n", encoding="utf-8"
    )
    (serr_folder / "cof-1_a_serr.cif").write_text(
        "data_dummy\n", encoding="utf-8"
    )

    existing = pd.DataFrame(
        [
            {
                "structure": "cof-1_old_serr",
                "stacking_mode": "serr",
                "energy_eV_per_layer": -10.0,
                "energy_rel_eV_per_layer": 0.0,
                "stopped_due_to_max_steps": False,
            },
            {
                "structure": "cof-1_old_incl",
                "stacking_mode": "incl",
                "energy_eV_per_layer": -9.0,
                "energy_rel_eV_per_layer": 1.0,
                "stopped_due_to_max_steps": False,
            },
        ]
    )
    csv_path = output_base / "cof-1_opt_energies_per_layer.csv"
    existing.to_csv(csv_path, index=False)

    def fake_read(path: str):
        if path.endswith("cof-1_a_incl.cif"):
            return _DummyAtoms(energy=-8.5)
        if path.endswith("cof-1_a_serr.cif"):
            return _DummyAtoms(energy=-20.0)
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(calc_mod, "read", fake_read)

    opt = object.__new__(GeometryOpt)
    opt.calc = EMT()

    out_csv = opt._write_optimized_energy_csv(
        cof_name="cof-1",
        mode_output_folders={"incl": incl_folder, "serr": serr_folder},
        output_base_path=output_base,
        convergence_map={
            ("incl", "cof-1_a_incl"): False,
            ("serr", "cof-1_a_serr"): True,
        },
    )

    assert out_csv == csv_path
    written = pd.read_csv(csv_path)

    keys = set(
        zip(written["stacking_mode"], written["structure"], strict=False)
    )
    assert ("incl", "cof-1_a_incl") in keys
    assert ("serr", "cof-1_a_serr") in keys
    assert ("incl", "cof-1_old_incl") not in keys
    assert ("serr", "cof-1_old_serr") not in keys

    serr_row = written[
        (written["stacking_mode"] == "serr")
        & (written["structure"] == "cof-1_a_serr")
    ].iloc[0]
    incl_row = written[
        (written["stacking_mode"] == "incl")
        & (written["structure"] == "cof-1_a_incl")
    ].iloc[0]

    assert float(serr_row["energy_eV_per_layer"]) == -10.0
    assert float(incl_row["energy_eV_per_layer"]) == -8.5
    assert bool(incl_row["stopped_due_to_max_steps"])
    assert not bool(serr_row["stopped_due_to_max_steps"])


@pytest.mark.unit
def test_run_preopt_uses_default_paths_and_restores_fix_z(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, str | bool] = {}

    def fake_optimize(
        self, input_path: str, output_path: str, **kwargs
    ) -> bool:
        assert kwargs["preopt_name"] == "COF-1"
        seen["input_path"] = input_path
        seen["output_path"] = output_path
        seen["fix_z_during_call"] = self._fix_z
        return True

    opt = GeometryOpt(fix_z=False, calculator=EMT())
    monkeypatch.setattr(
        opt, "_optimize_cof", fake_optimize.__get__(opt, GeometryOpt)
    )

    monkeypatch.chdir(tmp_path)

    converged = opt.run_preopt("COF-1")

    assert converged is True
    assert seen["input_path"] == "COF-1/1_COF-1_single_layer/COF-1_unopt.cif"
    assert seen["output_path"] == "COF-1/1_COF-1_single_layer/COF-1_preopt.cif"
    assert seen["fix_z_during_call"] is True
    assert opt._fix_z is False


@pytest.mark.unit
def test_run_preopt_can_disable_fix_z_for_single_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, bool] = {}

    def fake_optimize(
        self, _input_path: str, _output_path: str, **kwargs
    ) -> bool:
        assert kwargs["preopt_name"] == "COF-1"
        seen["fix_z_during_call"] = self._fix_z
        return True

    opt = GeometryOpt(fix_z=True, calculator=EMT())
    monkeypatch.setattr(
        opt, "_optimize_cof", fake_optimize.__get__(opt, GeometryOpt)
    )

    monkeypatch.chdir(tmp_path)

    converged = opt.run_preopt("COF-1", fix_z=False)

    assert converged is True
    assert seen["fix_z_during_call"] is False
    assert opt._fix_z is True


@pytest.mark.unit
def test_optimisation_preserves_cif_reader_warnings(tmp_path: Path) -> None:
    """A CIF format warning remains visible during and after an optimisation."""
    source = tmp_path / "copper.cif"
    write(source, bulk("Cu", "fcc", a=3.6, cubic=True))
    source.write_text("#\\#CIF_2.0\n" + source.read_text())
    opt = GeometryOpt(calculator=EMT(), max_steps=1, fmax=0.1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        opt.optimize_cof(str(source), str(tmp_path / "optimised.cif"))
        read(source)
    assert sum("CIF v2.0 file format" in str(w.message) for w in caught) == 2


@pytest.mark.unit
def test_optimize_cof_uses_full_cell_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures normal optimization uses a full cell filter."""
    seen: dict[str, bool] = {}
    atoms = _DummyAtoms()

    class _DummyOptimizer:
        def __init__(self, _filter: object) -> None:
            pass

        def run(self, **_kwargs: object) -> bool:
            return True

    def fake_filter(_atoms: object) -> object:
        seen["filter_called"] = True
        return object()

    monkeypatch.setattr(calc_mod, "read", lambda _path: atoms)
    monkeypatch.setattr(calc_mod, "FrechetCellFilter", fake_filter)
    monkeypatch.setattr(calc_mod, "LBFGS", _DummyOptimizer)
    opt = GeometryOpt(fix_z=False, calculator=EMT())
    opt.optimize_cof(str(tmp_path / "in.cif"), str(tmp_path / "out.cif"))
    assert seen["filter_called"]


@pytest.mark.unit
def test_run_fixed_uses_atoms_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures run_fixed sends the unfiltered atoms to LBFGS."""
    atoms = _DummyAtoms()
    seen: dict[str, object] = {}

    class _DummyOptimizer:
        def __init__(self, target: object) -> None:
            seen["target"] = target

        def run(self, **_kwargs: object) -> bool:
            return True

    def fail_filter(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fixed mode must not create a cell filter")

    monkeypatch.setattr(calc_mod, "read", lambda _path: atoms)
    monkeypatch.setattr(calc_mod, "FrechetCellFilter", fail_filter)
    monkeypatch.setattr(calc_mod, "LBFGS", _DummyOptimizer)
    input_dir = tmp_path / "cof-a" / "6_cof-a_scaling" / "scaling" / "serr"
    input_dir.mkdir(parents=True)
    (input_dir / "scaled.cif").write_text("data_dummy\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    opt = GeometryOpt(fix_z=False, calculator=EMT())
    assert opt.run_fixed("cof-a", "serr") == {"serr/scaled": True}
    assert seen["target"] is atoms


@pytest.mark.unit
def test_geometry_opt_run_routes_both_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures the renamed GeometryOpt.run preserves normal routing."""
    monkeypatch.chdir(tmp_path)
    opt = GeometryOpt(calculator=EMT())
    calls: list[tuple[str, str]] = []

    def record_process(
        input_folder: str, output_folder: str
    ) -> dict[str, bool]:
        calls.append((input_folder, output_folder))
        return {"candidate": Path(input_folder).name == "incl"}

    monkeypatch.setattr(opt, "process_cifs", record_process)
    assert opt.run(cof_name="cof-a", save_opt_energies_csv=False) is None  # type: ignore[func-returns-value]

    assert calls == [
        (
            "cof-a/3_cof-a_landscape/selection/serr",
            "cof-a/4_cof-a_optimization/serr",
        ),
        (
            "cof-a/3_cof-a_landscape/selection/incl",
            "cof-a/4_cof-a_optimization/incl",
        ),
    ]


@pytest.mark.unit
def test_run_routes_one_scaled_cif_to_postopt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures the public postopt workflow preserves the scaled CIF filename."""
    input_dir = tmp_path / "cof-a" / "6_cof-a_scaling" / "scaling" / "serr"
    input_dir.mkdir(parents=True)
    source = input_dir / "scaled_1.0237.cif"
    source.write_text("data_dummy\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    opt = GeometryOpt(calculator=EMT())
    seen: dict[str, str] = {}

    def fake_optimize(
        input_path: str,
        output_path: str,
        fixed_cell: bool = False,
    ) -> bool:
        _ = fixed_cell
        seen.update(input=input_path, output=output_path)
        return True

    monkeypatch.setattr(
        opt,
        "_optimize_cof",
        fake_optimize,
    )
    assert opt.run_fixed("cof-a", "serr") == {"serr/scaled_1.0237": True}
    assert seen == {
        "input": "cof-a/6_cof-a_scaling/scaling/serr/scaled_1.0237.cif",
        "output": "cof-a/6_cof-a_scaling/postopt/serr/scaled_1.0237.cif",
    }


@pytest.mark.unit
def test_run_requires_unambiguous_scaled_cif(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures postopt CIF selection is explicit when scaling outputs are ambiguous."""
    input_dir = tmp_path / "cof-a" / "6_cof-a_scaling" / "scaling" / "incl"
    input_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    opt = GeometryOpt(calculator=EMT())
    monkeypatch.setattr(opt, "_optimize_cof", lambda *_args, **_kwargs: True)
    with pytest.raises(FileNotFoundError, match="No CIF files"):
        opt.run_fixed("cof-a", "incl")
    for filename in ["a.cif", "b.cif"]:
        (input_dir / filename).write_text("data_dummy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple CIF"):
        opt.run_fixed("cof-a", "incl")
    assert opt.run_fixed("cof-a", "incl", source_cif="b.cif") == {
        "incl/b": True
    }


@pytest.mark.unit
def test_run_both_routes_modes_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures both mode folders are processed independently."""
    for mode in ["serr", "incl"]:
        input_dir = tmp_path / "cof-a" / "6_cof-a_scaling" / "scaling" / mode
        input_dir.mkdir(parents=True)
        (input_dir / f"{mode}.cif").write_text(
            "data_dummy\n", encoding="utf-8"
        )
    monkeypatch.chdir(tmp_path)
    opt = GeometryOpt(calculator=EMT())
    seen: list[tuple[str, str]] = []

    def record_optimization(
        input_path: str,
        output_path: str,
        fixed_cell: bool = False,
    ) -> bool:
        _ = fixed_cell
        seen.append((input_path, output_path))
        return True

    monkeypatch.setattr(
        opt,
        "_optimize_cof",
        record_optimization,
    )

    assert opt.run_fixed("cof-a", "both") == {
        "serr/serr": True,
        "incl/incl": True,
    }
    assert seen == [
        (
            "cof-a/6_cof-a_scaling/scaling/serr/serr.cif",
            "cof-a/6_cof-a_scaling/postopt/serr/serr.cif",
        ),
        (
            "cof-a/6_cof-a_scaling/scaling/incl/incl.cif",
            "cof-a/6_cof-a_scaling/postopt/incl/incl.cif",
        ),
    ]


@pytest.fixture
def preopt_atoms() -> Atoms:
    """Provide an asymmetric periodic cell for geometry and force checks."""
    return Atoms(
        "Cu3",
        scaled_positions=[[0.1, 0.2, 0.4], [0.5, 0.5, 0.4], [0.2, 0.8, 0.42]],
        cell=Cell.fromcellpar([4.5, 5.2, 16, 92, 94, 116]),
        pbc=True,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("topo", "gamma"),
    [("hcb", 120), ("hcb_ab", 120), ("kgm", 120), ("sql", 90), ("sql_ab", 90)],
)
def test_prepare_preopt_cell_preserves_fractional_positions(
    preopt_atoms: Atoms, topo: str, gamma: float
) -> None:
    """Angle preparation changes neither atom identity nor fractional placement."""
    original = preopt_atoms.copy()
    calc_mod._prepare_preopt_cell(preopt_atoms, topo)
    assert preopt_atoms.cell.lengths() == pytest.approx(
        [*original.cell.lengths()[:2], 15]
    )
    assert preopt_atoms.cell.angles() == pytest.approx([90, 90, gamma])
    np.testing.assert_allclose(
        preopt_atoms.get_scaled_positions(), original.get_scaled_positions()
    )
    assert (
        preopt_atoms.get_chemical_symbols() == original.get_chemical_symbols()
    )
    assert np.array_equal(preopt_atoms.pbc, original.pbc)


@pytest.mark.unit
@pytest.mark.parametrize("topo", ["sql", "hcb"])
@pytest.mark.parametrize("fix_z", [False, True])
def test_fixed_angle_filter_forces_match_finite_differences(
    preopt_atoms: Atoms, topo: str, fix_z: bool
) -> None:
    """Check atomic and log-length derivatives at nonzero anisotropic strain."""
    calc_mod._prepare_preopt_cell(preopt_atoms, topo)
    if fix_z:
        preopt_atoms.set_constraint(
            FixCartesian(range(len(preopt_atoms)), mask=[False, False, True])
        )
    preopt_atoms.calc = EMT()
    cell_filter = calc_mod._FixedAngleCellFilter(preopt_atoms)
    positions = cell_filter.get_positions()
    positions[-1, :2] = [0.31, -0.22]
    positions[0, 0] += 0.13
    positions[1, 1] -= 0.07
    cell_filter.set_positions(positions)
    analytic = cell_filter.get_forces()
    if topo == "hcb":
        # Ensure this case distinguishes the oblique-cell derivative from sigma_ii.
        cartesian = (
            -preopt_atoms.get_volume()
            * preopt_atoms.get_stress(voigt=False).diagonal()[:2]
            / len(preopt_atoms)
        )
        assert not np.allclose(analytic[-1, :2], cartesian, atol=1e-3)
    numerical = np.zeros_like(analytic)
    step = 1e-6
    for index in range(positions.size):
        plus = positions.copy()
        minus = positions.copy()
        plus.flat[index] += step
        minus.flat[index] -= step
        cell_filter.set_positions(plus)
        energy_plus = cell_filter.get_potential_energy()
        cell_filter.set_positions(minus)
        energy_minus = cell_filter.get_potential_energy()
        numerical.flat[index] = -(energy_plus - energy_minus) / (2 * step)
        cell_filter.set_positions(positions)
    np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=2e-5)


@pytest.mark.unit
@pytest.mark.parametrize(("topo", "gamma"), [("sql", 90), ("hcb", 120)])
@pytest.mark.parametrize("fix_z", [False, True])
def test_topology_preopt_does_not_constrain_later_full_cell_optimization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preopt_atoms: Atoms,
    topo: str,
    gamma: float,
    fix_z: bool,
) -> None:
    """Exercise CIF routing, independent lengths, atomic z, and mode isolation."""
    monkeypatch.chdir(tmp_path)
    folder = Path("cof/1_cof_single_layer")
    folder.mkdir(parents=True)
    source = folder / "cof_unopt.cif"
    write(source, preopt_atoms)
    original_bytes = source.read_bytes()
    original_lengths = preopt_atoms.cell.lengths()[:2].copy()
    targets = []
    limits = []
    initial_z: np.ndarray | None = None

    class RecordingOptimizer:
        def __init__(self, target) -> None:
            self.target = target
            targets.append(target)

        def run(self, fmax: float, steps: int) -> bool:
            nonlocal initial_z
            limits.append((fmax, steps))
            positions = self.target.get_positions()
            if isinstance(self.target, calc_mod._FixedAngleCellFilter):
                initial_z = self.target.atoms.positions[:, 2].copy()
                positions[-1, :2] = [0.24, -0.17]
                positions[0, 2] += 0.2
            else:
                positions[-3:] += self.target.exp_cell_factor * np.array(
                    [[0.01, 0.05, 0], [0.05, -0.02, 0], [0, 0, 0.1]]
                )
            self.target.set_positions(positions)
            return True

    monkeypatch.setattr(calc_mod, "LBFGS", RecordingOptimizer)
    opt = GeometryOpt(fix_z=False, calculator=EMT())
    assert opt.run_preopt("cof", fix_z=fix_z, topo=topo)
    assert opt._fix_z is False
    prepared = read(folder / "cof_preopt.cif")
    assert isinstance(prepared, Atoms)
    assert prepared.cell.angles() == pytest.approx([90, 90, gamma])
    assert prepared.cell.lengths()[2] == pytest.approx(15)
    np.testing.assert_allclose(
        prepared.cell.lengths()[:2],
        original_lengths * np.exp(np.array([0.24, -0.17]) / len(preopt_atoms)),
    )
    assert initial_z is not None
    expected_z = initial_z.copy()
    if not fix_z:
        expected_z[0] += 0.2
    np.testing.assert_allclose(prepared.positions[:, 2], expected_z)
    assert source.read_bytes() == original_bytes

    output = tmp_path / "full_opt.cif"
    assert opt.optimize_cof(str(folder / "cof_preopt.cif"), str(output))
    assert isinstance(targets[1], calc_mod.FrechetCellFilter)
    assert limits == [(0.03, 2000), (0.01, 2000)]
    assert not targets[1].atoms.constraints
    relaxed = read(output)
    assert isinstance(relaxed, Atoms)
    assert relaxed.cell.lengths()[2] != pytest.approx(15)
    assert relaxed.cell.angles()[2] != pytest.approx(gamma)


@pytest.mark.unit
def test_preopt_failures_preserve_optimizer_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preopt_atoms: Atoms
) -> None:
    """Reject invalid requests and restore fix_z after an unsuccessful call."""
    monkeypatch.chdir(tmp_path)
    opt = GeometryOpt(fix_z=False, calculator=EMT())
    with pytest.raises(
        ValueError, match="Unsupported preoptimization topology"
    ):
        opt.run_preopt("cof", topo="dia")
    preopt_atoms.set_cell(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="nondegenerate cell"):
        calc_mod._prepare_preopt_cell(preopt_atoms, "hcb")

    def fail_read(_path: str) -> None:
        raise RuntimeError("input read failed")

    monkeypatch.setattr(calc_mod, "read", fail_read)
    with pytest.raises(RuntimeError, match="input read failed"):
        opt.run_preopt("cof", fix_z=True, topo="hcb")
    assert opt._fix_z is False
    with pytest.raises(ValueError, match="cannot be combined"):
        opt._optimize_cof(
            "in.cif", "out.cif", fixed_cell=True, preopt_topo="hcb"
        )


@pytest.mark.unit
@pytest.mark.parametrize("workflow", ["script", "notebook"])
@pytest.mark.parametrize("topo", ["sql", "kgm"])
def test_standard_workflows_pass_preopt_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, topo: str
) -> None:
    """Run each standard preopt entry point without loading a MACE model."""
    root = Path(__file__).resolve().parents[1]
    seen: dict[str, object] = {}

    class PreoptReachedError(Exception):
        pass

    def record_preopt(self, cof_name: str, **kwargs: object) -> None:
        assert isinstance(self.calc, EMT)
        seen.update(cof_name=cof_name, **kwargs)
        raise PreoptReachedError

    monkeypatch.setattr(GeometryOpt, "run_preopt", record_preopt)
    monkeypatch.setattr(cl.BuildCOF2D, "build", lambda *_args, **_kwargs: [])
    if workflow == "script":
        driver = runpy.run_path(str(root / "example/cof-landscaper.py"))
        params = json.loads(
            (root / "example/cof-landscaper.params.json").read_text()
        )
        params.update(
            TOPOLOGY=topo,
            COF_NAME="workflow-test",
            CALCULATOR="emt",
            CALCULATOR_KWARGS={},
        )
        with pytest.raises(PreoptReachedError):
            driver["run_workflow"](params)
    else:
        notebook = json.loads(
            (root / "example/cof-landscaper_local.ipynb").read_text()
        )
        cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "run_preopt(" in "".join(cell["source"])
        ]
        assert len(cells) == 1
        cell_script = tmp_path / "preopt_cell.py"
        cell_script.write_text(cells[0])
        with pytest.raises(PreoptReachedError):
            runpy.run_path(
                str(cell_script),
                init_globals={
                    "cl": cl,
                    "COF_NAME": "workflow-test",
                    "TOPOLOGY": topo,
                    "CALCULATOR": EMT,
                    "CALCULATOR_KWARGS": {},
                },
            )
    assert seen["cof_name"] == "workflow-test"
    assert seen["topo"] == topo


@pytest.mark.unit
def test_geometry_opt3d_defaults_without_model_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 3D wrapper retains MACE defaults and always enables z relaxation."""
    received = {}

    def fake_preset(**kwargs):
        received.update(kwargs)
        return EMT()

    monkeypatch.setattr(calc_mod, "_default_mace", fake_preset)
    opt = cl.GeometryOpt3D()
    assert isinstance(opt.calc, EMT)
    assert opt._fix_z is False
    assert opt._fmax == 0.01
    assert opt._max_steps == 2000
    assert received == {}


@pytest.mark.unit
def test_geometry_opt3d_relaxes_atoms_and_full_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default 3D routing uses full-cell optimization without z constraints."""
    monkeypatch.chdir(tmp_path)
    source = Path("cof-a/1_cof-a_framework/cof-a_unopt.cif")
    source.parent.mkdir(parents=True)
    source.write_text("data_dummy\n", encoding="utf-8")
    atoms = Atoms("C", positions=[[1, 2, 3]], cell=[10, 11, 12], pbc=True)
    seen: dict[str, object] = {}
    filter_target = object()
    module = importlib.import_module(cl.GeometryOpt.__module__)

    def fake_filter(target: Atoms) -> object:
        seen["filter_atoms"] = target
        assert target.constraints == []
        return filter_target

    class FakeOptimizer:
        def __init__(self, target: object) -> None:
            seen["optimizer_target"] = target

        def run(self, fmax: float, steps: int) -> bool:
            seen["fmax"] = fmax
            seen["steps"] = steps
            atoms.positions[:, 2] += 1
            atoms.set_cell([10, 11, 13])
            return True

    monkeypatch.setattr(module, "read", lambda _path: atoms)
    monkeypatch.setattr(module, "FrechetCellFilter", fake_filter)
    monkeypatch.setattr(module, "LBFGS", FakeOptimizer)
    converged = cl.GeometryOpt3D(fmax=0.02, max_steps=7, calculator=EMT()).run(
        "cof-a"
    )
    assert converged is True
    assert seen == {
        "filter_atoms": atoms,
        "optimizer_target": filter_target,
        "fmax": 0.02,
        "steps": 7,
    }
    output = Path("cof-a/2_cof-a_optimization/cof-a_opt.cif")
    written = read(output)
    assert isinstance(written, Atoms)
    assert written.cell.cellpar() == pytest.approx([10, 11, 13, 90, 90, 90])
    assert written.positions[0, 2] == pytest.approx(4.0)


@pytest.mark.unit
def test_geometry_opt3d_custom_routing_and_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit routes preserve convergence and missing CIFs fail before optimization."""
    opt = cl.GeometryOpt3D(calculator=EMT())
    source = tmp_path / "source.cif"
    source.write_text("data_dummy\n", encoding="utf-8")
    output = tmp_path / "results"
    calls: list[tuple[str, str, bool]] = []

    def fake_optimize(input_path: str, output_path: str) -> bool:
        calls.append((input_path, output_path, opt._fix_z))
        return False

    monkeypatch.setattr(opt, "optimize_cof", fake_optimize)
    assert opt.run("cof-a", input_cif=source, output_folder=output) is False
    assert calls == [(str(source), str(output / "cof-a_opt.cif"), False)]
    with pytest.raises(FileNotFoundError, match=r"missing\.cif"):
        opt.run(
            "cof-a", input_cif=tmp_path / "missing.cif", output_folder=output
        )
    assert len(calls) == 1


@pytest.mark.parametrize("selector", [EMT, "emt", "ase.calculators.emt:EMT"])
def test_constructed_selector_matches_ase(selector) -> None:
    opts = {"asap_cutoff": True}
    sp = SinglePoint(calculator=selector, calculator_kwargs=opts)
    atoms = Atoms("Cu", cell=[3, 3, 3], pbc=True)
    reference = atoms.copy()
    reference.calc = EMT(**opts)
    assert sp.energy(atoms) == pytest.approx(reference.get_potential_energy())
    assert sp.calc is atoms.calc
    assert isinstance(sp.calc, EMT)
    assert sp.calc.parameters["asap_cutoff"] is True


def test_factory_and_instance_reused_for_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    calls = []
    backend = EMT()
    marker = object()

    def factory(**kwargs):
        calls.append(kwargs)
        return backend

    atoms = Atoms("Cu", cell=[3, 3, 3], pbc=True)
    for mode in ("serr", "incl"):
        folder = Path("cu/2_cu_matrix") / mode
        folder.mkdir(parents=True)
        write(folder / "cu_z30_L000.cif", atoms)
    sp = SinglePoint(calculator=factory, calculator_kwargs={"custom": marker})
    sp.run("cu", mode="both")
    assert calls == [{"custom": marker}]
    energy = sp.energy(atoms)
    for csv in ("serr", "incl"):
        data = pd.read_csv(f"cu/3_cu_landscape/cu_sp_energies_{csv}.csv")
        assert data.energy_eV.iloc[0] == pytest.approx(energy)
        assert data.energy_rel_eV.iloc[0] == 0
        assert list(data.columns) == [
            "structure",
            "z",
            "L",
            "energy_eV",
            "energy_rel_eV",
        ]
    assert GeometryOpt(calculator=sp.calc).single_point.calc is backend
    with pytest.raises(ValueError, match="existing calculator"):
        SinglePoint(calculator=backend, calculator_kwargs={"bad": 1})


def test_energy_only_and_missing_stress(tmp_path: Path) -> None:
    from ase.calculators.calculator import PropertyNotImplementedError

    class EnergyOnly:
        def get_potential_energy(self, atoms):
            return 1.0 * len(atoms)

    atoms = Atoms("Cu", cell=[3, 3, 3], pbc=True)
    source = tmp_path / "cu.cif"
    write(source, atoms)
    assert cl.SinglePoint(calculator=EnergyOnly()).energy(atoms) == 1.0
    with pytest.raises(
        PropertyNotImplementedError, match=r"forces.*EnergyOnly"
    ):
        GeometryOpt(calculator=EnergyOnly()).optimize_cof(
            str(source), str(tmp_path / "opt.cif")
        )

    class NoStress(EMT):
        implemented_properties = ("energy", "forces")

    with pytest.raises(PropertyNotImplementedError, match=r"stress.*NoStress"):
        GeometryOpt(calculator=NoStress()).run_preopt(
            "cu",
            topo="sql",
            input_path=str(source),
            output_path=str(tmp_path / "opt.cif"),
        )


def test_json_settings_and_invalid_selections() -> None:
    options = cl.utilities.get_calculator_options(
        {"CALCULATOR": "emt", "CALCULATOR_KWARGS": {"asap_cutoff": True}}
    )
    calc = GeometryOpt(**options).calc
    assert isinstance(calc, EMT)
    assert calc.parameters["asap_cutoff"] is True
    assert cl.utilities.get_calculator_options({}) == {
        "calculator": None,
        "calculator_kwargs": {},
    }
    with pytest.raises(TypeError, match="JSON object"):
        cl.utilities.get_calculator_options({"CALCULATOR_KWARGS": "cpu"})
    with pytest.raises(ValueError, match="MACE_HEAD"):
        cl.utilities.get_calculator_options({"MACE_HEAD": "omol"})
    with pytest.raises(TypeError, match="factory must return"):
        SinglePoint(calculator="builtins:object")


def test_partial_sp_and_stale_optimisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = Path("custom")
    folder.mkdir()
    write(
        folder / "c_z30_L000_incl.cif", Atoms("Cu", cell=[5, 5, 5], pbc=True)
    )
    sp = SinglePoint(calculator=EMT())
    sp.run("named", "incl", input_folder=str(folder))
    table = Path("named/3_named_landscape/named_sp_energies_incl.csv")
    original = table.read_bytes()
    (folder / "bad.cif").write_text("incomplete")
    with pytest.raises(RuntimeError, match="keeping existing"):
        sp.run("named", "incl", input_folder=str(folder))
    assert table.read_bytes() == original
    out = Path("optimised")
    out.mkdir()
    stale = out / "previous.cif"
    stale.write_text("previous run")
    with pytest.raises(ValueError, match="stale CIFs"):
        GeometryOpt(calculator=EMT()).process_cifs(str(folder), str(out))
    assert stale.read_text() == "previous run"
