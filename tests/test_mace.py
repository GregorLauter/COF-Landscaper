import importlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cell import Cell
from ase.constraints import FixCartesian
from ase.io import read, write

import coflandscaper as cl

mace_mod = importlib.import_module(cl.Mace.__module__)
MaceSP = cl.MaceSP
MaceOpt = cl.MaceOpt


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
def test_mace_sp_run_routes_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures the public MaceSP.run API preserves mode routing."""
    calls: list[tuple[str, str | None]] = []

    def fake_run_folder(
        _self: MaceSP,
        input_folder: str,
        output_csv_dir: str | None = None,
    ) -> None:
        calls.append((input_folder, output_csv_dir))

    monkeypatch.setattr(MaceSP, "_run_folder", fake_run_folder)
    MaceSP(verbose=False).run(
        cof_name="cof-a",
        mode="both",
        output_csv_dir="results",
    )

    assert calls == [
        ("cof-a/2_cof-a_matrix/serr", "results"),
        ("cof-a/2_cof-a_matrix/incl", "results"),
    ]


@pytest.mark.unit
def test_optimize_cof_respects_max_steps_and_warns_when_not_converged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures max step limit is forwarded and non-convergence does not abort writing."""
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

    monkeypatch.setattr(mace_mod, "read", lambda _p: dummy_atoms)
    monkeypatch.setattr(mace_mod, "FrechetCellFilter", lambda atoms: atoms)
    monkeypatch.setattr(mace_mod, "LBFGS", _DummyLBFGS)

    def fake_make_calc(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(MaceOpt, "_make_calc", fake_make_calc)
    opt = MaceOpt(fmax=0.01, max_steps=500, fix_z=False, verbose=False)

    with pytest.warns(UserWarning, match="did not converge within 500 steps"):
        converged = opt.optimize_cof(str(input_cif), str(output_cif))

    assert converged is False
    assert _DummyLBFGS.last_fmax == 0.01
    assert _DummyLBFGS.last_steps == 500
    assert dummy_atoms.written_to == str(output_cif)


@pytest.mark.unit
def test_calculator_settings_for_matpes_r2scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_mace_mp(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(mace_mod, "mace_mp", fake_mace_mp)

    base = object.__new__(mace_mod.Mace)
    base._verbose = False

    calc_settings = cl.calculator_settings_for_head("matpes_r2scan")
    base._make_calc(
        device="cpu",
        dtype="float64",
        model="mh-1",
        calc_settings=calc_settings,
    )

    assert captured["model"] == "mh-1"
    assert captured["default_dtype"] == "float64"
    assert captured["device"] == "cpu"
    assert captured["head"] == "matpes_r2scan"
    assert captured["dispersion"] is True
    assert captured["dispersion_xc"] == "r2scan"
    assert captured["dispersion_cutoff"] == 40.0


@pytest.mark.unit
def test_mace_opt_energy_csv_replaces_only_processed_mode(
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

    monkeypatch.setattr(mace_mod, "read", fake_read)

    opt = object.__new__(MaceOpt)
    opt.calc = object()

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

    def fake_optimize(self, input_path: str, output_path: str) -> bool:
        seen["input_path"] = input_path
        seen["output_path"] = output_path
        seen["fix_z_during_call"] = self._fix_z
        return True

    def fake_make_calc(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(MaceOpt, "_make_calc", fake_make_calc)
    opt = MaceOpt(fix_z=False, verbose=False)
    opt.optimize_cof = fake_optimize.__get__(opt, MaceOpt)

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

    def fake_optimize(self, _input_path: str, _output_path: str) -> bool:
        seen["fix_z_during_call"] = self._fix_z
        return True

    def fake_make_calc(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(MaceOpt, "_make_calc", fake_make_calc)
    opt = MaceOpt(fix_z=True, verbose=False)
    opt.optimize_cof = fake_optimize.__get__(opt, MaceOpt)

    monkeypatch.chdir(tmp_path)

    converged = opt.run_preopt("COF-1", fix_z=False)

    assert converged is True
    assert seen["fix_z_during_call"] is False
    assert opt._fix_z is True


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

    monkeypatch.setattr(mace_mod, "read", lambda _path: atoms)
    monkeypatch.setattr(mace_mod, "FrechetCellFilter", fake_filter)
    monkeypatch.setattr(mace_mod, "LBFGS", _DummyOptimizer)
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = MaceOpt(fix_z=False, verbose=False)
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

    monkeypatch.setattr(mace_mod, "read", lambda _path: atoms)
    monkeypatch.setattr(mace_mod, "FrechetCellFilter", fail_filter)
    monkeypatch.setattr(mace_mod, "LBFGS", _DummyOptimizer)
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    input_dir = tmp_path / "cof-a" / "6_cof-a_scaling" / "scaling" / "serr"
    input_dir.mkdir(parents=True)
    (input_dir / "scaled.cif").write_text("data_dummy\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    opt = MaceOpt(fix_z=False, verbose=False)
    assert opt.run_fixed("cof-a", "serr") == {"serr/scaled": True}
    assert seen["target"] is atoms


@pytest.mark.unit
def test_mace_opt_run_routes_both_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures the renamed MaceOpt.run preserves normal routing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = MaceOpt(verbose=False)
    calls: list[tuple[str, str]] = []

    def record_process(
        input_folder: str, output_folder: str
    ) -> dict[str, bool]:
        calls.append((input_folder, output_folder))
        return {}

    monkeypatch.setattr(opt, "process_cifs", record_process)
    opt.run(cof_name="cof-a", mode="both", save_opt_energies_csv=False)

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
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = MaceOpt(verbose=False)
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
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = MaceOpt(verbose=False)
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
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = MaceOpt(verbose=False)
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
    mace_mod._prepare_preopt_cell(preopt_atoms, topo)
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
    mace_mod._prepare_preopt_cell(preopt_atoms, topo)
    if fix_z:
        preopt_atoms.set_constraint(
            FixCartesian(range(len(preopt_atoms)), mask=[False, False, True])
        )
    preopt_atoms.calc = EMT()
    cell_filter = mace_mod._FixedAngleCellFilter(preopt_atoms)
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
    initial_z: np.ndarray | None = None

    class RecordingOptimizer:
        def __init__(self, target) -> None:
            self.target = target
            targets.append(target)

        def run(self, **_kwargs: object) -> bool:
            nonlocal initial_z
            positions = self.target.get_positions()
            if isinstance(self.target, mace_mod._FixedAngleCellFilter):
                initial_z = self.target.atoms.positions[:, 2].copy()
                positions[-1, :2] = [0.24, -0.17]
                positions[0, 2] += 0.2
            else:
                positions[-3:] += self.target.exp_cell_factor * np.array(
                    [[0.01, 0.05, 0], [0.05, -0.02, 0], [0, 0, 0.1]]
                )
            self.target.set_positions(positions)
            return True

    monkeypatch.setattr(mace_mod, "LBFGS", RecordingOptimizer)
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = MaceOpt(fix_z=False, verbose=False)
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
    assert isinstance(targets[1], mace_mod.FrechetCellFilter)
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
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = MaceOpt(fix_z=False, verbose=False)
    with pytest.raises(
        ValueError, match="Unsupported preoptimization topology"
    ):
        opt.run_preopt("cof", topo="dia")
    preopt_atoms.set_cell(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="nondegenerate cell"):
        mace_mod._prepare_preopt_cell(preopt_atoms, "hcb")

    def fail_read(_path: str) -> None:
        raise RuntimeError("input read failed")

    monkeypatch.setattr(mace_mod, "read", fail_read)
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
        _ = self
        seen.update(cof_name=cof_name, **kwargs)
        raise PreoptReachedError

    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(MaceOpt, "run_preopt", record_preopt)
    monkeypatch.setattr(cl.BuildCOF2D, "build", lambda *_args, **_kwargs: [])
    if workflow == "script":
        driver = runpy.run_path(str(root / "example/cof-landscaper.py"))
        params = json.loads(
            (root / "example/cof-landscaper.params.json").read_text()
        )
        params.update(TOPOLOGY=topo, COF_NAME="workflow-test")
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
                },
            )
    assert seen["cof_name"] == "workflow-test"
    assert seen["topo"] == topo


@pytest.mark.unit
def test_mace_opt3d_defaults_without_model_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 3D wrapper retains MACE defaults and always enables z relaxation."""
    calculator_settings: dict[str, object] = {}

    def fake_make_calc(_self, **kwargs) -> object:
        calculator_settings.update(kwargs)
        return object()

    monkeypatch.setattr(cl.MaceOpt, "_make_calc", fake_make_calc)
    opt = cl.MaceOpt3D(verbose=False)
    assert opt._fix_z is False
    assert opt._fmax == 0.01
    assert opt._max_steps == 2000
    assert calculator_settings["model"] == "mh-1"
    assert calculator_settings["dtype"] == "float64"
    assert calculator_settings["device"] == "cpu"
    settings = calculator_settings["calc_settings"]
    assert isinstance(settings, dict)
    assert settings["head"] == "spice_wB97M"


@pytest.mark.unit
def test_mace_opt3d_relaxes_atoms_and_full_cell(
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
    module = importlib.import_module(cl.MaceOpt.__module__)

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
    monkeypatch.setattr(
        cl.MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    converged = cl.MaceOpt3D(fmax=0.02, max_steps=7, verbose=False).run(
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
def test_mace_opt3d_custom_routing_and_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit routes preserve convergence and missing CIFs fail before optimization."""
    monkeypatch.setattr(
        cl.MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = cl.MaceOpt3D(verbose=False)
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
