import importlib
from pathlib import Path

import pandas as pd
import pytest

import coflandscaper as cl

mace_mod = importlib.import_module(cl.Mace.__module__)
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
def test_mace_opt_validates_cell_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """This test ensures MaceOpt accepts supported cell modes and rejects invalid ones."""
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    assert MaceOpt(cell_mode="full", verbose=False)._cell_mode == "full"
    assert (
        MaceOpt(cell_mode="out_of_plane", verbose=False)._cell_mode
        == "out_of_plane"
    )
    with pytest.raises(ValueError, match="cell_mode must be"):
        MaceOpt(cell_mode="invalid", verbose=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cell_mode", "expected_mask"),
    [
        ("full", None),
        ("out_of_plane", [False, False, True, False, False, False]),
    ],
)
def test_optimize_cof_uses_requested_cell_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell_mode: str,
    expected_mask: list[bool] | None,
) -> None:
    """This test ensures cell mode selects the expected FrechetCellFilter mask."""
    seen: dict[str, object] = {}
    atoms = _DummyAtoms()

    class _DummyOptimizer:
        def __init__(self, _filter: object) -> None:
            pass

        def run(self, **_kwargs: object) -> bool:
            return True

    def fake_filter(_atoms: object, **kwargs: object) -> object:
        seen["mask"] = kwargs.get("mask")
        return object()

    monkeypatch.setattr(mace_mod, "read", lambda _path: atoms)
    monkeypatch.setattr(mace_mod, "FrechetCellFilter", fake_filter)
    monkeypatch.setattr(mace_mod, "LBFGS", _DummyOptimizer)
    monkeypatch.setattr(
        MaceOpt, "_make_calc", lambda *_args, **_kwargs: object()
    )
    opt = MaceOpt(cell_mode=cell_mode, fix_z=False, verbose=False)
    opt.optimize_cof(str(tmp_path / "in.cif"), str(tmp_path / "out.cif"))
    assert seen["mask"] == expected_mask


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
    monkeypatch.setattr(
        opt,
        "optimize_cof",
        lambda input_path, output_path: (
            seen.update(input=input_path, output=output_path) or True
        ),
    )
    assert opt.run("cof-a", "serr") == {"serr/scaled_1.0237": True}
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
    monkeypatch.setattr(opt, "optimize_cof", lambda *_args: True)
    with pytest.raises(FileNotFoundError, match="No CIF files"):
        opt.run("cof-a", "incl")
    for filename in ["a.cif", "b.cif"]:
        (input_dir / filename).write_text("data_dummy\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple CIF"):
        opt.run("cof-a", "incl")
    assert opt.run("cof-a", "incl", source_cif="b.cif") == {"incl/b": True}


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

    def record_optimization(input_path: str, output_path: str) -> bool:
        seen.append((input_path, output_path))
        return True

    monkeypatch.setattr(
        opt,
        "optimize_cof",
        record_optimization,
    )

    assert opt.run("cof-a", "both") == {
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
