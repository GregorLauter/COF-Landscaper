from pathlib import Path

import pandas as pd
import pytest

import coflandscaper._internal.mace as mace_mod
from coflandscaper._internal.mace import MaceFullOpt, OptMACE


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
    monkeypatch.setattr(mace_mod, "UnitCellFilter", lambda atoms: atoms)
    monkeypatch.setattr(mace_mod, "LBFGS", _DummyLBFGS)

    opt = object.__new__(OptMACE)
    opt.fix_z = False
    opt.fmax = 0.01
    opt.max_steps = 500
    opt.calc = object()

    with pytest.warns(UserWarning, match="did not converge within 500 steps"):
        converged = opt.optimize_cof(str(input_cif), str(output_cif))

    assert converged is False
    assert _DummyLBFGS.last_fmax == 0.01
    assert _DummyLBFGS.last_steps == 500
    assert dummy_atoms.written_to == str(output_cif)


@pytest.mark.unit
def test_mace_fullopt_energy_csv_replaces_only_processed_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures rerunning one mode preserves other mode rows in the combined CSV."""
    output_base = tmp_path / "cof-1" / "4_cof-1_optimization"
    incl_folder = output_base / "incl"
    serr_folder = output_base / "serr"
    incl_folder.mkdir(parents=True)
    serr_folder.mkdir(parents=True)

    (incl_folder / "cof-1_a_incl.cif").write_text(
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
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(mace_mod, "read", fake_read)

    opt = object.__new__(MaceFullOpt)
    opt.calc = object()

    out_csv = opt._write_optimized_energy_csv(
        cof_name="cof-1",
        mode_output_folders={"incl": incl_folder},
        output_base_path=output_base,
        convergence_map={("incl", "cof-1_a_incl"): False},
    )

    assert out_csv == csv_path
    written = pd.read_csv(csv_path)

    keys = set(
        zip(written["stacking_mode"], written["structure"], strict=False)
    )
    assert ("serr", "cof-1_old_serr") in keys
    assert ("incl", "cof-1_a_incl") in keys
    assert ("incl", "cof-1_old_incl") not in keys

    stopped_map = {
        (row["stacking_mode"], row["structure"]): str(
            row["stopped_due_to_max_steps"]
        ).lower()
        in {"true", "1"}
        for _, row in written.iterrows()
    }
    assert stopped_map[("incl", "cof-1_a_incl")] is True
    assert stopped_map[("serr", "cof-1_old_serr")] is False
