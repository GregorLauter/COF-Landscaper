from pathlib import Path

import pandas as pd
import pytest

from coflandscaper._internal.landscape import SelectCofs


@pytest.mark.unit
def test_global_minima_returns_single_deterministic_pair(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "cof-1_sp_energies_serr.csv"
    pd.DataFrame(
        [
            {"z": 3.1, "L": 0.2, "energy_eV": -10.0},
            {"z": 3.0, "L": 0.1, "energy_eV": -10.0},
            {"z": 3.2, "L": 0.3, "energy_eV": -9.5},
        ]
    ).to_csv(csv_path, index=False)

    selections = SelectCofs()._global_minima_from_csv(csv_path)

    assert selections == [(3.0, 0.1)]


@pytest.mark.unit
def test_run_mode_autoselect_mode_routes_global_vs_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    cof_name = "cof-1"
    csv_dir = tmp_path / cof_name / f"3_{cof_name}_landscape"
    csv_dir.mkdir(parents=True)
    (csv_dir / f"{cof_name}_sp_energies_serr.csv").write_text(
        "z,L,energy_eV\n3.0,0.0,-1.0\n",
        encoding="utf-8",
    )

    input_folder = tmp_path / cof_name / f"2_{cof_name}_matrix" / "serr"
    input_folder.mkdir(parents=True)
    (input_folder / "dummy_z030_L000_serr.cif").write_text(
        "data_dummy\n",
        encoding="utf-8",
    )

    used_selections: list[list[tuple[float, float]]] = []

    def fake_run(
        _self: SelectCofs,
        input_folder: str,
        output_folder: str,
        selections: list[tuple[float, float]] | None = None,
        mode_label: str | None = None,
    ) -> None:
        del input_folder, output_folder, mode_label
        used_selections.append(list(selections or []))

    monkeypatch.setattr(SelectCofs, "run", fake_run)
    monkeypatch.setattr(
        SelectCofs,
        "_global_minima_from_csv",
        lambda _self, _csv: [(3.0, 0.0)],
    )
    monkeypatch.setattr(
        SelectCofs,
        "_local_minima_from_csv",
        lambda _self, _csv: [(3.0, 0.0), (3.2, 0.5)],
    )

    selector = SelectCofs()
    selector.run_mode(
        cof_name=cof_name,
        mode="serr",
        include_autoselect=True,
        autoselect_minima="global",
        input_folder=str(input_folder),
        output_folder=str(tmp_path / "out_global"),
    )
    selector.run_mode(
        cof_name=cof_name,
        mode="serr",
        include_autoselect=True,
        autoselect_minima="local",
        input_folder=str(input_folder),
        output_folder=str(tmp_path / "out_local"),
    )

    assert used_selections[0] == [(3.0, 0.0)]
    assert used_selections[1] == [(3.0, 0.0), (3.2, 0.5)]
