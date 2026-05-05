from pathlib import Path

import pandas as pd
import pytest

import coflandscaper as cl


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (
            [
                {"z": 3.4, "L": 2.0, "energy_eV": -70446.719266},
                {"z": 3.0, "L": 0.0, "energy_eV": -70443.344493},
                {"z": 3.0, "L": 1.0, "energy_eV": -70444.846268},
            ],
            [(3.4, 2.0)],
        ),
        (
            [
                {"z": 3.0, "L": 0.0, "energy_eV": -70443.344493},
                {"z": 3.4, "L": 2.0, "energy_eV": -70446.719266},
                {"z": 3.0, "L": 1.0, "energy_eV": -70444.846268},
            ],
            [(3.4, 2.0)],
        ),
        (
            [
                {"z": 3.0, "L": 0.0, "energy_eV": -70443.344493},
                {"z": 3.0, "L": 1.0, "energy_eV": -70444.846268},
                {"z": 3.4, "L": 2.0, "energy_eV": -70446.719266},
            ],
            [(3.4, 2.0)],
        ),
    ],
)
def test_global_minima_returns_lowest_energy_pair_independent_of_row_order(
    tmp_path: Path,
    rows: list[dict[str, float]],
    expected: list[tuple[float, float]],
) -> None:
    csv_path = tmp_path / "cof-1_sp_energies_serr.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    selections = cl.SelectCofs()._global_minima_from_csv(csv_path)

    assert selections == expected


@pytest.mark.unit
def test_run_copies_matching_cif_for_selected_pair(tmp_path: Path) -> None:
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    input_folder.mkdir()

    (input_folder / "dummy_z030_L000_serr.cif").write_text(
        "data_030_000\n",
        encoding="utf-8",
    )
    (input_folder / "dummy_z034_L020_serr.cif").write_text(
        "data_034_020\n",
        encoding="utf-8",
    )
    (input_folder / "dummy_z040_L040_serr.cif").write_text(
        "data_040_040\n",
        encoding="utf-8",
    )

    cl.SelectCofs().run(
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        selections=[(3.4, 2.0)],
    )

    copied_files = sorted(path.name for path in output_folder.glob("*.cif"))

    assert copied_files == ["dummy_z034_L020_serr.cif"]


@pytest.mark.unit
def test_run_raises_when_selected_pair_has_no_matching_cif(
    tmp_path: Path,
) -> None:
    input_folder = tmp_path / "input"
    output_folder = tmp_path / "output"
    input_folder.mkdir()

    (input_folder / "dummy_z030_L000_serr.cif").write_text(
        "data_030_000\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="No matching CIFs found"):
        cl.SelectCofs().run(
            input_folder=str(input_folder),
            output_folder=str(output_folder),
            selections=[(3.4, 2.0)],
        )


@pytest.mark.unit
def test_run_mode_global_autoselect_copies_global_minimum_cif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    cof_name = "cof-1"

    csv_dir = tmp_path / cof_name / f"3_{cof_name}_landscape"
    csv_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"z": 3.0, "L": 0.0, "energy_eV": -70443.344493},
            {"z": 3.4, "L": 2.0, "energy_eV": -70446.719266},
            {"z": 4.0, "L": 4.0, "energy_eV": -70445.000000},
        ]
    ).to_csv(csv_dir / f"{cof_name}_sp_energies_serr.csv", index=False)

    input_folder = tmp_path / cof_name / f"2_{cof_name}_matrix" / "serr"
    input_folder.mkdir(parents=True)
    (input_folder / "dummy_z030_L000_serr.cif").write_text(
        "data_030_000\n",
        encoding="utf-8",
    )
    (input_folder / "dummy_z034_L020_serr.cif").write_text(
        "data_034_020\n",
        encoding="utf-8",
    )
    (input_folder / "dummy_z040_L040_serr.cif").write_text(
        "data_040_040\n",
        encoding="utf-8",
    )

    output_folder = tmp_path / "selected"

    cl.SelectCofs().run_mode(
        cof_name=cof_name,
        mode="serr",
        include_autoselect=True,
        autoselect_minima="global",
        input_folder=str(input_folder),
        output_folder=str(output_folder),
    )

    copied_files = sorted(path.name for path in output_folder.glob("*.cif"))

    assert copied_files == ["dummy_z034_L020_serr.cif"]


@pytest.mark.unit
def test_run_mode_manual_selection_works_without_autoselect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    cof_name = "cof-1"
    input_folder = tmp_path / cof_name / f"2_{cof_name}_matrix" / "serr"
    input_folder.mkdir(parents=True)
    (input_folder / "dummy_z034_L020_serr.cif").write_text(
        "data_034_020\n",
        encoding="utf-8",
    )

    output_folder = tmp_path / "selected"

    cl.SelectCofs().run_mode(
        cof_name=cof_name,
        mode="serr",
        include_autoselect=False,
        selections_serr=[(3.4, 2.0)],
        input_folder=str(input_folder),
        output_folder=str(output_folder),
    )

    copied_files = sorted(path.name for path in output_folder.glob("*.cif"))

    assert copied_files == ["dummy_z034_L020_serr.cif"]


@pytest.mark.unit
def test_run_mode_without_autoselect_or_manual_selections_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    cof_name = "cof-1"
    input_folder = tmp_path / cof_name / f"2_{cof_name}_matrix" / "serr"
    input_folder.mkdir(parents=True)

    with pytest.raises(ValueError, match="No selections provided"):
        cl.SelectCofs().run_mode(
            cof_name=cof_name,
            mode="serr",
            include_autoselect=False,
            input_folder=str(input_folder),
            output_folder=str(tmp_path / "selected"),
        )


@pytest.mark.unit
def test_run_mode_both_routes_serr_and_incl_selections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    calls: list[
        tuple[str, str, list[tuple[float, float]] | None, str | None]
    ] = []

    def fake_run(
        _self: cl.SelectCofs,
        input_folder: str,
        output_folder: str,
        selections: list[tuple[float, float]] | None = None,
        mode_label: str | None = None,
    ) -> None:
        calls.append((input_folder, output_folder, selections, mode_label))

    monkeypatch.setattr(cl.SelectCofs, "run", fake_run)

    cl.SelectCofs().run_mode(
        cof_name="cof-1",
        mode="both",
        include_autoselect=False,
        selections_serr=[(3.4, 2.0)],
        selections_incl=[(3.5, 1.0)],
        input_base=str(tmp_path / "input_base"),
        output_base=str(tmp_path / "output_base"),
    )

    assert len(calls) == 2

    serr_input, serr_output, serr_selections, serr_label = calls[0]
    assert set(serr_selections or []) == {(3.4, 2.0)}
    assert serr_label == "Serrated"
    assert serr_input.endswith("/serr")
    assert serr_output.endswith("/serr")

    incl_input, incl_output, incl_selections, incl_label = calls[1]
    assert set(incl_selections or []) == {(3.5, 1.0)}
    assert incl_label == "Inclined"
    assert incl_input.endswith("/incl")
    assert incl_output.endswith("/incl")
