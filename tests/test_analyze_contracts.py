import csv
from pathlib import Path

import pytest

from coflandscaper import AnalyzeStacking


def test_resolve_modes_accepts_supported_values() -> None:
    """This test ensures mode parsing stays stable for valid CLI/API inputs."""
    analyzer = AnalyzeStacking()

    assert analyzer._resolve_modes("incl") == ["incl"]
    assert analyzer._resolve_modes("serr") == ["serr"]
    assert analyzer._resolve_modes("both") == ["serr", "incl"]


def test_resolve_modes_rejects_invalid_mode() -> None:
    """This test ensures invalid mode values fail fast with a clear error."""
    analyzer = AnalyzeStacking()

    with pytest.raises(
        ValueError, match="mode must be 'incl', 'serr', or 'both'"
    ):
        analyzer._resolve_modes("bad")


def test_load_energy_map_skips_malformed_rows(tmp_path: Path) -> None:
    """This test ensures CSV parsing keeps only valid rows used by downstream analysis."""
    analyzer = AnalyzeStacking()
    csv_path = tmp_path / "cof-a_opt_energies_per_layer.csv"
    csv_path.write_text(
        "structure,stacking_mode,energy_eV_per_layer,energy_rel_eV_per_layer\n"
        "good_serr,serr,-10.0,0.0\n"
        "good_incl,incl,-9.9,0.1\n"
        "bad_mode,other,-9.0,0.2\n"
        "missing_structure,,-8.0,0.3\n"
        "bad_float,serr,NaN,nope\n"
    )

    result = analyzer._load_energy_map(
        cof_name="cof-a", input_base_path=tmp_path, dft=False
    )

    assert result == {
        ("serr", "good_serr"): (-10.0, 0.0),
        ("incl", "good_incl"): (-9.9, 0.1),
    }


def test_run_writes_expected_csv_schema_for_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures the exported analysis CSV keeps the expected reviewer-facing schema."""
    analyzer = AnalyzeStacking()

    def fake_collect(_folder: Path) -> list[str]:
        return [str(tmp_path / "mock-a.cif")]

    def fake_metrics(
        _input_file: str, selected_mode: str
    ) -> tuple[float, float]:
        return (1.0, 2.0) if selected_mode == "serr" else (3.0, 4.0)

    def fake_energy_map(**_kwargs):
        return {
            ("serr", "mock-a"): (-10.0, 0.0),
            ("incl", "mock-a"): (-9.0, 1.0),
        }

    monkeypatch.setattr(analyzer, "_collect_cifs", fake_collect)
    monkeypatch.setattr(analyzer, "_compute_metrics", fake_metrics)
    monkeypatch.setattr(analyzer, "_load_energy_map", fake_energy_map)

    out_dir = tmp_path / "analysis"
    analyzer.run(
        cof_name="cof-a",
        mode="both",
        input_base=tmp_path / "in",
        output_base=out_dir,
        print_values=False,
    )

    output_csv = out_dir / "final_structures.csv"
    assert output_csv.exists()

    with output_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "Stacking",
        "filename",
        "ILD",
        "ILS",
        "energy_eV_per_layer",
        "energy_rel_eV_per_layer",
    ]
    assert len(rows) == 2
    assert {row["Stacking"] for row in rows} == {"serr", "incl"}
    assert {row["filename"] for row in rows} == {"mock-a.cif"}


def test_run_dft_mode_uses_dft_folder_and_filename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures DFT analysis writes to the dft-specific output contract."""
    analyzer = AnalyzeStacking()
    seen_folders: list[str] = []

    def fake_collect(folder: Path) -> list[str]:
        seen_folders.append(folder.name)
        return [str(tmp_path / "mock-dft.cif")]

    monkeypatch.setattr(analyzer, "_collect_cifs", fake_collect)
    monkeypatch.setattr(
        analyzer, "_compute_metrics", lambda *_args, **_kwargs: (1.0, 2.0)
    )
    monkeypatch.setattr(analyzer, "_load_energy_map", lambda **_kwargs: {})

    out_dir = tmp_path / "analysis"
    analyzer.run(
        cof_name="cof-a",
        mode="serr",
        input_base=tmp_path / "in",
        output_base=out_dir,
        dft=True,
        print_values=False,
    )

    assert seen_folders == ["dft_serr"]
    assert (out_dir / "final_structures_dft.csv").exists()


def test_run_merges_with_existing_final_structures_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures single-mode reruns update only their rows and preserve others."""
    analyzer = AnalyzeStacking()

    def fake_collect(folder: Path) -> list[str]:
        if folder.name == "serr":
            return [str(tmp_path / "serr_new.cif")]
        return [str(tmp_path / "incl_keep.cif")]

    def fake_metrics(
        input_file: str, _selected_mode: str
    ) -> tuple[float, float]:
        if input_file.endswith("serr_new.cif"):
            return (3.3, 0.7)
        return (4.4, 1.1)

    monkeypatch.setattr(analyzer, "_collect_cifs", fake_collect)
    monkeypatch.setattr(analyzer, "_compute_metrics", fake_metrics)
    monkeypatch.setattr(analyzer, "_load_energy_map", lambda **_kwargs: {})

    out_dir = tmp_path / "analysis"
    out_dir.mkdir(parents=True)
    output_csv = out_dir / "final_structures.csv"
    output_csv.write_text(
        "Stacking,filename,ILD,ILS,energy_eV_per_layer,energy_rel_eV_per_layer\n"
        "serr,serr_old.cif,1.0,2.0,-10.0,0.0\n"
        "incl,incl_keep.cif,5.0,6.0,-9.0,1.0\n",
        encoding="utf-8",
    )

    analyzer.run(
        cof_name="cof-a",
        mode="serr",
        input_base=tmp_path / "in",
        output_base=out_dir,
        print_values=False,
    )

    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_key = {(row["Stacking"], row["filename"]): row for row in rows}

    assert ("incl", "incl_keep.cif") in by_key
    assert ("serr", "serr_new.cif") in by_key
    assert ("serr", "serr_old.cif") not in by_key
    assert by_key[("incl", "incl_keep.cif")]["ILD"] == "5.0"
