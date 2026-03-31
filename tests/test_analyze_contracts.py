import csv
from pathlib import Path

import pytest

from coflandscaper import Analyze


def test_resolve_modes_accepts_supported_values() -> None:
    """This test ensures mode parsing stays stable for valid CLI/API inputs."""
    analyzer = Analyze()

    assert analyzer._resolve_modes("incl") == ["incl"]
    assert analyzer._resolve_modes("serr") == ["serr"]
    assert analyzer._resolve_modes("both") == ["serr", "incl"]


def test_resolve_modes_rejects_invalid_mode() -> None:
    """This test ensures invalid mode values fail fast with a clear error."""
    analyzer = Analyze()

    with pytest.raises(
        ValueError, match="mode must be 'incl', 'serr', or 'both'"
    ):
        analyzer._resolve_modes("bad")


def test_load_energy_map_skips_malformed_rows(tmp_path: Path) -> None:
    """This test ensures CSV parsing keeps only valid rows used by downstream analysis."""
    analyzer = Analyze()
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
    analyzer = Analyze()

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
    analyzer = Analyze()
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
