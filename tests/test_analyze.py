import csv
from pathlib import Path

import pytest

from coflandscaper import AnalyzeStacking, VisualizeCOF


def test_resolve_modes() -> None:
    """This test ensures mode parsing stays stable for valid CLI/API inputs."""
    analyzer = AnalyzeStacking()

    assert analyzer._resolve_modes("incl") == ["incl"]
    assert analyzer._resolve_modes("serr") == ["serr"]
    assert analyzer._resolve_modes("both") == ["serr", "incl"]

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


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected_input", "expected_output"),
    [
        ("opt", "4_cof-a_optimization", "5_cof-a_analysis"),
        ("postopt", "6_cof-a_scaling/postopt", "7_cof-a_postanalysis"),
    ],
)
def test_analyze_source_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected_input: str,
    expected_output: str,
) -> None:
    """This test ensures analysis routes each supported source to its stage folders."""
    analyzer = AnalyzeStacking()
    seen: dict[str, Path] = {}

    def fake_collect(folder: Path) -> list[str]:
        seen["input"] = folder.parent
        return [str(tmp_path / "sample.cif")]

    monkeypatch.setattr(analyzer, "_collect_cifs", fake_collect)
    monkeypatch.setattr(
        analyzer, "_compute_metrics", lambda *_args: (1.0, 2.0)
    )
    monkeypatch.setattr(analyzer, "_load_energy_map", lambda **_kwargs: {})
    monkeypatch.chdir(tmp_path)
    analyzer.run("cof-a", mode="serr", source=source, print_values=False)
    assert seen["input"] == Path("cof-a") / expected_input
    assert (
        tmp_path / "cof-a" / expected_output / "final_structures.csv"
    ).exists()


@pytest.mark.unit
def test_analyze_explicit_paths_override_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures explicit analysis paths take precedence over source defaults."""
    analyzer = AnalyzeStacking()
    seen: list[Path] = []

    def fake_collect(folder: Path) -> list[str]:
        seen.append(folder)
        return [str(tmp_path / "a.cif")]

    monkeypatch.setattr(
        analyzer,
        "_collect_cifs",
        fake_collect,
    )
    monkeypatch.setattr(
        analyzer, "_compute_metrics", lambda *_args: (1.0, 2.0)
    )
    monkeypatch.setattr(analyzer, "_load_energy_map", lambda **_kwargs: {})
    analyzer.run(
        "cof-a",
        mode="serr",
        source="postopt",
        input_base=tmp_path / "input",
        output_base=tmp_path / "output",
        print_values=False,
    )
    assert seen == [tmp_path / "input" / "serr"]
    assert (tmp_path / "output" / "final_structures.csv").exists()
    with pytest.raises(ValueError, match="source must be"):
        analyzer.run("cof-a", source="invalid")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected_base"),
    [("opt", "4_cof-a_optimization"), ("postopt", "6_cof-a_scaling/postopt")],
)
def test_visualize_cof_source_routing(
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    expected_base: str,
) -> None:
    """This test ensures visualization selects the source-specific input base without rendering."""
    visualizer = VisualizeCOF()
    seen: list[Path] = []

    def fake_collect(_self: AnalyzeStacking, folder: Path) -> list[str]:
        seen.append(folder)
        return []

    monkeypatch.setattr(
        AnalyzeStacking,
        "_collect_cifs",
        fake_collect,
    )
    visualizer.visualize_cof("cof-a", mode="serr", source=source)
    assert seen == [Path("cof-a") / expected_base / "serr"]
    with pytest.raises(ValueError, match="source must be"):
        visualizer.visualize_cof("cof-a", source="invalid")
