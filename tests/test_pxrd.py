import importlib
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read, write

import coflandscaper as cl


@pytest.mark.unit
def test_resolve_modes() -> None:
    """This test ensures PXRD mode parsing accepts supported options and rejects invalid ones."""
    pxrd = cl.PXRD()
    assert pxrd._resolve_modes("incl") == ["incl"]
    assert pxrd._resolve_modes("serr") == ["serr"]
    assert pxrd._resolve_modes("both") == ["serr", "incl"]

    with pytest.raises(ValueError, match="mode must be"):
        pxrd._resolve_modes("bad")


@pytest.mark.unit
def test_run_default_routing_both_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures default PXRD run routing targets expected input and output folders."""
    pxrd = cl.PXRD()
    calls: list[tuple[Path, Path]] = []

    def fake_produce_xy(
        _self: cl.PXRD,
        input_folder: str | Path,
        output_folder: str | Path | None = None,
    ) -> str:
        assert output_folder is not None
        in_path = Path(input_folder)
        out_path = Path(output_folder)
        calls.append((in_path, out_path))
        return str(out_path)

    monkeypatch.setattr(cl.PXRD, "produce_xy", fake_produce_xy)

    outputs = pxrd.run(cof_name="cof-a", mode="both", dft=False)

    assert outputs == {
        "serr": "cof-a/5_cof-a_analysis/pxrd_xy/serr",
        "incl": "cof-a/5_cof-a_analysis/pxrd_xy/incl",
    }
    assert calls == [
        (
            Path("cof-a/4_cof-a_optimization/serr"),
            Path("cof-a/5_cof-a_analysis/pxrd_xy/serr"),
        ),
        (
            Path("cof-a/4_cof-a_optimization/incl"),
            Path("cof-a/5_cof-a_analysis/pxrd_xy/incl"),
        ),
    ]


@pytest.mark.unit
def test_run_custom_parent_folder_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures custom parent folders are respected for DFT PXRD run routing."""
    pxrd = cl.PXRD()
    calls: list[tuple[Path, Path]] = []

    def fake_produce_xy(
        _self: cl.PXRD,
        input_folder: str | Path,
        output_folder: str | Path | None = None,
    ) -> str:
        assert output_folder is not None
        in_path = Path(input_folder)
        out_path = Path(output_folder)
        calls.append((in_path, out_path))
        return str(out_path)

    monkeypatch.setattr(cl.PXRD, "produce_xy", fake_produce_xy)

    outputs = pxrd.run(
        cof_name="cof-a",
        mode="both",
        dft=True,
        input_folder="my_inputs",
        output_folder="my_outputs",
    )

    assert outputs == {
        "serr": "my_outputs/serr",
        "incl": "my_outputs/incl",
    }
    assert calls == [
        (Path("my_inputs/dft_serr"), Path("my_outputs/serr")),
        (Path("my_inputs/dft_incl"), Path("my_outputs/incl")),
    ]


@pytest.mark.unit
def test_run_single_mode_uses_mode_output_subfolder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures run always appends the selected mode to output roots."""
    pxrd = cl.PXRD()
    calls: list[tuple[Path, Path]] = []

    def fake_produce_xy(
        _self: cl.PXRD,
        input_folder: str | Path,
        output_folder: str | Path | None = None,
    ) -> str:
        assert output_folder is not None
        in_path = Path(input_folder)
        out_path = Path(output_folder)
        calls.append((in_path, out_path))
        return str(out_path)

    monkeypatch.setattr(cl.PXRD, "produce_xy", fake_produce_xy)

    default_outputs = pxrd.run(cof_name="cof-a", mode="incl")
    custom_outputs = pxrd.run(
        cof_name="cof-a",
        mode="incl",
        output_folder="my_outputs",
    )

    assert default_outputs == {"incl": "cof-a/5_cof-a_analysis/pxrd_xy/incl"}
    assert custom_outputs == {"incl": "my_outputs/incl"}
    assert calls == [
        (
            Path("cof-a/4_cof-a_optimization/incl"),
            Path("cof-a/5_cof-a_analysis/pxrd_xy/incl"),
        ),
        (
            Path("cof-a/4_cof-a_optimization/incl"),
            Path("my_outputs/incl"),
        ),
    ]


@pytest.mark.unit
def test_plot_sim_default_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures PXRD plot_sim routes each mode to structure PDFs."""
    monkeypatch.chdir(tmp_path)
    pxrd = cl.PXRD()
    for selected_mode in ("serr", "incl"):
        xy_dir = (
            Path("cof-b") / "5_cof-b_analysis" / "pxrd_xy_dft" / selected_mode
        )
        xy_dir.mkdir(parents=True)
        np.savetxt(xy_dir / "structure-a.xy", [[5.0, 1.0], [10.0, 3.0]])

    outputs = pxrd.plot_sim(
        cof_name="cof-b",
        mode="both",
        dft=True,
        show_stacking_values=False,
        show=False,
    )

    assert outputs == [
        "cof-b/5_cof-b_analysis/pxrd_plots/simulated/serr/structure-a.pdf",
        "cof-b/5_cof-b_analysis/pxrd_plots/simulated/incl/structure-a.pdf",
    ]


@pytest.mark.unit
def test_plot_sim_vs_exp_default_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures plot_sim_vs_exp uses the default exp and simulated folder layout."""
    monkeypatch.chdir(tmp_path)

    exp_dir = tmp_path / "experimental_pxrd"
    exp_dir.mkdir()
    np.savetxt(
        exp_dir / "sample.xy",
        np.array([[5.0, 10.0], [10.0, 18.0], [15.0, 6.0]]),
    )

    sim_serr_dir = tmp_path / "cof-c" / "5_cof-c_analysis" / "pxrd_xy" / "serr"
    sim_incl_dir = tmp_path / "cof-c" / "5_cof-c_analysis" / "pxrd_xy" / "incl"
    sim_serr_dir.mkdir(parents=True)
    sim_incl_dir.mkdir(parents=True)
    np.savetxt(
        sim_serr_dir / "sim_1.xy",
        np.array([[5.0, 2.0], [10.0, 5.0], [15.0, 1.0]]),
    )
    np.savetxt(
        sim_incl_dir / "sim_2.xy",
        np.array([[5.0, 1.0], [10.0, 4.0], [15.0, 3.0]]),
    )

    pxrd = cl.PXRD()
    output = pxrd.plot_sim_vs_exp(
        cof_name="cof-c",
        mode="both",
        show=False,
        save=False,
    )

    assert output == [
        "cof-c/5_cof-c_analysis/pxrd_plots/serr/sim_1.pdf",
        "cof-c/5_cof-c_analysis/pxrd_plots/incl/sim_2.pdf",
    ]


class _FakePattern:
    def __init__(self) -> None:
        self.x = np.array([5.0, 10.0, 15.0])
        self.y = np.array([10.0, 25.0, 5.0])


class _FakeCalculator:
    def __init__(self, wavelength: str) -> None:
        self.wavelength = wavelength

    def get_pattern(
        self,
        _structure: object,
        two_theta_range: tuple[float, float],
    ) -> _FakePattern:
        assert two_theta_range == (1.5, 30.0)
        return _FakePattern()


@pytest.mark.unit
def test_produce_xy_writes_xy_for_each_cif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures one XY pattern file is produced for each discovered CIF input."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "a.cif").write_text("data_a\n", encoding="utf-8")
    (input_dir / "b.cif").write_text("data_b\n", encoding="utf-8")
    output_dir = tmp_path / "xy"

    module = importlib.import_module(cl.PXRD.__module__)
    monkeypatch.setattr(module, "XRDCalculator", _FakeCalculator)
    monkeypatch.setattr(module.Structure, "from_file", lambda _path: object())

    pxrd = cl.PXRD()
    output = pxrd.produce_xy(input_folder=input_dir, output_folder=output_dir)

    assert output == str(output_dir)
    assert (output_dir / "a.xy").exists()
    assert (output_dir / "b.xy").exists()


@pytest.mark.unit
def test_produce_xy_raises_for_missing_folder() -> None:
    """This test ensures produce_xy fails clearly when the input CIF folder is missing."""
    pxrd = cl.PXRD()
    with pytest.raises(FileNotFoundError, match="CIF folder not found"):
        pxrd.produce_xy("/definitely/not/there")


@pytest.mark.unit
def test_produce_xy_raises_for_empty_folder(tmp_path: Path) -> None:
    """This test ensures produce_xy rejects empty folders with no CIF files."""
    pxrd = cl.PXRD()
    with pytest.raises(FileNotFoundError, match=r"No \.cif files found"):
        pxrd.produce_xy(tmp_path)


@pytest.mark.unit
def test_plot_xy_creates_output(tmp_path: Path) -> None:
    """This test ensures plot_xy writes a non-empty stacked image from XY inputs."""
    xy_dir = tmp_path / "xy"
    xy_dir.mkdir()
    np.savetxt(
        xy_dir / "first.xy",
        np.array([[5.0, 10.0], [10.0, 20.0], [15.0, 5.0]]),
    )
    np.savetxt(
        xy_dir / "second.xy",
        np.array([[5.0, 6.0], [10.0, 9.0], [15.0, 4.0]]),
    )
    output = tmp_path / "plots" / "stacked.png"

    pxrd = cl.PXRD()
    out_path = pxrd.plot_xy(xy_folder=xy_dir, output_path=output, show=False)

    assert out_path == str(output)
    assert output.exists()
    assert output.stat().st_size > 0


@pytest.mark.unit
def test_extract_peaks_single_mode_default_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures extract_peaks uses default XY routing for one mode."""
    xy_dir = tmp_path / "cof-a" / "5_cof-a_analysis" / "pxrd_xy" / "serr"
    xy_dir.mkdir(parents=True)
    np.savetxt(xy_dir / "sample.xy", np.array([[5.0, 10.0], [10.0, 20.0]]))
    monkeypatch.chdir(tmp_path)

    pxrd = cl.PXRD()
    outputs = pxrd.extract_peaks(
        cof_name="cof-a",
        mode="serr",
        print_peaks=False,
        save_csv=True,
    )

    assert list(outputs.keys()) == ["serr"]
    df = outputs["serr"]
    assert list(df.columns) == [
        "structure",
        "rank",
        "two_theta_deg",
        "relative_intensity",
    ]
    assert (
        tmp_path
        / "cof-a"
        / "5_cof-a_analysis"
        / "pxrd_peaks"
        / "serr"
        / "sample_all.csv"
    ).exists()


@pytest.mark.unit
def test_extract_peaks_both_writes_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures extract_peaks writes per-mode CSV outputs."""
    serr_dir = tmp_path / "cof-a" / "5_cof-a_analysis" / "pxrd_xy" / "serr"
    incl_dir = tmp_path / "cof-a" / "5_cof-a_analysis" / "pxrd_xy" / "incl"
    serr_dir.mkdir(parents=True)
    incl_dir.mkdir(parents=True)
    np.savetxt(serr_dir / "serr.xy", np.array([[5.0, 10.0], [10.0, 20.0]]))
    np.savetxt(incl_dir / "incl.xy", np.array([[5.0, 10.0], [10.0, 20.0]]))
    monkeypatch.chdir(tmp_path)

    pxrd = cl.PXRD()
    pxrd.extract_peaks(
        cof_name="cof-a",
        mode="both",
        print_peaks=False,
        save_csv=True,
    )

    assert (
        tmp_path
        / "cof-a"
        / "5_cof-a_analysis"
        / "pxrd_peaks"
        / "serr"
        / "serr_all.csv"
    ).exists()
    assert (
        tmp_path
        / "cof-a"
        / "5_cof-a_analysis"
        / "pxrd_peaks"
        / "incl"
        / "incl_all.csv"
    ).exists()


@pytest.mark.unit
def test_extract_peaks_single_mode_uses_custom_mode_subfolders(
    tmp_path: Path,
) -> None:
    """This test ensures extract_peaks treats custom folders as mode roots."""
    xy_root = tmp_path / "xy"
    xy_dir = xy_root / "incl"
    xy_dir.mkdir(parents=True)
    np.savetxt(xy_dir / "sample.xy", np.array([[5.0, 10.0], [10.0, 20.0]]))
    output_root = tmp_path / "peaks"

    outputs = cl.PXRD().extract_peaks(
        cof_name="cof-a",
        mode="incl",
        xy_folder=xy_root,
        output_folder=output_root,
        print_peaks=False,
    )

    assert list(outputs) == ["incl"]
    assert (output_root / "incl" / "sample_all.csv").exists()


@pytest.mark.unit
def test_extract_peaks_filters_and_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures extract_peaks filters, ranks, and caps peak lists."""
    xy_dir = tmp_path / "cof-a" / "5_cof-a_analysis" / "pxrd_xy" / "serr"
    xy_dir.mkdir(parents=True)
    two_theta = np.arange(5.0, 17.0)
    intensity = np.r_[np.arange(100.0, 0.0, -10.0), [5.0, 0.5]]
    np.savetxt(xy_dir / "sample.xy", np.column_stack([two_theta, intensity]))
    monkeypatch.chdir(tmp_path)

    pxrd = cl.PXRD()
    outputs = pxrd.extract_peaks(
        cof_name="cof-a",
        mode="serr",
        max_peaks=10,
        min_relative_intensity=1.0,
        print_peaks=False,
        save_csv=False,
    )
    df = outputs["serr"]

    assert len(df) == 10
    assert df["rank"].tolist() == list(range(1, 11))
    assert df["two_theta_deg"].tolist() == list(np.arange(5.0, 15.0))
    assert 15.0 not in df["two_theta_deg"].tolist()
    assert 16.0 not in df["two_theta_deg"].tolist()
    assert df["relative_intensity"].iloc[0] == pytest.approx(100.0)
    assert df["relative_intensity"].iloc[-1] == pytest.approx(10.0)


@pytest.mark.unit
def test_extract_peaks_raises_for_missing_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures extract_peaks fails clearly for missing folders."""
    monkeypatch.chdir(tmp_path)
    pxrd = cl.PXRD()
    with pytest.raises(FileNotFoundError, match="XY folder not found"):
        pxrd.extract_peaks(
            cof_name="cof-a",
            mode="serr",
            print_peaks=False,
            save_csv=False,
        )


@pytest.mark.unit
def test_extract_peaks_raises_for_empty_folder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures extract_peaks rejects folders without XY files."""
    xy_dir = tmp_path / "cof-a" / "5_cof-a_analysis" / "pxrd_xy" / "serr"
    xy_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    pxrd = cl.PXRD()
    with pytest.raises(FileNotFoundError, match=r"No \.xy files found"):
        pxrd.extract_peaks(
            cof_name="cof-a",
            mode="serr",
            print_peaks=False,
            save_csv=False,
        )


@pytest.mark.unit
def test_postopt_run_routing_and_source_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures postopt PXRD routing uses its dedicated stage folders."""
    calls: list[tuple[Path, Path]] = []

    def fake_produce_xy(
        _self: cl.PXRD,
        input_folder: str | Path,
        output_folder: str | Path | None,
    ) -> str:
        assert output_folder is not None
        calls.append((Path(input_folder), Path(output_folder)))
        return str(output_folder)

    monkeypatch.setattr(cl.PXRD, "produce_xy", fake_produce_xy)
    pxrd = cl.PXRD()
    assert pxrd.run("cof-a", mode="serr", source="postopt") == {
        "serr": "cof-a/7_cof-a_postanalysis/pxrd_xy/serr"
    }
    assert calls == [
        (
            Path("cof-a/6_cof-a_scaling/postopt/serr"),
            Path("cof-a/7_cof-a_postanalysis/pxrd_xy/serr"),
        )
    ]
    with pytest.raises(ValueError, match="source must be"):
        pxrd.run("cof-a", source="invalid")


@pytest.mark.unit
def test_extract_peak_regions_normalizes_per_region_and_tracks_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures regional peaks use local normalization and source-aware state."""
    xy_dir = tmp_path / "cof-a" / "7_cof-a_postanalysis" / "pxrd_xy" / "serr"
    xy_dir.mkdir(parents=True)
    np.savetxt(xy_dir / "sample.xy", [[3.0, 0.011], [3.5, 14.006], [8.0, 5.0]])
    monkeypatch.chdir(tmp_path)
    pxrd = cl.PXRD()
    monkeypatch.setattr(
        pxrd,
        "_fit_exp_peak",
        lambda **_kwargs: {"two_theta": 3.4, "intensity": 10.0, "sigma": 0.1},
    )
    peak_data = pxrd.extract_peak_regions(
        cof_name="cof-a",
        mode="serr",
        peak_regions=[(2.5, 4.0), (7.5, 8.5)],
        source="postopt",
    )
    assert peak_data is pxrd._peak_data_by_structure
    assert pxrd._peak_data_source == "postopt"
    structure_data = peak_data["sample"]
    assert list(structure_data.columns) == [
        "region",
        "region_min",
        "region_max",
        "source",
        "two_theta",
        "intensity",
        "relative_intensity",
    ]
    exp_rows = structure_data[structure_data["source"] == "exp"]
    assert len(exp_rows) == 2
    assert exp_rows["intensity"].isna().all()
    assert exp_rows["relative_intensity"].isna().all()
    sim_rows = structure_data[structure_data["source"] == "sim"]
    assert len(sim_rows) == 2
    assert sim_rows.loc[
        sim_rows["two_theta"] == 3.5, "relative_intensity"
    ].item() == pytest.approx(100.0)
    assert sim_rows.loc[
        sim_rows["two_theta"] == 8.0, "relative_intensity"
    ].item() == pytest.approx(100.0)
    assert (
        tmp_path
        / "cof-a"
        / "7_cof-a_postanalysis"
        / "pxrd_peaks"
        / "serr"
        / "sample_regions.csv"
    ).exists()


@pytest.mark.unit
def test_sim_peak_centroid_and_median_scale_factor() -> None:
    """This test ensures PXRD centroids and Bragg-law median scaling remain deterministic."""
    pxrd = cl.PXRD()
    two_peaks = pd.DataFrame(
        {
            "two_theta_deg": [3.8846, 3.8948],
            "relative_intensity": [100.0, 99.97],
        }
    )
    assert pxrd._sim_peak_centroid(two_peaks) == pytest.approx(
        3.8897, abs=0.0001
    )
    assert pxrd._sim_peak_centroid(two_peaks.iloc[:1]) == pytest.approx(3.8846)
    with pytest.raises(ValueError, match="No retained"):
        pxrd._sim_peak_centroid(two_peaks.iloc[:0])

    factors = [1.0245, 1.0152, 1.0230, 1.0246]
    rows: list[dict[str, float | int | str]] = []
    for region, factor in enumerate(factors, start=1):
        exp_theta = 4.0
        sim_theta = np.degrees(
            2 * np.arcsin(factor * np.sin(np.radians(exp_theta / 2)))
        )
        rows.extend(
            [
                {
                    "region": region,
                    "source": "exp",
                    "two_theta": exp_theta,
                    "relative_intensity": np.nan,
                },
                {
                    "region": region,
                    "source": "sim",
                    "two_theta": sim_theta,
                    "relative_intensity": 100.0,
                },
            ]
        )
    assert pxrd._median_scale_factor(pd.DataFrame(rows)) == pytest.approx(
        1.02375, abs=0.0001
    )
    with pytest.raises(ValueError, match="No valid"):
        pxrd._median_scale_factor(
            pd.DataFrame(
                columns=["region", "source", "two_theta", "relative_intensity"]
            )
        )


@pytest.mark.unit
def test_generate_scaled_cif_scales_in_plane_vectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures scaled CIF generation changes only in-plane cell vectors."""
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "cof-a" / "4_cof-a_optimization" / "serr"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "original_structure.cif"
    original = Atoms(
        "C", positions=[[1.0, 2.0, 3.0]], cell=[10.0, 20.0, 30.0], pbc=True
    )
    write(source_path, original)
    pxrd = cl.PXRD()
    pxrd._peak_data_by_structure = {
        "original_structure": pd.DataFrame(
            {
                "region": [1],
                "source": ["exp"],
                "two_theta": [4.0],
                "relative_intensity": [np.nan],
            }
        )
    }
    monkeypatch.setattr(pxrd, "_median_scale_factor", lambda _data: 1.0237)
    output = Path(
        pxrd.generate_scaled_cif("cof-a", "serr")["original_structure"]
    )
    scaled = cast("Atoms", read(output))
    assert output == Path(
        "cof-a/6_cof-a_scaling/scaling/serr/original_structure_1.0237.cif"
    )
    assert scaled.cell.lengths() == pytest.approx([10.237, 20.474, 30.0])
    assert scaled.positions[0] == pytest.approx([1.0237, 2.0474, 3.0])


@pytest.mark.unit
def test_generate_scaled_cif_requires_peak_data_and_explicit_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures scaled CIF selection uses current peak-data state."""
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "cof-a" / "4_cof-a_optimization" / "serr"
    source_dir.mkdir(parents=True)
    pxrd = cl.PXRD()
    pxrd._peak_data_by_structure = {
        "b": pd.DataFrame({"region": [1]}),
    }
    with pytest.raises(FileNotFoundError, match="No CIF files"):
        pxrd.generate_scaled_cif("cof-a", "serr")
    for name in ["a.cif", "b.cif"]:
        write(source_dir / name, Atoms("C", cell=[5, 5, 5], pbc=True))
    monkeypatch.setattr(pxrd, "_median_scale_factor", lambda _data: 1.0)
    assert (
        Path(
            pxrd.generate_scaled_cif("cof-a", "serr", source_cif="b.cif")["b"]
        ).name
        == "b_1.0000.cif"
    )


@pytest.mark.unit
def test_generate_scaled_cif_uses_selected_scale_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures only selected extracted regions reach scaling."""
    pxrd = cl.PXRD()
    peak_data = pd.DataFrame(
        {
            "region": [1, 2],
            "source": ["exp", "exp"],
            "two_theta": [4.0, 5.0],
            "relative_intensity": [np.nan, np.nan],
        }
    )
    pxrd._peak_data_by_structure = {"structure": peak_data}
    captured: list[pd.DataFrame] = []

    def capture_scale_factor(data: pd.DataFrame) -> float:
        captured.append(data)
        return 1.0

    monkeypatch.setattr(pxrd, "_median_scale_factor", capture_scale_factor)
    source_dir = tmp_path / "cof-a" / "4_cof-a_optimization" / "serr"
    source_dir.mkdir(parents=True)
    write(source_dir / "structure.cif", Atoms("C", cell=[1, 1, 1], pbc=True))
    monkeypatch.chdir(tmp_path)
    pxrd.generate_scaled_cif("cof-a", "serr", scale_regions=[2])
    assert captured[0]["region"].unique().tolist() == [2]


@pytest.mark.unit
def test_plot_sim_vs_exp_rejects_stale_peak_data_source() -> None:
    """This test ensures postopt plots never annotate peak data extracted from opt."""
    pxrd = cl.PXRD()
    pxrd._peak_data_by_structure = {"sample": pd.DataFrame({"region": [1]})}
    pxrd._peak_data_source = "opt"
    with pytest.raises(ValueError, match="belongs to source 'opt'"):
        pxrd.plot_sim_vs_exp(
            "cof-a", mode="serr", source="postopt", show=False
        )


@pytest.mark.unit
def test_plot_region_selector_requires_current_peak_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures region selectors use stored regions and reject invalid state."""
    pxrd = cl.PXRD()
    with pytest.raises(ValueError, match="No peak-region data"):
        pxrd.plot_sim_vs_exp(
            "cof-a", mode="serr", xlim="region[1]", show=False
        )
    xy_dir = tmp_path / "cof-a" / "5_cof-a_analysis" / "pxrd_xy" / "serr"
    xy_dir.mkdir(parents=True)
    np.savetxt(xy_dir / "sample.xy", [[3.0, 1.0]])
    exp_file = tmp_path / "experimental.xy"
    np.savetxt(exp_file, [[3.0, 1.0]])
    monkeypatch.chdir(tmp_path)
    pxrd._peak_data_by_structure = {
        "sample": pd.DataFrame(
            {"region": [1], "region_min": [3.0], "region_max": [5.0]}
        )
    }
    pxrd._peak_data_source = "opt"
    with pytest.raises(ValueError, match="does not exist"):
        pxrd.plot_sim_vs_exp(
            "cof-a",
            mode="serr",
            xlim="region[2]",
            exp_xy_file=exp_file,
            show=False,
        )
    with pytest.raises(ValueError, match="must use the form"):
        pxrd.plot_sim_vs_exp("cof-a", mode="serr", xlim="region-1", show=False)


@pytest.mark.unit
def test_extract_peaks_postopt_uses_postanalysis_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures postopt peak CSV extraction uses postanalysis input and output roots."""
    xy_dir = tmp_path / "cof-a" / "7_cof-a_postanalysis" / "pxrd_xy" / "incl"
    xy_dir.mkdir(parents=True)
    np.savetxt(xy_dir / "sample.xy", [[5.0, 1.0]])
    monkeypatch.chdir(tmp_path)
    cl.PXRD().extract_peaks(
        "cof-a", mode="incl", source="postopt", print_peaks=False
    )
    assert (
        tmp_path
        / "cof-a"
        / "7_cof-a_postanalysis"
        / "pxrd_peaks"
        / "incl"
        / "sample_all.csv"
    ).exists()
