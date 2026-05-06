import importlib
from pathlib import Path

import numpy as np
import pytest

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
def test_plot_sim_default_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """This test ensures PXRD plot_sim routing writes mode-specific output image paths."""
    pxrd = cl.PXRD()
    calls: list[tuple[Path, Path, tuple[float, float], bool]] = []

    def fake_plot_xy(
        _self: cl.PXRD,
        xy_folder: str | Path,
        output_path: str | Path,
        xlim: tuple[float, float] = (1.5, 60.0),
        show: bool = True,
        save: bool = True,
    ) -> str:
        xy_path = Path(xy_folder)
        out_path = Path(output_path)
        calls.append((xy_path, out_path, xlim, show))
        _ = save
        return str(out_path)

    monkeypatch.setattr(cl.PXRD, "plot_xy", fake_plot_xy)

    outputs = pxrd.plot_sim(
        cof_name="cof-b", mode="both", dft=True, show=False
    )

    assert outputs == {
        "serr": "cof-b/5_cof-b_analysis/cof-b_sim_serr.png",
        "incl": "cof-b/5_cof-b_analysis/cof-b_sim_incl.png",
    }
    assert calls == [
        (
            Path("cof-b/5_cof-b_analysis/pxrd_xy_dft/serr"),
            Path("cof-b/5_cof-b_analysis/cof-b_sim_serr.png"),
            (1.5, 60.0),
            False,
        ),
        (
            Path("cof-b/5_cof-b_analysis/pxrd_xy_dft/incl"),
            Path("cof-b/5_cof-b_analysis/cof-b_sim_incl.png"),
            (1.5, 60.0),
            False,
        ),
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

    assert output == "cof-c/5_cof-c_analysis/cof-c_both.png"


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
        assert two_theta_range == (1.5, 60.0)
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
def test_plot_xy_raises_for_missing_folder() -> None:
    """This test ensures plot_xy fails clearly when the XY folder does not exist."""
    pxrd = cl.PXRD()
    with pytest.raises(FileNotFoundError, match="XY folder not found"):
        pxrd.plot_xy("/definitely/not/there", "out.png", show=False)


@pytest.mark.unit
def test_plot_xy_raises_for_empty_folder(tmp_path: Path) -> None:
    """This test ensures plot_xy rejects folders that contain no XY pattern files."""
    pxrd = cl.PXRD()
    with pytest.raises(FileNotFoundError, match=r"No \.xy files found"):
        pxrd.plot_xy(tmp_path, tmp_path / "out.png", show=False)


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
        save_csv=False,
    )

    assert list(outputs.keys()) == ["serr"]
    df = outputs["serr"]
    assert list(df.columns) == [
        "structure",
        "rank",
        "two_theta_deg",
        "relative_intensity",
    ]


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
        / "pxrd_peaks.csv"
    ).exists()
    assert (
        tmp_path
        / "cof-a"
        / "5_cof-a_analysis"
        / "pxrd_peaks"
        / "incl"
        / "pxrd_peaks.csv"
    ).exists()


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
