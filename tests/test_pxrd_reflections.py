"""Reflection metadata contracts and plotting regression tests."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from pymatgen.core import Lattice, Structure

import coflandscaper as cl
from coflandscaper._internal.reflections import format_hkl, load_reflections


@pytest.fixture
def simulated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Export distinguishable reflections that round to the same XY angle."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    cif = input_dir / "sample.cif"
    Structure(Lattice.cubic(4), ["C"], [[0, 0, 0]]).to(filename=cif)
    pattern = SimpleNamespace(
        x=np.array([12.0000041, 8.0, 12.0000042, 29.0]),
        y=np.array([10.0, 90.0, 70.0, 1.0]),
        d_hkls=[7.400001, 11.0, 7.400002, 3.1],
        hkls=[
            [{"hkl": (0, 0, 1), "multiplicity": 2}],
            [{"hkl": (1, 0, 0), "multiplicity": 4}],
            [
                {"hkl": (1, -1, 2), "multiplicity": 8},
                {"hkl": (1, 0, -1, 2), "multiplicity": 6},
            ],
            [{"hkl": (2, 2, 1), "multiplicity": 4}],
        ],
    )
    module = importlib.import_module(cl.PXRD.__module__)
    monkeypatch.setattr(
        module.XRDCalculator, "get_pattern", lambda *_a, **_k: pattern
    )
    output = tmp_path / "xy" / "serr"
    cl.PXRD().produce_xy(input_dir, output)
    return output / "sample.xy"


def test_export_keeps_full_precision_and_indexing_basis(
    simulated: Path,
) -> None:
    data = json.loads(simulated.with_suffix(".reflections.json").read_text())
    assert data["schema_version"] == 1
    assert data["source_cif"]["name"] == "sample.cif"
    assert data["unit_cell"]["lengths_angstrom"] == [4, 4, 4]
    assert data["calculation_settings"] == {
        "wavelength": "CuKa",
        "two_theta_range": [1.5, 30],
        "scaled": True,
        "symprec": 0,
    }
    assert np.loadtxt(simulated).shape == (4, 2)
    assert data["peaks"][0]["two_theta_deg"] == 12.0000041
    assert data["peaks"][2]["two_theta_deg"] == 12.0000042
    assert data["peaks"][2]["peak_index"] == 2
    assert data["peaks"][2]["families"][1]["hkl"] == [1, 0, -1, 2]


def test_extraction_alignment_and_csv_lists(simulated: Path) -> None:
    pxrd = cl.PXRD()
    df = pxrd.extract_peaks(
        "unused",
        mode="serr",
        xy_folder=simulated.parent.parent,
        output_folder=simulated.parents[2] / "peaks",
        max_peaks=2,
        min_relative_intensity=20,
    )["serr"]
    assert list(df.relative_intensity) == pytest.approx([100, 70 / 90 * 100])
    assert list(df.d_spacing_angstrom) == [11, 7.400002]
    assert json.loads(df.iloc[1].hkls) == [[1, -1, 2], [1, 0, -1, 2]]
    assert json.loads(df.iloc[1].multiplicities) == [8, 6]
    # Including the weaker near-coincident reflection must keep its own label.
    all_peaks = pxrd.extract_peaks(
        "unused",
        mode="serr",
        xy_folder=simulated.parent.parent,
        max_peaks=0,
        min_relative_intensity=0,
        save_csv=False,
    )["serr"]
    assert json.loads(all_peaks.iloc[2].hkls) == [[0, 0, 1]]


@pytest.mark.parametrize(
    "damage", ["xy", "source", "index", "angle", "families", "schema", "json"]
)
def test_stale_or_invalid_metadata_rejected(
    simulated: Path, damage: str
) -> None:
    path = simulated.with_suffix(".reflections.json")
    data = json.loads(path.read_text())
    if damage == "xy":
        simulated.write_text(simulated.read_text() + "30 5\n")
    elif damage == "source":
        cif = Path(data["source_cif"]["path"])
        cif.write_text(cif.read_text() + "# changed\n")
    elif damage == "index":
        data["peaks"][0]["peak_index"] = 2
    elif damage == "angle":
        data["peaks"][0]["two_theta_deg"] = 15
    elif damage == "families":
        data["peaks"][0]["families"] = []
    elif damage == "schema":
        data["schema_version"] = 999
    if damage == "json":
        path.write_text("{broken")
    else:
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Invalid reflection metadata"):
        cl.PXRD().extract_peaks(
            "unused",
            mode="serr",
            xy_folder=simulated.parent.parent,
            save_csv=False,
        )


def test_relocated_dataset_retains_metadata(simulated: Path) -> None:
    data = json.loads(simulated.with_suffix(".reflections.json").read_text())
    Path(data["source_cif"]["path"]).unlink()
    xy = np.loadtxt(simulated)
    assert load_reflections(simulated, xy[:, 0], xy[:, 1]) is not None


def test_legacy_metadata_unavailable_and_requested_labels_warn(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "serr"
    folder.mkdir()
    np.savetxt(folder / "old.xy", [[5, 20], [10, 1]])
    pxrd = cl.PXRD()
    df = pxrd.extract_peaks(
        "unused",
        mode="serr",
        xy_folder=tmp_path,
        save_csv=False,
    )["serr"]
    assert (
        df[["d_spacing_angstrom", "hkls", "multiplicities"]].isna().all().all()
    )
    with pytest.warns(UserWarning, match="No reflection metadata.*skipping"):
        pxrd.plot_sim(
            "unused",
            mode="serr",
            xy_folder=tmp_path,
            show=False,
            save=False,
            annotate_hkl=True,
        )


@pytest.mark.parametrize("compared", [False, True])
def test_annotation_selection_and_all_families(
    simulated: Path,
    monkeypatch: pytest.MonkeyPatch,
    compared: bool,
) -> None:
    captured = []
    original = Axes.annotate

    def record(self: Axes, text: str, *args: Any, **kwargs):
        captured.append(text)
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", record)
    options: dict[str, Any] = {
        "cof_name": "unused",
        "mode": "serr",
        "xlim": (10, 30),
        "annotate_hkl": True,
        "max_hkl_labels": 1,
        "min_hkl_intensity": 50,
        "show": False,
        "save": False,
    }
    pxrd = cl.PXRD()
    if compared:
        exp = simulated.parents[2] / "exp.xy"
        np.savetxt(exp, [[10, 1], [12, 20], [30, 2]])
        pxrd.plot_sim_vs_exp(
            **options,
            exp_xy_file=exp,
            simulated_xy_folder=simulated.parent,
        )
    else:
        pxrd.plot_sim(**options, xy_folder=simulated.parent.parent)
    assert captured == [r"$(1\overline{1}2)$" + "\n" + "$(102)$"]
    assert plt.get_fignums() == []


def test_threshold_uses_full_pattern_maximum(
    simulated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(Axes, "annotate", lambda *_a, **_k: calls.append(1))
    cl.PXRD().plot_sim(
        "unused",
        mode="serr",
        xy_folder=simulated.parent.parent,
        xlim=(10, 30),
        annotate_hkl=True,
        min_hkl_intensity=80,
        show=False,
        save=False,
    )
    # The visible maximum is 70, but the full pattern maximum is 90.
    assert calls == []


@pytest.mark.parametrize("source", ["opt", "postopt"])
def test_both_mode_metadata_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    for mode in ["serr", "incl"]:
        folder = (
            Path("tiny")
            / (
                "4_tiny_optimization"
                if source == "opt"
                else "6_tiny_scaling/postopt"
            )
            / mode
        )
        folder.mkdir(parents=True)
        Structure(Lattice.cubic(4), ["C"], [[0, 0, 0]]).to(
            filename=folder / "tiny.cif"
        )
    pxrd = cl.PXRD(two_theta_range=(3, 60))
    outputs = pxrd.run("tiny", mode="both", source=source)
    for folder in outputs.values():
        assert (Path(folder) / "tiny.reflections.json").is_file()
    peaks = pxrd.extract_peaks("tiny", mode="both", source=source)
    assert all(df.hkls.notna().all() for df in peaks.values())
    paths = pxrd.plot_sim(
        "tiny", mode="both", source=source, annotate_hkl=True, show=False
    )
    assert len(paths) == 2
    assert all(Path(p).is_file() for p in paths)


@pytest.mark.parametrize(
    "kwargs", [{"max_hkl_labels": -1}, {"min_hkl_intensity": 101}]
)
def test_invalid_annotation_options(simulated: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="hkl"):
        cl.PXRD().plot_sim(
            "unused",
            mode="serr",
            xy_folder=simulated.parent.parent,
            show=False,
            save=False,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ([1, 0, 0], "$(100)$"),
        ([1, 0, -1, 0], "$(100)$"),
        ([1, 0, -1, 1], "$(101)$"),
        ([0, 0, 0, 2], "$(002)$"),
        ([2, -1, -1, 0], r"$(2\overline{1}0)$"),
        ([1, -1, 2], r"$(1\overline{1}2)$"),
        ([-1, 0, 1, -2], r"$(\overline{1}0\overline{2})$"),
        ([12, -3, 0], r"$(12,\ \overline{3},\ 0)$"),
        ([0, 0, -12], r"$(0,\ 0,\ \overline{12})$"),
    ],
)
def test_classical_miller_label(indices: list[int], expected: str) -> None:
    original = indices.copy()
    assert format_hkl(indices) == expected
    assert indices == original


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ([1, 0, 0, 1], "(1, 0, 0, 1)"),
        ([1, -1], "(1, -1)"),
        ([1, 0, -1, 0, 2], "(1, 0, -1, 0, 2)"),
        ([1.5, 0, -1], "(1.5, 0, -1)"),
    ],
)
def test_unexpected_indices_warn_and_keep_original(
    indices: list,
    expected: str,
) -> None:
    with pytest.warns(
        UserWarning, match="Unexpected Miller indices.*retaining"
    ):
        assert format_hkl(indices) == expected


def test_plot_labels_leave_exports_unchanged(simulated: Path) -> None:
    metadata_path = simulated.with_suffix(".reflections.json")
    metadata_before = metadata_path.read_bytes()
    pxrd = cl.PXRD()
    tables_before = pxrd.extract_peaks(
        "unused",
        mode="serr",
        xy_folder=simulated.parent.parent,
        output_folder=simulated.parents[2] / "peaks",
    )
    csv_path = simulated.parents[2] / "peaks/serr/sample_all.csv"
    csv_before = csv_path.read_bytes()
    pxrd.plot_sim(
        "unused",
        mode="serr",
        xy_folder=simulated.parent.parent,
        annotate_hkl=True,
        show=False,
        save=False,
    )
    tables_after = pxrd.extract_peaks(
        "unused",
        mode="serr",
        xy_folder=simulated.parent.parent,
        output_folder=simulated.parents[2] / "peaks",
    )
    assert tables_before["serr"].equals(tables_after["serr"])
    assert csv_path.read_bytes() == csv_before
    assert metadata_path.read_bytes() == metadata_before
    assert json.loads(tables_after["serr"].iloc[1].hkls)[1] == [1, 0, -1, 2]


@pytest.mark.parametrize("compared", [False, True])
def test_three_dimensional_plots_share_classical_labels(
    simulated: Path,
    monkeypatch: pytest.MonkeyPatch,
    compared: bool,
) -> None:
    labels = []
    original = Axes.annotate

    def record(self: Axes, text: str, *args: Any, **kwargs):
        labels.append(text)
        return original(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "annotate", record)
    pxrd = cl.PXRD3D()
    options: dict[str, Any] = {
        "cof_name": "unused",
        "annotate_hkl": True,
        "max_hkl_labels": 1,
        "min_hkl_intensity": 50,
        "xlim": (10, 30),
        "show": False,
        "save": False,
    }
    if compared:
        exp = simulated.parents[2] / "experiment_3d.xy"
        np.savetxt(exp, [[10, 1], [12, 20], [30, 2]])
        pxrd.plot_sim_vs_exp(
            **options,
            exp_xy_file=exp,
            simulated_xy_folder=simulated.parent,
        )
    else:
        pxrd.plot_sim(**options, xy_folder=simulated.parent)
    assert labels == [r"$(1\overline{1}2)$" + "\n" + "$(102)$"]
