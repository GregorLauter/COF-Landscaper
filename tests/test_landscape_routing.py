from pathlib import Path

import pytest
from matplotlib.figure import Figure

import coflandscaper as cl


@pytest.mark.unit
def test_landscape_run_uses_standard_csv_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures run checks standard CSV names when dft is disabled."""
    base = tmp_path / "cof-1" / "3_cof-1_landscape"
    (base / "serr").mkdir(parents=True)
    (base / "incl").mkdir(parents=True)
    (base / "cof-1_sp_energies_serr.csv").write_text(
        "structure,z,L,energy_eV\n",
        encoding="utf-8",
    )
    (base / "cof-1_sp_energies_incl.csv").write_text(
        "structure,z,L,energy_eV\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, bool]] = []

    def fake_run(
        _self: cl.Landscape,
        input_folder: str,
        cof_name: str | None = None,
        dft: bool = False,
        output_folder: str | None = None,
        colorscheme: str = "viridis",
        rel_energy_max: float | None = None,
        show_minima_markers: bool = True,
        minima_mode: str = "global",
        show_header: bool = True,
        show_title_block: bool = True,
        show: bool = False,
        method_label: str | None = None,
    ) -> None:
        assert method_label == "EMT"
        assert show
        del (
            cof_name,
            output_folder,
            colorscheme,
            rel_energy_max,
            show_minima_markers,
            minima_mode,
            show_header,
            show_title_block,
            show,
        )
        calls.append((input_folder, dft))

    monkeypatch.setattr(cl.Landscape, "_plot_single_mode", fake_run)

    monkeypatch.setattr(
        "coflandscaper._internal.landscape._in_notebook", lambda: True
    )
    cl.Landscape().run(
        cof_name="cof-1",
        method_label="EMT",
        input_folder=str(base),
    )

    assert calls == [
        (str(base / "serr"), False),
        (str(base / "incl"), False),
    ]


@pytest.mark.unit
def test_landscape_run_uses_dft_csv_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures run checks _dft CSV names and forwards dft=True."""
    base = tmp_path / "cof-1" / "3_cof-1_landscape"
    (base / "serr").mkdir(parents=True)
    (base / "incl").mkdir(parents=True)
    (base / "cof-1_sp_energies_serr_dft.csv").write_text(
        "structure,z,L,energy_eV\n",
        encoding="utf-8",
    )
    (base / "cof-1_sp_energies_incl_dft.csv").write_text(
        "structure,z,L,energy_eV\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, bool]] = []

    def fake_run(
        _self: cl.Landscape,
        input_folder: str,
        cof_name: str | None = None,
        dft: bool = False,
        output_folder: str | None = None,
        colorscheme: str = "viridis",
        rel_energy_max: float | None = None,
        show_minima_markers: bool = True,
        minima_mode: str = "global",
        show_header: bool = True,
        show_title_block: bool = True,
        show: bool = False,
        method_label: str | None = None,
    ) -> None:
        assert method_label is None
        assert not show
        del (
            cof_name,
            output_folder,
            colorscheme,
            rel_energy_max,
            show_minima_markers,
            minima_mode,
            show_header,
            show_title_block,
            show,
        )
        calls.append((input_folder, dft))

    monkeypatch.setattr(cl.Landscape, "_plot_single_mode", fake_run)

    cl.Landscape().run(
        cof_name="cof-1",
        mode="both",
        input_folder=str(base),
        dft=True,
        show=False,
    )

    assert calls == [
        (str(base / "serr"), True),
        (str(base / "incl"), True),
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dft", "method_label", "expected"),
    [
        (False, None, []),
        (False, "EMT", ["Level of Theory: EMT"]),
        (True, "unused", ["Level of Theory: DFT"]),
    ],
)
def test_landscape_level_of_theory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dft: bool,
    method_label: str | None,
    expected: list[str],
) -> None:
    """The title block uses DFT or an explicit ASE label, even for older CSVs."""
    monkeypatch.chdir(tmp_path)
    base = Path("cof/3_cof_landscape")
    (base / "incl").mkdir(parents=True)
    suffix = "_dft" if dft else ""
    (base / f"cof_sp_energies_incl{suffix}.csv").write_text(
        "z,L,energy_eV,calculator\n3,0,1,old\n4,0,2,old\n4,1,3,old\n",
        encoding="utf-8",
    )
    labels: list[str] = []

    def capture(figure: Figure, *_args: object, **_kwargs: object) -> None:
        labels.extend(
            text.get_text()
            for text in figure.axes[0].texts
            if text.get_text().startswith("Level of Theory:")
        )

    monkeypatch.setattr(Figure, "savefig", capture)
    cl.Landscape().run(
        "cof",
        mode="incl",
        dft=dft,
        method_label=method_label,
        show_title_block=True,
        show=False,
    )
    assert labels == expected
