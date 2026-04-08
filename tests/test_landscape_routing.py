from pathlib import Path

import pytest

from coflandscaper._internal.landscape import Landscape


@pytest.mark.unit
def test_landscape_run_mode_uses_standard_csv_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures run_mode checks standard CSV names when dft is disabled."""
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
        _self: Landscape,
        input_folder: str,
        cof_name: str | None = None,
        dft: bool = False,
        output_folder: str | None = None,
        colorscheme: str = "viridis",
        plot_mode: str = "both",
        rel_energy_max: float | None = None,
        show_minima_markers: bool = True,
        show_header: bool = True,
    ) -> None:
        del (
            cof_name,
            output_folder,
            colorscheme,
            plot_mode,
            rel_energy_max,
            show_minima_markers,
            show_header,
        )
        calls.append((input_folder, dft))

    monkeypatch.setattr(Landscape, "run", fake_run)

    Landscape().run_mode(
        cof_name="cof-1",
        mode="both",
        input_folder=str(base),
    )

    assert calls == [
        (str(base / "serr"), False),
        (str(base / "incl"), False),
    ]


@pytest.mark.unit
def test_landscape_run_mode_uses_dft_csv_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures run_mode checks _dft CSV names and forwards dft=True."""
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
        _self: Landscape,
        input_folder: str,
        cof_name: str | None = None,
        dft: bool = False,
        output_folder: str | None = None,
        colorscheme: str = "viridis",
        plot_mode: str = "both",
        rel_energy_max: float | None = None,
        show_minima_markers: bool = True,
        show_header: bool = True,
    ) -> None:
        del (
            cof_name,
            output_folder,
            colorscheme,
            plot_mode,
            rel_energy_max,
            show_minima_markers,
            show_header,
        )
        calls.append((input_folder, dft))

    monkeypatch.setattr(Landscape, "run", fake_run)

    Landscape().run_mode(
        cof_name="cof-1",
        mode="both",
        input_folder=str(base),
        dft=True,
    )

    assert calls == [
        (str(base / "serr"), True),
        (str(base / "incl"), True),
    ]
