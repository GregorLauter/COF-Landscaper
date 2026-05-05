"""Simulate PXRD patterns and generate publication-style comparison plots.

This module provides end-to-end utilities to convert optimized CIF structures
into simulated PXRD `.xy` files and to render stacked simulated or
simulated-vs-experimental visualizations for selected stacking modes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure


class PXRD:
    """Simulate PXRD patterns from optimized CIFs and create stacked plots.

    Default workflow:

    * Read CIFs from ``{cof_name}/4_{cof_name}_optimization/{serr|incl}``
        (or ``dft_{serr|incl}`` when ``dft=True``).
    * Write ``.xy`` files under
        ``{cof_name}/5_{cof_name}_analysis/pxrd_xy``
        (or ``pxrd_xy_dft`` when ``dft=True``).
    * Write plots under ``{cof_name}/5_{cof_name}_analysis``.
    """

    def __init__(
        self,
        wavelength: str = "CuKa",
        two_theta_range: tuple[float, float] = (1.5, 60.0),
    ) -> None:
        """Initialize PXRD simulation settings.

        Args:
            wavelength: X-ray wavelength preset accepted by `XRDCalculator`.
                Defaults to `"CuKa"`.
            two_theta_range: Simulated 2-theta range in degrees.
                Defaults to `(1.5, 60.0)`.

        Returns:
            None.
        """
        self._wavelength = wavelength
        self._two_theta_range = two_theta_range

    def _resolve_modes(self, mode: str) -> list[str]:
        """Normalize mode selector to one or two concrete mode tags.

        Args:
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`.

        Returns:
            List of mode tags (`["incl"]`, `["serr"]`, or `["serr", "incl"]`).

        Raises:
            ValueError: If `mode` is invalid.
        """
        mode_lower = mode.lower()
        if mode_lower not in {"incl", "serr", "both"}:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")
        return ["serr", "incl"] if mode_lower == "both" else [mode_lower]

    @staticmethod
    def _read_xy(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Read a two-column PXRD `.xy` file with small header tolerance.

        Args:
            file_path: Path to `.xy` data file.

        Returns:
            Tuple `(x, y)` arrays.

        Raises:
            ValueError: If file cannot be parsed as numeric XY data.
        """
        for skip in (0, 1, 2):
            try:
                arr = np.loadtxt(file_path, skiprows=skip)
                if arr.ndim == 1 and arr.size >= 2:
                    arr = arr.reshape(-1, 2)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    return arr[:, 0], arr[:, 1]
            except Exception:
                continue
        raise ValueError(f"Could not parse XY data from {file_path}")

    @staticmethod
    def _sim_label(file_path: str | Path) -> str:
        """Create a human-readable label from a simulated file path.

        Args:
            file_path: Simulated `.xy` file path.

        Returns:
            Cleaned label string for plotting.
        """
        stem = Path(file_path).stem
        label = stem.replace("Inc_", "Inclined ")
        label = label.replace("Ser_", "Serrated ")
        label = label.replace("Serr_", "Serrated ")
        return label.replace("_", " ")

    def run(
        self,
        cof_name: str,
        mode: str = "both",
        dft: bool = False,
        input_folder: str | Path | None = None,
        output_folder: str | Path | None = None,
    ) -> dict[str, str]:
        """Generate simulated .xy files for one or both stacking modes.

        Args:
            cof_name: COF name used for default path construction.
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`. Defaults to `"both"`.
            dft: If `True`, default input folders use `dft_{mode}`.
                Defaults to `False`.
            input_folder: Optional explicit input folder. For mode="both",
                this is treated as a parent folder and per-mode subfolders are used.
                Defaults to `None`.
            output_folder: Optional explicit output folder. Defaults to
                {cof_name}/5_{cof_name}_analysis/pxrd_xy or pxrd_xy_dft.
                For mode="both", this is treated as a parent folder and
                per-mode subfolders are used.
                Defaults to `None`.

        Returns:
            Mapping of mode to generated XY folder path.

        Notes:
            - mode='both' writes into {output_root}/serr and {output_root}/incl.
            - mode='incl' or mode='serr' writes directly into output_folder.
        """
        modes = self._resolve_modes(mode)

        outputs: dict[str, str] = {}
        default_xy_root = Path(
            f"{cof_name}/5_{cof_name}_analysis/"
            f"{'pxrd_xy_dft' if dft else 'pxrd_xy'}"
        )
        for selected_mode in modes:
            if input_folder is None:
                cif_dir = Path(
                    f"{cof_name}/4_{cof_name}_optimization/"
                    f"{f'dft_{selected_mode}' if dft else selected_mode}"
                )
            elif len(modes) == 1:
                cif_dir = Path(input_folder)
            else:
                cif_dir = Path(input_folder) / (
                    f"dft_{selected_mode}" if dft else selected_mode
                )

            if output_folder is None:
                target_output = (
                    default_xy_root / selected_mode
                    if len(modes) > 1
                    else default_xy_root
                )
            elif len(modes) == 1:
                target_output = Path(output_folder)
            else:
                target_output = Path(output_folder) / selected_mode

            outputs[selected_mode] = self.produce_xy(
                input_folder=cif_dir,
                output_folder=target_output,
            )

        return outputs

    def produce_xy(
        self,
        input_folder: str | Path,
        output_folder: str | Path | None = None,
    ) -> str:
        """Simulate PXRD from all CIF files in a folder and save .xy files.

        Args:
            input_folder: Folder containing .cif files.
            output_folder: Folder for generated .xy files. If None, uses
                "simulated_xy" inside input_folder.
                Defaults to `None`.

        Returns:
            Path to the output folder containing generated .xy files.
        """
        cif_dir = Path(input_folder)
        if not cif_dir.exists() or not cif_dir.is_dir():
            raise FileNotFoundError(f"CIF folder not found: {cif_dir}")

        cifs = sorted(cif_dir.glob("*.cif"))
        if not cifs:
            raise FileNotFoundError(f"No .cif files found in: {cif_dir}")

        xy_dir = (
            Path(output_folder) if output_folder else cif_dir / "simulated_xy"
        )
        xy_dir.mkdir(parents=True, exist_ok=True)

        calculator = XRDCalculator(wavelength=self._wavelength)
        for cif_path in cifs:
            structure = Structure.from_file(str(cif_path))
            pattern = calculator.get_pattern(
                structure,
                two_theta_range=self._two_theta_range,
            )
            xy = np.column_stack((pattern.x, pattern.y))
            np.savetxt(xy_dir / f"{cif_path.stem}.xy", xy, fmt="%.5f %.3f")

        return str(xy_dir)

    def plot_xy(
        self,
        xy_folder: str | Path,
        output_path: str | Path,
        xlim: tuple[float, float] = (1.5, 60.0),
        show: bool = True,
        save: bool = True,
    ) -> str:
        """Plot all .xy files in one stacked figure and save it.

        Args:
            xy_folder: Folder containing simulated .xy files.
            output_path: Path for the output image file.
            xlim: X-axis bounds as (min_2theta, max_2theta) in degrees.
                Defaults to `(1.5, 60.0)`.
            show: If `True`, display the plot in the active notebook/session.
                Defaults to `True`.
            save: If `True`, write the figure to `output_path`.
                Defaults to `True`.

        Returns:
            Output image path as a string.

        Notes:
            Each subplot is labeled with its CIF stem in the top-right corner.
        """
        xy_dir = Path(xy_folder)
        if not xy_dir.exists() or not xy_dir.is_dir():
            raise FileNotFoundError(f"XY folder not found: {xy_dir}")

        xy_files = sorted(xy_dir.glob("*.xy"))
        if not xy_files:
            raise FileNotFoundError(f"No .xy files found in: {xy_dir}")

        figure_width = 9.0
        figure_height_per_pattern = 1.8
        line_color = "black"
        line_width = 0.9
        dpi = 300

        nrows = len(xy_files)
        figure_height = max(2.0, figure_height_per_pattern * nrows)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(figure_width, figure_height),
            sharex=True,
        )

        axes_list = [axes] if nrows == 1 else list(axes)

        for ax, xy_file in zip(axes_list, xy_files, strict=True):
            data = np.loadtxt(xy_file)
            data_2d = np.atleast_2d(data)
            x_vals = data_2d[:, 0]
            y_vals = data_2d[:, 1]

            ax.vlines(
                x_vals,
                0.0,
                y_vals,
                color=line_color,
                linewidth=line_width,
            )
            y_max = float(np.max(y_vals)) if y_vals.size else 1.0
            ax.set_ylim(0.0, y_max * 1.1)
            ax.text(
                0.98,
                0.88,
                xy_file.stem,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelbottom=True)

        axes_list[-1].set_xlim(*xlim)
        axes_list[-1].set_xlabel(r"2$\theta$ (deg)")
        fig.supylabel("Intensity (a.u.)")
        fig.tight_layout(rect=(0.07, 0.03, 1.0, 1.0), h_pad=1.1)

        output = Path(output_path)
        if save:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        return str(output)

    def plot_sim(
        self,
        cof_name: str,
        mode: str = "both",
        dft: bool = False,
        xy_folder: str | Path | None = None,
        output_folder: str | Path | None = None,
        xlim: tuple[float, float] = (1.5, 60.0),
        show: bool = True,
        save: bool = True,
    ) -> dict[str, str]:
        """Plot stacked simulated PXRD patterns for one or both modes.

        Args:
            cof_name: COF name used for default path construction.
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`. Defaults to `"both"`.
            dft: If `True`, default XY folders are read from `dft_{mode}`.
                Defaults to `False`.
            xy_folder: Optional explicit XY folder. For mode="both",
                this is treated as a parent folder with per-mode subfolders.
                Defaults to `None`.
            output_folder: Optional explicit output folder for plot image(s).
                Defaults to `None` (uses `{cof_name}/5_{cof_name}_analysis`).
            xlim: X-axis bounds as (min_2theta, max_2theta) in degrees.
                Defaults to `(1.5, 60.0)`.
            show: If `True`, display generated plot(s) in the notebook/session.
                Defaults to `True`.
            save: If `True`, write figure(s) to disk. Defaults to `True`.

        Returns:
            Mapping of mode to output plot path.

        Notes:
            Output files are named {cof_name}_sim_{mode}.png.
        """
        modes = self._resolve_modes(mode)

        outputs: dict[str, str] = {}
        default_xy_root = Path(
            f"{cof_name}/5_{cof_name}_analysis/"
            f"{'pxrd_xy_dft' if dft else 'pxrd_xy'}"
        )
        output_root = (
            Path(output_folder)
            if output_folder is not None
            else Path(f"{cof_name}/5_{cof_name}_analysis")
        )
        for selected_mode in modes:
            if xy_folder is None:
                xy_dir = (
                    default_xy_root / selected_mode
                    if len(modes) > 1
                    else default_xy_root
                )
            elif len(modes) == 1:
                xy_dir = Path(xy_folder)
            else:
                xy_dir = Path(xy_folder) / selected_mode

            target_output = output_root / f"{cof_name}_sim_{selected_mode}.png"

            outputs[selected_mode] = self.plot_xy(
                xy_folder=xy_dir,
                output_path=target_output,
                xlim=xlim,
                show=show,
                save=save,
            )

        return outputs

    def plot_sim_vs_exp(
        self,
        cof_name: str,
        mode: str,
        dft: bool = False,
        exp_xy_file: str | Path | None = None,
        simulated_xy_folder: str | Path | None = None,
        output_folder: str | Path | None = None,
        xlim: tuple[float, float] = (1.5, 60.0),
        show: bool = True,
        save: bool = True,
    ) -> str:
        """Plot one experimental PXRD pattern against each simulated pattern.

        Args:
            cof_name: COF name used for default path construction.
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`; selects simulated folder(s).
            dft: If `True`, default simulated folder uses `pxrd_xy_dft`.
                Defaults to `False`.
            exp_xy_file: Path to experimental .xy file. If None, searches the
                'experimental_pxrd' folder for exactly one .xy file.
                If multiple files exist, you must specify the path explicitly.
                To customize the label displayed in the plot, rename the .xy file.
                Defaults to `None`.
            simulated_xy_folder: Folder containing simulated .xy files. If None,
                defaults to {cof_name}/5_{cof_name}_analysis/pxrd_xy/{mode}
                or pxrd_xy_dft/{mode} when dft=True.
                Defaults to `None`.
            output_folder: Optional folder for the output image. Defaults to
                `None` (uses `{cof_name}/5_{cof_name}_analysis`).
            xlim: X-axis bounds as (min_2theta, max_2theta) in degrees.
                Defaults to `(1.5, 60.0)`.
            show: If `True`, display the figure. Defaults to `True`.
            save: If `True`, write the figure to disk. Defaults to `True`.

        Returns:
            Output image path as a string.
        """
        mode_lower = mode.lower()
        if mode_lower not in {"incl", "serr", "both"}:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")

        if exp_xy_file is None:
            exp_dir = Path("experimental_pxrd")
            if not exp_dir.exists() or not exp_dir.is_dir():
                raise FileNotFoundError(
                    f"Experimental folder not found: {exp_dir}"
                )
            exp_files = sorted(exp_dir.glob("*.xy"))
            if not exp_files:
                raise FileNotFoundError(
                    f"No experimental .xy files found in: {exp_dir}"
                )
            if len(exp_files) != 1:
                raise ValueError(
                    f"Expected exactly one experimental .xy file in {exp_dir}, got {len(exp_files)}. "
                    f"Please specify the path explicitly using exp_xy_file parameter."
                )
            exp_path = exp_files[0]
        else:
            exp_path = Path(exp_xy_file)
            if not exp_path.exists():
                raise FileNotFoundError(
                    f"Experimental .xy file not found: {exp_path}"
                )

        sim_files: list[Path] = []
        if simulated_xy_folder is None:
            sim_root = Path(
                f"{cof_name}/5_{cof_name}_analysis/"
                f"{'pxrd_xy_dft' if dft else 'pxrd_xy'}"
            )
            if mode_lower == "both":
                sim_dirs = [sim_root / "serr", sim_root / "incl"]
            else:
                sim_dirs = [sim_root / mode_lower]
        else:
            sim_root = Path(simulated_xy_folder)
            if mode_lower == "both":
                direct_files = (
                    sorted(sim_root.glob("*.xy")) if sim_root.is_dir() else []
                )
                if direct_files:
                    sim_files = direct_files
                    sim_dirs = []
                else:
                    sim_dirs = [sim_root / "serr", sim_root / "incl"]
            else:
                sim_dirs = [sim_root]

        if not sim_files:
            for sim_dir in sim_dirs:
                if not sim_dir.exists() or not sim_dir.is_dir():
                    raise FileNotFoundError(
                        f"Simulated XY folder not found: {sim_dir}"
                    )
                mode_files = sorted(sim_dir.glob("*.xy"))
                if not mode_files:
                    raise FileNotFoundError(
                        f"No simulated .xy files found in: {sim_dir}"
                    )
                sim_files.extend(mode_files)

        output_root = (
            Path(output_folder)
            if output_folder is not None
            else Path(f"{cof_name}/5_{cof_name}_analysis")
        )
        output_path = output_root / f"{cof_name}_{mode_lower}.png"

        x_exp, y_exp = self._read_xy(exp_path)
        x_exp = np.asarray(x_exp, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)
        exp_shifted = y_exp - np.nanmin(y_exp)
        exp_max = (
            float(np.nanmax(exp_shifted))
            if np.nanmax(exp_shifted) > 0
            else 1.0
        )

        figure_width = 8.0
        figure_height_per_pattern = 2.1
        dpi = 300

        nrows = len(sim_files)
        figure_height = max(2.0, figure_height_per_pattern * nrows)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(figure_width, figure_height),
            sharex=True,
        )
        axes_list = [axes] if nrows == 1 else list(axes)

        for ax, sim_file in zip(axes_list, sim_files, strict=True):
            x_sim, y_sim = self._read_xy(sim_file)
            x_sim = np.asarray(x_sim, dtype=float)
            y_sim = np.asarray(y_sim, dtype=float)
            order = np.argsort(x_sim)
            x_sim = x_sim[order]
            y_sim = y_sim[order]

            sim_shifted = y_sim - np.nanmin(y_sim)
            sim_max = float(np.nanmax(sim_shifted))
            if sim_max <= 0:
                continue

            y_sim_scaled = (sim_shifted / sim_max) * exp_max

            ax.plot(
                x_exp,
                exp_shifted,
                color="red",
                linewidth=1.6,
                alpha=0.95,
            )
            ax.vlines(
                x_sim,
                0.0,
                y_sim_scaled,
                color="black",
                linewidth=1.5,
                alpha=0.9,
            )

            y_max = float(max(np.nanmax(exp_shifted), np.nanmax(y_sim_scaled)))
            ax.set_ylim(0.0, y_max * 1.15 if y_max > 0 else 1.0)
            ax.text(
                0.98,
                0.88,
                self._sim_label(sim_file),
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_yticks([])
            ax.tick_params(axis="y", length=0)

        axes_list[-1].set_xlim(*xlim)
        axes_list[-1].set_xlabel(r"2$\theta$ (deg)")
        fig.supylabel("Intensity (a.u.)", fontsize=12)
        fig.tight_layout(rect=(0.07, 0.03, 1.0, 1.0), h_pad=1.2)

        if save:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        return str(output_path)
