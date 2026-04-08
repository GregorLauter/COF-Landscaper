"""PXRD simulation and stacked plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure


class Pxrd:
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
        self.wavelength = wavelength
        self.two_theta_range = two_theta_range

    def _resolve_modes(self, mode: str) -> list[str]:
        mode_lower = mode.lower()
        if mode_lower not in {"incl", "serr", "both"}:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")
        return ["serr", "incl"] if mode_lower == "both" else [mode_lower]

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
            mode: "incl", "serr", or "both".
            dft: If True, default input folders are dft_{mode}.
            input_folder: Optional explicit input folder. For mode="both",
                this is treated as a parent folder and per-mode subfolders are used.
            output_folder: Optional explicit output folder. Defaults to
                {cof_name}/5_{cof_name}_analysis/pxrd_xy or pxrd_xy_dft.
                For mode="both", this is treated as a parent folder and
                per-mode subfolders are used.

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

        calculator = XRDCalculator(wavelength=self.wavelength)
        for cif_path in cifs:
            structure = Structure.from_file(str(cif_path))
            pattern = calculator.get_pattern(
                structure,
                two_theta_range=self.two_theta_range,
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
    ) -> str:
        """Plot all .xy files in one stacked figure and save it.

        Args:
            xy_folder: Folder containing simulated .xy files.
            output_path: Path for the output image file.
            xlim: X-axis bounds as (min_2theta, max_2theta) in degrees.
            show: If True, display the plot in the active notebook/session.

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
        figure_height_per_pattern = 1.2
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

        axes_list[-1].set_xlim(*xlim)
        axes_list[-1].set_xlabel(r"2$\theta$ (deg)")
        fig.supylabel("Intensity (a.u.)")
        fig.tight_layout(rect=(0.07, 0.03, 1.0, 1.0))

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        return str(output)

    def plot(
        self,
        cof_name: str,
        mode: str = "both",
        dft: bool = False,
        xy_folder: str | Path | None = None,
        output_folder: str | Path | None = None,
        xlim: tuple[float, float] = (1.5, 60.0),
        show: bool = True,
    ) -> dict[str, str]:
        """Plot stacked PXRD patterns for one or both modes.

        Args:
            cof_name: COF name used for default path construction.
            mode: "incl", "serr", or "both".
            dft: If True, default XY folders are read from dft_{mode}.
            xy_folder: Optional explicit XY folder. For mode="both",
                this is treated as a parent folder with per-mode subfolders.
            output_folder: Optional explicit output folder for plot image(s).
                Defaults to {cof_name}/5_{cof_name}_analysis.
            xlim: X-axis bounds as (min_2theta, max_2theta) in degrees.
            show: If True, display generated plot(s) in the notebook/session.

        Returns:
            Mapping of mode to output plot path.

        Notes:
            Output files are named pxrd_stacked_{mode}.png and use
            pxrd_stacked_{mode}_dft.png when dft=True.
        """
        modes = self._resolve_modes(mode)

        outputs: dict[str, str] = {}
        default_xy_root = Path(
            f"{cof_name}/5_{cof_name}_analysis/"
            f"{'pxrd_xy_dft' if dft else 'pxrd_xy'}"
        )
        dft_suffix = "_dft" if dft else ""
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

            target_output = output_root / (
                f"pxrd_stacked_{selected_mode}{dft_suffix}.png"
            )

            outputs[selected_mode] = self.plot_xy(
                xy_folder=xy_dir,
                output_path=target_output,
                xlim=xlim,
                show=show,
            )

        return outputs
