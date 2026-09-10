"""Simulate, compare, and refine COF structures using PXRD data.

This module provides utilities to simulate PXRD patterns from optimized COF
structures, compare simulated and experimental diffraction data, extract
user-defined peak regions, derive PXRD-guided in-plane lattice scaling factors,
generate scaled CIF structures, and analyse postoptimized structures.

The workflow supports both normally optimized structures and structures obtained
after PXRD-guided scaling and fixed-cell postoptimization.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read, write
from matplotlib.lines import Line2D
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from ase.atoms import Atoms


class PXRD:
    """Simulate PXRD patterns and perform PXRD-guided structure refinement.

    The class provides the PXRD workflow used after COF structure optimization:

    - simulate PXRD patterns from optimized CIF structures;
    - extract simulated peak tables;
    - compare simulated and experimental PXRD patterns;
    - fit experimental features within user-defined 2-theta regions;
    - extract corresponding simulated reflections;
    - derive an in-plane lattice scaling factor from selected regions;
    - generate scaled CIF structures;
    - repeat PXRD analysis after fixed-cell postoptimization.

    The ``source`` argument distinguishes between the normal optimized structure
    workflow (``"opt"``) and the postoptimization workflow (``"postopt"``).

    Default optimized-structure workflow:

    - input CIFs:
        ``{cof_name}/4_{cof_name}_optimization/{serr|incl}``
    - simulated XY files:
        ``{cof_name}/5_{cof_name}_analysis/pxrd_xy/{serr|incl}``
    - extracted peak tables:
        ``{cof_name}/5_{cof_name}_analysis/pxrd_peaks/{serr|incl}``
    - comparison plots:
        ``{cof_name}/5_{cof_name}_analysis/pxrd_plots/{serr|incl}``

    Default postoptimization workflow:

    - input CIFs:
        ``{cof_name}/6_{cof_name}_scaling/postopt/{serr|incl}``
    - simulated XY files:
        ``{cof_name}/7_{cof_name}_postanalysis/pxrd_xy/{serr|incl}``
    - extracted peak tables:
        ``{cof_name}/7_{cof_name}_postanalysis/pxrd_peaks/{serr|incl}``
    - comparison plots:
        ``{cof_name}/7_{cof_name}_postanalysis/pxrd_plots/{serr|incl}``
    """

    def __init__(
        self,
        wavelength: str = "CuKa",
        two_theta_range: tuple[float, float] = (1.5, 30.0),
    ) -> None:
        """Initialize PXRD simulation settings.

        Args:
            wavelength: X-ray wavelength preset accepted by `XRDCalculator`.
                Defaults to `"CuKa"`.
            two_theta_range: Simulated 2-theta range in degrees.
                Defaults to `(1.5, 30.0)`.
        """
        self._wavelength = wavelength
        self._two_theta_range = two_theta_range
        self._peak_data_by_structure: dict[str, pd.DataFrame] = {}
        self._peak_data_source: str | None = None
        self._simulated_structures: set[str] = set()

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
    def _resolve_source(source: str) -> tuple[str, str]:
        """Return default CIF and analysis roots for a PXRD source."""
        if source not in {"opt", "postopt"}:
            raise ValueError("source must be 'opt' or 'postopt'.")
        if source == "opt":
            return "4_{cof_name}_optimization", "5_{cof_name}_analysis"
        return "6_{cof_name}_scaling/postopt", "7_{cof_name}_postanalysis"

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
    def _resolve_exp_xy_file(exp_xy_file: str | Path | None) -> Path:
        """Return an explicit or automatically discovered experimental file.

        Args:
            exp_xy_file: Path to experimental .xy data. If None, find the
                single .xy file in ``experimental_pxrd``.

        Returns:
            Path to the experimental .xy file.

        Raises:
            FileNotFoundError: If the requested experimental data is missing.
            ValueError: If automatic discovery finds multiple files.
        """
        if exp_xy_file is not None:
            exp_path = Path(exp_xy_file)
            if not exp_path.exists():
                raise FileNotFoundError(
                    f"Experimental .xy file not found: {exp_path}"
                )
            return exp_path

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
                f"Expected exactly one experimental .xy file in {exp_dir}, "
                f"got {len(exp_files)}. Please specify the path explicitly "
                "using exp_xy_file parameter."
            )
        return exp_files[0]

    @staticmethod
    def _validate_peak_range(
        peak_range: tuple[float, float],
    ) -> tuple[float, float]:
        """Validate and normalize a manually selected 2-theta interval."""
        if len(peak_range) != 2:
            raise ValueError(
                "peak_range must contain exactly two 2-theta bounds."
            )
        lower, upper = map(float, peak_range)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(
                "peak_range must contain two finite bounds with lower < upper."
            )
        return lower, upper

    @staticmethod
    def _pseudo_voigt_with_baseline(
        x_vals: np.ndarray,
        amplitude: float,
        centre: float,
        fwhm: float,
        eta: float,
        slope: float,
        baseline: float,
    ) -> np.ndarray:
        """Evaluate a pseudo-Voigt diffraction feature with a linear baseline."""
        scaled_distance = (x_vals - centre) / fwhm
        gaussian = np.exp(-4.0 * np.log(2.0) * scaled_distance**2)
        lorentzian = 1.0 / (1.0 + 4.0 * scaled_distance**2)
        return (
            amplitude * (eta * lorentzian + (1.0 - eta) * gaussian)
            + slope * x_vals
            + baseline
        )

    @staticmethod
    def _visible_y_max(
        x_vals: np.ndarray,
        y_vals: np.ndarray,
        xlim: tuple[float, float],
        fallback: float = 1.0,
    ) -> float:
        """Return max y value within the visible x-range.

        Args:
            x_vals: X coordinates.
            y_vals: Y values aligned with x_vals.
            xlim: Visible x-range (xmin, xmax).
            fallback: Value returned when no visible points are available.

        Returns:
            Maximum finite y value inside xlim, or fallback if unavailable.
        """
        x_arr = np.asarray(x_vals, dtype=float)
        y_arr = np.asarray(y_vals, dtype=float)
        mask = (x_arr >= float(xlim[0])) & (x_arr <= float(xlim[1]))
        if not np.any(mask):
            return float(fallback)

        visible = y_arr[mask]
        visible = visible[np.isfinite(visible)]
        if visible.size == 0:
            return float(fallback)

        vmax = float(np.max(visible))
        return vmax if vmax > 0 else float(fallback)

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
        source: str = "opt",
        input_folder: str | Path | None = None,
        output_folder: str | Path | None = None,
    ) -> dict[str, str]:
        """Simulate PXRD patterns for selected stacking mode(s).

        The method converts CIF structures into simulated ``.xy`` diffraction patterns
        using ``pymatgen.analysis.diffraction.xrd.XRDCalculator``.

        By default, ``source="opt"`` reads normally optimized structures from
        ``{cof_name}/4_{cof_name}_optimization`` and writes simulated patterns under
        ``{cof_name}/5_{cof_name}_analysis/pxrd_xy``.

        With ``source="postopt"``, structures are read from the fixed-cell
        postoptimization stage and simulated patterns are written under
        ``{cof_name}/7_{cof_name}_postanalysis/pxrd_xy``.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
                Defaults to ``"both"``.
            dft: If ``True``, use the DFT-specific default input and PXRD output
                folders. Defaults to ``False``.
            source: Structure source: ``"opt"`` for normally optimized structures or
                ``"postopt"`` for fixed-cell postoptimized structures. Defaults to
                ``"opt"``.
            input_folder: Optional explicit input folder. For ``mode="both"``, this is
                treated as a parent folder containing the mode-specific subfolders.
                Defaults to ``None``.
            output_folder: Optional output root for generated XY files. A mode-specific
                ``serr`` or ``incl`` subfolder is always appended. Defaults to the
                source-specific analysis folder.

        Returns:
            Mapping from stacking mode to generated XY-folder path.

        Raises:
            ValueError: If ``mode`` or ``source`` is invalid.
        """
        modes = self._resolve_modes(mode)
        input_root_template, analysis_root_template = self._resolve_source(
            source
        )

        outputs: dict[str, str] = {}
        default_xy_root = Path(
            f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}/"
            f"{'pxrd_xy_dft' if dft else 'pxrd_xy'}"
        )
        for selected_mode in modes:
            if input_folder is None:
                cif_dir = Path(
                    f"{cof_name}/{input_root_template.format(cof_name=cof_name)}/"
                    f"{f'dft_{selected_mode}' if dft else selected_mode}"
                )
            elif len(modes) == 1:
                cif_dir = Path(input_folder)
            else:
                cif_dir = Path(input_folder) / (
                    f"dft_{selected_mode}" if dft else selected_mode
                )

            output_root = (
                default_xy_root
                if output_folder is None
                else Path(output_folder)
            )
            target_output = output_root / selected_mode

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
        """Simulate PXRD patterns for all CIF structures in one folder.

        Each CIF is converted into one two-column ``.xy`` file containing simulated
        2-theta positions and relative intensities over the range configured when
        constructing ``PXRD``.

        Args:
            input_folder: Folder containing CIF files.
            output_folder: Optional folder for generated ``.xy`` files. Defaults to
                ``simulated_xy`` inside ``input_folder``.

        Returns:
            Path to the folder containing the generated ``.xy`` files.

        Raises:
            FileNotFoundError: If the input folder is missing or contains no CIF files.
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
        xlim: tuple[float, float] = (1.5, 30.0),
        show: bool = True,
        save: bool = True,
    ) -> str:
        """Plot all .xy files in one stacked figure and save it.

        Args:
            xy_folder: Folder containing simulated .xy files.
            output_path: Path for the output image file.
            xlim: X-axis bounds as (min_2theta, max_2theta) in degrees.
                Defaults to `(1.5, 30.0)`.
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
            ax.tick_params(axis="both", labelsize=11)

        axes_list[-1].set_xlim(*xlim)
        axes_list[-1].set_xlabel(r"2$\theta$ (°)", fontsize=14)
        fig.supylabel("Intensity (a.u.)", fontsize=14)
        fig.tight_layout(rect=(0.07, 0.03, 1.0, 1.0), h_pad=1.1)

        output = Path(output_path)
        if save:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output), dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        return str(output)

    def extract_peaks(
        self,
        cof_name: str,
        mode: str = "both",
        dft: bool = False,
        source: str = "opt",
        xy_folder: str | Path | None = None,
        output_folder: str | Path | None = None,
        max_peaks: int = 100,
        min_relative_intensity: float = 1.0,
        print_peaks: bool = False,
        save_csv: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Extract simulated PXRD peak tables for selected stacking mode(s).

        For each simulated structure, intensities are normalized relative to the
        strongest reflection in the complete simulated pattern. Reflections below
        ``min_relative_intensity`` are discarded, the remaining reflections are ranked
        by relative intensity, and up to ``max_peaks`` entries are retained.

        When ``save_csv=True``, one structure-specific ``<structure>_all.csv`` file is
        written to the mode-specific peak folder.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
                Defaults to ``"both"``.
            dft: If ``True``, use the DFT-specific PXRD folders. Defaults to ``False``.
            source: PXRD source: ``"opt"`` or ``"postopt"``. Defaults to ``"opt"``.
            xy_folder: Optional root folder containing simulated XY mode folders.
                Defaults to the source-specific PXRD XY root.
            output_folder: Optional root folder for peak CSV files. Defaults to the
                source-specific PXRD peak root.
            max_peaks: Maximum number of simulated reflections retained per structure.
                Defaults to ``100``.
            min_relative_intensity: Minimum relative intensity threshold in percent.
                Defaults to ``1.0``.
            print_peaks: Whether to print the extracted peak tables. Defaults to
                ``False``.
            save_csv: Whether to write structure-specific ``_all.csv`` files. Defaults
                to ``True``.

        Returns:
            Mapping from stacking mode to a DataFrame containing ``structure``,
            ``rank``, ``two_theta_deg``, and ``relative_intensity``.

        Raises:
            FileNotFoundError: If a required simulated XY folder is missing or empty.
        """
        modes = self._resolve_modes(mode)
        _, analysis_root_template = self._resolve_source(source)
        default_xy_root = Path(
            f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}/"
            f"{'pxrd_xy_dft' if dft else 'pxrd_xy'}"
        )
        default_output_root = Path(
            f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}/"
            f"{'pxrd_peaks_dft' if dft else 'pxrd_peaks'}"
        )

        outputs: dict[str, pd.DataFrame] = {}
        for selected_mode in modes:
            xy_root = default_xy_root if xy_folder is None else Path(xy_folder)
            output_root = (
                default_output_root
                if output_folder is None
                else Path(output_folder)
            )
            xy_dir = xy_root / selected_mode
            target_output = output_root / selected_mode

            if not xy_dir.exists() or not xy_dir.is_dir():
                raise FileNotFoundError(f"XY folder not found: {xy_dir}")

            xy_files = sorted(xy_dir.glob("*.xy"))
            if not xy_files:
                raise FileNotFoundError(f"No .xy files found in: {xy_dir}")

            rows: list[dict[str, object]] = []
            for xy_file in xy_files:
                two_theta, intensity = self._read_xy(xy_file)
                max_intensity = (
                    float(np.max(intensity)) if intensity.size else 0.0
                )
                if max_intensity > 0.0:
                    rel_intensity = (intensity / max_intensity) * 100.0
                else:
                    rel_intensity = np.zeros_like(intensity, dtype=float)

                data = np.column_stack((two_theta, rel_intensity))
                data = data[data[:, 1] >= float(min_relative_intensity)]
                data = data[np.argsort(data[:, 1])[::-1]]
                if max_peaks > 0:
                    data = data[:max_peaks]

                for rank, (theta, rel) in enumerate(data, start=1):
                    rows.append(
                        {
                            "structure": xy_file.stem,
                            "rank": rank,
                            "two_theta_deg": float(theta),
                            "relative_intensity": float(rel),
                        }
                    )

            df = pd.DataFrame(
                rows,
                columns=[
                    "structure",
                    "rank",
                    "two_theta_deg",
                    "relative_intensity",
                ],
            )

            if print_peaks:
                for structure, group in df.groupby("structure", sort=False):
                    print(f"Structure: {structure}")
                    print(group.to_string(index=False))
                    print()

            if save_csv:
                target_output.mkdir(parents=True, exist_ok=True)
                for structure, structure_df in df.groupby(
                    "structure", sort=False
                ):
                    structure_df.to_csv(
                        target_output / f"{structure}_all.csv", index=False
                    )

            outputs[selected_mode] = df

        return outputs

    def _fit_exp_peak(
        self,
        cof_name: str,
        peak_range: tuple[float, float],
        exp_xy_file: str | Path | None = None,
    ) -> dict[str, float]:
        """Fit one experimental PXRD feature within a selected 2-theta range.

        The experimental data inside ``peak_range`` are fitted with a pseudo-Voigt
        profile and linear baseline. The fitted peak centre is used as the experimental
        reference position during PXRD-guided refinement.

        Args:
            cof_name: COF name retained for API consistency.
            peak_range: Lower and upper 2-theta limits enclosing the experimental
                diffraction feature.
            exp_xy_file: Optional experimental ``.xy`` file. Defaults to automatic
                discovery of the single ``.xy`` file in ``experimental_pxrd/``.

        Returns:
            Dictionary containing the fitted ``two_theta`` centre, fitted peak
            ``intensity``, and Gaussian-equivalent ``sigma``.

        Raises:
            ValueError: If the selected range is invalid or contains insufficient or
                non-varying experimental data.
            RuntimeError: If the pseudo-Voigt fit cannot be completed.
        """
        del cof_name
        lower, upper = self._validate_peak_range(peak_range)
        x_exp, y_exp = self._read_xy(self._resolve_exp_xy_file(exp_xy_file))
        x_exp = np.asarray(x_exp, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)
        mask = (
            (x_exp >= lower)
            & (x_exp <= upper)
            & np.isfinite(x_exp)
            & np.isfinite(y_exp)
        )
        x_local = x_exp[mask]
        y_local = y_exp[mask]
        if x_local.size < 5:
            raise ValueError(
                "peak_range must contain at least five finite experimental data points."
            )

        order = np.argsort(x_local)
        x_local = x_local[order]
        y_local = y_local[order]
        local_min = float(np.min(y_local))
        local_max = float(np.max(y_local))
        intensity_span = local_max - local_min
        if intensity_span <= 0.0:
            raise ValueError(
                "Experimental data in peak_range must vary in intensity."
            )

        peak_index = int(np.argmax(y_local))
        interval = upper - lower
        slope = float((y_local[-1] - y_local[0]) / (x_local[-1] - x_local[0]))
        baseline = float(y_local[0] - slope * x_local[0])
        peak_background = slope * x_local[peak_index] + baseline
        initial = (
            max(
                float(y_local[peak_index] - peak_background),
                intensity_span / 10.0,
            ),
            float(x_local[peak_index]),
            interval / 6.0,
            0.5,
            slope,
            baseline,
        )
        lower_bounds = (0.0, lower, interval / 1000.0, 0.0, -np.inf, -np.inf)
        upper_bounds = (
            10.0 * intensity_span,
            upper,
            interval,
            1.0,
            np.inf,
            np.inf,
        )
        try:
            params, _ = curve_fit(
                self._pseudo_voigt_with_baseline,
                x_local,
                y_local,
                p0=initial,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10_000,
            )
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"Could not fit an experimental peak in range {peak_range}."
            ) from error

        amplitude, centre, fwhm, _eta, slope, baseline = params
        return {
            "two_theta": float(centre),
            "intensity": float(amplitude + slope * centre + baseline),
            "sigma": float(fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))),
        }

    def _extract_sim_peaks(
        self,
        cof_name: str,
        mode: str,
        peak_range: tuple[float, float],
        dft: bool = False,
        source: str = "opt",
        xy_folder: str | Path | None = None,
    ) -> pd.DataFrame:
        """Extract simulated reflections within one selected 2-theta region.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            peak_range: Lower and upper 2-theta limits defining the region.
            dft: If ``True``, use DFT-specific simulated PXRD folders. Defaults to
                ``False``.
            source: PXRD source: ``"opt"`` or ``"postopt"``. Defaults to ``"opt"``.
            xy_folder: Optional root folder for simulated XY files.

        Returns:
            DataFrame containing the simulated structure identifier, 2-theta position,
            raw simulated intensity, and intensity relative to the strongest reflection
            in the complete simulated pattern.
        """
        lower, upper = self._validate_peak_range(peak_range)
        modes = self._resolve_modes(mode)
        _, analysis_root_template = self._resolve_source(source)
        default_xy_root = Path(
            f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}/"
            f"{'pxrd_xy_dft' if dft else 'pxrd_xy'}"
        )
        xy_root = default_xy_root if xy_folder is None else Path(xy_folder)
        rows: list[dict[str, float | str]] = []
        self._simulated_structures = set()
        for selected_mode in modes:
            xy_dir = xy_root / selected_mode
            if not xy_dir.exists() or not xy_dir.is_dir():
                raise FileNotFoundError(f"XY folder not found: {xy_dir}")
            xy_files = sorted(xy_dir.glob("*.xy"))
            if not xy_files:
                raise FileNotFoundError(f"No .xy files found in: {xy_dir}")

            for xy_file in xy_files:
                self._simulated_structures.add(xy_file.stem)
                two_theta, intensity = self._read_xy(xy_file)
                max_intensity = (
                    float(np.max(intensity)) if intensity.size else 0.0
                )
                mask = (two_theta >= lower) & (two_theta <= upper)
                for theta, value in zip(
                    two_theta[mask], intensity[mask], strict=True
                ):
                    rows.append(
                        {
                            "structure": xy_file.stem,
                            "two_theta_deg": float(theta),
                            "intensity": float(value),
                            "relative_intensity": (
                                float(value / max_intensity * 100.0)
                                if max_intensity > 0.0
                                else 0.0
                            ),
                        }
                    )

        return pd.DataFrame(
            rows,
            columns=[
                "structure",
                "two_theta_deg",
                "intensity",
                "relative_intensity",
            ],
        )

    @staticmethod
    def _sim_peak_centroid(sim_peaks: pd.DataFrame) -> float:
        """Return the representative simulated peak position for one region.

        If several simulated reflections remain within the region, their representative
        position is calculated as the relative-intensity-weighted centroid.

        Args:
            sim_peaks: Retained simulated reflections for one peak region.

        Returns:
            Relative-intensity-weighted simulated 2-theta position.

        Raises:
            ValueError: If no usable simulated reflections are available.
        """
        if sim_peaks.empty:
            raise ValueError(
                "No retained simulated reflections in peak region."
            )

        weights = sim_peaks["relative_intensity"].to_numpy(dtype=float)
        if float(np.sum(weights)) <= 0.0:
            raise ValueError(
                "Simulated reflection weights must sum to a positive value."
            )
        positions = sim_peaks["two_theta_deg"].to_numpy(dtype=float)
        return float(np.average(positions, weights=weights))

    @classmethod
    def _median_scale_factor(cls, peak_data: pd.DataFrame) -> float:
        """Calculate the recommended in-plane scale factor from selected peak regions.

        For each region containing both an experimental fitted peak and simulated
        reflection data, a lattice scale factor is obtained from the corresponding
        Bragg-law peak positions. The final scale factor is the median of the valid
        regional factors.

        Args:
            peak_data: Region-specific experimental and simulated peak data.

        Returns:
            Median in-plane lattice scale factor.

        Raises:
            ValueError: If no valid regional scale factors can be calculated.
        """
        scale_factors: list[float] = []
        for _, region_data in peak_data.groupby("region", sort=True):
            exp_peaks = region_data[region_data["source"] == "exp"]
            sim_peaks = region_data[region_data["source"] == "sim"]
            if exp_peaks.empty or sim_peaks.empty:
                continue
            exp_two_theta = float(exp_peaks.iloc[0]["two_theta"])
            sim_two_theta = cls._sim_peak_centroid(
                sim_peaks.rename(columns={"two_theta": "two_theta_deg"})
            )
            theta_exp = np.radians(exp_two_theta / 2.0)
            theta_sim = np.radians(sim_two_theta / 2.0)
            scale_factors.append(float(np.sin(theta_sim) / np.sin(theta_exp)))

        if not scale_factors:
            raise ValueError("No valid regional scale factors are available.")
        return float(np.median(scale_factors))

    def extract_peak_regions(
        self,
        cof_name: str,
        mode: str,
        peak_regions: list[tuple[float, float]],
        dft: bool = False,
        source: str = "opt",
        exp_xy_file: str | Path | None = None,
        xy_folder: str | Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fit experimental PXRD features and extract corresponding simulated peaks.

        The user defines one or more physically meaningful 2-theta regions after
        inspecting the simulated-versus-experimental PXRD comparison. Each region should
        contain an experimental diffraction feature and the corresponding simulated
        reflection or reflections.

        Within each region, the experimental feature is fitted with a pseudo-Voigt
        profile and the simulated reflections are extracted from the same interval.
        Simulated reflections are then renormalized within that region and reflections
        below 10% of the region maximum are discarded.

        The extracted data are retained internally for subsequent plotting and
        PXRD-guided scaling and are also written as structure-specific
        ``<structure>_regions.csv`` files.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            peak_regions: User-defined list of ``(lower, upper)`` 2-theta intervals.
            dft: If ``True``, use DFT-specific simulated PXRD folders. Defaults to
                ``False``.
            source: PXRD source: ``"opt"`` or ``"postopt"``. Defaults to ``"opt"``.
            exp_xy_file: Optional experimental ``.xy`` file. Defaults to automatic
                discovery from ``experimental_pxrd/``.
            xy_folder: Optional root folder for simulated XY files.

        Returns:
            Mapping from simulated structure identifier to a DataFrame containing
            region limits, source (experimental or simulated), 2-theta position, raw
            simulated intensity where applicable, and region-normalized simulated
            relative intensity.

        Raises:
            ValueError: If no peak regions are provided or a region is invalid.
        """
        if not peak_regions:
            raise ValueError("peak_regions must contain at least one region.")

        structure_rows: dict[str, list[dict[str, float | int | str]]] = {}
        experimental_rows: dict[int, dict[str, float | int | str]] = {}
        for region_index, peak_range in enumerate(peak_regions, start=1):
            lower, upper = self._validate_peak_range(peak_range)
            exp_peak = self._fit_exp_peak(
                cof_name=cof_name,
                peak_range=(lower, upper),
                exp_xy_file=exp_xy_file,
            )
            experimental_rows[region_index] = {
                "region": region_index,
                "region_min": lower,
                "region_max": upper,
                "source": "exp",
                "two_theta": exp_peak["two_theta"],
                "intensity": np.nan,
                "relative_intensity": np.nan,
            }
            sim_peaks = self._extract_sim_peaks(
                cof_name=cof_name,
                mode=mode,
                peak_range=(lower, upper),
                dft=dft,
                source=source,
                xy_folder=xy_folder,
            )
            for structure in self._simulated_structures:
                structure_rows.setdefault(structure, [])
            for structure, structure_peaks in sim_peaks.groupby(
                "structure", sort=True
            ):
                structure_rows.setdefault(str(structure), []).append(
                    {
                        "region": region_index,
                        "region_min": lower,
                        "region_max": upper,
                        "source": "exp",
                        "two_theta": exp_peak["two_theta"],
                        "intensity": np.nan,
                        "relative_intensity": np.nan,
                    }
                )
                region_max_intensity = float(
                    structure_peaks["intensity"].max()
                )
                if region_max_intensity > 0.0:
                    filtered_peaks = structure_peaks.copy()
                    filtered_peaks["relative_intensity"] = (
                        filtered_peaks["intensity"]
                        / region_max_intensity
                        * 100.0
                    )
                    filtered_peaks = filtered_peaks[
                        filtered_peaks["relative_intensity"] >= 10.0
                    ]
                else:
                    filtered_peaks = structure_peaks.iloc[0:0]
                for _, peak in filtered_peaks.iterrows():
                    structure_rows[str(structure)].append(
                        {
                            "region": region_index,
                            "region_min": lower,
                            "region_max": upper,
                            "source": "sim",
                            "two_theta": float(peak["two_theta_deg"]),
                            "intensity": float(peak["intensity"]),
                            "relative_intensity": float(
                                peak["relative_intensity"]
                            ),
                        }
                    )

        for rows in structure_rows.values():
            existing_regions = {
                int(row["region"]) for row in rows if row["source"] == "exp"
            }
            for region_index, experimental_row in experimental_rows.items():
                if region_index not in existing_regions:
                    rows.append(experimental_row.copy())

        peak_data_by_structure = {
            structure: pd.DataFrame(
                rows,
                columns=[
                    "region",
                    "region_min",
                    "region_max",
                    "source",
                    "two_theta",
                    "intensity",
                    "relative_intensity",
                ],
            )
            for structure, rows in structure_rows.items()
        }
        _, analysis_root_template = self._resolve_source(source)
        output_root = Path(
            f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}/"
            f"{'pxrd_peaks_dft' if dft else 'pxrd_peaks'}"
        )
        for selected_mode in self._resolve_modes(mode):
            mode_output = output_root / selected_mode
            mode_output.mkdir(parents=True, exist_ok=True)
            for structure, structure_data in peak_data_by_structure.items():
                structure_data.to_csv(
                    mode_output / f"{structure}_regions.csv", index=False
                )
        self._peak_data_by_structure = peak_data_by_structure
        self._peak_data_source = source
        return peak_data_by_structure

    def generate_scaled_cif(
        self,
        cof_name: str,
        mode: str,
        source_cif: str | None = None,
        scale_regions: list[int] | None = None,
    ) -> dict[str, str]:
        """Generate PXRD-guided in-plane scaled CIF structures.

        This method uses peak-region data previously generated by
        ``extract_peak_regions()``. A Bragg-law scale factor is calculated for each
        selected region and the median factor is applied uniformly to the first two
        unit-cell vectors. The third cell vector is left unchanged, and atomic
        coordinates are scaled consistently with the modified in-plane cell.

        By default, all extracted regions are used. ``scale_regions`` can be supplied
        to restrict the scaling to selected region IDs when some experimental features
        are broad, overlapping, or otherwise unsuitable for reliable comparison.

        The resulting structure is written to the mode-specific scaling folder and is
        intended to undergo subsequent fixed-cell relaxation with ``MaceOpt.run_fixed``.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"serr"``, ``"incl"``, or ``"both"``.
            source_cif: Optional source CIF filename in the normal optimization mode
                folder. If omitted, the available CIF must be matched unambiguously to
                the stored peak-region data.
            scale_regions: Optional list of region IDs to use for determining the final
                scale factor. ``None`` uses all extracted regions. Defaults to ``None``.

        Returns:
            Mapping from structure identifier to generated scaled CIF path.

        Raises:
            FileNotFoundError: If the optimization folder or requested source CIF is
                missing.
            ValueError: If peak-region data are unavailable, cannot be matched
                unambiguously to a source CIF, requested regions are unavailable, or no
                valid scale factor can be calculated.
        """
        modes = self._resolve_modes(mode)
        if not self._peak_data_by_structure:
            raise ValueError(
                "No peak-region data available. Run extract_peak_regions() first."
            )
        outputs: dict[str, str] = {}
        for selected_mode in modes:
            optimization_dir = Path(
                f"{cof_name}/4_{cof_name}_optimization/{selected_mode}"
            )
            if not optimization_dir.exists() or not optimization_dir.is_dir():
                raise FileNotFoundError(
                    f"Optimization folder not found: {optimization_dir}"
                )
            cif_files = sorted(optimization_dir.glob("*.cif"))
            if source_cif is not None:
                cif_files = [optimization_dir / source_cif]
                if not cif_files[0].is_file():
                    raise FileNotFoundError(
                        f"Source CIF not found in {optimization_dir}: {source_cif}"
                    )
            elif not cif_files:
                raise FileNotFoundError(
                    f"No CIF files found in optimization folder: {optimization_dir}"
                )

            structure_items = [
                (structure, peak_data)
                for structure, peak_data in self._peak_data_by_structure.items()
                if structure in {path.stem for path in cif_files}
            ]
            if not structure_items:
                raise ValueError(
                    f"No peak-region data matches CIFs in {optimization_dir}."
                )
            if (
                source_cif is None
                and len(cif_files) > 1
                and len(structure_items) == 1
            ):
                cif_files = [optimization_dir / f"{structure_items[0][0]}.cif"]

            for structure, peak_data in structure_items:
                source_path = next(
                    (path for path in cif_files if path.stem == structure),
                    None,
                )
                if source_path is None:
                    raise ValueError(
                        f"Could not unambiguously match peak data for '{structure}' "
                        f"to a CIF in {optimization_dir}."
                    )
                structure_regions = {
                    int(region) for region in peak_data["region"].dropna()
                }
                if scale_regions is None:
                    scaling_peak_data = peak_data
                else:
                    requested_regions = {
                        int(region) for region in scale_regions
                    }
                    missing_regions = requested_regions - structure_regions
                    if missing_regions:
                        raise ValueError(
                            f"Requested scale regions {sorted(requested_regions)} "
                            f"are not available for structure '{structure}'. "
                            f"Available regions: {sorted(structure_regions)}."
                        )
                    scaling_peak_data = peak_data[
                        peak_data["region"].isin(requested_regions)
                    ]
                    if scaling_peak_data.empty:
                        raise ValueError(
                            f"No usable scale regions remain for structure "
                            f"'{structure}' after selecting "
                            f"{sorted(requested_regions)}."
                        )
                final_scale_factor = self._median_scale_factor(
                    scaling_peak_data
                )
                atoms = cast("Atoms", read(source_path))
                cell = atoms.cell.array.copy()
                cell[0] *= final_scale_factor
                cell[1] *= final_scale_factor
                atoms.set_cell(cell, scale_atoms=True)
                output_dir = Path(
                    f"{cof_name}/6_{cof_name}_scaling/scaling/{selected_mode}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / (
                    f"{source_path.stem}_{final_scale_factor:.4f}.cif"
                )
                write(output_path, atoms)
                outputs[structure] = str(output_path)
        return outputs

    def plot_sim(
        self,
        cof_name: str,
        mode: str = "both",
        dft: bool = False,
        source: str = "opt",
        xy_folder: str | Path | None = None,
        output_folder: str | Path | None = None,
        xlim: tuple[float, float] | None = None,
        show: bool = True,
        save: bool = True,
        show_stacking_values: bool = True,
    ) -> list[str]:
        """Plot simulated PXRD patterns without experimental data.

        This simulation-only plotting utility generates one PDF per simulated structure
        and is useful when experimental PXRD data are unavailable. The normal
        simulated-versus-experimental workflow should use ``plot_sim_vs_exp()``.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
                Defaults to ``"both"``.
            dft: If ``True``, use DFT-specific simulated PXRD folders. Defaults to
                ``False``.
            source: PXRD source: ``"opt"`` or ``"postopt"``. Defaults to ``"opt"``.
            xy_folder: Optional root folder containing simulated XY files.
            output_folder: Optional root folder for generated PDFs.
            xlim: Optional plotted 2-theta interval. Defaults to ``(1.5, 30.0)``.
            show: Whether to display generated figures. Defaults to ``True``.
            save: Whether to save PDF figures. Defaults to ``True``.
            show_stacking_values: Whether to display analyzed ILD and ILS values when
                available. Defaults to ``True``.

        Returns:
            List of generated PDF paths.
        """
        modes = self._resolve_modes(mode)
        _, analysis_root_template = self._resolve_source(source)
        default_xy_root = Path(
            f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}/"
            f"{'pxrd_xy_dft' if dft else 'pxrd_xy'}"
        )
        if output_folder is None:
            output_root = Path(
                f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}"
                "/pxrd_plots/simulated"
            )
        else:
            output_root = Path(output_folder)

        plot_xlim = (1.5, 30.0) if xlim is None else xlim
        sim_files: list[tuple[str, Path]] = []
        for selected_mode in modes:
            xy_root = default_xy_root if xy_folder is None else Path(xy_folder)
            xy_dir = xy_root / selected_mode
            if not xy_dir.exists() or not xy_dir.is_dir():
                raise FileNotFoundError(f"XY folder not found: {xy_dir}")
            mode_files = sorted(xy_dir.glob("*.xy"))
            if not mode_files:
                raise FileNotFoundError(
                    f"No simulated .xy files found in: {xy_dir}"
                )
            sim_files.extend((selected_mode, path) for path in mode_files)

        figure_width = 15.0
        figure_height = 5.0
        dpi = 500
        outputs: list[str] = []

        for selected_mode, sim_file in sim_files:
            x_sim, y_sim = self._read_xy(sim_file)
            x_sim = np.asarray(x_sim, dtype=float)
            y_sim = np.asarray(y_sim, dtype=float)
            order = np.argsort(x_sim)
            x_sim = x_sim[order]
            y_sim = y_sim[order]
            sim_shifted = y_sim - np.nanmin(y_sim)
            y_max = self._visible_y_max(x_sim, sim_shifted, plot_xlim)

            fig, ax = plt.subplots(figsize=(figure_width, figure_height))
            ax.vlines(
                x_sim,
                0.0,
                sim_shifted,
                color="black",
                linewidth=1.5,
                alpha=0.9,
            )

            handles: list[Line2D] = []
            if show_stacking_values:
                stacking_values = self._load_stacking_values(
                    cof_name=cof_name,
                    source=source,
                    mode=selected_mode,
                    structure_filename=f"{sim_file.stem}.cif",
                    dft=dft,
                )
                if stacking_values is not None:
                    ild, ils = stacking_values
                    handles.extend(
                        [
                            Line2D(
                                [],
                                [],
                                color="none",
                                label=f"ILD = {ild:.2f} Å",
                            ),
                            Line2D(
                                [],
                                [],
                                color="none",
                                label=f"ILS = {ils:.2f} Å",
                            ),
                        ]
                    )
            if handles:
                ax.legend(
                    handles=handles,
                    loc="upper right",
                    fontsize=11,
                    frameon=False,
                    handletextpad=0.4,
                    borderaxespad=0.2,
                )

            ax.set_ylim(0.0, y_max * 1.15 if y_max > 0 else 1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_yticks([])
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(*plot_xlim)
            ax.set_xlabel(r"2$\theta$ (°)", fontsize=14)
            ax.set_ylabel("Intensity (a.u.)", fontsize=14)
            ax.tick_params(axis="both", labelsize=11)
            fig.tight_layout()

            output_path = output_root / selected_mode / f"{sim_file.stem}.pdf"
            if save:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            outputs.append(str(output_path))
            print(f"Plotted simulated PXRD pattern for {sim_file.stem}.cif")

        return outputs

    def _load_stacking_values(
        self,
        cof_name: str,
        source: str,
        mode: str,
        structure_filename: str,
        dft: bool,
    ) -> tuple[float, float] | None:
        """Return analyzed ILD/ILS values for one plotted structure, if present."""
        _, analysis_root_template = self._resolve_source(source)
        analysis_root = Path(
            f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}"
        )
        csv_path = analysis_root / (
            "final_structures_dft.csv" if dft else "final_structures.csv"
        )
        if not csv_path.is_file():
            return None
        try:
            analysis = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError):
            return None
        required = {"Stacking", "filename", "ILD", "ILS"}
        if not required.issubset(analysis.columns):
            return None
        matches = analysis[
            (analysis["Stacking"] == mode)
            & (analysis["filename"] == structure_filename)
        ]
        if matches.empty:
            return None
        row = matches.iloc[0]
        try:
            ild = float(row["ILD"])
            ils = float(row["ILS"])
        except (TypeError, ValueError):
            return None
        if not np.isfinite(ild) or not np.isfinite(ils):
            return None
        return ild, ils

    def plot_sim_vs_exp(
        self,
        cof_name: str,
        mode: str,
        dft: bool = False,
        exp_xy_file: str | Path | None = None,
        simulated_xy_folder: str | Path | None = None,
        output_folder: str | Path | None = None,
        xlim: tuple[float, float] | str | None = None,
        source: str = "opt",
        show_color_legend: bool = True,
        show_stacking_values: bool = True,
        show: bool = True,
        save: bool = True,
    ) -> list[str]:
        """Compare experimental PXRD data with each simulated structure.

        The experimental pattern is plotted as a continuous profile and the simulated
        pattern as vertical diffraction reflections. Simulated intensities are rescaled
        within the displayed 2-theta window for visual comparison.

        If peak-region data have previously been generated with
        ``extract_peak_regions()``, fitted experimental peak positions can be displayed
        as dashed indicators. ``xlim="region[N]"`` can also be used to focus the plot
        on one stored peak region.

        This method is the main visual comparison step used before selecting peak
        regions for PXRD-guided lattice refinement.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            dft: If ``True``, use DFT-specific simulated PXRD folders. Defaults to
                ``False``.
            exp_xy_file: Optional experimental ``.xy`` file. If omitted, exactly one
                ``.xy`` file must exist in ``experimental_pxrd/``.
            simulated_xy_folder: Optional simulated XY folder override.
            output_folder: Optional output folder for generated PDFs.
            xlim: Plotted 2-theta interval, or ``"region[N]"`` to display a previously
                extracted peak region with 5% padding. Defaults to ``(1.5, 30.0)``.
            source: PXRD source: ``"opt"`` or ``"postopt"``. Defaults to ``"opt"``.
            show_color_legend: Whether to display experimental/simulated legend entries.
                Defaults to ``True``.
            show_stacking_values: Whether to display analyzed ILD and ILS values when
                available. Defaults to ``True``.
            show: Whether to display generated figures. Defaults to ``True``.
            save: Whether to save PDF figures. Defaults to ``True``.

        Returns:
            List of generated PDF paths, one for each simulated structure.

        Raises:
            FileNotFoundError: If required experimental or simulated data are missing.
            ValueError: If ``mode``, ``source``, a region selector, or stored
                peak-region state is invalid.
        """
        mode_lower = mode.lower()
        if mode_lower not in {"incl", "serr", "both"}:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")
        _, analysis_root_template = self._resolve_source(source)
        if self._peak_data_by_structure and self._peak_data_source != source:
            raise ValueError(
                f"Stored peak-region data belongs to source "
                f"'{self._peak_data_source}', not '{source}'. "
                "Run extract_peak_regions() for the requested source."
            )

        region_selector: int | None = None
        if xlim is None:
            xlim = (1.5, 30.0)
        elif isinstance(xlim, str):
            region_match = re.fullmatch(r"region\[(\d+)\]", xlim)
            if region_match is None:
                raise ValueError(
                    "xlim region selector must use the form 'region[N]'."
                )
            if not self._peak_data_by_structure:
                raise ValueError(
                    "No peak-region data available. Run extract_peak_regions() first."
                )
            region_selector = int(region_match.group(1))

        exp_path = self._resolve_exp_xy_file(exp_xy_file)

        sim_files: list[Path] = []
        if simulated_xy_folder is None:
            sim_root = Path(
                f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}/"
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

        default_output = output_folder is None
        output_root = (
            Path(output_folder)
            if output_folder is not None
            else Path(
                f"{cof_name}/{analysis_root_template.format(cof_name=cof_name)}"
                "/pxrd_plots"
            )
        )

        x_exp, y_exp = self._read_xy(exp_path)
        x_exp = np.asarray(x_exp, dtype=float)
        y_exp = np.asarray(y_exp, dtype=float)
        exp_shifted = y_exp - np.nanmin(y_exp)
        figure_width = 15.0
        figure_height = 5.0
        dpi = 500
        outputs: list[str] = []

        for sim_file in sim_files:
            structure_peak_data = self._peak_data_by_structure.get(
                sim_file.stem
            )
            if region_selector is not None:
                if structure_peak_data is None:
                    raise ValueError(
                        f"No peak-region data available for structure "
                        f"'{sim_file.stem}'. Run extract_peak_regions() first."
                    )
                selected_region = structure_peak_data[
                    structure_peak_data["region"] == region_selector
                ]
                if selected_region.empty:
                    raise ValueError(
                        f"Region {region_selector} does not exist for "
                        f"structure '{sim_file.stem}'."
                    )
                x_min = float(selected_region["region_min"].min())
                x_max = float(selected_region["region_max"].max())
                padding = 0.05 * (x_max - x_min)
                plot_xlim = (x_min - padding, x_max + padding)
            else:
                plot_xlim = xlim
            x_sim, y_sim = self._read_xy(sim_file)
            x_sim = np.asarray(x_sim, dtype=float)
            y_sim = np.asarray(y_sim, dtype=float)
            order = np.argsort(x_sim)
            x_sim = x_sim[order]
            y_sim = y_sim[order]

            sim_shifted = y_sim - np.nanmin(y_sim)
            sim_max = self._visible_y_max(x_sim, sim_shifted, plot_xlim)
            if sim_max <= 0:
                continue

            exp_max = self._visible_y_max(x_exp, exp_shifted, plot_xlim)
            y_sim_scaled = (sim_shifted / sim_max) * exp_max
            fig, ax = plt.subplots(figsize=(figure_width, figure_height))

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

            if structure_peak_data is not None:
                exp_peaks = structure_peak_data[
                    structure_peak_data["source"] == "exp"
                ]
                for _, peak in exp_peaks.iterrows():
                    exp_centre = float(peak["two_theta"])
                    if not (plot_xlim[0] <= exp_centre <= plot_xlim[1]):
                        continue
                    exp_index = int(np.argmin(np.abs(x_exp - exp_centre)))
                    exp_intensity = float(exp_shifted[exp_index])
                    ax.vlines(
                        exp_centre,
                        0.0,
                        exp_intensity,
                        color="red",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.8,
                    )

            handles: list[Line2D] = []
            if show_color_legend:
                handles.extend(
                    [
                        Line2D([], [], color="red", label="Experimental"),
                        Line2D([], [], color="black", label="Simulated"),
                    ]
                )
            if show_stacking_values:
                stacking_values = self._load_stacking_values(
                    cof_name=cof_name,
                    source=source,
                    mode=(
                        mode_lower
                        if mode_lower != "both"
                        else sim_file.parent.name
                    ),
                    structure_filename=f"{sim_file.stem}.cif",
                    dft=dft,
                )
                if stacking_values is not None:
                    ild, ils = stacking_values
                    handles.extend(
                        [
                            Line2D(
                                [],
                                [],
                                color="none",
                                label=f"ILD = {ild:.2f} Å",
                            ),
                            Line2D(
                                [],
                                [],
                                color="none",
                                label=f"ILS = {ils:.2f} Å",
                            ),
                        ]
                    )
            if handles:
                ax.legend(
                    handles=handles,
                    loc="upper right",
                    fontsize=11,
                    frameon=False,
                    handletextpad=0.4,
                    borderaxespad=0.2,
                )

            y_max = max(
                self._visible_y_max(x_exp, exp_shifted, plot_xlim),
                self._visible_y_max(x_sim, y_sim_scaled, plot_xlim),
            )
            ax.set_ylim(0.0, y_max * 1.15 if y_max > 0 else 1.0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_yticks([])
            ax.tick_params(axis="y", length=0)

            ax.set_xlim(*plot_xlim)
            ax.set_xlabel(r"2$\theta$ (°)", fontsize=14)
            ax.set_ylabel("Intensity (a.u.)", fontsize=14)
            ax.tick_params(axis="both", labelsize=11)
            fig.tight_layout()

            if default_output:
                output_path = (
                    output_root / sim_file.parent.name / f"{sim_file.stem}.pdf"
                )
            else:
                output_path = output_root / f"{sim_file.stem}.pdf"
            if save:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            outputs.append(str(output_path))
            print(
                "Plotted simulated and experimental PXRD pattern for "
                f"{sim_file.stem}.cif"
            )

        return outputs
