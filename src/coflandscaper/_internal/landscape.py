import os
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches

from .ild_ils_utils import get_mode_folders


class Landscape:
    """Generate potential energy landscape plots from single-point CSV data."""

    def _resolve_input_csv(
        self,
        input_folder: str,
        cof_name: str | None,
        dft: bool = False,
    ) -> tuple[Path, Path, str, str | None]:
        input_path = Path(input_folder)
        folder_tag = input_path.name

        if folder_tag not in {"serr", "incl"}:
            raise ValueError(
                "input_folder must point to a mode folder named 'serr' or 'incl'."
            )

        csv_dir = input_path.parent
        if cof_name is None:
            raise ValueError("cof_name must be provided explicitly.")
        dft_suffix = "_dft" if dft else ""
        csv_path = (
            csv_dir / f"{cof_name}_sp_energies_{folder_tag}{dft_suffix}.csv"
        )

        return csv_dir, csv_path, folder_tag, cof_name

    def run(
        self,
        input_folder: str,
        cof_name: str | None = None,
        dft: bool = False,
        output_folder: str | None = None,
        colorscheme: str = "viridis",
        plot_mode: str = "both",
        rel_energy_max: float | None = None,
        show_minima_markers: bool = True,
        show_header: bool = True,
        show_title_block: bool = True,
    ) -> None:
        """Build PES plots for one stacking mode.

        Args:
            input_folder: Mode folder path (serr or incl) used to infer mode;
                its parent must contain {cof_name}_sp_energies_{mode}.csv
                (or {cof_name}_sp_energies_{mode}_dft.csv when dft=True).
            dft: If True, read input CSVs with _dft suffix.
            output_folder: Optional output folder for plots.
                Defaults to {cof_name}/3_{cof_name}_landscape.
            colorscheme: Any valid Matplotlib colormap name.
                Defaults to "viridis".
            plot_mode: "heatmap", "isolines", or "both".
            rel_energy_max: Optional max value (eV) to cap relative energies.
                Values above this are clipped in the plots.
            show_minima_markers: If True (default), mark global minima in red and
                local minima in green on heatmap/isolines.
            show_header: If True (default), draw title and header text.
            show_title_block: If True (default), draw title plus two header lines
                (stacking mode and level of theory when available).

        Returns:
            None.
        """
        Path(input_folder)
        csv_dir, csv_path, folder_tag, cof_name = self._resolve_input_csv(
            input_folder, cof_name, dft=dft
        )
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        lot_suffix = None
        match = re.match(
            r"(.+)_sp_energies_(serr|incl)_(.+?)(?:\.csv)?$", csv_path.name
        )
        if match:
            lot_suffix = match.group(3)

        lot_label: str | None = "DFT" if dft else "MACE-MH-1"
        if lot_label is None and lot_suffix:
            lot_label = "DFT" if lot_suffix.lower() == "dft" else lot_suffix

        use_mode_naming = (
            folder_tag in {"serr", "incl"} and cof_name is not None
        )

        if output_folder:
            heatmap_dir = Path(output_folder)
        elif use_mode_naming:
            heatmap_dir = Path(f"{cof_name}/3_{cof_name}_landscape")
        else:
            heatmap_dir = Path(f"heatmaps/{folder_tag}")
        os.makedirs(heatmap_dir, exist_ok=True)

        lot_tag = f"_{lot_suffix}" if lot_suffix else ""

        if use_mode_naming:
            heatmap_path = (
                heatmap_dir
                / f"pes_{cof_name}_{folder_tag}_heatmap{lot_tag}.png"
            )
            isolines_path = (
                heatmap_dir
                / f"pes_{cof_name}_{folder_tag}_isolines{lot_tag}.png"
            )
            write_rel_csv = False
        else:
            rel_grid_csv_path = csv_dir / "energy_relative.csv"
            heatmap_path = heatmap_dir / f"heatmap{lot_tag}.png"
            isolines_path = heatmap_dir / f"isolines{lot_tag}.png"
            write_rel_csv = True

        df = pd.read_csv(csv_path)
        df2 = df.dropna(subset=["z", "L", "energy_eV"]).copy()
        if df2.empty:
            raise ValueError(
                "No entries with parsed z/L. Check naming like ..._z30_..._L020.cif"
            )

        abs_grid = df2.pivot_table(
            index="z", columns="L", values="energy_eV", aggfunc="last"
        ).sort_index()

        if abs_grid.empty:
            raise ValueError(
                "No matching rows for a z/L grid. Check CSV content."
            )

        vals = np.array(abs_grid.to_numpy(), dtype=float)
        mask = np.isfinite(vals)
        if not mask.any():
            raise ValueError("Grid has no finite energies (unexpected).")

        global_min = vals[mask].min()
        rel_grid = abs_grid - global_min
        if rel_energy_max is not None:
            rel_grid = rel_grid.clip(lower=0.0, upper=float(rel_energy_max))

        if write_rel_csv:
            rel_grid.to_csv(rel_grid_csv_path, index=True)

        plt.figure(figsize=(10, 6))
        data = rel_grid.to_numpy()

        cmap = self._resolve_cmap(colorscheme)
        mode = (plot_mode or "heatmap").lower()
        nrows, ncols = data.shape
        vmax = float(rel_energy_max) if rel_energy_max is not None else None

        def _style_axes() -> None:
            plt.xlim(-0.5, ncols - 0.5)
            plt.ylim(-0.5, nrows - 0.5)
            plt.xticks(
                range(len(rel_grid.columns)),
                [f"{c:.1f}" for c in rel_grid.columns],
                rotation=45,
                ha="right",
                fontsize=10,
            )
            plt.yticks(
                range(len(rel_grid.index)),
                [f"{r:.1f}" for r in rel_grid.index],
                fontsize=10,
            )
            plt.xlabel("Inter Layer Slipping [Å]", fontsize=12)
            plt.ylabel("Inter Layer Distance [Å]", fontsize=12)
            if show_header and show_title_block:
                title_name = cof_name or "COF"
                title_prefix = (
                    "Potential Energy Landscape (DFT)"
                    if dft
                    else "Potential Energy Landscape"
                )
                plt.title(
                    f"{title_prefix} - {title_name}",
                    fontsize=14,
                    pad=36,
                )
                mode_label = None
                if folder_tag == "serr":
                    mode_label = "Serrated"
                elif folder_tag == "incl":
                    mode_label = "Inclined"
                if mode_label:
                    plt.text(
                        0.5,
                        1.06,
                        f"Stacking Mode: {mode_label}",
                        transform=plt.gca().transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                if lot_label:
                    plt.text(
                        0.5,
                        1.02,
                        f"Level of Theory: {lot_label}",
                        transform=plt.gca().transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        def _mark_minima(use_rect: bool) -> None:
            finite_vals = np.array(data, dtype=float)
            min_pos = np.unravel_index(
                np.nanargmin(finite_vals), finite_vals.shape
            )
            y, x = min_pos
            if use_rect:
                rect = patches.Rectangle(
                    (float(x) - 0.5, float(y) - 0.5),
                    1,
                    1,
                    linewidth=2.5,
                    edgecolor="red",
                    facecolor="none",
                )
                plt.gca().add_patch(rect)
            else:
                plt.scatter(
                    [x], [y], marker="x", color="red", s=120, linewidths=2.5
                )

            local_minima = self._find_local_minima(finite_vals)
            if local_minima:
                if use_rect:
                    for ly, lx in local_minima:
                        rect = patches.Rectangle(
                            (lx - 0.5, ly - 0.5),
                            1,
                            1,
                            linewidth=2.0,
                            edgecolor="green",
                            facecolor="none",
                        )
                        plt.gca().add_patch(rect)
                else:
                    ys, xs = zip(*local_minima, strict=False)
                    plt.scatter(
                        xs,
                        ys,
                        marker="x",
                        color="green",
                        s=220,
                        linewidths=3.0,
                    )

        paths: list[Path] = []
        if mode in {"heatmap", "both"}:
            plt.figure(figsize=(10, 6))
            im = plt.imshow(
                data,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                extent=(-0.5, ncols - 0.5, -0.5, nrows - 0.5),
                vmin=0.0,
                vmax=vmax,
            )
            cbar = plt.colorbar(im, pad=0.02)
            cbar.set_label("Relative energy (eV)", labelpad=18, fontsize=12)
            _style_axes()
            if show_minima_markers:
                _mark_minima(use_rect=True)
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=200)
            plt.show()
            print(f"Saved: {heatmap_path}")
            paths.append(heatmap_path)

        if mode in {"isolines", "contour", "contours", "both"}:
            plt.figure(figsize=(10, 6))
            levels = np.linspace(0.0, vmax, 12) if vmax is not None else 12
            im = plt.contour(data, levels=levels, cmap=cmap)
            cbar = plt.colorbar(im, pad=0.02)
            cbar.set_label("Relative energy (eV)", labelpad=18, fontsize=12)
            _style_axes()
            if show_minima_markers:
                _mark_minima(use_rect=False)
            plt.tight_layout()
            plt.savefig(isolines_path, dpi=200)
            plt.show()
            print(f"Saved: {isolines_path}")
            paths.append(isolines_path)

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        dft: bool = False,
        colorscheme: str = "viridis",
        plot_mode: str = "both",
        rel_energy_max: float | None = None,
        show_minima_markers: bool = True,
        show_header: bool = True,
        show_title_block: bool = True,
        input_folder: str | None = None,
        output_folder: str | None = None,
    ) -> None:
        """Generate landscapes for selected stacking mode(s).

        Args:
            cof_name: COF name used for folder naming.
            mode: "incl", "serr", or "both".
            dft: If True, read input CSVs with _dft suffix.
            colorscheme: Any valid Matplotlib colormap name.
                Defaults to "viridis".
            plot_mode: "heatmap", "isolines", or "both".
            rel_energy_max: Optional max value for relative energies.
            show_minima_markers: If True (default), mark minima on plots.
            show_header: If True (default), draw title and header text.
            show_title_block: If True (default), draw title plus two header lines.
            input_folder: Optional base folder containing mode folders and
                {cof_name}_sp_energies_{mode}.csv files, or
                {cof_name}_sp_energies_{mode}_dft.csv when dft=True.
                Defaults to {cof_name}/3_{cof_name}_landscape.
            output_folder: Optional output folder for plots.
                Defaults to {cof_name}/3_{cof_name}_landscape.

        Returns:
            None.
        """
        mode_norm = (mode or "").strip().lower()
        mode_tags = (
            ["serr", "incl"]
            if mode_norm == "both"
            else [mode_norm]
            if mode_norm in {"serr", "incl"}
            else []
        )
        if not mode_tags:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")

        base_path = Path(input_folder or f"{cof_name}/3_{cof_name}_landscape")
        if not base_path.exists() or not base_path.is_dir():
            raise FileNotFoundError(f"Input folder not found: {base_path}")

        missing_csvs: list[str] = []
        dft_suffix = "_dft" if dft else ""
        for mode_tag in mode_tags:
            expected_csv = (
                base_path
                / f"{cof_name}_sp_energies_{mode_tag}{dft_suffix}.csv"
            )
            if not expected_csv.exists():
                missing_csvs.append(str(expected_csv))
                continue

            self.run(
                input_folder=str(base_path / mode_tag),
                cof_name=cof_name,
                dft=dft,
                output_folder=output_folder,
                colorscheme=colorscheme,
                plot_mode=plot_mode,
                rel_energy_max=rel_energy_max,
                show_minima_markers=show_minima_markers,
                show_header=show_header,
                show_title_block=show_title_block,
            )

        if missing_csvs:
            raise FileNotFoundError(
                "Missing expected CSV(s): " + ", ".join(missing_csvs)
            )

    def _find_local_minima(self, data: np.ndarray) -> list[tuple[int, int]]:
        minima: list[tuple[int, int]] = []
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                val = data[i, j]
                if not np.isfinite(val):
                    continue
                neighbors = []
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            nval = data[ni, nj]
                            if np.isfinite(nval):
                                neighbors.append(nval)
                if not neighbors:
                    continue
                if val < min(neighbors):
                    minima.append((i, j))
        return minima

    def _resolve_cmap(self, colorscheme: str):
        raw = colorscheme or "viridis"
        try:
            plt.get_cmap(raw)
            return raw
        except ValueError as exc:
            raise ValueError(
                "Unknown colorscheme. Use any valid Matplotlib colormap name "
                "(e.g. 'viridis', 'plasma', 'magma', 'cividis', 'coolwarm')."
            ) from exc


class LandscapeDifference(Landscape):
    """Generate PES difference plots from two subfolder-based SP energy CSVs."""

    def _normalize_subfolder(self, subfolder: str) -> str:
        cleaned = (subfolder or "").strip()
        if not cleaned:
            raise ValueError("subfolder must be a non-empty string")
        folder_path = Path(cleaned)
        if folder_path.is_absolute():
            raise ValueError(
                "subfolder must be a relative path inside the landscape folder"
            )
        if any(part == ".." for part in folder_path.parts):
            raise ValueError("subfolder must not contain '..'")
        return cleaned

    def _resolve_base_csv_path(
        self, input_folder: str | None, cof_name: str | None, mode: str
    ) -> tuple[Path, str, str]:
        if cof_name is None:
            raise ValueError("cof_name must be provided explicitly.")

        mode_tag = (mode or "").strip().lower()
        if mode_tag not in {"serr", "incl"}:
            raise ValueError("mode must be 'serr' or 'incl'.")

        base_dir = Path(input_folder or f"{cof_name}/3_{cof_name}_landscape")
        if not base_dir.exists() or not base_dir.is_dir():
            raise FileNotFoundError(f"Input folder not found: {base_dir}")

        standard_csv_name = f"{cof_name}_sp_energies_{mode_tag}.csv"
        return base_dir, standard_csv_name, mode_tag

    def _relative_grid_from_csv(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        value_col = (
            "energy_rel_eV" if "energy_rel_eV" in df.columns else "energy_eV"
        )
        df2 = df.dropna(subset=["z", "L", value_col]).copy()
        if df2.empty:
            raise ValueError(
                f"No entries with parsed z/L in: {csv_path}. "
                "Check naming like ..._z30_..._L020.cif"
            )

        abs_grid = df2.pivot_table(
            index="z", columns="L", values=value_col, aggfunc="last"
        ).sort_index()

        if abs_grid.empty:
            raise ValueError(
                f"No matching rows for a z/L grid in: {csv_path}. Check CSV content."
            )

        vals = np.array(abs_grid.to_numpy(), dtype=float)
        mask = np.isfinite(vals)
        if not mask.any():
            raise ValueError(f"Grid has no finite energies: {csv_path}")

        if value_col == "energy_rel_eV":
            return abs_grid

        global_min = vals[mask].min()
        return abs_grid - global_min

    def run(
        self,
        input_folder: str | None,
        cof_name: str | None,
        mode: str,
        subfolder_1: str,
        subfolder_2: str,
        output_folder: str | None = None,
        colorscheme: str = "viridis",
        plot_mode: str = "both",
        rel_energy_max: float | None = None,
        show_header: bool = True,
    ) -> None:
        """Build PES difference plots from two subfolder-based CSVs.

        Args:
            input_folder: Base landscape folder containing theory subfolders.
                Defaults to {cof_name}/3_{cof_name}_landscape.
            mode: "serr" or "incl".
            subfolder_1: First subfolder containing the standard CSV name.
            subfolder_2: Second subfolder containing the standard CSV name.
            output_folder: Optional output folder for plots.
            colorscheme: Heatmap colorscheme.
            plot_mode: "heatmap", "isolines", or "both".
            rel_energy_max: Optional symmetric cap (eV) for difference values.
            show_header: If True (default), draw title and header text.

        Returns:
            None.
        """
        normalized_subfolder_1 = self._normalize_subfolder(subfolder_1)
        normalized_subfolder_2 = self._normalize_subfolder(subfolder_2)
        if normalized_subfolder_1 == normalized_subfolder_2:
            raise ValueError("subfolder_1 and subfolder_2 must be different")

        base_dir, standard_csv_name, mode_tag = self._resolve_base_csv_path(
            input_folder, cof_name, mode
        )
        csv_path_1 = base_dir / normalized_subfolder_1 / standard_csv_name
        csv_path_2 = base_dir / normalized_subfolder_2 / standard_csv_name

        rel_grid_1 = self._relative_grid_from_csv(csv_path_1)
        rel_grid_2 = self._relative_grid_from_csv(csv_path_2)

        rel_grid_1, rel_grid_2 = rel_grid_1.align(rel_grid_2, join="inner")
        if rel_grid_1.empty or rel_grid_2.empty:
            raise ValueError(
                "No common z/L grid points between both CSVs after alignment."
            )

        diff_grid = rel_grid_1 - rel_grid_2
        if rel_energy_max is not None:
            diff_grid = diff_grid.clip(
                lower=-float(rel_energy_max), upper=float(rel_energy_max)
            )

        heatmap_dir = Path(output_folder) if output_folder else base_dir
        os.makedirs(heatmap_dir, exist_ok=True)

        subfolder_1_tag = re.sub(r"[\\/]+", "__", normalized_subfolder_1)
        subfolder_2_tag = re.sub(r"[\\/]+", "__", normalized_subfolder_2)
        comparison_tag = f"diff_{subfolder_1_tag}_vs_{subfolder_2_tag}"

        heatmap_path = (
            heatmap_dir
            / f"pes_{cof_name}_{mode_tag}_heatmap_{comparison_tag}.png"
        )
        isolines_path = (
            heatmap_dir
            / f"pes_{cof_name}_{mode_tag}_isolines_{comparison_tag}.png"
        )

        diff_csv_dir = Path(output_folder) if output_folder else base_dir
        os.makedirs(diff_csv_dir, exist_ok=True)
        diff_csv_path = diff_csv_dir / f"energy_relative_{comparison_tag}.csv"
        diff_grid.to_csv(diff_csv_path, index=True)

        data = diff_grid.to_numpy()
        cmap = self._resolve_cmap(colorscheme)
        mode = (plot_mode or "heatmap").lower()
        nrows, ncols = data.shape

        finite_vals = np.array(data, dtype=float)
        finite_vals = finite_vals[np.isfinite(finite_vals)]
        if finite_vals.size == 0:
            raise ValueError("Difference grid contains no finite values")

        vmin = float(np.min(finite_vals))
        vmax = float(np.max(finite_vals))
        if vmin == vmax:
            vmax = vmin + 1e-12

        def _style_axes() -> None:
            plt.xlim(-0.5, ncols - 0.5)
            plt.ylim(-0.5, nrows - 0.5)
            plt.xticks(
                range(len(diff_grid.columns)),
                [f"{c:.1f}" for c in diff_grid.columns],
                rotation=45,
                ha="right",
                fontsize=10,
            )
            plt.yticks(
                range(len(diff_grid.index)),
                [f"{r:.1f}" for r in diff_grid.index],
                fontsize=10,
            )
            plt.xlabel("Inter Layer Slipping [Å]", fontsize=12)
            plt.ylabel("Inter Layer Distance [Å]", fontsize=12)
            if show_header:
                title_name = cof_name or "COF"
                plt.title(
                    f"Potential Energy Difference Landscape - {title_name}",
                    fontsize=14,
                    pad=36,
                )
                mode_label = None
                if mode_tag == "serr":
                    mode_label = "Serrated"
                elif mode_tag == "incl":
                    mode_label = "Inclined"
                if mode_label:
                    plt.text(
                        0.5,
                        1.06,
                        f"Stacking Mode: {mode_label}",
                        transform=plt.gca().transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                plt.text(
                    0.5,
                    1.03,
                    f"Difference: {normalized_subfolder_1} - {normalized_subfolder_2}",
                    transform=plt.gca().transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        if mode in {"heatmap", "both"}:
            plt.figure(figsize=(10, 6))
            im = plt.imshow(
                data,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                extent=(-0.5, ncols - 0.5, -0.5, nrows - 0.5),
                vmin=vmin,
                vmax=vmax,
            )
            cbar = plt.colorbar(im, pad=0.02)
            cbar.set_label(
                "Relative energy difference (eV): "
                f"{normalized_subfolder_1} - {normalized_subfolder_2}",
                labelpad=18,
                fontsize=12,
            )
            _style_axes()
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=200)
            plt.show()
            print(f"Saved: {heatmap_path}")

        if mode in {"isolines", "contour", "contours", "both"}:
            plt.figure(figsize=(10, 6))
            levels = np.linspace(vmin, vmax, 12) if vmax > vmin else 12
            im = plt.contour(data, levels=levels, cmap=cmap)
            cbar = plt.colorbar(im, pad=0.02)
            cbar.set_label(
                "Relative energy difference (eV): "
                f"{normalized_subfolder_1} - {normalized_subfolder_2}",
                labelpad=18,
                fontsize=12,
            )
            _style_axes()
            plt.tight_layout()
            plt.savefig(isolines_path, dpi=200)
            plt.show()
            print(f"Saved: {isolines_path}")

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        subfolder_1: str,
        subfolder_2: str,
        colorscheme: str = "viridis",
        plot_mode: str = "both",
        rel_energy_max: float | None = None,
        show_header: bool = True,
        input_folder: str | None = None,
        output_folder: str | None = None,
    ) -> None:
        """Generate difference landscapes for selected mode(s)."""
        mode_norm = (mode or "").strip().lower()
        mode_tags = (
            ["serr", "incl"]
            if mode_norm == "both"
            else [mode_norm]
            if mode_norm in {"serr", "incl"}
            else []
        )
        if not mode_tags:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")

        for mode_tag in mode_tags:
            self.run(
                input_folder=input_folder,
                cof_name=cof_name,
                mode=mode_tag,
                subfolder_1=subfolder_1,
                subfolder_2=subfolder_2,
                output_folder=output_folder,
                colorscheme=colorscheme,
                plot_mode=plot_mode,
                rel_energy_max=rel_energy_max,
                show_header=show_header,
            )


class BenchmarkOverview:
    """Create a benchmark overview slide from existing PES PNGs."""

    def _sanitize_level_tag(self, level: str) -> str:
        return re.sub(r"[\\/]+", "__", level)

    def _mode_label(self, mode: str) -> str:
        if mode == "serr":
            return "Serrated"
        if mode == "incl":
            return "Inclined"
        return mode

    def _draw_image(
        self, ax, image_path: Path, title: str | None = None
    ) -> None:
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12, pad=8)
        if image_path.exists():
            image = plt.imread(str(image_path))
            ax.imshow(image)
        else:
            ax.text(
                0.5,
                0.5,
                f"Missing:\n{image_path.name}",
                ha="center",
                va="center",
                fontsize=10,
                transform=ax.transAxes,
            )

    def _read_diff_stats(
        self, base_dir: Path, level_2: str, level_1: str
    ) -> tuple[float | None, float | None, float | None]:
        level_1_tag = self._sanitize_level_tag(level_1)
        level_2_tag = self._sanitize_level_tag(level_2)
        candidate_paths = [
            base_dir
            / level_2
            / f"energy_relative_diff_{level_1}_vs_{level_2}.csv",
            base_dir
            / level_2
            / f"energy_relative_diff_{level_1_tag}_vs_{level_2_tag}.csv",
            base_dir / f"energy_relative_diff_{level_1}_vs_{level_2}.csv",
            base_dir
            / f"energy_relative_diff_{level_1_tag}_vs_{level_2_tag}.csv",
        ]
        for csv_path in candidate_paths:
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path, index_col=0)
                vals = np.array(df.to_numpy(), dtype=float)
                finite = vals[np.isfinite(vals)]
                if finite.size == 0:
                    continue
                return (
                    float(np.min(finite)),
                    float(np.max(finite)),
                    float(np.mean(finite)),
                )
            except Exception:
                continue
        return None, None, None

    def _read_minima_tuples(
        self, base_dir: Path, cof_name: str, level: str, mode: str
    ) -> list[tuple[float, float]]:
        level_dir = base_dir / level
        candidate_paths = [level_dir / f"{cof_name}_sp_energies_{mode}.csv"]
        candidate_paths.extend(
            sorted(level_dir.glob(f"{cof_name}_sp_energies_{mode}_*.csv"))
        )

        csv_path = next(
            (path for path in candidate_paths if path.exists()), None
        )
        if csv_path is None:
            return []

        try:
            df = pd.read_csv(csv_path)
            value_col = (
                "energy_rel_eV"
                if "energy_rel_eV" in df.columns
                else "energy_eV"
            )
            df2 = df.dropna(subset=["z", "L", value_col]).copy()
            if df2.empty:
                return []

            grid = df2.pivot_table(
                index="z", columns="L", values=value_col, aggfunc="last"
            ).sort_index()
            if grid.empty:
                return []

            data = np.array(grid.to_numpy(), dtype=float)
            minima_idx = Landscape()._find_local_minima(data)
            if not minima_idx:
                return []

            z_vals = list(grid.index)
            L_vals = list(grid.columns)
            minima = [
                (float(z_vals[i]), float(L_vals[j])) for i, j in minima_idx
            ]
            return sorted(set(minima), key=lambda pair: (pair[0], pair[1]))
        except Exception:
            return []

    def _format_minima_text(
        self, minima: list[tuple[float, float]], max_line_len: int = 45
    ) -> str:
        if not minima:
            return "minima: []"
        tuples = [f"[{z:.1f},{L:.1f}]" for z, L in minima]
        lines: list[str] = []
        current = ""
        for item in tuples:
            candidate = item if not current else f"{current}, {item}"
            if len(candidate) <= max_line_len:
                current = candidate
            else:
                lines.append(current)
                current = item
        if current:
            lines.append(current)

        if not lines:
            return "minima: []"

        return (
            "minima: "
            + lines[0]
            + ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else "")
        )

    def run(
        self,
        cof_name: str,
        mode: str,
        level_1: str,
        level_2_list: list[str],
        heading_text: str | None = None,
        input_folder: str | None = None,
        output_folder: str | None = None,
        output_name: str | None = None,
    ) -> Path:
        """Build one benchmark overview PNG in a 4-column layout.

        Layout:
            Row 0: level_1 label | level_1 heatmap | level_1 isolines | empty
            Row i>0: level_2 name | level_2 heatmap | level_2 isolines | diff heatmap
        """
        if mode not in {"serr", "incl"}:
            raise ValueError("mode must be 'serr' or 'incl'")
        if not level_2_list:
            raise ValueError("level_2_list must not be empty")
        if len(level_2_list) > 4:
            raise ValueError(
                "level_2_list supports up to 4 entries to keep a 4x5 overview layout"
            )

        base_dir = Path(input_folder or f"{cof_name}/3_{cof_name}_landscape")
        out_dir = Path(output_folder or base_dir)
        os.makedirs(out_dir, exist_ok=True)

        output_file = (
            output_name
            or f"benchmark_overview_{cof_name}_{mode}_{level_1}.png"
        )
        output_path = out_dir / output_file

        rows = 1 + len(level_2_list)
        cols = 4
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(20, 3.0 * rows),
        )

        if rows == 1:
            axes = np.array([axes])

        mode_label = self._mode_label(mode)
        heading = heading_text or f"{cof_name} — {mode_label}"
        fig.suptitle(heading, fontsize=20, y=0.985)

        title_ax = axes[0, 0]
        title_ax.axis("off")
        level_1_minima = self._read_minima_tuples(
            base_dir, cof_name, level_1, mode
        )
        level_1_minima_text = self._format_minima_text(level_1_minima)
        title_ax.text(
            0.5,
            0.5,
            f"{level_1}\n{level_1_minima_text}",
            ha="center",
            va="center",
            fontsize=14,
            transform=title_ax.transAxes,
        )

        level_1_dir = base_dir / level_1
        self._draw_image(
            axes[0, 1],
            level_1_dir / f"pes_{cof_name}_{mode}_heatmap.png",
        )
        self._draw_image(
            axes[0, 2],
            level_1_dir / f"pes_{cof_name}_{mode}_isolines.png",
        )
        axes[0, 3].axis("off")

        for row_idx, level_2 in enumerate(level_2_list, start=1):
            row_title_ax = axes[row_idx, 0]
            row_title_ax.axis("off")
            min_v, max_v, avg_v = self._read_diff_stats(
                base_dir, level_2, level_1
            )
            level_2_minima = self._read_minima_tuples(
                base_dir, cof_name, level_2, mode
            )
            level_2_minima_text = self._format_minima_text(level_2_minima)
            if min_v is None or max_v is None or avg_v is None:
                stats_text = "min: n/a\nmax: n/a\navg: n/a"
            else:
                stats_text = (
                    f"min: {min_v:.3f} eV\n"
                    f"max: {max_v:.3f} eV\n"
                    f"avg: {avg_v:.3f} eV"
                )
            row_title_ax.text(
                0.5,
                0.5,
                f"{level_2}\n{stats_text}\n{level_2_minima_text}",
                ha="center",
                va="center",
                fontsize=13,
                transform=row_title_ax.transAxes,
            )

            level_2_dir = base_dir / level_2
            self._draw_image(
                axes[row_idx, 1],
                level_2_dir / f"pes_{cof_name}_{mode}_heatmap.png",
            )
            self._draw_image(
                axes[row_idx, 2],
                level_2_dir / f"pes_{cof_name}_{mode}_isolines.png",
            )
            self._draw_image(
                axes[row_idx, 3],
                (
                    level_2_dir
                    / f"pes_{cof_name}_{mode}_heatmap_diff_{self._sanitize_level_tag(level_1)}_vs_{self._sanitize_level_tag(level_2)}.png"
                    if (
                        level_2_dir
                        / f"pes_{cof_name}_{mode}_heatmap_diff_{self._sanitize_level_tag(level_1)}_vs_{self._sanitize_level_tag(level_2)}.png"
                    ).exists()
                    else level_2_dir
                    / f"pes_{cof_name}_{mode}_heatmap_diff_{level_1}_vs_{level_2}.png"
                ),
            )

        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
        fig.subplots_adjust(wspace=0.02, hspace=0.08)
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {output_path}")
        return output_path


class SelectCofs:
    """Select CIFs for downstream optimization based on ILD/ILS pairs."""

    def _dedupe_selections(
        self, selections: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        seen: set[tuple[float, float]] = set()
        out: list[tuple[float, float]] = []
        for z, L in selections:
            key = (float(z), float(L))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _global_minima_from_csv(
        self, csv_path: Path
    ) -> list[tuple[float, float]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df2 = df.dropna(subset=["z", "L", "energy_eV"]).copy()
        if df2.empty:
            raise ValueError(f"CSV has no valid z/L/energy rows: {csv_path}")
        min_val = df2["energy_eV"].min()
        sel = df2[df2["energy_eV"] == min_val]
        selections = list(
            zip(sel["z"].astype(float), sel["L"].astype(float), strict=False)
        )
        return self._dedupe_selections(selections)

    def _local_minima_from_csv(
        self, csv_path: Path
    ) -> list[tuple[float, float]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df2 = df.dropna(subset=["z", "L", "energy_eV"]).copy()
        if df2.empty:
            raise ValueError(f"CSV has no valid z/L/energy rows: {csv_path}")

        abs_grid = df2.pivot_table(
            index="z", columns="L", values="energy_eV", aggfunc="last"
        ).sort_index()
        if abs_grid.empty:
            return []

        data = np.array(abs_grid.to_numpy(), dtype=float)
        minima_idx = Landscape()._find_local_minima(data)
        if not minima_idx:
            return []

        z_vals = list(abs_grid.index)
        L_vals = list(abs_grid.columns)
        selections = [
            (float(z_vals[i]), float(L_vals[j])) for i, j in minima_idx
        ]
        return self._dedupe_selections(selections)

    def _parse_z_L_from_stem(self, stem: str):
        mz = re.search(r"_z(\d+)", stem)
        mL = re.search(r"_L(\d+)", stem)
        if not (mz and mL):
            return None, None
        z = float(mz.group(1)) / 10.0
        L = float(mL.group(1)) / 10.0
        return z, L

    def run(
        self,
        input_folder: str,
        output_folder: str,
        selections: list[tuple[float, float]] | None = None,
        mode_label: str | None = None,
    ) -> None:
        if not selections:
            raise ValueError(
                "selections must be a non-empty list of (z, L) tuples"
            )

        in_path = Path(input_folder)
        out_path = Path(output_folder)
        os.makedirs(out_path, exist_ok=True)

        cif_files = sorted(in_path.glob("*.cif"))
        if not cif_files:
            raise FileNotFoundError(
                f"No .cif files found in: {in_path.resolve()}"
            )

        remaining = set(selections)
        selected_rows: list[dict[str, object]] = []
        for cif_path in cif_files:
            z, L = self._parse_z_L_from_stem(cif_path.stem)
            if z is None or L is None:
                continue
            for z_sel, L_sel in list(remaining):
                if z == z_sel and L_sel == L:
                    shutil.copy2(cif_path, out_path / cif_path.name)
                    selected_rows.append(
                        {
                            "structure": cif_path.name,
                            "ILD (Å)": z,
                            "ILS (Å)": L,
                            "output_folder": str(out_path),
                        }
                    )
                    remaining.discard((z_sel, L_sel))

        if remaining:
            missing = ", ".join([f"(z={z}, L={L})" for z, L in remaining])
            raise FileNotFoundError(f"No matching CIFs found for: {missing}")

        if selected_rows:
            df = pd.DataFrame(selected_rows)
            df = df[["ILD (Å)", "ILS (Å)"]]
            df = df.sort_values(["ILD (Å)", "ILS (Å)"]).reset_index(drop=True)
            if mode_label:
                print(f"\nSelected ILD/ILS pairs ({mode_label}):")
            else:
                print("\nSelected ILD/ILS pairs:")
            print(df.to_string(index=False))

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        selections_serr: list[tuple[float, float]] | None = None,
        selections_incl: list[tuple[float, float]] | None = None,
        include_autoselect: bool = False,
        input_base: str | None = None,
        output_base: str | None = None,
        input_folder: str | None = None,
        output_folder: str | None = None,
    ) -> None:
        """Select CIFs for selected mode(s) and copy them into selection folders.

        Args:
            cof_name: COF name used for folder naming.
            mode: "incl", "serr", or "both".
            selections_serr: Extra selections for serrated only.
            selections_incl: Extra selections for inclined only.
            include_autoselect: If True, auto-select local minima per mode.
            input_base: Optional base folder containing mode subfolders.
                Defaults to {cof_name}/2_{cof_name}_matrix.
            output_base: Optional base folder for selected CIFs.
                Defaults to {cof_name}/3_{cof_name}_landscape/selection.
            input_folder: Optional explicit folder for one mode (serr or incl).
                If set, this folder is used directly and `input_base`/`mode`
                folder expansion is not used.
            output_folder: Optional explicit output folder for selected CIFs.
                Used with `input_folder` for single-folder selection.
        """

        def _build_mode_selections(mode_tag: str) -> list[tuple[float, float]]:
            mode_selections: list[tuple[float, float]] = []
            if include_autoselect:
                csv_dir = Path(f"{cof_name}/3_{cof_name}_landscape")
                matches = list(
                    csv_dir.glob(f"{cof_name}_sp_energies_{mode_tag}_*.csv")
                )
                if matches:
                    csv_path = max(matches, key=lambda p: p.stat().st_mtime)
                else:
                    csv_path = (
                        csv_dir / f"{cof_name}_sp_energies_{mode_tag}.csv"
                    )
                mode_selections.extend(self._local_minima_from_csv(csv_path))
            if mode_tag == "serr" and selections_serr:
                mode_selections.extend(selections_serr)
            if mode_tag == "incl" and selections_incl:
                mode_selections.extend(selections_incl)

            mode_selections = self._dedupe_selections(mode_selections)
            if not mode_selections:
                raise ValueError(
                    "No selections provided. Use include_autoselect=True or provide selections_serr/selections_incl."
                )
            return mode_selections

        if input_folder:
            mode_tag = Path(input_folder).name.replace("dft_", "")
            if mode_tag not in {"serr", "incl"}:
                raise ValueError(
                    "input_folder must point to a serr or incl mode folder."
                )

            target_output = output_folder
            if target_output is None:
                base = (
                    output_base
                    or f"{cof_name}/3_{cof_name}_landscape/selection"
                )
                target_output = f"{base}/{mode_tag}"

            selections = _build_mode_selections(mode_tag)
            label = "Serrated" if mode_tag == "serr" else "Inclined"
            self.run(
                input_folder=input_folder,
                output_folder=target_output,
                selections=selections,
                mode_label=label,
            )
            return

        if input_base is None:
            input_base = f"{cof_name}/2_{cof_name}_matrix"
        if output_base is None:
            output_base = f"{cof_name}/3_{cof_name}_landscape/selection"
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = Path(folder).name
            out_folder = f"{output_base}/{mode_tag}"
            mode_selections = _build_mode_selections(mode_tag)

            label = (
                "Serrated"
                if mode_tag == "serr"
                else "Inclined"
                if mode_tag == "incl"
                else None
            )
            self.run(
                input_folder=f"{input_base}/{mode_tag}",
                output_folder=out_folder,
                selections=mode_selections,
                mode_label=label,
            )
