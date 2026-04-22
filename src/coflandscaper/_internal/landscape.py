"""Plot PES landscapes and select candidate CIFs from ILD/ILS grids.

This module provides visualization utilities for single-point energy grids and
selection utilities for copying structures corresponding to global or local
minima into optimization-ready folders.
"""

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
        """Resolve mode folder and expected CSV path for one landscape run.

        Args:
            input_folder: Mode folder path (`serr` or `incl`).
            cof_name: COF identifier used in CSV naming.
            dft: If `True`, resolve CSV with `_dft` suffix. Defaults to `False`.

        Returns:
            Tuple `(csv_dir, csv_path, mode_tag, cof_name)`.

        Raises:
            ValueError: If `input_folder` is not a `serr`/`incl` mode folder.
            ValueError: If `cof_name` is not provided.
        """
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
        minima_mode: str = "global",
        show_header: bool = True,
        show_title_block: bool = False,
        show: bool = False,
    ) -> None:
        """Build PES plots for one stacking mode.

        Args:
            input_folder: Mode folder path (serr or incl) used to infer mode;
                its parent must contain {cof_name}_sp_energies_{mode}.csv
                (or {cof_name}_sp_energies_{mode}_dft.csv when dft=True).
            cof_name: COF identifier used for expected CSV naming. Defaults to
                `None` (must be provided explicitly for this method).
            dft: If `True`, read input CSVs with `_dft` suffix. Defaults to `False`.
            output_folder: Optional output folder for plots.
                Defaults to `None` (uses `{cof_name}/3_{cof_name}_landscape`).
            colorscheme: Any valid Matplotlib colormap name.
                Defaults to `"viridis"`.
            plot_mode: Plot variant selector. Allowed values are `"heatmap"`,
                `"isolines"`, or `"both"`. Defaults to `"both"`.
            rel_energy_max: Optional max value (eV) to cap relative energies.
                Values above this are clipped in the plots. Defaults to `None`.
            show_minima_markers: If True (default), mark global and local minima
                in red on heatmap/isolines. Defaults to `True`.
            minima_mode: Minima marker mode: "global" (default) marks only the
                single global minimum; "local" marks local minima as well.
                Defaults to `"global"`.
            show_header: If `True`, draw title and header text. Defaults to `True`.
            show_title_block: If True, draw title plus two header lines
                (stacking mode and level of theory when available).
                Defaults to `False`.
            show: If `True`, display plots interactively via Matplotlib.
                Defaults to `False` for non-interactive/batch workflows.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the expected input CSV is missing.
            ValueError: If minima mode is invalid or no valid grid data exist.
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
        rel_grid_csv_path: Path | None = None

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
            if rel_grid_csv_path is None:
                raise RuntimeError(
                    "Internal error: rel_grid_csv_path is unset"
                )
            rel_grid.to_csv(rel_grid_csv_path, index=True)

        plt.figure(figsize=(10, 6))
        data = rel_grid.to_numpy()

        cmap = self._resolve_cmap(colorscheme)
        mode = (plot_mode or "heatmap").lower()
        minima_mode_norm = (minima_mode or "global").strip().lower()
        if minima_mode_norm not in {"global", "local"}:
            raise ValueError("minima_mode must be 'global' or 'local'.")
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

            if minima_mode_norm == "global":
                return

            local_minima = self._find_local_minima(finite_vals)
            if local_minima:
                if use_rect:
                    for ly, lx in local_minima:
                        rect = patches.Rectangle(
                            (lx - 0.5, ly - 0.5),
                            1,
                            1,
                            linewidth=2.0,
                            edgecolor="red",
                            facecolor="none",
                        )
                        plt.gca().add_patch(rect)
                else:
                    ys, xs = zip(*local_minima, strict=False)
                    plt.scatter(
                        xs,
                        ys,
                        marker="x",
                        color="red",
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
            if show:
                plt.show()
            else:
                plt.close()
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
            if show:
                plt.show()
            else:
                plt.close()
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
        minima_mode: str = "global",
        show_header: bool = True,
        show_title_block: bool = False,
        show: bool = False,
        input_folder: str | None = None,
        output_folder: str | None = None,
    ) -> None:
        """Generate landscapes for selected stacking mode(s).

        Args:
            cof_name: COF name used for folder naming.
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`.
            dft: If `True`, read input CSVs with `_dft` suffix.
                Defaults to `False`.
            colorscheme: Any valid Matplotlib colormap name.
                Defaults to `"viridis"`.
            plot_mode: Plot variant selector. Allowed values are `"heatmap"`,
                `"isolines"`, or `"both"`. Defaults to `"both"`.
            rel_energy_max: Optional max value for relative energies.
                Defaults to `None`.
            show_minima_markers: If `True`, mark minima on plots.
                Defaults to `True`.
            minima_mode: "global" (default) marks only one global minimum;
                "local" includes local minima markers too. Defaults to `"global"`.
            show_header: If `True`, draw title and header text. Defaults to `True`.
            show_title_block: If `True`, draw title plus two header lines.
                Defaults to `False`.
            show: If `True`, display plots interactively. Defaults to `False`
                for cluster/batch runs.
            input_folder: Optional base folder containing mode folders and
                {cof_name}_sp_energies_{mode}.csv files, or
                {cof_name}_sp_energies_{mode}_dft.csv when dft=True.
                Defaults to `None` (uses `{cof_name}/3_{cof_name}_landscape`).
            output_folder: Optional output folder for plots.
                Defaults to `None` (uses `{cof_name}/3_{cof_name}_landscape`).

        Returns:
            None.

        Raises:
            ValueError: If `mode` is invalid.
            FileNotFoundError: If base input folder or expected CSVs are missing.
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
                minima_mode=minima_mode,
                show_header=show_header,
                show_title_block=show_title_block,
                show=show,
            )

        if missing_csvs:
            raise FileNotFoundError(
                "Missing expected CSV(s): " + ", ".join(missing_csvs)
            )

    def _find_local_minima(self, data: np.ndarray) -> list[tuple[int, int]]:
        """Find strict local minima on a 2D finite-valued grid.

        Args:
            data: 2D array of energy values.

        Returns:
            List of `(row, col)` minima indices.
        """
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
        """Validate and return a Matplotlib colormap name.

        Args:
            colorscheme: Requested colormap name.

        Returns:
            Valid colormap name string.

        Raises:
            ValueError: If the colormap name is unknown.
        """
        raw = colorscheme or "viridis"
        try:
            plt.get_cmap(raw)
            return raw
        except ValueError as exc:
            raise ValueError(
                "Unknown colorscheme. Use any valid Matplotlib colormap name "
                "(e.g. 'viridis', 'plasma', 'magma', 'cividis', 'coolwarm')."
            ) from exc


class SelectCofs:
    """Select CIFs for downstream optimization based on ILD/ILS pairs."""

    def _dedupe_selections(
        self, selections: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Remove duplicate `(ILD, ILS)` tuples while preserving order.

        Args:
            selections: Input selection tuples.

        Returns:
            Deduplicated selection list.
        """
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
        """Select the global minimum `(z, L)` pair from one energy CSV.

        Args:
            csv_path: CSV path with `z`, `L`, and `energy_eV` columns.

        Returns:
            One-element list containing the global-minimum pair.

        Raises:
            FileNotFoundError: If CSV is missing.
            ValueError: If CSV has no valid `z/L/energy` rows.
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df2 = df.dropna(subset=["z", "L", "energy_eV"]).copy()
        if df2.empty:
            raise ValueError(f"CSV has no valid z/L/energy rows: {csv_path}")
        sel = (
            df2.sort_values(["energy_eV", "z", "L"])
            .head(1)
            .reset_index(drop=True)
        )
        z_num = pd.to_numeric(sel["z"], errors="coerce").iloc[0]
        l_num = pd.to_numeric(sel["L"], errors="coerce").iloc[0]
        if pd.isna(z_num) or pd.isna(l_num):
            raise ValueError(
                f"CSV global minimum has non-numeric z/L values: {csv_path}"
            )
        return [(float(z_num), float(l_num))]

    def _local_minima_from_csv(
        self, csv_path: Path
    ) -> list[tuple[float, float]]:
        """Select local-minimum `(z, L)` pairs from one energy CSV.

        Args:
            csv_path: CSV path with `z`, `L`, and `energy_eV` columns.

        Returns:
            Deduplicated local-minimum selection list.

        Raises:
            FileNotFoundError: If CSV is missing.
            ValueError: If CSV has no valid `z/L/energy` rows.
        """
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
        """Parse `_z` and `_L` tags from a CIF stem.

        Args:
            stem: CIF filename stem.

        Returns:
            Tuple `(z, L)` in Angstrom, or `(None, None)` when missing.
        """
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
        """Copy CIFs that match requested `(ILD, ILS)` tuples.

        Args:
            input_folder: Source folder containing CIF files.
            output_folder: Destination folder for selected CIF files.
            selections: Selection tuples `(z, L)` in Angstrom.
                Defaults to `None` (must be non-empty at runtime).
            mode_label: Optional display label used in console output.
                Defaults to `None`.

        Returns:
            None.

        Raises:
            ValueError: If `selections` is empty.
            FileNotFoundError: If no CIF files exist or requested pairs are missing.
        """
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
        include_autoselect: bool = True,
        autoselect_minima: str = "global",
        input_base: str | None = None,
        output_base: str | None = None,
        input_folder: str | None = None,
        output_folder: str | None = None,
    ) -> None:
        """Select CIFs for selected mode(s) and copy them into selection folders.

        Args:
            cof_name: COF name used for folder naming.
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`.
            selections_serr: Extra selections for serrated only.
                Defaults to `None`.
            selections_incl: Extra selections for inclined only.
                Defaults to `None`.
            include_autoselect: If `True`, include automatically selected minima.
                Defaults to `True`.
            autoselect_minima: Minima mode for auto-selection:
                "global" (default) selects one global minimum,
                "local" selects all local minima. Defaults to `"global"`.
            input_base: Optional base folder containing mode subfolders.
                Defaults to `None` (uses `{cof_name}/2_{cof_name}_matrix`).
            output_base: Optional base folder for selected CIFs.
                Defaults to `None`
                (uses `{cof_name}/3_{cof_name}_landscape/selection`).
            input_folder: Optional explicit folder for one mode (serr or incl).
                If set, this folder is used directly and `input_base`/`mode`
                folder expansion is not used. Defaults to `None`.
            output_folder: Optional explicit output folder for selected CIFs.
                Used with `input_folder` for single-folder selection.
                Defaults to `None`.

        Returns:
            None.

        Raises:
            ValueError: If minima mode is invalid or no selections are available.
            ValueError: If explicit `input_folder` is not a mode folder.
        """
        autoselect_mode = (autoselect_minima or "global").strip().lower()
        if autoselect_mode not in {"global", "local"}:
            raise ValueError("autoselect_minima must be 'global' or 'local'.")

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
                if autoselect_mode == "global":
                    mode_selections.extend(
                        self._global_minima_from_csv(csv_path)
                    )
                else:
                    mode_selections.extend(
                        self._local_minima_from_csv(csv_path)
                    )
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
