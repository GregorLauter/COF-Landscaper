"""Generate simplified stacking-energy landscapes and select structures for optimization.

This module provides utilities for visualizing the single-point energy grid
generated from interlayer distance (ILD) and interlayer slipping (ILS) scans,
identifying global or local minima, and selecting corresponding CIF structures
for subsequent full geometry optimization.

The resulting potential energy landscape is a reduced-dimensional screening
representation because the matrix structures are evaluated without full geometry
relaxation.
"""

import re
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from .ild_ils_utils import get_mode_folders


class Landscape:
    """Generate simplified potential energy landscapes from ILD/ILS energy grids.

    The landscape is constructed from MACE or DFT single-point energies evaluated
    for the generated stacking matrix. Relative energies are mapped as a function
    of interlayer distance (ILD) and interlayer slipping (ILS).

    Because the individual matrix structures are not fully relaxed at this stage,
    the resulting PES should be interpreted as a reduced-dimensional screening
    landscape used to identify promising stacking configurations for subsequent
    full geometry optimization.
    """

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

    def _plot_single_mode(
        self,
        input_folder: str,
        cof_name: str | None = None,
        dft: bool = False,
        output_folder: str | None = None,
        colorscheme: str = "viridis",
        rel_energy_max: float | None = None,
        show_minima_markers: bool = True,
        minima_mode: str = "global",
        show_header: bool = True,
        show_title_block: bool = False,
        show: bool = False,
    ) -> None:
        """Build one fixed-style hybrid PES plot for a single stacking mode.

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
            rel_energy_max: Optional max value (eV) to cap relative energies.
                Values above this are clipped in the plots. Defaults to `None`.
            show_minima_markers: If `True`, mark the global minimum with a red
                star and (for `minima_mode="local"`) additional local minima
                with orange stars. Defaults to `True`.
            minima_mode: Minima marker mode: "global" (default) marks only the
                single global minimum; "local" marks local minima as well.
                Defaults to `"global"`.
            show_header: If `True`, draw title and header text. Defaults to `True`.
            show_title_block: If True, draw title plus two header lines
                (stacking mode and level of theory when available).
                Defaults to `False`.
            show: If `True`, display plots interactively via Matplotlib.
                Defaults to `False` for non-interactive/batch workflows.

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
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        lot_tag = f"_{lot_suffix}" if lot_suffix else ""
        rel_grid_csv_path: Path | None = None

        if use_mode_naming:
            figure_path = (
                heatmap_dir / f"pes_{cof_name}_{folder_tag}{lot_tag}.png"
            )
            write_rel_csv = False
        else:
            rel_grid_csv_path = csv_dir / "energy_relative.csv"
            figure_path = heatmap_dir / f"pes_{folder_tag}{lot_tag}.png"
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

        data = rel_grid.to_numpy(dtype=float)

        cmap = self._resolve_cmap(colorscheme)
        minima_mode_norm = (minima_mode or "global").strip().lower()
        if minima_mode_norm not in {"global", "local"}:
            raise ValueError("minima_mode must be 'global' or 'local'.")
        vmax = float(rel_energy_max) if rel_energy_max is not None else None
        x_vals = rel_grid.columns.to_numpy(dtype=float)
        y_vals = rel_grid.index.to_numpy(dtype=float)
        xx, yy = np.meshgrid(x_vals, y_vals)
        finite = np.isfinite(data)
        if int(finite.sum()) < 3:
            raise ValueError(
                "Need at least three finite points to triangulate PES."
            )
        x_points = xx[finite]
        y_points = yy[finite]
        e_points = data[finite]

        tri = mtri.Triangulation(x_points, y_points)

        vmin = float(np.nanmin(e_points))
        vmax_data = float(np.nanmax(e_points))
        if np.isclose(vmin, vmax_data):
            vmax_data = vmin + 1e-9
        if vmax is not None:
            vmax_data = min(vmax_data, vmax)

        fill_levels = np.linspace(vmin, vmax_data, 24)
        line_levels = np.linspace(vmin, vmax_data, 14)

        def _style_axes(ax: plt.Axes) -> None:
            ax.set_xlabel("Inter Layer Slipping [Å]", fontsize=12)
            ax.set_ylabel("Inter Layer Distance [Å]", fontsize=12)
            ax.set_xlim(float(np.nanmin(x_points)), float(np.nanmax(x_points)))
            ax.set_ylim(float(np.nanmin(y_points)), float(np.nanmax(y_points)))
            ax.grid(visible=True, alpha=0.16, linewidth=0.6)
            if show_header and show_title_block:
                title_name = cof_name or "COF"
                title_prefix = (
                    "Potential Energy Landscape (DFT)"
                    if dft
                    else "Potential Energy Landscape"
                )
                ax.set_title(
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
                    ax.text(
                        0.5,
                        1.06,
                        f"Stacking Mode: {mode_label}",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
                if lot_label:
                    ax.text(
                        0.5,
                        1.02,
                        f"Level of Theory: {lot_label}",
                        transform=ax.transAxes,
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        cf = ax.tricontourf(
            tri,
            e_points,
            levels=fill_levels,
            cmap=cmap,
            extend="both",
            antialiased=True,
        )
        cs = ax.tricontour(
            tri,
            e_points,
            levels=line_levels,
            colors="k",
            linewidths=0.65,
            alpha=0.45,
        )
        ax.clabel(cs, inline=True, fmt="%.1f", fontsize=10)

        if show_minima_markers:
            min_pos = np.unravel_index(np.nanargmin(data), data.shape)
            g_row, g_col = min_pos
            ax.scatter(
                [x_vals[g_col]],
                [y_vals[g_row]],
                marker="*",
                s=250,
                facecolor="red",
                edgecolor="white",
                linewidth=1.4,
                zorder=7,
                label="Global Minimum",
            )

            if minima_mode_norm == "local":
                local_minima = [
                    (ly, lx)
                    for ly, lx in self._find_local_minima(data)
                    if (ly, lx) != (g_row, g_col)
                ]
                if local_minima:
                    ys, xs = zip(*local_minima, strict=False)
                    ax.scatter(
                        [x_vals[ix] for ix in xs],
                        [y_vals[iy] for iy in ys],
                        marker="*",
                        s=180,
                        facecolor="orange",
                        edgecolor="white",
                        linewidth=1.2,
                        zorder=6,
                        label="Local Minima",
                    )

            ax.legend(loc="upper left", frameon=True, framealpha=0.95)

        _style_axes(ax)
        cbar = fig.colorbar(cf, ax=ax, pad=0.02)
        cbar.set_label("Relative energy [eV]", fontsize=12)
        cbar.ax.yaxis.set_major_formatter(
            mpl.ticker.FormatStrFormatter("%.1f")
        )

        fig.savefig(figure_path, dpi=200)
        if show:
            plt.show()
        else:
            plt.close(fig)
        print(f"Saved: {figure_path}")

    def run(
        self,
        cof_name: str,
        mode: str,
        dft: bool = False,
        colorscheme: str = "viridis",
        rel_energy_max: float | None = None,
        show_minima_markers: bool = True,
        minima_mode: str = "global",
        show_header: bool = True,
        show_title_block: bool = False,
        show: bool = False,
        input_folder: str | None = None,
        output_folder: str | None = None,
    ) -> None:
        """Generate potential energy landscapes for selected stacking mode(s).

        The method reads mode-specific single-point energy CSV files, converts absolute
        energies to relative energies, and plots the resulting ILD/ILS potential energy
        landscape. The global minimum can be marked automatically, and additional local
        minima can be shown when ``minima_mode="local"``.

        By default, energy data are read from
        ``{cof_name}/3_{cof_name}_landscape`` and the generated PES figures are written
        to the same directory.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            dft: If ``True``, read DFT single-point CSV files with the ``_dft`` suffix.
                Defaults to ``False``.
            colorscheme: Matplotlib colormap used for the PES. Defaults to ``"viridis"``.
            rel_energy_max: Optional upper limit for displayed relative energies.
                Values above the limit are clipped. Defaults to ``None``.
            show_minima_markers: Whether to mark detected minima on the PES. Defaults to
                ``True``.
            minima_mode: Minima mode used for plotting: ``"global"`` marks only the
                global minimum, while ``"local"`` also marks local minima. Defaults to
                ``"global"``.
            show_header: Whether plot header information is enabled. Defaults to
                ``True``.
            show_title_block: Whether to display the title, stacking mode, and level
                of theory above the plot. Defaults to ``False``.
            show: Whether to display generated figures interactively. Defaults to
                ``False`` for batch/HPC-compatible execution.
            input_folder: Optional base folder containing the mode-specific energy CSV
                files and ``serr``/``incl`` subfolders. Defaults to
                ``{cof_name}/3_{cof_name}_landscape``.
            output_folder: Optional output folder for PES figures. Defaults to
                ``{cof_name}/3_{cof_name}_landscape``.

        Raises:
            ValueError: If ``mode`` or ``minima_mode`` is invalid, or the energy grid
                does not contain sufficient valid data.
            FileNotFoundError: If the input folder or required energy CSV files are
                missing.
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

            self._plot_single_mode(
                input_folder=str(base_path / mode_tag),
                cof_name=cof_name,
                dft=dft,
                output_folder=output_folder,
                colorscheme=colorscheme,
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
    """Select stacking structures for subsequent full geometry optimization.

    Structures can be selected automatically from global or local minima of the
    simplified ILD/ILS energy landscape and supplemented with user-defined sampling
    points. Selected CIF files are copied from the matrix folders into the
    optimization-selection folders.
    """

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

    def _copy_selected_cifs(
        self,
        input_folder: str,
        output_folder: str,
        selections: list[tuple[float, float]] | None = None,
        mode_label: str | None = None,
    ) -> None:
        """Copy CIFs matching explicit `(ILD, ILS)` selections.

        Args:
            input_folder: Source folder containing CIF files.
            output_folder: Destination folder for selected CIF files.
            selections: Selection tuples `(z, L)` in Angstrom.
                Defaults to `None` (must be non-empty at runtime).
            mode_label: Optional display label used in console output.
                Defaults to `None`.

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
        out_path.mkdir(parents=True, exist_ok=True)

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

    def run(
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
        """Select candidate CIF structures for full geometry optimization.

        By default, the global minimum of each requested stacking mode is selected
        automatically from the corresponding single-point energy CSV. Setting
        ``autoselect_minima="local"`` selects local minima instead.

        Additional ILD/ILS sampling points can be supplied independently for serrated
        and inclined stacking using ``selections_serr`` and ``selections_incl``. These
        manual selections are combined with the automatically selected minima when
        ``include_autoselect=True``.

        Default folder behavior:

        - matrix structures:
          ``{cof_name}/2_{cof_name}_matrix/{mode}``
        - selected structures:
          ``{cof_name}/3_{cof_name}_landscape/selection/{mode}``

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            selections_serr: Optional additional ``(ILD, ILS)`` pairs for serrated
                stacking. Defaults to ``None``.
            selections_incl: Optional additional ``(ILD, ILS)`` pairs for inclined
                stacking. Defaults to ``None``.
            include_autoselect: Whether automatically detected minima are included in
                addition to any manual selections. Defaults to ``True``.
            autoselect_minima: Automatic selection mode: ``"global"`` selects the
                global minimum, while ``"local"`` selects detected local minima.
                Defaults to ``"global"``.
            input_base: Optional matrix input base folder. Defaults to
                ``{cof_name}/2_{cof_name}_matrix``.
            output_base: Optional selection output base folder. Defaults to
                ``{cof_name}/3_{cof_name}_landscape/selection``.
            input_folder: Optional explicit single-mode input folder. When provided,
                mode-folder expansion from ``input_base`` is bypassed.
            output_folder: Optional explicit output folder used together with
                ``input_folder``.

        Raises:
            ValueError: If the minima mode is invalid, an explicit input folder does
                not correspond to ``serr`` or ``incl``, or no automatic/manual
                selections are available.
            FileNotFoundError: If required energy CSV files or requested matrix CIF
                structures are missing.
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
            self._copy_selected_cifs(
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
            self._copy_selected_cifs(
                input_folder=f"{input_base}/{mode_tag}",
                output_folder=out_folder,
                selections=mode_selections,
                mode_label=label,
            )
