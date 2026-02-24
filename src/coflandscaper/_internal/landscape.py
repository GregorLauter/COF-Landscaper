from pathlib import Path
import os
import re
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .ild_ils_utils import get_mode_folders

EH_TO_KJMOL = 2625.5

class Landscape:
    """Generate potential energy landscapes from SP energy CSVs."""

    def run(
        self,
        input_folder: str,
        output_folder: str | None = None,
        colorscheme: str = "viridis",
        plot_mode: str = "both",
        rel_energy_max: float | None = None,
    ) -> None:
        """Build the PES plots for a given stacking folder.

        Args:
            input_folder: Folder containing the CIFs for a stacking mode.
                Defaults to {cof_name}/2_{cof_name}_matrix/{serr|incl} when
                used via `run_mode`.
            output_folder: Optional output folder for plots.
                Defaults to {cof_name}/3_{cof_name}_landscape.
            colorscheme: Heatmap colorscheme. Options: "grey",
                "colorblind" ("cividis"), or "viridis" (default).
            plot_mode: "heatmap", "isolines", or "both".
            rel_energy_max: Optional max value (kJ/mol) to cap relative energies.
                Values above this are clipped in the plots.

        Returns:
            None.
        """
        input_path = Path(input_folder)
        folder_tag = input_path.name
        if input_path.parent.name.endswith("_matrix"):
            cof_name = input_path.parents[1].name
            csv_dir = Path(f"{cof_name}/3_{cof_name}_landscape")
            csv_matches = list(csv_dir.glob(f"{cof_name}_sp_energies_{folder_tag}_*.csv"))
            if csv_matches:
                csv_path = max(csv_matches, key=lambda p: p.stat().st_mtime)
            else:
                csv_path = csv_dir / f"{cof_name}_sp_energies_{folder_tag}.csv"
        else:
            cof_name = None
            csv_dir = Path(f"csvs/{folder_tag}")
            csv_path = csv_dir / "energy_absolute.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        lot_suffix = None
        match = re.match(r"(.+)_sp_energies_(serr|incl)_(.+?)(?:\.csv)?$", csv_path.name)
        if match:
            lot_suffix = match.group(3)

        if output_folder:
            heatmap_dir = Path(output_folder)
        elif Path(input_folder).parent.name.endswith("_matrix") and cof_name:
            heatmap_dir = Path(f"{cof_name}/3_{cof_name}_landscape")
        else:
            heatmap_dir = Path(f"heatmaps/{folder_tag}")
        os.makedirs(heatmap_dir, exist_ok=True)

        lot_tag = f"_{lot_suffix}" if lot_suffix else ""

        if Path(input_folder).parent.name.endswith("_matrix") and folder_tag in {"serr", "incl"}:
            heatmap_path = heatmap_dir / f"pes_{cof_name}_{folder_tag}_heatmap{lot_tag}.png"
            isolines_path = heatmap_dir / f"pes_{cof_name}_{folder_tag}_isolines{lot_tag}.png"
            write_rel_csv = False
        else:
            rel_grid_csv_path = csv_dir / "energy_relative.csv"
            heatmap_path = heatmap_dir / f"heatmap{lot_tag}.png"
            isolines_path = heatmap_dir / f"isolines{lot_tag}.png"
            write_rel_csv = True

        df = pd.read_csv(csv_path)
        df2 = df.dropna(subset=["z", "L", "energy_Eh"]).copy()
        if df2.empty:
            raise ValueError("No entries with parsed z/L. Check naming like ..._z30_..._L020.cif")

        abs_grid = df2.pivot_table(index="z", columns="L", values="energy_Eh", aggfunc="last").sort_index()

        if abs_grid.empty:
            raise ValueError("No matching rows for a z/L grid. Check CSV content.")

        vals = np.array(abs_grid.values, dtype=float)
        mask = np.isfinite(vals)
        if not mask.any():
            raise ValueError("Grid has no finite energies (unexpected).")

        global_min = vals[mask].min()
        rel_grid = (abs_grid - global_min) * EH_TO_KJMOL
        if rel_energy_max is not None:
            rel_grid = rel_grid.clip(lower=0.0, upper=float(rel_energy_max))

        if write_rel_csv:
            rel_grid.to_csv(rel_grid_csv_path, index=True)

        plt.figure(figsize=(10, 6))
        data = rel_grid.values

        cmap = self._resolve_cmap(colorscheme)
        mode = (plot_mode or "heatmap").lower()
        nrows, ncols = data.shape
        vmax = float(rel_energy_max) if rel_energy_max is not None else None
        def _style_axes():
            plt.xlim(-0.5, ncols - 0.5)
            plt.ylim(-0.5, nrows - 0.5)
            plt.xticks(
                range(len(rel_grid.columns)),
                [f"{c:.1f}" for c in rel_grid.columns],
                rotation=45,
                ha="right",
                fontsize=10,
            )
            plt.yticks(range(len(rel_grid.index)), [f"{r:.1f}" for r in rel_grid.index], fontsize=10)
            plt.xlabel("Inter Layer Slipping [Å]", fontsize=12)
            plt.ylabel("Inter Layer Distance [Å]", fontsize=12)
            title_name = cof_name or "COF"
            plt.title(f"Potential Energy Landscape - {title_name}", fontsize=14, pad=36)
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
            if lot_suffix:
                plt.text(
                    0.5,
                    1.02,
                    f"Level of Theory: {lot_suffix}",
                    transform=plt.gca().transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        def _mark_minima(use_rect: bool) -> None:
            finite_vals = np.array(data, dtype=float)
            min_pos = np.unravel_index(np.nanargmin(finite_vals), finite_vals.shape)
            y, x = min_pos
            if use_rect:
                rect = patches.Rectangle(
                    (x - 0.5, y - 0.5),
                    1,
                    1,
                    linewidth=2.5,
                    edgecolor="red",
                    facecolor="none",
                )
                plt.gca().add_patch(rect)
            else:
                plt.scatter([x], [y], marker="x", color="red", s=120, linewidths=2.5)

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
                    ys, xs = zip(*local_minima)
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
            cbar.set_label("Relative energy (kJ/mol)", labelpad=18, fontsize=12)
            _style_axes()
            _mark_minima(use_rect=True)
            plt.tight_layout()
            plt.savefig(heatmap_path, dpi=200)
            plt.show()
            print(f"Saved: {heatmap_path}")
            paths.append(heatmap_path)

        if mode in {"isolines", "contour", "contours", "both"}:
            plt.figure(figsize=(10, 6))
            if vmax is not None:
                levels = np.linspace(0.0, vmax, 12)
            else:
                levels = 12
            im = plt.contour(data, levels=levels, cmap=cmap)
            cbar = plt.colorbar(im, pad=0.02)
            cbar.set_label("Relative energy (kJ/mol)", labelpad=18, fontsize=12)
            _style_axes()
            _mark_minima(use_rect=False)
            plt.tight_layout()
            plt.savefig(isolines_path, dpi=200)
            plt.show()
            print(f"Saved: {isolines_path}")
            paths.append(isolines_path)
        return None

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        colorscheme: str = "viridis",
        plot_mode: str = "both",
        rel_energy_max: float | None = None,
    ) -> None:
        """Generate landscapes for the selected mode(s) using MaceSP CSVs.

        Args:
            cof_name: COF name used for folder naming.
            mode: "incl", "serr", or "both".
            colorscheme: Heatmap colorscheme. Options: "grey",
                "colorblind" ("cividis"), or "viridis" (default).
            plot_mode: "heatmap", "isolines", or "both".
            rel_energy_max: Optional max value for relative energies.

        Returns:
            None.
        """
        for folder in get_mode_folders(cof_name, mode):
            self.run(
                input_folder=folder,
                colorscheme=colorscheme,
                plot_mode=plot_mode,
                rel_energy_max=rel_energy_max,
            )
        return None

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
        key = (colorscheme or "viridis").lower()
        if key in {"grey", "gray", "greys"}:
            return "Greys"
        if key in {"colorblind", "cb", "cividis"}:
            return "cividis"
        if key in {"viridis"}:
            return "viridis"
        raise ValueError(
            "Unknown colorscheme. Use 'grey', 'colorblind', 'cividis', or 'viridis'."
        )

class SelectCofs:
    """Select CIFs based on ILD/ILS pairs."""

    def _dedupe_selections(self, selections: list[tuple[float, float]]) -> list[tuple[float, float]]:
        seen: set[tuple[float, float]] = set()
        out: list[tuple[float, float]] = []
        for z, L in selections:
            key = (float(z), float(L))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _global_minima_from_csv(self, csv_path: Path) -> list[tuple[float, float]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df2 = df.dropna(subset=["z", "L", "energy_Eh"]).copy()
        if df2.empty:
            raise ValueError(f"CSV has no valid z/L/energy rows: {csv_path}")
        min_val = df2["energy_Eh"].min()
        sel = df2[df2["energy_Eh"] == min_val]
        selections = list(zip(sel["z"].astype(float), sel["L"].astype(float)))
        return self._dedupe_selections(selections)

    def _local_minima_from_csv(self, csv_path: Path) -> list[tuple[float, float]]:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df2 = df.dropna(subset=["z", "L", "energy_Eh"]).copy()
        if df2.empty:
            raise ValueError(f"CSV has no valid z/L/energy rows: {csv_path}")

        abs_grid = (
            df2.pivot_table(index="z", columns="L", values="energy_Eh", aggfunc="last")
            .sort_index()
        )
        if abs_grid.empty:
            return []

        data = np.array(abs_grid.values, dtype=float)
        minima_idx = Landscape()._find_local_minima(data)
        if not minima_idx:
            return []

        z_vals = list(abs_grid.index)
        L_vals = list(abs_grid.columns)
        selections = [(float(z_vals[i]), float(L_vals[j])) for i, j in minima_idx]
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
            raise ValueError("selections must be a non-empty list of (z, L) tuples")

        in_path = Path(input_folder)
        out_path = Path(output_folder)
        os.makedirs(out_path, exist_ok=True)

        cif_files = sorted(in_path.glob("*.cif"))
        if not cif_files:
            raise FileNotFoundError(f"No .cif files found in: {in_path.resolve()}")

        remaining = set(selections)
        selected_rows: list[dict[str, object]] = []
        for cif_path in cif_files:
            z, L = self._parse_z_L_from_stem(cif_path.stem)
            if z is None or L is None:
                continue
            for (z_sel, L_sel) in list(remaining):
                if z == z_sel and L == L_sel:
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
        input_base: str | None = None,
        output_base: str | None = None,
    ) -> None:
        """Select CIFs for the selected mode(s).

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
        """
        if input_base is None:
            input_base = f"{cof_name}/2_{cof_name}_matrix"
        if output_base is None:
            output_base = f"{cof_name}/3_{cof_name}_landscape/selection"
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = Path(folder).name
            out_folder = f"{output_base}/{mode_tag}"
            mode_selections: list[tuple[float, float]] = []
            if include_autoselect:
                csv_dir = Path(f"{cof_name}/3_{cof_name}_landscape")
                matches = list(csv_dir.glob(f"{cof_name}_sp_energies_{mode_tag}_*.csv"))
                if matches:
                    csv_path = max(matches, key=lambda p: p.stat().st_mtime)
                else:
                    csv_path = csv_dir / f"{cof_name}_sp_energies_{mode_tag}.csv"
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

            label = "Serrated" if mode_tag == "serr" else "Inclined" if mode_tag == "incl" else None
            self.run(
                input_folder=f"{input_base}/{mode_tag}",
                output_folder=out_folder,
                selections=mode_selections,
                mode_label=label,
            )

    __call__ = run
