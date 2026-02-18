from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

EH_TO_KJMOL = 2625.5


class Landscape:
    def run(
        self,
        input_folder: str | None = None,
        input_csv: str | None = None,
        output_heatmap_dir: str | None = None,
        colorscheme: str = "default",
        plot_mode: str = "heatmap",
        rel_energy_max: float | None = None,
    ) -> Path:
        if not input_folder and not input_csv:
            raise ValueError("Provide input_folder or input_csv.")

        if input_csv:
            csv_path = Path(input_csv)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")
            csv_dir = csv_path.parent
            folder_tag = csv_dir.name
        else:
            input_path = Path(input_folder)
            folder_tag = input_path.name
            if input_path.parent.name.endswith("_matrix"):
                cof_name = input_path.parents[1].name
                csv_dir = Path(f"{cof_name}/3_{cof_name}_landscape")
                csv_path = csv_dir / f"{cof_name}_energies_{folder_tag}.csv"
            else:
                csv_dir = Path(f"csvs/{folder_tag}")
                csv_path = csv_dir / "energy_absolute.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")

        if output_heatmap_dir:
            heatmap_dir = Path(output_heatmap_dir)
        elif input_folder and Path(input_folder).parent.name.endswith("_matrix"):
            cof_name = Path(input_folder).parents[1].name
            heatmap_dir = Path(f"{cof_name}/3_{cof_name}_landscape/heatmaps_{folder_tag}")
        else:
            heatmap_dir = Path(f"heatmaps/{folder_tag}")
        os.makedirs(heatmap_dir, exist_ok=True)

        if input_folder and Path(input_folder).parent.name.endswith("_matrix"):
            cof_name = Path(input_folder).parents[1].name
            rel_grid_csv_path = csv_dir / f"{cof_name}_energies_relative_{folder_tag}.csv"
        else:
            rel_grid_csv_path = csv_dir / "energy_relative.csv"
        heatmap_path = heatmap_dir / "heatmap.png"

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

        rel_grid.to_csv(rel_grid_csv_path, index=True)

        plt.figure(figsize=(10, 6))
        data = rel_grid.values

        cmap = self._resolve_cmap(colorscheme)
        mode = (plot_mode or "heatmap").lower()
        nrows, ncols = data.shape
        vmax = float(rel_energy_max) if rel_energy_max is not None else None
        if mode == "heatmap":
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
        elif mode in {"isolines", "contour", "contours"}:
            if vmax is not None:
                levels = np.linspace(0.0, vmax, 12)
            else:
                levels = 12
            im = plt.contour(data, levels=levels, cmap=cmap)
            cbar = plt.colorbar(im, pad=0.02)
        else:
            raise ValueError("plot_mode must be 'heatmap' or 'isolines'.")
        cbar.set_label("Relative energy (kJ/mol)", labelpad=18, fontsize=12)

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
        plt.title("Potential Energy Landscape (MACE)", fontsize=14)

        finite_vals = np.array(data, dtype=float)
        min_pos = np.unravel_index(np.nanargmin(finite_vals), finite_vals.shape)
        y, x = min_pos
        if mode == "heatmap":
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
            if mode == "heatmap":
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

        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=200)
        plt.show()

        print(f"Saved: {heatmap_path}")
        return heatmap_path

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
        key = (colorscheme or "default").lower()
        if key in {"default", "normal"}:
            return None
        if key in {"grey", "gray", "greys"}:
            return "Greys"
        if key in {"colorblind", "cb", "cividis"}:
            return "cividis"
        if key in {"viridis"}:
            return "viridis"
        raise ValueError(
            "Unknown colorscheme. Use 'default', 'grey', 'colorblind', 'cividis', or 'viridis'."
        )
