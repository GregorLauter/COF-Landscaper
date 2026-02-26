"""Analysis helpers for ILD/ILS, structure checks, and visualization."""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pymatgen.core import Structure

from .ild_ils_utils import (
    _calculate_ild,
    _unwrap_fractional_z,
    list_cifs,
    parse_xyz_from_atom_line,
    pick_lower_left_pair_from_lines,
    wrap01,
)

ModelFormat = Literal["cif", "xyz", "pdb", "mol", "mol2", "sdf"]


class CalcIlsDl:
    def run(self, input_folder: str):
        for input_file in list_cifs(input_folder):
            struct = Structure.from_file(input_file)

            with open(input_file) as f:
                lines = f.readlines()

            atom_lines = []
            in_atom_loop = False
            saw_atom_headers = False

            for line in lines:
                ls = line.strip()
                if ls.startswith("loop_"):
                    in_atom_loop = False
                    saw_atom_headers = False
                    continue

                if ls.startswith("_atom_site_"):
                    saw_atom_headers = True
                    in_atom_loop = True
                    continue

                if in_atom_loop and saw_atom_headers:
                    if not ls or ls.startswith("_") or ls.startswith("loop_"):
                        break
                    if ls and ls[0].isalpha():
                        atom_lines.append(ls)

            if not atom_lines:
                raise ValueError("No atom lines found in CIF")

            pair_idx, (lower_line, upper_line), (xl, yl, zl), key = (
                pick_lower_left_pair_from_lines(atom_lines)
            )
            xu, yu, zu = parse_xyz_from_atom_line(upper_line)
            xu_w, yu_w = wrap01(xu), wrap01(yu)

            dxu = xu_w - xl
            if dxu < 0.0:
                dxu += 1.0
            dyu = yu_w - yl
            if dyu < 0.0:
                dyu += 1.0

            a_vec, b_vec, _ = struct.lattice.matrix
            a_xy = np.array([a_vec[0], a_vec[1]])
            b_xy = np.array([b_vec[0], b_vec[1]])

            slip_vec_xy = dxu * a_xy + dyu * b_xy
            slip_mag = float(np.linalg.norm(slip_vec_xy))

            print(f"{os.path.basename(input_file)}: ILS = {slip_mag:.2f} Å")

    __call__ = run


class CalcIlsSl:
    def run(self, input_folder: str):
        for input_file in list_cifs(input_folder):
            struct = Structure.from_file(input_file)
            lat = struct.lattice
            a, b, c = lat.abc
            alpha_deg, beta_deg, gamma_deg = lat.angles
            ar = math.radians(alpha_deg)
            br = math.radians(beta_deg)
            gr = math.radians(gamma_deg)

            cx = c * math.cos(br)
            cy = (
                c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / math.sin(gr)
            )
            slip = math.sqrt(cx**2 + cy**2)

            print(f"{os.path.basename(input_file)}: ILS = {slip:.2f} Å")

    __call__ = run


class CheckIld:
    def run(self, input_folder: str):
        for input_file in list_cifs(input_folder):
            struct = Structure.from_file(input_file)
            ild = _calculate_ild(struct.lattice)

            fz = struct.frac_coords[:, 2]
            z0 = _unwrap_fractional_z(fz)
            fz_unwrapped = np.mod(fz - z0, 1.0)
            thickness = (np.max(fz_unwrapped) - np.min(fz_unwrapped)) * ild

            print(
                f"{os.path.basename(input_file)}: ILD = {ild:.6f} Å, Layer Thickness = {thickness:.6f} Å"
            )

    __call__ = run


class Analyze:
    def _collect_cifs(self, folder: Path) -> list[str]:
        files: list[str] = []
        if not folder.exists():
            raise FileNotFoundError(f"Input folder not found: {folder}")

        for entry in sorted(folder.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".cif":
                files.append(str(entry))
            elif entry.is_dir():
                candidate = entry / f"{entry.name}.cif"
                if candidate.exists():
                    files.append(str(candidate))

        if not files:
            raise FileNotFoundError(f"No .cif files found in: {folder}")
        return files

    def _calc_ils_dl(self, input_file: str) -> float:
        struct = Structure.from_file(input_file)

        with open(input_file) as f:
            lines = f.readlines()

        atom_lines = []
        in_atom_loop = False
        saw_atom_headers = False

        for line in lines:
            ls = line.strip()
            if ls.startswith("loop_"):
                in_atom_loop = False
                saw_atom_headers = False
                continue

            if ls.startswith("_atom_site_"):
                saw_atom_headers = True
                in_atom_loop = True
                continue

            if in_atom_loop and saw_atom_headers:
                if not ls or ls.startswith("_") or ls.startswith("loop_"):
                    break
                if ls and ls[0].isalpha():
                    atom_lines.append(ls)

        if not atom_lines:
            raise ValueError("No atom lines found in CIF")

        _, (lower_line, upper_line), (xl, yl, _), _ = (
            pick_lower_left_pair_from_lines(atom_lines)
        )
        xu, yu, _ = parse_xyz_from_atom_line(upper_line)
        xu_w, yu_w = wrap01(xu), wrap01(yu)

        dxu = xu_w - xl
        if dxu < 0.0:
            dxu += 1.0
        dyu = yu_w - yl
        if dyu < 0.0:
            dyu += 1.0

        a_vec, b_vec, _ = struct.lattice.matrix
        a_xy = np.array([a_vec[0], a_vec[1]])
        b_xy = np.array([b_vec[0], b_vec[1]])

        slip_vec_xy = dxu * a_xy + dyu * b_xy
        return float(np.linalg.norm(slip_vec_xy))

    def _calc_ils_sl(self, input_file: str) -> float:
        struct = Structure.from_file(input_file)
        lat = struct.lattice
        _, _, c = lat.abc
        alpha_deg, beta_deg, gamma_deg = lat.angles
        ar = math.radians(alpha_deg)
        br = math.radians(beta_deg)
        gr = math.radians(gamma_deg)

        cx = c * math.cos(br)
        cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / math.sin(gr)
        return math.sqrt(cx**2 + cy**2)

    def _calc_ild(self, input_file: str, *, divide_by_two: bool) -> float:
        struct = Structure.from_file(input_file)
        ild = _calculate_ild(struct.lattice)
        return ild / 2.0 if divide_by_two else ild

    def run(
        self,
        cof_name: str,
        mode: str = "both",
        input_base: str | Path | None = None,
        output_base: str | Path | None = None,
        print_values: bool = True,
    ):
        """Compute ILD/ILS metrics for optimized CIFs and write a summary CSV.

        Args:
            cof_name: COF name used for default folder naming.
            mode: "incl", "serr", or "both".
            input_base: Optional base folder containing per-mode subfolders.
                Defaults to {cof_name}/4_{cof_name}_final_structures.
            output_base: Optional folder for the output CSV.
                Defaults to the input base folder.
            print_values: If True, print ILD/ILS values to stdout.
        """
        mode_lower = mode.lower()
        if mode_lower not in {"incl", "serr", "both"}:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")

        base = (
            Path(input_base)
            if input_base
            else Path(f"{cof_name}/4_{cof_name}_final_structures")
        )
        output_base_path = Path(output_base) if output_base else base
        modes = ["serr", "incl"] if mode_lower == "both" else [mode_lower]

        rows: list[dict[str, float | str]] = []
        for selected_mode in modes:
            folder = base / selected_mode
            files = self._collect_cifs(folder)
            label = "Serrated" if selected_mode == "serr" else "Inclined"
            if print_values:
                print(f"{label}:")
                print(" ILD (Å)  ILS (Å)")
            for input_file in files:
                ild = self._calc_ild(
                    input_file, divide_by_two=selected_mode == "serr"
                )
                if selected_mode == "serr":
                    ils = self._calc_ils_dl(input_file)
                else:
                    ils = self._calc_ils_sl(input_file)

                if print_values:
                    print(f" {ild:6.1f}  {ils:7.1f}")
                rows.append(
                    {
                        "Stacking": selected_mode,
                        "filename": os.path.basename(input_file),
                        "ILD": float(ild),
                        "ILS": float(ils),
                    }
                )

        output_csv = output_base_path / "final_structures.csv"
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["Stacking", "filename", "ILD", "ILS"]
            )
            writer.writeheader()
            writer.writerows(rows)

    __call__ = run


def analyze(
    cof_name: str,
    mode: str = "both",
    input_base: str | Path | None = None,
    output_base: str | Path | None = None,
    print_values: bool = True,
):
    """Compute ILD/ILS metrics and write a summary CSV.

    Args:
        cof_name: COF name used for default folder naming.
        mode: "incl", "serr", or "both".
        input_base: Optional base folder containing per-mode subfolders.
            Defaults to {cof_name}/4_{cof_name}_final_structures.
        output_base: Optional folder for the output CSV.
            Defaults to the input base folder.
        print_values: If True, print ILD/ILS values to stdout.

    Returns:
        None.
    """
    return Analyze().run(
        cof_name=cof_name,
        mode=mode,
        input_base=input_base,
        output_base=output_base,
        print_values=print_values,
    )


@dataclass
class VisualizeCOF:
    width: int = 800
    height: int = 600
    background: str = "white"
    style: str = "stick"

    def _find_files(self, path: Path) -> list[Path]:
        candidates: list[Path] = []
        for ext in ("cif", "xyz", "pdb", "mol", "mol2", "sdf"):
            candidates.extend(sorted(path.glob(f"*.{ext}")))
        return candidates

    def _resolve_model(
        self, source: str | Path | Structure, model_format: ModelFormat | None
    ) -> tuple[str, str]:
        if isinstance(source, Structure):
            fmt = model_format or "cif"
            data = source.to(fmt=fmt)
            return data, fmt

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(
                f"Structure file or folder not found: {path}"
            )

        if path.is_dir():
            candidates = self._find_files(path)
            if not candidates:
                raise FileNotFoundError(
                    f"No structure files found in folder: {path}"
                )
            path = candidates[0]

        fmt = model_format or path.suffix.lower().lstrip(".")
        if fmt == "":
            raise ValueError(
                "Unable to infer model format. Pass model_format explicitly."
            )

        data = path.read_text()
        return data, fmt

    def view(
        self,
        folder: str | Path,
        model_format: ModelFormat | None = None,
        add_unit_cell: bool = True,
        style: str | dict[str, Any] | None = None,
        print_names: bool = True,
    ):
        path = Path(folder)
        if not path.exists():
            raise FileNotFoundError(f"Structure folder not found: {path}")
        if not path.is_dir():
            raise ValueError("view_all expects a folder path.")

        files = self._find_files(path)
        if not files:
            raise FileNotFoundError(
                f"No structure files found in folder: {path}"
            )

        views = []
        for file_path in files:
            if print_names:
                print(file_path.name)
            view = self._view_single(
                source=file_path,
                model_format=model_format,
                add_unit_cell=add_unit_cell,
                style=style,
            )
            view.show()
            views.append(view)
        return views

    def _view_single(
        self,
        source: str | Path | Structure,
        model_format: ModelFormat | None = None,
        add_unit_cell: bool = True,
        style: str | dict[str, Any] | None = None,
    ):
        try:
            import py3Dmol
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "py3Dmol is required for visualization. Install with: pip install py3Dmol"
            ) from exc

        data, fmt = self._resolve_model(source, model_format)
        view = py3Dmol.view(width=self.width, height=self.height)
        view.addModel(data, fmt)
        resolved_style = style or self.style
        if isinstance(resolved_style, dict):
            view.setStyle(resolved_style)
        else:
            view.setStyle({resolved_style: {}})
        if add_unit_cell:
            view.addUnitCell()
        view.setBackgroundColor(self.background)
        view.zoomTo()
        return view


def visualize_cof(
    folder: str | Path,
    model_format: ModelFormat | None = None,
    add_unit_cell: bool = True,
    width: int = 800,
    height: int = 600,
    background: str = "white",
    style: str | dict[str, Any] = "stick",
    print_names: bool = True,
):
    """Visualize all structures in a folder with py3Dmol.

    Args:
        folder: Folder containing structure files.
        model_format: Optional format override (e.g., "cif").
        add_unit_cell: If True, draw the unit cell.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
        background: Viewer background color.
        style: py3Dmol style string or dict (default "stick").
        print_names: If True, print file names as they are shown.

    Returns:
        List of py3Dmol views.
    """
    return VisualizeCOF(
        width=width,
        height=height,
        background=background,
        style=style if isinstance(style, str) else "stick",
    ).view(
        folder=folder,
        model_format=model_format,
        add_unit_cell=add_unit_cell,
        style=style,
        print_names=print_names,
    )


def visualizecof(
    cof_name: str,
    mode: str = "both",
    input_base: str | Path | None = None,
    width: int = 800,
    height: int = 600,
    background: str = "white",
    style: str | dict[str, Any] = "stick",
    add_unit_cell: bool = True,
    supercell_size_serr: tuple[int, int, int] = (2, 2, 1),
    supercell_size_incl: tuple[int, int, int] = (2, 2, 2),
):
    """Visualize optimized COF structures for the selected stacking mode(s).

    Args:
        cof_name: COF name used for folder naming.
        mode: "incl", "serr", or "both".
        input_base: Optional base folder containing per-mode subfolders.
            Defaults to {cof_name}/4_{cof_name}_final_structures.
        width: Viewer width in pixels.
        height: Viewer height in pixels.
        background: Viewer background color.
        style: py3Dmol style string or dict (default "stick").
        add_unit_cell: If True, draw the unit cell.
        supercell_size_serr: Supercell size for serrated structures.
        supercell_size_incl: Supercell size for inclined structures.

    Returns:
        List of py3Dmol views.
    """
    analyzer = Analyze()
    mode_lower = mode.lower()
    if mode_lower not in {"incl", "serr", "both"}:
        raise ValueError("mode must be 'incl', 'serr', or 'both'.")

    base = (
        Path(input_base)
        if input_base
        else Path(f"{cof_name}/4_{cof_name}_final_structures")
    )
    modes = ["serr", "incl"] if mode_lower == "both" else [mode_lower]

    viewer = VisualizeCOF(
        width=width,
        height=height,
        background=background,
        style=style if isinstance(style, str) else "stick",
    )
    views = []

    def _collect_cifs(folder: Path) -> list[str]:
        files: list[str] = []
        if not folder.exists():
            raise FileNotFoundError(f"Input folder not found: {folder}")

        for entry in sorted(folder.iterdir()):
            if entry.is_file() and entry.suffix.lower() == ".cif":
                files.append(str(entry))
            elif entry.is_dir():
                candidate = entry / f"{entry.name}.cif"
                if candidate.exists():
                    files.append(str(candidate))

        if not files:
            raise FileNotFoundError(f"No .cif files found in: {folder}")
        return files

    for selected_mode in modes:
        folder = base / selected_mode
        files = _collect_cifs(folder)
        label = "Serrated" if selected_mode == "serr" else "Inclined"
        supercell_size = (
            supercell_size_serr
            if selected_mode == "serr"
            else supercell_size_incl
        )
        for input_file in files:
            ild = analyzer._calc_ild(
                input_file, divide_by_two=selected_mode == "serr"
            )
            if selected_mode == "serr":
                ils = analyzer._calc_ils_dl(input_file)
            else:
                ils = analyzer._calc_ils_sl(input_file)

            name = os.path.basename(input_file)
            print(f"{label} | {name}: ILD = {ild:.2f} Å, ILS = {ils:.2f} Å")

            struct = Structure.from_file(input_file)
            supercell_struct = struct * supercell_size

            view = viewer._view_single(
                source=supercell_struct,
                add_unit_cell=add_unit_cell,
                style=style,
            )
            view.show()
            views.append(view)
    return views
