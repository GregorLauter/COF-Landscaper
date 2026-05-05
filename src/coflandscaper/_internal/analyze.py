"""Analyze ILD/ILS metrics and visualize optimized COF structures.

This module provides utilities to compute interlayer metrics from optimized
structures, merge those metrics with per-layer energies, and visualize CIF
structures (including supercell-expanded views) for selected stacking modes.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

from .ild_ils_utils import (
    _calculate_ild,
    list_cifs,
    parse_xyz_from_atom_line,
    pick_lower_left_pair_from_lines,
)
from .utilities import read_cif_atom_lines


class Supercell:
    r"""Build a supercell $a\times b\times c$ from each input unit cell.

    Physically, this replicates the periodic unit cell in-plane and along $c$
    to create a larger slab for visualization or downstream calculations.
    """

    def run(
        self,
        input_folder: str,
        output_folder: str,
        supercell_size: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        """Expand all CIF structures in a folder into supercell CIF files.

        Args:
            input_folder: Folder containing input `.cif` files.
            output_folder: Destination folder for expanded `.cif` files.
            supercell_size: Supercell replication `(a, b, c)`.
                Defaults to `(2, 2, 2)`.
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            outname = f"{base}_supercell_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.cif"
            outpath = os.path.join(output_folder, outname)
            struct = Structure.from_file(input_file)
            supercell = struct * supercell_size
            CifWriter(supercell).write_file(outpath, mode="wt")


class AnalyzeStacking:
    _FINAL_STRUCT_FIELDS: ClassVar[list[str]] = [
        "Stacking",
        "filename",
        "ILD",
        "ILS",
        "energy_eV_per_layer",
        "energy_rel_eV_per_layer",
    ]

    def _collect_cifs(self, folder: Path) -> list[str]:
        """Collect CIF files from a mode folder.

        Args:
            folder: Mode folder path.

        Returns:
            List of CIF file paths.

        Raises:
            FileNotFoundError: If folder is missing or contains no valid CIFs.
        """
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

    def _ils_registry(self, input_file: str) -> float:
        """Compute registry-based in-plane slip from lower/upper atom offset.

        Args:
            input_file: CIF file path.

        Returns:
            Interlayer slipping length in Angstrom.

        Raises:
            ValueError: If atom lines cannot be parsed from CIF text.
        """
        struct = Structure.from_file(input_file)

        atom_lines = read_cif_atom_lines(input_file)

        _, (_lower_line, upper_line), (xl, yl, _), _ = (
            pick_lower_left_pair_from_lines(atom_lines)
        )
        xu, yu, _ = cast(
            "tuple[float, float, float]", parse_xyz_from_atom_line(upper_line)
        )

        # Use minimum-image deltas in fractional space so shifts across
        # periodic boundaries remain small (e.g., -0.08 instead of 0.92).
        dxu = ((xu - xl + 0.5) % 1.0) - 0.5
        dyu = ((yu - yl + 0.5) % 1.0) - 0.5

        a_vec, b_vec, _ = struct.lattice.matrix
        a_xy = np.array([a_vec[0], a_vec[1]])
        b_xy = np.array([b_vec[0], b_vec[1]])

        slip_vec_xy = dxu * a_xy + dyu * b_xy
        return float(np.linalg.norm(slip_vec_xy))

    def _ils_tilt(self, input_file: str) -> float:
        """Compute in-plane slip from projected c-vector components.

        Args:
            input_file: CIF file path.

        Returns:
            Interlayer slipping length in Angstrom.
        """
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
        """Compute ILD from lattice geometry with optional bilayer scaling.

        Args:
            input_file: CIF file path.
            divide_by_two: If `True`, return half ILD (used for serrated bilayers).

        Returns:
            Interlayer distance in Angstrom.
        """
        struct = Structure.from_file(input_file)
        ild = _calculate_ild(struct.lattice)
        return ild / 2.0 if divide_by_two else ild

    def _resolve_modes(self, mode: str) -> list[str]:
        """Normalize mode selector into concrete mode tags.

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

    def _compute_metrics(
        self, input_file: str, selected_mode: str
    ) -> tuple[float, float]:
        """Compute `(ILD, ILS)` metrics for one structure and mode.

        Args:
            input_file: CIF file path.
            selected_mode: Mode tag (`"serr"` or `"incl"`).

        Returns:
            Tuple `(ild, ils)` in Angstrom.
        """
        ild = self._calc_ild(input_file, divide_by_two=selected_mode == "serr")
        if selected_mode == "serr":
            slip_registry = self._ils_registry(input_file)
            slip_tilt = self._ils_tilt(input_file)
            ils = slip_registry + (slip_tilt / 2.0)
        else:
            ils = self._ils_tilt(input_file)
        return float(ild), float(ils)

    def _load_energy_map(
        self,
        cof_name: str,
        input_base_path: Path,
        dft: bool,
    ) -> dict[tuple[str, str], tuple[float, float]]:
        """Load per-structure per-mode energies from optimization CSV.

        Args:
            cof_name: COF identifier used for CSV naming.
            input_base_path: Base optimization folder.
            dft: If `True`, load DFT energy CSV variant.

        Returns:
            Mapping `(mode, structure) -> (absolute_energy, relative_energy)`.
        """
        filename = (
            f"{cof_name}_opt_energies_per_layer_dft.csv"
            if dft
            else f"{cof_name}_opt_energies_per_layer.csv"
        )
        csv_path = input_base_path / filename
        if not csv_path.exists():
            return {}

        energies: dict[tuple[str, str], tuple[float, float]] = {}
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                structure = (row.get("structure") or "").strip()
                mode = (row.get("stacking_mode") or "").strip()
                abs_e = row.get("energy_eV_per_layer")
                rel_e = row.get("energy_rel_eV_per_layer")
                if not structure or mode not in {"serr", "incl"}:
                    continue
                if abs_e is None or rel_e is None:
                    continue
                try:
                    energies[(mode, structure)] = (float(abs_e), float(rel_e))
                except ValueError:
                    continue
        return energies

    def _merge_existing_rows(
        self,
        output_csv: Path,
        new_rows: list[dict[str, float | str]],
    ) -> list[dict[str, float | str]]:
        """Merge new analysis rows with existing CSV rows by stacking mode.

        Args:
            output_csv: Existing output CSV path.
            new_rows: Freshly computed rows for current mode selection.

        Returns:
            Sorted merged row list preserving untouched modes.
        """
        modes_to_replace = {
            str(row.get("Stacking", ""))
            for row in new_rows
            if str(row.get("Stacking", ""))
        }

        preserved: list[dict[str, float | str]] = []
        if output_csv.exists():
            with output_csv.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    mode = str(row.get("Stacking", ""))
                    if mode in modes_to_replace:
                        continue
                    preserved.append(
                        {
                            field: row.get(field, "")
                            for field in self._FINAL_STRUCT_FIELDS
                        }
                    )

        merged = preserved + new_rows
        merged.sort(
            key=lambda row: (
                str(row.get("Stacking", "")),
                str(row.get("filename", "")),
            )
        )
        return merged

    def analyze(
        self,
        cof_name: str,
        mode: str = "both",
        input_base: str | Path | None = None,
        output_base: str | Path | None = None,
        dft: bool = False,
        print_values: bool = True,
    ) -> None:
        """Compute ILD/ILS metrics and write an analysis CSV.

        Args:
            cof_name: COF name used for default folder naming.
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`. Defaults to `"both"`.
            input_base: Optional base folder containing per-mode subfolders.
                Defaults to `None`
                (uses `{cof_name}/4_{cof_name}_optimization`).
            output_base: Optional folder for the output CSV.
                Defaults to `None` (uses `{cof_name}/5_{cof_name}_analysis`).
            dft: If True, analyze dft_{mode} subfolders and write
                final_structures_dft.csv. Defaults to `False`.
            print_values: If `True`, print ILD/ILS values to stdout.
                Defaults to `True`.

        Notes:
                        - dft=False reads from `{input_base}/{serr|incl}` and writes
                            `final_structures.csv`.
                        - dft=True reads from `{input_base}/dft_{serr|incl}` and writes
                            `final_structures_dft.csv`.
        """
        base = (
            Path(input_base)
            if input_base
            else Path(f"{cof_name}/4_{cof_name}_optimization")
        )
        output_base_path = (
            Path(output_base)
            if output_base
            else Path(f"{cof_name}/5_{cof_name}_analysis")
        )
        modes = self._resolve_modes(mode)
        energy_map = self._load_energy_map(
            cof_name=cof_name,
            input_base_path=base,
            dft=dft,
        )

        rows: list[dict[str, float | str]] = []
        for selected_mode in modes:
            folder = base / (f"dft_{selected_mode}" if dft else selected_mode)
            files = self._collect_cifs(folder)
            label = "Serrated" if selected_mode == "serr" else "Inclined"
            if print_values:
                print(f"{label}:")
                print(" ILD (Å)  ILS (Å)  Erel (eV)")
            for input_file in files:
                ild, ils = self._compute_metrics(input_file, selected_mode)
                structure_name = os.path.splitext(
                    os.path.basename(input_file)
                )[0]
                energy_abs, energy_rel = energy_map.get(
                    (selected_mode, structure_name),
                    (float("nan"), float("nan")),
                )

                if print_values:
                    rel_display = (
                        "--" if np.isnan(energy_rel) else f"{energy_rel:.1f}"
                    )
                    print(f" {ild:6.1f}  {ils:7.1f}  {rel_display:>9}")
                rows.append(
                    {
                        "Stacking": selected_mode,
                        "filename": os.path.basename(input_file),
                        "ILD": float(ild),
                        "ILS": float(ils),
                        "energy_eV_per_layer": energy_abs,
                        "energy_rel_eV_per_layer": energy_rel,
                    }
                )

        output_base_path.mkdir(parents=True, exist_ok=True)
        output_csv_name = (
            "final_structures_dft.csv" if dft else "final_structures.csv"
        )
        output_csv = output_base_path / output_csv_name

        merged_rows = self._merge_existing_rows(
            output_csv=output_csv,
            new_rows=rows,
        )

        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=self._FINAL_STRUCT_FIELDS,
            )
            writer.writeheader()
            writer.writerows(merged_rows)

    def run(
        self,
        cof_name: str,
        mode: str = "both",
        input_base: str | Path | None = None,
        output_base: str | Path | None = None,
        dft: bool = False,
        print_values: bool = True,
    ) -> None:
        """Backward-compatible alias for :meth:`analyze`.

        Args:
            cof_name: COF name used for default folder naming.
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`. Defaults to `"both"`.
            input_base: Optional base folder containing per-mode subfolders.
                Defaults to `None`.
            output_base: Optional output folder for analysis CSV files.
                Defaults to `None`.
            dft: If `True`, analyze DFT-mode folders. Defaults to `False`.
            print_values: If `True`, print ILD/ILS values. Defaults to `True`.
        """
        return self.analyze(
            cof_name=cof_name,
            mode=mode,
            input_base=input_base,
            output_base=output_base,
            dft=dft,
            print_values=print_values,
        )


@dataclass
class VisualizeCOF:
    """Visualize optimized COF structures with py3Dmol.

    The viewer can render single files, folders, or selected mode outputs and
    supports optional supercell expansion before display.
    """

    width: int = 800
    height: int = 600
    background: str = "white"
    style: str = "stick"

    def _find_files(self, path: Path) -> list[Path]:
        """List CIF files in a folder.

        Args:
            path: Folder path.

        Returns:
            Sorted list of `.cif` file paths.
        """
        return sorted(path.glob("*.cif"))

    def _resolve_model(
        self, source: str | Path | Structure
    ) -> tuple[str, str]:
        """Resolve supported structure source into py3Dmol model payload.

        Args:
            source: CIF path, folder path, or pymatgen `Structure`.

        Returns:
            Tuple `(model_text, format)` ready for `py3Dmol.addModel`.

        Raises:
            FileNotFoundError: If path source does not exist.
            FileNotFoundError: If folder source has no CIF files.
            ValueError: If source file is not `.cif`.
        """
        if isinstance(source, Structure):
            return source.to(fmt="cif"), "cif"

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

        if path.suffix.lower() != ".cif":
            raise ValueError(f"Only .cif files are supported: {path}")

        data = path.read_text()
        return data, "cif"

    def view(
        self,
        folder: str | Path,
        add_unit_cell: bool = True,
        style: str | dict[str, Any] | None = None,
        print_names: bool = True,
    ):
        """Visualize every CIF in one folder and return py3Dmol views.

        Args:
            folder: Folder containing CIF files.
            add_unit_cell: If `True`, draw the unit cell. Defaults to `True`.
            style: Optional style override. Defaults to `None`
                (uses instance default style).
            print_names: If `True`, print filenames during rendering.
                Defaults to `True`.

        Returns:
            List of py3Dmol view objects.

        Raises:
            FileNotFoundError: If folder is missing or has no CIF files.
            ValueError: If `folder` is not a directory.
        """
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
                add_unit_cell=add_unit_cell,
                style=style,
            )
            view.show()
            views.append(view)
        return views

    def _view_single(
        self,
        source: str | Path | Structure,
        add_unit_cell: bool = True,
        style: str | dict[str, Any] | None = None,
    ):
        """Create one py3Dmol view from a structure source.

        Args:
            source: CIF path, folder path, or pymatgen `Structure`.
            add_unit_cell: If `True`, draw the unit cell. Defaults to `True`.
            style: Optional style override. Defaults to `None`
                (uses instance default style).

        Returns:
            py3Dmol view object.

        Raises:
            ModuleNotFoundError: If `py3Dmol` is not installed.
        """
        try:
            import py3Dmol
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "py3Dmol is required for visualization. Install with: pip install py3Dmol"
            ) from exc

        data, fmt = self._resolve_model(source)
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
        self,
        cof_name: str,
        mode: str = "both",
        input_base: str | Path | None = None,
        dft: bool = False,
        add_unit_cell: bool = True,
        supercell_size_serr: tuple[int, int, int] = (2, 2, 1),
        supercell_size_incl: tuple[int, int, int] = (2, 2, 2),
    ):
        """Visualize optimized COF structures for the selected stacking mode(s).

        Args:
            cof_name: COF name used for folder naming.
            mode: Mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`. Defaults to `"both"`.
            input_base: Optional base folder containing per-mode subfolders.
                Defaults to `None`
                (uses `{cof_name}/4_{cof_name}_optimization`).
            dft: If `True`, read structures from `dft_{mode}` subfolders.
                Defaults to `False`.
            add_unit_cell: If `True`, draw the unit cell. Defaults to `True`.
            supercell_size_serr: Supercell size for serrated structures.
                Defaults to `(2, 2, 1)`.
            supercell_size_incl: Supercell size for inclined structures.
                Defaults to `(2, 2, 2)`.

        Returns:
            List of py3Dmol views.

        Notes:
            Viewer appearance is fixed to defaults
            (width=800, height=600, background="white", style="stick").
        """
        analyzer = AnalyzeStacking()

        base = (
            Path(input_base)
            if input_base
            else Path(f"{cof_name}/4_{cof_name}_optimization")
        )
        modes = analyzer._resolve_modes(mode)

        views = []

        for selected_mode in modes:
            folder = base / (f"dft_{selected_mode}" if dft else selected_mode)
            files = analyzer._collect_cifs(folder)
            label = "Serrated" if selected_mode == "serr" else "Inclined"
            supercell_size = (
                supercell_size_serr
                if selected_mode == "serr"
                else supercell_size_incl
            )
            for input_file in files:
                ild, ils = analyzer._compute_metrics(input_file, selected_mode)

                name = os.path.basename(input_file)
                print(
                    f"{label} | {name}: ILD = {ild:.2f} Å, ILS = {ils:.2f} Å"
                )

                struct = Structure.from_file(input_file)
                supercell_struct = struct * supercell_size

                view = self._view_single(
                    source=supercell_struct,
                    add_unit_cell=add_unit_cell,
                )
                view.show()
                views.append(view)
        return views

    def visualize_single_layer(
        self,
        input_folder: str | Path = "1_ILCOF-1_single_layer",
        add_unit_cell: bool = True,
        supercell_size: tuple[int, int, int] = (2, 2, 1),
    ):
        """Visualize all single-layer CIFs in one folder.

        Args:
            input_folder: Folder containing single-layer .cif files.
                Defaults to `"1_ILCOF-1_single_layer"`.
            add_unit_cell: If `True`, draw the unit cell. Defaults to `True`.
            supercell_size: Supercell size applied before visualization.
                Defaults to `(2, 2, 1)`.

        Returns:
            List of py3Dmol views.

        Notes:
            - No mode argument is used.
            - All .cif files are read directly from input_folder
              (no subfolder traversal).
            - ILS is computed with inclined logic for reporting.
        """
        folder = Path(input_folder)
        if not folder.exists():
            raise FileNotFoundError(f"Input folder not found: {folder}")
        if not folder.is_dir():
            raise ValueError(f"input_folder must be a directory: {folder}")

        files = self._find_files(folder)
        if not files:
            raise FileNotFoundError(f"No .cif files found in: {folder}")

        analyzer = AnalyzeStacking()
        views = []

        for file_path in files:
            input_file = str(file_path)
            ild, ils = analyzer._compute_metrics(input_file, "incl")
            print(
                f"Single layer | {file_path.name}: ILD = {ild:.2f} Å, ILS = {ils:.2f} Å"
            )

            struct = Structure.from_file(input_file)
            supercell_struct = struct * supercell_size
            view = self._view_single(
                source=supercell_struct,
                add_unit_cell=add_unit_cell,
            )
            view.show()
            views.append(view)

        return views
