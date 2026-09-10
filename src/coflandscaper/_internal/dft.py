"""Generate and parse optional CRYSTAL23 inputs and outputs.

This module provides the optional CRYSTAL23 workflow used for DFT-based
single-point calculations and geometry optimizations. It contains utilities for
converting COF CIF structures into CRYSTAL ``.d12`` inputs, extracting converged
energies from CRYSTAL ``.out`` files, reconstructing optimized structures, and
writing workflow-compatible CSV outputs.

The CRYSTAL workflow is retained as an alternative to the primary MACE-based
workflow for benchmarking or calculations requiring an explicit DFT treatment.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from ase.data import atomic_numbers

if TYPE_CHECKING:
    from ase.atoms import Atoms

HARTREE_TO_EV = 27.211386245988


def guess_symbol(raw: str) -> str | None:
    """Guess a chemical symbol from a CIF-like raw token.

    Args:
        raw: Raw symbol or label text.

    Returns:
        Normalized element symbol if recognized, else `None`.
    """
    s = re.sub(r"[^A-Za-z]", "", raw)
    if not s:
        return None
    s = s[0].upper() + s[1:].lower()
    if s in atomic_numbers:
        return s
    if len(s) > 1 and s[:2] in atomic_numbers:
        return s[:2]
    if s[0] in atomic_numbers:
        return s[0]
    return None


def parse_cell(text: str) -> dict[str, float]:
    """Parse unit-cell lengths and angles from CIF text.

    Args:
        text: Full CIF text.

    Returns:
        Dictionary with keys `a`, `b`, `c`, `alpha`, `beta`, and `gamma`.

    Raises:
        ValueError: If any required cell parameter is missing.
    """

    def grab(key: str) -> float:
        m = re.search(rf"{key}\s+([0-9.+\-Ee()]+)", text)
        if not m:
            raise ValueError(f"Missing cell parameter: {key}")
        return float(re.sub(r"\([^)]*\)", "", m.group(1)))

    return {
        "a": grab(r"_cell_length_a"),
        "b": grab(r"_cell_length_b"),
        "c": grab(r"_cell_length_c"),
        "alpha": grab(r"_cell_angle_alpha"),
        "beta": grab(r"_cell_angle_beta"),
        "gamma": grab(r"_cell_angle_gamma"),
    }


def extract_atoms(lines: list[str]) -> list[tuple[int, float, float, float]]:
    """Extract fractional atom sites from a CIF atom loop block.

    Args:
        lines: CIF file lines.

    Returns:
        List of tuples `(Z, x, y, z)` with atomic number and fractional coords.

    Raises:
        ValueError: If no supported fractional-coordinate atom loop is found.
    """
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("loop_"):
            j = i + 1
            headers: list[str] = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                headers.append(lines[j].strip())
                j += 1
            if not any("fract_x" in h.lower() for h in headers):
                continue

            hdr = {h.split()[0].lower(): idx for idx, h in enumerate(headers)}
            xk = next(k for k in hdr if "_fract_x" in k)
            yk = next(k for k in hdr if "_fract_y" in k)
            zk = next(k for k in hdr if "_fract_z" in k)
            lbl = next(
                (k for k in hdr if "_type_symbol" in k or "_label" in k), None
            )

            atoms: list[tuple[int, float, float, float]] = []
            while j < len(lines):
                s = lines[j].strip()
                if not s or s.startswith(("loop_", "_")):
                    break
                parts = lines[j].split()
                if len(parts) >= len(headers):
                    lab = parts[hdr[lbl]] if lbl else "X"
                    sym = guess_symbol(lab) or "X"
                    Z = atomic_numbers.get(sym, -1)
                    x = float(re.sub(r"\([^)]*\)", "", parts[hdr[xk]]))
                    y = float(re.sub(r"\([^)]*\)", "", parts[hdr[yk]]))
                    z = float(re.sub(r"\([^)]*\)", "", parts[hdr[zk]]))
                    atoms.append((Z, x, y, z))
                j += 1
            return atoms
    raise ValueError("No atom site loop with fractional coordinates.")


def find_last_occurrence(lines: list[str], keyword: str) -> int:
    """Return the index of the last occurrence of a keyword.

    Args:
        lines: Text lines from a CRYSTAL output.
        keyword: Substring to search for.

    Returns:
        Index of the last matching line, or -1 if not found.
    """
    for i in range(len(lines) - 1, -1, -1):
        if keyword in lines[i]:
            return i
    return -1


def parse_atom_lines(
    lines: list[str], start_idx: int
) -> tuple[list[str], list[list[float]]]:
    """Parse atom symbols and XYZ coordinates from a line block.

    Args:
        lines: Text lines from a CRYSTAL output.
        start_idx: Index where the atom block begins.

    Returns:
        Tuple of (symbols, coordinates), where coordinates are raw XYZ values.
    """
    symbols: list[str] = []
    raw_coords: list[list[float]] = []

    for i in range(start_idx, len(lines)):
        parts = lines[i].split()
        if len(parts) < 4:
            if symbols:
                break
            continue

        try:
            sym = parts[-4]
            if sym.capitalize() not in atomic_numbers:
                if symbols:
                    break
                continue

            x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            symbols.append(sym)
            raw_coords.append([x, y, z])
        except (ValueError, IndexError):
            if symbols:
                break
            continue

    return symbols, raw_coords


def _parse_primary_structure(lines: list[str]) -> Atoms | None:
    """Parse a structure using DIRECT LATTICE and PRIMITIVE CELL blocks.

    Args:
        lines: Text lines from a CRYSTAL output.

    Returns:
        ASE Atoms if parsing succeeds, otherwise None.
    """
    lat_idx = find_last_occurrence(lines, "DIRECT LATTICE")
    prim_idx = find_last_occurrence(lines, "PRIMITIVE CELL")

    if lat_idx == -1 and prim_idx == -1:
        return None

    try:
        cell = None
        if lat_idx != -1:
            v_start = lat_idx + 2
            cell = [
                list(map(float, lines[v_start].split())),
                list(map(float, lines[v_start + 1].split())),
                list(map(float, lines[v_start + 2].split())),
            ]

        symbols: list[str] = []
        positions: list[list[float]] = []
        if prim_idx != -1:
            symbols, positions = parse_atom_lines(lines, prim_idx + 4)

        if not symbols and cell is None:
            return None

        try:
            from ase import Atoms
        except Exception as exc:
            raise ModuleNotFoundError(
                "ase is required for CRYSTAL structure extraction. Install with: pip install ase"
            ) from exc

        return Atoms(
            symbols=symbols,
            cell=cell,
            positions=positions,
            pbc=(cell is not None),
        )
    except (ValueError, IndexError, KeyError):
        return None


def _parse_fallback_structure(lines: list[str]) -> Atoms | None:
    """Parse a structure using LATTICE PARAMETERS and ATOM blocks.

    Args:
        lines: Text lines from a CRYSTAL output.

    Returns:
        ASE Atoms if parsing succeeds, otherwise None.
    """
    lat_header_sig = "LATTICE PARAMETERS (ANGSTROMS AND DEGREES)"
    lat_idx = find_last_occurrence(lines, lat_header_sig)

    if lat_idx == -1:
        return None

    try:
        param_line_idx = -1
        for i in range(lat_idx, min(lat_idx + 10, len(lines))):
            if "ALPHA" in lines[i] and "BETA" in lines[i]:
                param_line_idx = i + 1
                break

        if param_line_idx == -1 or param_line_idx >= len(lines):
            return None

        params = list(map(float, lines[param_line_idx].split()))
        if len(params) != 6:
            return None

        try:
            from ase.geometry import cellpar_to_cell
        except Exception as exc:
            raise ModuleNotFoundError(
                "ase is required for CRYSTAL structure extraction. Install with: pip install ase"
            ) from exc

        cell = cellpar_to_cell(params)

        atom_header_idx = -1
        for i in range(param_line_idx, len(lines)):
            if "ATOM" in lines[i] and "Z" in lines[i]:
                atom_header_idx = i
                break
        if atom_header_idx == -1:
            return None

        header = lines[atom_header_idx]
        x_is_frac = "/A" in header or "/a" in header
        y_is_frac = "/B" in header or "/b" in header
        z_is_frac = "/C" in header or "/c" in header

        symbols, raw_coords = parse_atom_lines(lines, atom_header_idx + 2)
        if not symbols:
            return None

        final_positions: list[list[float]] = []
        cell_mat = np.array(cell, dtype=float)

        for x, y, z in raw_coords:
            if x_is_frac and y_is_frac and z_is_frac:
                pos = np.dot([x, y, z], cell_mat)
            elif (not x_is_frac) and (not y_is_frac) and (not z_is_frac):
                pos = [x, y, z]
            else:
                frac_vec = [
                    x if x_is_frac else 0.0,
                    y if y_is_frac else 0.0,
                    z if z_is_frac else 0.0,
                ]
                cart_vec = [
                    x if not x_is_frac else 0.0,
                    y if not y_is_frac else 0.0,
                    z if not z_is_frac else 0.0,
                ]
                part_a = np.dot(frac_vec, cell_mat)
                pos = [float(part_a[k] + cart_vec[k]) for k in range(3)]
            final_positions.append(
                [float(pos[0]), float(pos[1]), float(pos[2])]
            )

        try:
            from ase import Atoms
        except Exception as exc:
            raise ModuleNotFoundError(
                "ase is required for CRYSTAL structure extraction. Install with: pip install ase"
            ) from exc

        return Atoms(
            symbols=symbols, positions=final_positions, cell=cell, pbc=True
        )
    except Exception:
        return None


def parse_z_L_from_stem(stem: str) -> tuple[float, float]:
    """Parse `_z` and `_L` tags from a structure stem.

    Args:
        stem: Structure filename stem.

    Returns:
        Tuple `(z, L)` in Angstrom, or `(nan, nan)` when tags are missing.
    """
    mz = re.search(r"_z(\d+)", stem)
    mL = re.search(r"_L(\d+)", stem)
    if not (mz and mL):
        return np.nan, np.nan
    z = float(mz.group(1)) / 10.0
    L = float(mL.group(1)) / 10.0
    return z, L


class Crystal:
    """Base CRYSTAL23 input-generation interface.

    The class converts CIF structures into CRYSTAL ``.d12`` files using P1
    symmetry and appends a user-defined CRYSTAL input block containing the chosen
    electronic-structure or optimization settings.

    Returns:
        None.
    """

    def __init__(self, post_block: str) -> None:
        """Initialize a CRYSTAL input generator.

        Args:
            post_block: CRYSTAL input text appended after the generated structural
                section. This typically contains basis-set, DFT, SHRINK, or geometry-
                optimization settings.
        """
        self._post_block = post_block

    def _convert_one(
        self, cif_path: Path, output_path: Path | None = None
    ) -> Path:
        """Convert one CIF file into one CRYSTAL `.d12` file.

        Args:
            cif_path: Input CIF path.
            output_path: Optional output `.d12` path. Defaults to `None`
                (writes next to input with `.d12` suffix).

        Returns:
            Path to the generated `.d12` file.
        """
        txt = cif_path.read_text(errors="ignore")
        lines = txt.splitlines()
        cell = parse_cell(txt)
        atoms = extract_atoms(lines)

        title = cif_path.stem
        out_path = output_path or cif_path.with_suffix(".d12")

        out: list[str] = []
        out.append(title)
        out.append("CRYSTAL")
        out.append("0 0 0")
        out.append("1")  # P1 symmetry
        out.append(
            f"{cell['a']:.6f} {cell['b']:.6f} {cell['c']:.6f} "
            f"{cell['alpha']:.6f} {cell['beta']:.6f} {cell['gamma']:.6f}"
        )
        out.append(str(len(atoms)))
        for Z, x, y, z in atoms:
            if Z < 0:
                out.append(f"0 {x:.9f} {y:.9f} {z:.9f}")
            else:
                out.append(f"{Z} {x:.9f} {y:.9f} {z:.9f}")
        if self._post_block.strip():
            out.append(self._post_block.strip())

        out_path.write_text("\n".join(out) + "\n")
        return out_path

    def run(
        self,
        input_folder: str,
        output_folder: str | None = None,
    ) -> None:
        """Convert all CIF structures in one folder into CRYSTAL input files.

        Each CIF is written to a structure-specific subfolder containing one ``.d12``
        input file.

        Args:
            input_folder: Folder containing CIF structures.
            output_folder: Optional destination folder for generated CRYSTAL inputs.
                Defaults to ``input_folder``.

        Raises:
            FileNotFoundError: If ``input_folder`` does not exist.
        """
        in_path = Path(input_folder)
        if not in_path.exists():
            raise FileNotFoundError(f"Input folder not found: {in_path}")

        out_path = Path(output_folder) if output_folder else in_path
        out_path.mkdir(parents=True, exist_ok=True)

        for cif in sorted(in_path.glob("*.cif")):
            try:
                subdir = out_path / cif.stem
                subdir.mkdir(parents=True, exist_ok=True)
                target = subdir / (cif.stem + ".d12")
                self._convert_one(cif, output_path=target)
            except Exception:
                continue

    def generate_input(
        self,
        cof_name: str,
        mode: str,
        input_base_folder: str | None = None,
        output_base_folder: str | None = None,
    ) -> None:
        """Generate CRYSTAL single-point inputs for selected stacking mode(s).

        By default, structures are read from the generated ILD/ILS matrix and CRYSTAL
        input directories are created alongside the matrix using ``dft_serr`` and/or
        ``dft_incl`` subfolders.

        Args:
            cof_name: COF name used for workflow folder naming.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            input_base_folder: Optional input base-folder override. Defaults to
                ``{cof_name}/2_{cof_name}_matrix``.
            output_base_folder: Optional output base-folder override. Defaults to
                ``{cof_name}/2_{cof_name}_matrix``.

        Raises:
            ValueError: If ``mode`` is invalid.
        """
        mode_lower = mode.lower()
        if mode_lower not in {"incl", "serr", "both"}:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")

        mode_tags = ["serr", "incl"] if mode_lower == "both" else [mode_lower]
        input_base_folder_used = (
            input_base_folder or f"{cof_name}/2_{cof_name}_matrix"
        )
        output_base_folder_used = (
            output_base_folder or f"{cof_name}/2_{cof_name}_matrix"
        )

        for mode_tag in mode_tags:
            self.run(
                input_folder=f"{input_base_folder_used}/{mode_tag}",
                output_folder=f"{output_base_folder_used}/dft_{mode_tag}",
            )


class CrystalSP(Crystal):
    """Generate and parse CRYSTAL single-point calculations.

    ``CrystalSP`` prepares CRYSTAL single-point input files and extracts converged
    total energies from completed CRYSTAL output files. Energies are converted from
    Hartree to electronvolts and written in the same CSV format used by the
    stacking-landscape workflow.

    Returns:
        None.
    """

    def _extract_energy_au(self, text: str) -> float | None:
        """Extract final converged total energy from CRYSTAL output text.

        Args:
            text: Full `.out` text content.

        Returns:
            Converged energy in Hartree, or `None` when not found.
        """
        lines = text.splitlines()
        if len(lines) < 2:
            return None
        if "TELAPSE" not in lines[-2]:
            return None

        energy_label = "TOTAL ENERGY + DISP + GCP (AU)"
        num_re = re.compile(r"([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")
        last_val: float | None = None

        for i, line in enumerate(lines):
            if energy_label in line:
                tail = line.split(energy_label, 1)[1]
                m = num_re.search(tail)
                if m:
                    last_val = float(m.group(1))
                elif i + 1 < len(lines):
                    m2 = num_re.search(lines[i + 1])
                    if m2:
                        last_val = float(m2.group(1))

        return last_val

    def __init__(
        self,
        basisset: str = "SOLDEF2MSVP",
        functional: str = "HSESOL3C",
        shrink: str = "2 2 8",
        post_block: str | None = None,
    ) -> None:
        """Configure CRYSTAL single-point calculation settings.

        Args:
            basisset: CRYSTAL basis-set keyword. Defaults to ``"SOLDEF2MSVP"``.
            functional: CRYSTAL density-functional keyword. Defaults to
                ``"HSESOL3C"``.
            shrink: CRYSTAL SHRINK-grid specification. Defaults to ``"2 2 8"``.
            post_block: Optional complete CRYSTAL input-tail override. Defaults to
                ``None``, which generates the basis-set, DFT, and SHRINK sections from
                the supplied settings.
        """
        if post_block is None:
            post_block = f"""BASISSET
{basisset}
DFT
{functional}
END
SHRINK
0 8
{shrink}
END"""
        super().__init__(post_block=post_block)

    def read(
        self,
        input_folder: str,
        output_csv_dir: str | None = None,
        output_filename_suffix: str = "",
    ) -> Path:
        """Extract converged CRYSTAL single-point energies from one folder.

        Valid CRYSTAL output files are parsed, total energies are converted from
        Hartree to electronvolts, and relative energies are calculated with respect to
        the lowest successfully parsed structure.

        Args:
            input_folder: Folder containing CRYSTAL output files.
            output_csv_dir: Optional output directory for the generated energy CSV.
                Defaults to ``{cof_name}/3_{cof_name}_landscape``.
            output_filename_suffix: Optional suffix appended to the generated CSV
                filename before ``.csv``. Defaults to an empty string.

        Returns:
            Path to the generated single-point energy CSV.

        Raises:
            FileNotFoundError: If no valid CRYSTAL output files are found.
        """
        input_path = Path(input_folder)
        folder_tag = input_path.name
        mode_tag = (
            folder_tag.replace("dft_", "")
            if folder_tag.startswith("dft_")
            else folder_tag
        )

        cof_name = folder_tag
        if input_path.parent.name.endswith("_matrix"):
            cof_name = input_path.parents[1].name
        elif input_path.parent.name:
            cof_name = input_path.parent.name

        out_files: list[Path] = []
        for out_path in sorted(input_path.rglob("*.out")):
            if out_path.name.lower().startswith("slurm"):
                continue
            if out_path.parent == input_path:
                out_files.append(out_path)
                continue
            if out_path.stem == out_path.parent.name:
                out_files.append(out_path)

        if not out_files:
            raise FileNotFoundError(
                f"No valid .out files found in: {input_path.resolve()} (expected system_name.out)"
            )

        csv_dir = Path(output_csv_dir or f"{cof_name}/3_{cof_name}_landscape")
        csv_dir.mkdir(parents=True, exist_ok=True)

        energies_csv_path = (
            csv_dir
            / f"{cof_name}_sp_energies_{mode_tag}{output_filename_suffix}.csv"
        )

        rows = []
        failed = []
        for out_path in out_files:
            try:
                text = out_path.read_text(errors="ignore")
                energy_au = self._extract_energy_au(text)
                if energy_au is None:
                    raise ValueError("Energy not found in output")
                energy_ev = energy_au * HARTREE_TO_EV
                z, L = parse_z_L_from_stem(out_path.stem)
                rows.append(
                    {
                        "structure": out_path.stem,
                        "z": z,
                        "L": L,
                        "energy_eV": energy_ev,
                    }
                )
            except Exception as exc:
                failed.append((str(out_path), repr(exc)))

        df = pd.DataFrame(rows).sort_values("structure").reset_index(drop=True)
        if not df.empty:
            min_e = float(df["energy_eV"].min())
            df["energy_rel_eV"] = df["energy_eV"] - min_e
        df.to_csv(energies_csv_path, index=False)

        if failed:
            print("\nFailed files (first 10):")
            for p, err in failed[:10]:
                print(" -", p)
                print("   ", err)
            print("Total failed:", len(failed))

        return energies_csv_path

    def read_output(
        self,
        cof_name: str,
        mode: str,
        output_base_folder: str | None = None,
        input_base_folder: str | None = None,
    ) -> list[Path]:
        """Extract CRYSTAL single-point energies for selected stacking mode(s).

        The method processes ``dft_serr`` and/or ``dft_incl`` output folders and writes
        mode-specific energy CSV files with the ``_dft`` suffix for subsequent
        landscape generation.

        Args:
            cof_name: COF name used for folder naming.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            output_base_folder: Optional output directory for generated CSV files.
                Defaults to ``{cof_name}/3_{cof_name}_landscape``.
            input_base_folder: Optional base folder containing ``dft_{mode}``
                subfolders. Defaults to ``{cof_name}/2_{cof_name}_matrix``.

        Returns:
            List of generated energy-CSV paths.
        """
        from .ild_ils_utils import get_mode_folders

        input_base_used = (
            input_base_folder or f"{cof_name}/2_{cof_name}_matrix"
        )
        csv_paths: list[Path] = []
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = Path(folder).name
            csv_paths.append(
                self.read(
                    input_folder=f"{input_base_used}/dft_{mode_tag}",
                    output_csv_dir=output_base_folder,
                    output_filename_suffix="_dft",
                )
            )
        return csv_paths


class CrystalOpt(Crystal):
    """Generate and parse CRYSTAL geometry optimizations.

    ``CrystalOpt`` prepares CRYSTAL geometry-optimization input files, extracts
    converged optimized energies, and reconstructs optimized CIF structures from
    completed CRYSTAL outputs.

    Serrated structures are represented as bilayers in the COF-Landscaper
    workflow, so their reported optimization energies are divided by two when
    stored as per-layer energies.

    Returns:
        None.
    """

    def _extract_energy_au(self, text: str) -> float | None:
        """Extract final converged total energy from CRYSTAL output text.

        Args:
            text: Full `.out` text content.

        Returns:
            Converged energy in Hartree, or `None` when not found.
        """
        lines = text.splitlines()
        if len(lines) < 2:
            return None
        if "TELAPSE" not in lines[-2]:
            return None

        energy_label = "TOTAL ENERGY + DISP + GCP (AU)"
        num_re = re.compile(r"([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")
        last_val: float | None = None

        for i, line in enumerate(lines):
            if energy_label in line:
                tail = line.split(energy_label, 1)[1]
                m = num_re.search(tail)
                if m:
                    last_val = float(m.group(1))
                elif i + 1 < len(lines):
                    m2 = num_re.search(lines[i + 1])
                    if m2:
                        last_val = float(m2.group(1))

        return last_val

    def __init__(
        self,
        basisset: str = "SOLDEF2MSVP",
        functional: str = "HSESOL3C",
        shrink: str = "2 2 8",
        maxtradius: str = "0.5",
        post_block: str | None = None,
    ) -> None:
        """Configure CRYSTAL geometry-optimization settings.

        Args:
            basisset: CRYSTAL basis-set keyword. Defaults to ``"SOLDEF2MSVP"``.
            functional: CRYSTAL density-functional keyword. Defaults to
                ``"HSESOL3C"``.
            shrink: CRYSTAL SHRINK-grid specification. Defaults to ``"2 2 8"``.
            maxtradius: ``MAXTRADIUS`` value used in the ``OPTGEOM`` block.
                Defaults to ``"0.5"``.
            post_block: Optional complete CRYSTAL input-tail override. Defaults to
                ``None``, which generates the optimization, basis-set, DFT, and SHRINK
                sections from the supplied settings.
        """
        if post_block is None:
            post_block = f"""OPTGEOM
MAXTRADIUS
{maxtradius}
ENDOPT
BASISSET
{basisset}
DFT
{functional}
END
SHRINK
0 8
{shrink}
END"""
        super().__init__(post_block=post_block)

    def generate_input(
        self,
        cof_name: str,
        mode: str,
        input_base_folder: str | None = None,
        output_base_folder: str | None = None,
    ) -> None:
        """Generate CRYSTAL geometry-optimization inputs for selected structures.

        By default, selected structures are read from the landscape-selection folders
        and CRYSTAL optimization inputs are written to ``dft_serr`` and/or
        ``dft_incl`` subfolders of the normal optimization stage.

        Args:
            cof_name: COF name used for folder naming.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            input_base_folder: Optional input base-folder override. Defaults to
                ``{cof_name}/3_{cof_name}_landscape/selection``.
            output_base_folder: Optional output base-folder override. Defaults to
                ``{cof_name}/4_{cof_name}_optimization``.
        """
        from .ild_ils_utils import get_mode_folders

        if input_base_folder is None:
            input_base_folder = f"{cof_name}/3_{cof_name}_landscape/selection"
        if output_base_folder is None:
            output_base_folder = f"{cof_name}/4_{cof_name}_optimization"

        for folder in get_mode_folders(cof_name, mode):
            mode_tag = Path(folder).name
            self.run(
                input_folder=f"{input_base_folder}/{mode_tag}",
                output_folder=f"{output_base_folder}/dft_{mode_tag}",
            )

    def read(
        self,
        input_folder: str,
        output_csv_dir: str | None = None,
    ) -> Path:
        """Extract optimized CRYSTAL energies from one output folder.

        Converged total energies are converted from Hartree to electronvolts and
        reported on a per-layer basis. Serrated bilayer energies are divided by two.

        Args:
            input_folder: Folder containing CRYSTAL optimization output files.
            output_csv_dir: Optional destination directory for the optimization-energy
                CSV. Defaults to ``{cof_name}/4_{cof_name}_optimization``.

        Returns:
            Path to ``{cof_name}_opt_energies_per_layer_dft.csv``.

        Raises:
            FileNotFoundError: If no valid CRYSTAL output files are found.
        """
        input_path = Path(input_folder)
        folder_tag = input_path.name
        mode_tag = (
            folder_tag.replace("dft_", "")
            if folder_tag.startswith("dft_")
            else folder_tag
        )

        cof_name = folder_tag
        if input_path.parent.name.endswith(
            ("_final_structures", "_optimization")
        ):
            cof_name = input_path.parents[1].name
        elif input_path.parent.name:
            cof_name = input_path.parent.name

        out_files = self._collect_out_files([input_path])

        if not out_files:
            raise FileNotFoundError(
                f"No valid .out files found in: {input_path.resolve()} (expected system_name.out)"
            )

        csv_dir = Path(
            output_csv_dir or f"{cof_name}/4_{cof_name}_optimization"
        )
        csv_dir.mkdir(parents=True, exist_ok=True)

        energies_csv_path = (
            csv_dir / f"{cof_name}_opt_energies_per_layer_dft.csv"
        )

        rows: list[dict[str, str | float]] = []
        failed = []
        for out_path in out_files:
            try:
                text = out_path.read_text(errors="ignore")
                energy_au = self._extract_energy_au(text)
                if energy_au is None:
                    raise ValueError("Energy not found or not converged")
                energy_ev = energy_au * HARTREE_TO_EV
                energy_ev_per_layer = (
                    energy_ev / 2.0 if mode_tag == "serr" else energy_ev
                )
                rows.append(
                    {
                        "structure": out_path.stem,
                        "stacking_mode": mode_tag,
                        "energy_eV_per_layer": energy_ev_per_layer,
                    }
                )
            except Exception as exc:
                failed.append((str(out_path), repr(exc)))

        df = pd.DataFrame(rows).sort_values("structure").reset_index(drop=True)
        if not df.empty:
            min_e = float(df["energy_eV_per_layer"].min())
            df["energy_rel_eV_per_layer"] = df["energy_eV_per_layer"] - min_e
        df.to_csv(energies_csv_path, index=False)

        if failed:
            print("\nFailed files (first 10):")
            for p, err in failed[:10]:
                print(" -", p)
                print("   ", err)
            print("Total failed:", len(failed))

        return energies_csv_path

    def _collect_out_files(self, input_paths: list[Path]) -> list[Path]:
        """Collect valid CRYSTAL .out files from one or more folders.

        Args:
            input_paths: List of folders to search.

        Returns:
            Sorted list of valid .out files.
        """
        out_files: list[Path] = []
        for input_path in input_paths:
            for out_path in sorted(input_path.rglob("*.out")):
                if out_path.name.lower().startswith("slurm"):
                    continue
                if out_path.parent == input_path:
                    out_files.append(out_path)
                    continue
                if out_path.stem == out_path.parent.name:
                    out_files.append(out_path)
        return out_files

    def read_output(
        self,
        cof_name: str,
        mode: str,
        output_base_folder: str | None = None,
        input_base_folder: str | None = None,
    ) -> list[Path]:
        """Extract CRYSTAL optimization energies for selected stacking mode(s).

        The method reads completed ``dft_serr`` and/or ``dft_incl`` calculations,
        combines their per-layer energies into one CSV, and preserves rows belonging to
        an unprocessed stacking mode when only one mode is rerun.

        Args:
            cof_name: COF name used for folder naming.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            output_base_folder: Optional destination directory for the combined energy
                CSV. Defaults to ``{cof_name}/4_{cof_name}_optimization``.
            input_base_folder: Optional base folder containing ``dft_{mode}``
                subfolders. Defaults to ``{cof_name}/4_{cof_name}_optimization``.

        Returns:
            One-element list containing the path to
            ``{cof_name}_opt_energies_per_layer_dft.csv``.

        Raises:
            FileNotFoundError: If no valid CRYSTAL output files are found.
        """
        from .ild_ils_utils import get_mode_folders

        if input_base_folder is None:
            input_base_folder = f"{cof_name}/4_{cof_name}_optimization"
        mode_tags = [
            Path(folder).name for folder in get_mode_folders(cof_name, mode)
        ]
        input_paths = [
            Path(f"{input_base_folder}/dft_{mode_tag}")
            for mode_tag in mode_tags
        ]
        out_files = self._collect_out_files(input_paths)
        if not out_files:
            raise FileNotFoundError(
                f"No valid .out files found in: {input_base_folder} (expected system_name.out)"
            )

        csv_dir = Path(
            output_base_folder or f"{cof_name}/4_{cof_name}_optimization"
        )
        csv_dir.mkdir(parents=True, exist_ok=True)
        energies_csv_path = (
            csv_dir / f"{cof_name}_opt_energies_per_layer_dft.csv"
        )

        rows: list[dict[str, str | float]] = []
        failed = []
        for out_path in out_files:
            try:
                text = out_path.read_text(errors="ignore")
                energy_au = self._extract_energy_au(text)
                if energy_au is None:
                    raise ValueError("Energy not found or not converged")
                energy_ev = energy_au * HARTREE_TO_EV
                mode_from_parent = out_path.parent.name
                if mode_from_parent.startswith("dft_"):
                    mode_from_parent = mode_from_parent.replace("dft_", "")
                if mode_from_parent not in {"serr", "incl"}:
                    mode_from_parent = (
                        "serr" if out_path.stem.endswith("_serr") else "incl"
                    )

                energy_ev_per_layer = (
                    energy_ev / 2.0
                    if mode_from_parent == "serr"
                    else energy_ev
                )
                rows.append(
                    {
                        "structure": out_path.stem,
                        "stacking_mode": mode_from_parent,
                        "energy_eV_per_layer": energy_ev_per_layer,
                    }
                )
            except Exception as exc:
                failed.append((str(out_path), repr(exc)))

        df_new = pd.DataFrame(rows)

        # Merge with existing CSV so rerunning a single mode updates only that
        # mode and preserves rows for other modes.
        if energies_csv_path.exists():
            try:
                df_existing = pd.read_csv(energies_csv_path)
            except Exception:
                df_existing = pd.DataFrame()
        else:
            df_existing = pd.DataFrame()

        if not df_existing.empty and "stacking_mode" in df_existing.columns:
            df_existing = df_existing[
                ~df_existing["stacking_mode"].isin(mode_tags)
            ]

        if df_existing.empty:
            df = df_new
        elif df_new.empty:
            df = df_existing
        else:
            df = pd.concat([df_existing, df_new], ignore_index=True)

        if not df.empty and {"stacking_mode", "structure"}.issubset(
            df.columns
        ):
            df = df.sort_values(["stacking_mode", "structure"]).reset_index(
                drop=True
            )
        if not df.empty:
            min_e = float(df["energy_eV_per_layer"].min())
            df["energy_rel_eV_per_layer"] = df["energy_eV_per_layer"] - min_e
        df.to_csv(energies_csv_path, index=False)

        if failed:
            print("\nFailed files (first 10):")
            for p, err in failed[:10]:
                print(" -", p)
                print("   ", err)
            print("Total failed:", len(failed))

        return [energies_csv_path]

    def extract_structures(
        self,
        input_folder: str,
        output_folder: str | None = None,
    ) -> list[Path]:
        """Extract optimized structures from CRYSTAL output files.

        The final structure is reconstructed from the last available CRYSTAL geometry
        information and written as a CIF file. A secondary parser is used when the
        primary ``DIRECT LATTICE`` / ``PRIMITIVE CELL`` representation is unavailable.

        Args:
            input_folder: Folder containing CRYSTAL optimization output files.
            output_folder: Optional destination folder for generated CIF structures.
                Defaults to the corresponding CRYSTAL output directory.

        Returns:
            List of generated CIF paths.

        Raises:
            FileNotFoundError: If the input folder is missing or contains no valid
                CRYSTAL output files.
            RuntimeError: If the final optimized structure cannot be reconstructed from
                a CRYSTAL output.
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_path}")

        out_files: list[Path] = []
        for out_path in sorted(input_path.rglob("*.out")):
            if out_path.name.lower().startswith("slurm"):
                continue
            if out_path.parent == input_path:
                out_files.append(out_path)
                continue
            if out_path.stem == out_path.parent.name:
                out_files.append(out_path)

        if not out_files:
            raise FileNotFoundError(
                f"No valid .out files found in: {input_path.resolve()} (expected system_name.out)"
            )

        try:
            from ase.io import write
        except Exception as exc:
            raise ModuleNotFoundError(
                "ase is required for CRYSTAL structure extraction. Install with: pip install ase"
            ) from exc

        outputs: list[Path] = []
        for out_path in out_files:
            text = out_path.read_text(errors="ignore")
            lines = text.splitlines()
            structure = _parse_primary_structure(lines)
            if structure is None:
                structure = _parse_fallback_structure(lines)
            if structure is None:
                raise RuntimeError(
                    f"Could not parse structure from: {out_path}"
                )

            out_dir = Path(output_folder) if output_folder else out_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            out_cif = out_dir / f"{out_path.stem}.cif"
            write(str(out_cif), structure)
            outputs.append(out_cif)

        return outputs

    def extract_cif(
        self,
        cof_name: str,
        mode: str,
        output_base_folder: str | None = None,
        input_base_folder: str | None = None,
    ) -> list[Path]:
        """Extract optimized CIF structures for selected stacking mode(s).

        The method processes the ``dft_serr`` and/or ``dft_incl`` optimization output
        folders and reconstructs the final optimized CIF structures in the
        corresponding DFT optimization folders.

        Args:
            cof_name: COF name used for folder naming.
            mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            output_base_folder: Optional output base-folder override. Defaults to
                ``{cof_name}/4_{cof_name}_optimization``.
            input_base_folder: Optional base folder containing ``dft_{mode}``
                subfolders. Defaults to ``{cof_name}/4_{cof_name}_optimization``.

        Returns:
            List of reconstructed optimized CIF paths.
        """
        from .ild_ils_utils import get_mode_folders

        if input_base_folder is None:
            input_base_folder = f"{cof_name}/4_{cof_name}_optimization"
        if output_base_folder is None:
            output_base_folder = input_base_folder

        outputs: list[Path] = []
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = Path(folder).name
            outputs.extend(
                self.extract_structures(
                    input_folder=f"{input_base_folder}/dft_{mode_tag}",
                    output_folder=f"{output_base_folder}/dft_{mode_tag}",
                )
            )
        return outputs
