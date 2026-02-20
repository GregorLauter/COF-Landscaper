"""CRYSTAL input generation from CIF files.

Currently supports Crystal23 only. We plan to add three VASP classes later
(two analogous to the CRYSTAL SP/OPT variants).

Provides a base class `Crystal` and two derived classes:
- `CrystalSP`: single-point input blocks
- `CrystalOpt`: geometry optimization input blocks
"""

from __future__ import annotations
from pathlib import Path
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from ase.data import atomic_numbers

def guess_symbol(raw: str) -> Optional[str]:
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
            lbl = next((k for k in hdr if "_type_symbol" in k or "_label" in k), None)

            atoms: list[tuple[int, float, float, float]] = []
            while j < len(lines):
                s = lines[j].strip()
                if not s or s.startswith("loop_") or s.startswith("_"):
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

def _parse_z_L_from_stem(stem: str) -> tuple[float, float]:
    mz = re.search(r"_z(\d+)", stem)
    mL = re.search(r"_L(\d+)", stem)
    if not (mz and mL):
        return np.nan, np.nan
    z = float(mz.group(1)) / 10.0
    L = float(mL.group(1)) / 10.0
    return z, L

class Crystal:
    """Base class to convert CIF files into CRYSTAL .d12 inputs."""

    def __init__(self, post_block: str) -> None:
        self.post_block = post_block

    def _convert_one(self, cif_path: Path, output_path: Path | None = None) -> Path:
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
        if self.post_block.strip():
            out.append(self.post_block.strip())

        out_path.write_text("\n".join(out) + "\n")
        return out_path

    def run(
        self,
        input_folder: str,
        output_folder: str | None = None,
        verbose: bool = False,
    ) -> list[Path]:
        """Convert all CIF files in a folder to .d12.

        Args:
            input_folder: Folder containing .cif files.
            output_folder: Optional output folder for .d12 files.

        Returns:
            List of .d12 output paths.
        """
        in_path = Path(input_folder)
        if not in_path.exists():
            raise FileNotFoundError(f"Input folder not found: {in_path}")

        out_path = Path(output_folder) if output_folder else in_path
        out_path.mkdir(parents=True, exist_ok=True)

        outputs: list[Path] = []
        for cif in sorted(in_path.glob("*.cif")):
            try:
                subdir = out_path / cif.stem
                subdir.mkdir(parents=True, exist_ok=True)
                target = subdir / (cif.stem + ".d12")
                outputs.append(self._convert_one(cif, output_path=target))
            except Exception as exc:
                if verbose:
                    print(f"✖ Failed {cif.name}: {exc}")
        return outputs

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        verbose: bool = False,
        return_paths: bool = False,
    ) -> list[Path] | None:
        """Convert CIFs for stacking modes and write to dft_{mode} folders.

        Args:
            cof_name: COF name used for folder naming.
            mode: "incl", "serr", or "both".

        Returns:
            List of .d12 output paths.
        """
        from .ild_ils_utils import get_mode_folders

        outputs: list[Path] = []
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = Path(folder).name
            outputs.extend(
                self.run(
                    input_folder=folder,
                    output_folder=f"{cof_name}/2_{cof_name}_matrix/dft_{mode_tag}",
                    verbose=verbose,
                )
            )
        return outputs if return_paths else None

class CrystalSP(Crystal):
    """CRYSTAL single-point input generator."""

    def _extract_energy_au(self, text: str) -> float | None:
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
                else:
                    if i + 1 < len(lines):
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
        if post_block is None:
            post_block = """BASISSET
{basisset}
DFT
{functional}
END
SHRINK
            0 8
            {shrink}
END""".format(
                basisset=basisset,
                functional=functional,
                shrink=shrink,
            )
        super().__init__(post_block=post_block)

    def read(
        self,
        input_folder: str,
        output_csv_dir: str | None = None,
    ) -> Path:
        input_path = Path(input_folder)
        folder_tag = input_path.name
        mode_tag = folder_tag.replace("dft_", "") if folder_tag.startswith("dft_") else folder_tag

        cof_name = folder_tag
        if input_path.parent.name.endswith("_matrix"):
            cof_name = input_path.parents[1].name
        elif input_path.parent.name:
            cof_name = input_path.parent.name

        csv_dir = Path(output_csv_dir or f"{cof_name}/3_{cof_name}_landscape")
        os.makedirs(csv_dir, exist_ok=True)

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

        energies_csv_path = csv_dir / f"{cof_name}_sp_energies_{mode_tag}.csv"

        rows = []
        failed = []
        for out_path in out_files:
            try:
                text = out_path.read_text(errors="ignore")
                energy = self._extract_energy_au(text)
                if energy is None:
                    raise ValueError("Energy not found in output")
                z, L = _parse_z_L_from_stem(out_path.stem)
                rows.append(
                    {
                        "structure": out_path.stem,
                        "z": z,
                        "L": L,
                        "energy_Eh": energy,
                    }
                )
            except Exception as exc:
                failed.append((str(out_path), repr(exc)))

        df = pd.DataFrame(rows).sort_values("structure").reset_index(drop=True)
        if not df.empty:
            min_e = float(df["energy_Eh"].min())
            df["energy_rel_Eh"] = df["energy_Eh"] - min_e
        df.to_csv(energies_csv_path, index=False)

        if failed:
            print("\nFailed files (first 10):")
            for p, err in failed[:10]:
                print(" -", p)
                print("   ", err)
            print("Total failed:", len(failed))

        return energies_csv_path

    def read_mode(
        self,
        cof_name: str,
        mode: str,
        output_csv_dir: str | None = None,
    ) -> list[Path]:
        from .ild_ils_utils import get_mode_folders

        csv_paths: list[Path] = []
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = Path(folder).name
            csv_paths.append(
                self.read(
                    input_folder=f"{cof_name}/2_{cof_name}_matrix/dft_{mode_tag}",
                    output_csv_dir=output_csv_dir,
                )
            )
        return csv_paths

class CrystalOpt(Crystal):
    """CRYSTAL geometry-optimization input generator."""

    def __init__(
        self,
        basisset: str = "SOLDEF2MSVP",
        functional: str = "HSESOL3C",
        shrink: str = "2 2 8",
        maxtradius: str = "0.8",
        post_block: str | None = None,
    ) -> None:
        if post_block is None:
            post_block = """OPTGEOM
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
END""".format(
                maxtradius=maxtradius,
                basisset=basisset,
                functional=functional,
                shrink=shrink,
            )
        super().__init__(post_block=post_block)

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        verbose: bool = False,
        return_paths: bool = False,
        input_base: str | None = None,
        output_base: str | None = None,
    ) -> list[Path] | None:
        """Convert CIFs for stacking modes and write to dft_{mode} folders.

        Args:
            cof_name: COF name used for folder naming.
            mode: "incl", "serr", or "both".
            input_base: Optional base folder containing per-mode input subfolders.
            output_base: Optional base folder for outputs (relative to cof_name).

        Returns:
            List of .d12 output paths.
        """
        from .ild_ils_utils import get_mode_folders

        if input_base is None:
            input_base = f"{cof_name}/3_{cof_name}_landscape/selection"
        if output_base is None:
            output_base = f"{cof_name}/4_{cof_name}_final_structures"

        outputs: list[Path] = []
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = Path(folder).name
            outputs.extend(
                self.run(
                    input_folder=f"{input_base}/{mode_tag}",
                    output_folder=f"{output_base}/dft_{mode_tag}",
                    verbose=verbose,
                )
            )
        return outputs if return_paths else None
