"""MACE base and derived classes for COF workflows."""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from ase.constraints import FixCartesian
from ase.filters import UnitCellFilter
from ase.io import read
from ase.optimize import LBFGS
from mace.calculators import mace_mp

EV_TO_HARTREE = 1.0 / 27.211386245988

def _parse_z_L_from_stem(stem: str) -> tuple[float, float]:
    mz = re.search(r"_z(\d+)", stem)
    mL = re.search(r"_L(\d+)", stem)
    if not (mz and mL):
        return np.nan, np.nan
    z = float(mz.group(1)) / 10.0
    L = float(mL.group(1)) / 10.0
    return z, L

def _default_dispersion_for_head(head: str) -> bool:
    head_key = (head or "").lower()
    if head_key in {"omat_pbe", "matpes_r2scan"}:
        return True
    if head_key in {"omol", "spice_wb97m", "spice"}:
        return False
    return False

class Mace:
    """Base class for MACE calculators.

    Stores common configuration and calculator construction.

    Args:
        device: Torch device.
        dtype: Default dtype for the model.
        head: Model head name.
        dispersion: Whether to enable dispersion. If None, defaults by head.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        head: str = "omat_pbe",
        dispersion: bool | None = None,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.head = head
        self.dispersion = dispersion

    def _resolve_params(
        self,
        device: str | None = None,
        dtype: str | None = None,
        head: str | None = None,
        dispersion: bool | None = None,
    ) -> tuple[str, str, str, bool]:
        resolved_head = head or self.head
        if dispersion is None:
            if self.dispersion is None:
                resolved_dispersion = _default_dispersion_for_head(resolved_head)
            else:
                resolved_dispersion = self.dispersion
        else:
            resolved_dispersion = dispersion
        return (
            device or self.device,
            dtype or self.dtype,
            resolved_head,
            resolved_dispersion,
        )

    def _make_calc(
        self,
        device: str,
        dtype: str,
        head: str,
        dispersion: bool,
    ):
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected*",
            )
            warnings.filterwarnings("ignore", category=UserWarning, module="e3nn")
            return self._make_calc_inner(device, dtype, head, dispersion)

    def _make_calc_inner(self, device: str, dtype: str, head: str, dispersion: bool):
        try:
            import torch
            from mace.modules.models import ScaleShiftMACE

            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals([ScaleShiftMACE])
        except Exception:
            pass

        return mace_mp(
            default_dtype=dtype,
            device=device,
            head=head,
            dispersion=dispersion,
        )

class MaceSP(Mace):
    """Single-point MACE energies over a folder of CIFs.

    This computes energies for a set of CIFs (no geometry optimization). It is
    intended for evaluating ILD/ILS grids and generating energy landscapes.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        head: str = "omat_pbe",
        dispersion: bool | None = None,
    ) -> None:
        """Configure the MACE single-point calculator.
        
    Args:
            dtype: Numerical precision; use "float64" (more accurate, slower)
                or "float32" (faster, less accurate).
            head: MACE head to use (e.g., "omat_pbe", "omol",
                "spice_wB97M", "matpes_r2scan").
            device: Compute device, e.g. "cpu" or "cuda" (if available).
        dispersion: Whether to enable dispersion. If None, defaults by head.
        """
        super().__init__(device=device, dtype=dtype, head=head, dispersion=dispersion)

    def _run_folder(
        self,
        input_folder: str,
        output_csv_dir: str | None = None,
    ) -> Path:
        input_path = Path(input_folder)
        folder_tag = input_path.name
        mode_tag = folder_tag if folder_tag in {"serr", "incl"} else folder_tag

        cof_name = folder_tag
        if input_path.parent.name.endswith("_matrix"):
            cof_name = input_path.parents[1].name
        elif input_path.parent.name:
            cof_name = input_path.parent.name

        csv_dir = Path(output_csv_dir or f"{cof_name}/3_{cof_name}_landscape")
        os.makedirs(csv_dir, exist_ok=True)

        cif_files = sorted(input_path.glob("*.cif"))
        if not cif_files:
            raise FileNotFoundError(f"No .cif files found in: {input_path.resolve()}")

        device, dtype, head, dispersion = self._resolve_params()

        energies_csv_path = csv_dir / f"{cof_name}_sp_energies_{mode_tag}.csv"

        calc = self._make_calc(device, dtype, head, dispersion)

        rows = []
        failed = []

        for i, cif_path in enumerate(cif_files, start=1):
            try:
                atoms = read(str(cif_path))
                atoms.calc = calc

                e_ev = float(atoms.get_potential_energy())
                e_eh = e_ev * EV_TO_HARTREE

                z, L = _parse_z_L_from_stem(cif_path.stem)

                rows.append(
                    {
                        "structure": cif_path.stem,
                        "z": z,
                        "L": L,
                        "energy_Eh": e_eh,
                    }
                )
            except Exception as e:
                failed.append((str(cif_path), repr(e)))

            if i % 25 == 0 or i == len(cif_files):
                print(f"[single-point] {i}/{len(cif_files)} done | ok={len(rows)} | failed={len(failed)}")

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

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        input_folder: str | None = None,
        output_csv_dir: str | None = None,
    ) -> None:
        """Run single-point energies for stacking modes or a specific folder.

        Args:
            cof_name: COF name used for folder naming.
            mode: "incl", "serr", or "both".
            input_folder: Optional folder containing CIFs to process.
                Defaults to {cof_name}/2_{cof_name}_matrix/{serr|incl}.
            output_csv_dir: Optional output folder for CSVs.
                Defaults to {cof_name}/3_{cof_name}_landscape.

        Returns:
            None.
        """
        from .ild_ils_utils import get_mode_folders

        if input_folder:
            self._run_folder(input_folder=input_folder, output_csv_dir=output_csv_dir)
            return None

        for folder in get_mode_folders(cof_name, mode):
            self._run_folder(input_folder=folder, output_csv_dir=output_csv_dir)
        return None

class OptMACE(Mace):
    """Geometry optimization using MACE."""

    def __init__(
        self,
        fmax: float = 0.05,
        dtype: str = "float64",
            head: str = "omat_pbe",
        device: str = "cpu",
        fix_z: bool = True,
        dispersion: bool | None = None,
    ) -> None:
        super().__init__(device=device, dtype=dtype, head=head, dispersion=dispersion)
        self.fmax = fmax
        self.fix_z = fix_z
        self.calc = self._make_calc(
            device=self.device,
            dtype=self.dtype,
            head=self.head,
            dispersion=self.dispersion,
        )

    def _apply_constraints(self, atoms) -> None:
        if self.fix_z:
            indices = range(len(atoms))
            con = FixCartesian(indices, mask=[False, False, True])
            atoms.set_constraint(con)

    def optimize_cof(self, input_path: str, output_path: str) -> None:
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="ase.io.cif",
        )
        atoms = read(input_path)
        self._apply_constraints(atoms)
        atoms.calc = self.calc

        def _print_step_info():
            f = atoms.get_forces()
            max_force = np.abs(f).max()
            _ = atoms.get_potential_energy()
            _ = max_force

        ucf = UnitCellFilter(atoms)
        dyn = LBFGS(ucf)
        dyn.attach(_print_step_info, interval=1)
        dyn.run(fmax=self.fmax)
        atoms.write(output_path)

    def process_cifs(self, input_folder: str, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".cif"):
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, file_name)
                self.optimize_cof(input_path, output_path)

    def run(self, input_folder: str, output_folder: str) -> None:
        """Alias for batch optimization over a folder of CIFs."""
        self.process_cifs(input_folder=input_folder, output_folder=output_folder)

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        output_base: str | None = None,
        input_base: str | None = None,
    ) -> None:
        """Perform full 3D MACE relaxations for stacking-mode CIFs and write relaxed structures.

        Args:
            cof_name: COF name used for folder naming.
            mode: "incl", "serr", or "both".
            output_base: Base folder for outputs (relative to cof_name).
            input_base: Optional base folder containing per-mode input subfolders.
        """
        from .ild_ils_utils import get_mode_folders

        if input_base is None:
            input_base = f"{cof_name}/3_{cof_name}_landscape/selection"
        if output_base is None:
            output_base = f"4_{cof_name}_final_structures"

        for folder in get_mode_folders(cof_name, mode):
            mode_tag = os.path.basename(folder)
            self.run(
                input_folder=f"{input_base}/{mode_tag}",
                output_folder=f"{cof_name}/{output_base}/{mode_tag}",
            )

class MacePreopt(OptMACE):
    """Pre-optimize a single CIF to improve the subsequent energy landscape.

    This step refines the CIF produced by pormake so that later landscape
    calculations are more stable. Assumes input is in
    1_{cof_name}_single_layer/{cof_name}_unopt.cif and writes
    1_{cof_name}_single_layer/{cof_name}_preopt.cif.

    Defaults:
        fmax=0.01, dtype="float64", head="omat_pbe", device="cpu",
        fix_z=True, dispersion=False.

    Options:
        - fix_z: Whether to constrain Z during optimization.
        - head: Any available MACE head for the mace-mh-1 model.

    MACE model & heads:
        Uses mace-mh-1 (https://huggingface.co/mace-foundations/mace-mh-1).
        Recommended heads: "omat_pbe", "omol", "spice_wB97M", "matpes_r2scan".
    """

    def __init__(
        self,
        fmax: float = 0.01,
        dtype: str = "float64",
            head: str = "omat_pbe",
        device: str = "cpu",
        fix_z: bool = True,
    ) -> None:
        super().__init__(
            fmax=fmax,
            dtype=dtype,
            head=head,
            device=device,
            fix_z=fix_z,
            dispersion=False,
        )

    def run(
        self,
        cof_name: str,
        input_folder: str | None = None,
        output_folder: str | None = None,
    ) -> None:
        """Run the pre-optimization step on a single CIF.

        Args:
            cof_name: COF name used for default input/output naming.
            input_folder: Optional folder containing {cof_name}_unopt.cif.
            output_folder: Optional folder for {cof_name}_preopt.cif.
        """
        default_folder = os.path.join(cof_name, f"1_{cof_name}_single_layer")
        input_folder_used = input_folder or default_folder
        output_folder_used = output_folder or default_folder
        input_path = os.path.join(input_folder_used, f"{cof_name}_unopt.cif")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing input CIF: {input_path}")
        os.makedirs(output_folder_used, exist_ok=True)
        output_path = os.path.join(output_folder_used, f"{cof_name}_preopt.cif")
        self.optimize_cof(input_path, output_path)

class MaceFullOpt(OptMACE):
    """Full geometry optimization with MACE allowing unconstrained 3D relaxation.

    Processes all CIFs in a folder without fixed Z, optional dispersion.
    """

    def __init__(
        self,
        fmax: float = 0.01,
        dtype: str = "float64",
            head: str = "omat_pbe",
        device: str = "cpu",
        dispersion: bool | None = None,
    ) -> None:
        super().__init__(
            fmax=fmax,
            dtype=dtype,
            head=head,
            device=device,
            fix_z=False,
            dispersion=dispersion,
        )

