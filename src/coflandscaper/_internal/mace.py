"""MACE base and derived classes for COF workflows."""

from __future__ import annotations

import os
import re
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch
from ase.constraints import FixCartesian
from ase.filters import UnitCellFilter
from ase.io import read
from ase.optimize import LBFGS
from mace.calculators import mace_mp
from mace.modules.models import ScaleShiftMACE

if TYPE_CHECKING:
    from ase.atoms import Atoms


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
        model: MACE checkpoint key or local path. If None, inferred from head.
        dispersion: Whether to enable dispersion. If None, defaults by head.
        verbose: If True, write raw MACE output and runtime summary to
            `mace_calculator.log`.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        head: str = "omol",
        model: str | None = None,
        dispersion: bool | None = None,
        verbose: bool = True,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.head = head
        self.model = model
        self.dispersion = dispersion
        self.verbose = verbose

    def _resolve_params(
        self,
        device: str | None = None,
        dtype: str | None = None,
        head: str | None = None,
        model: str | None = None,
        dispersion: bool | None = None,
    ) -> tuple[str, str, str, str, bool]:
        resolved_head = head or self.head
        resolved_model = model or self.model or "mh-1"
        if dispersion is None:
            if self.dispersion is None:
                resolved_dispersion = _default_dispersion_for_head(
                    resolved_head
                )
            else:
                resolved_dispersion = self.dispersion
        else:
            resolved_dispersion = dispersion
        return (
            device or self.device,
            dtype or self.dtype,
            resolved_head,
            resolved_model,
            resolved_dispersion,
        )

    def _make_calc(
        self,
        device: str,
        dtype: str,
        head: str,
        model: str,
        dispersion: bool,
    ):
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected*",
            )
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="e3nn"
            )
            return self._make_calc_inner(
                device,
                dtype,
                head,
                model,
                dispersion,
            )

    def _make_calc_inner(
        self,
        device: str,
        dtype: str,
        head: str,
        model: str,
        dispersion: bool,
    ):
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([ScaleShiftMACE])

        if self.verbose:
            mace_stdout = StringIO()
            with redirect_stdout(mace_stdout):
                calc = mace_mp(
                    model=model,
                    default_dtype=dtype,
                    device=device,
                    head=head,
                    dispersion=dispersion,
                )

            raw_mace_output = mace_stdout.getvalue()
            log_path = Path("mace_calculator.log")
            with log_path.open("w", encoding="utf-8") as f:
                if raw_mace_output:
                    f.write(raw_mace_output)
                    if not raw_mace_output.endswith("\n"):
                        f.write("\n")

                metadata_source = calc
                mixer = getattr(calc, "mixer", None)
                mixer_calcs = getattr(mixer, "calcs", None)
                if isinstance(mixer_calcs, list):
                    for wrapped_calc in mixer_calcs:
                        if hasattr(wrapped_calc, "available_heads"):
                            metadata_source = wrapped_calc
                            break

                effective_head = getattr(metadata_source, "head", "unknown")
                available_heads = getattr(
                    metadata_source, "available_heads", None
                )
                effective_device = getattr(
                    metadata_source, "device", "unknown"
                )
                d3_detected = "Using TorchDFTD3Calculator" in raw_mace_output
                f.write(
                    "[runtime]\n"
                    f"head={effective_head}\n"
                    f"available_heads={available_heads}\n"
                    f"device={effective_device}\n"
                    f"d3_detected={d3_detected}\n"
                    f"requested_dispersion={dispersion}\n"
                )
        else:
            calc = mace_mp(
                model=model,
                default_dtype=dtype,
                device=device,
                head=head,
                dispersion=dispersion,
            )
        return calc


class MaceSP(Mace):
    """Single-point MACE energies over a folder of CIFs.

    This computes energies for a set of CIFs (no geometry optimization). It is
    intended for evaluating ILD/ILS grids and generating energy landscapes.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        head: str = "omol",
        model: str | None = None,
        dispersion: bool | None = None,
        verbose: bool = True,
    ) -> None:
        """Configure the MACE single-point calculator.

        Args:
                dtype: Numerical precision; use "float64" (more accurate, slower)
                    or "float32" (faster, less accurate).
                head: MACE head to use (e.g., "omat_pbe", "omol",
                    "spice_wB97M", "matpes_r2scan").
                model: Optional MACE checkpoint key/path. If omitted, inferred
                    from `head` (e.g., "medium-omat-0" for "omat_pbe").
                device: Compute device, e.g. "cpu" or "cuda" (if available).
            dispersion: Whether to enable dispersion. If None, defaults by head.
        """
        super().__init__(
            device=device,
            dtype=dtype,
            head=head,
            model=model,
            dispersion=dispersion,
            verbose=verbose,
        )

    def _run_folder(
        self,
        input_folder: str,
        output_csv_dir: str | None = None,
    ) -> Path:
        input_path = Path(input_folder)
        folder_tag = input_path.name
        mode_tag = folder_tag

        cof_name = folder_tag
        if input_path.parent.name.endswith("_matrix"):
            cof_name = input_path.parents[1].name
        elif input_path.parent.name:
            cof_name = input_path.parent.name

        csv_dir = Path(output_csv_dir or f"{cof_name}/3_{cof_name}_landscape")
        os.makedirs(csv_dir, exist_ok=True)

        cif_files = sorted(input_path.glob("*.cif"))
        if not cif_files:
            raise FileNotFoundError(
                f"No .cif files found in: {input_path.resolve()}"
            )

        device, dtype, head, model, dispersion = self._resolve_params()

        energies_csv_path = csv_dir / f"{cof_name}_sp_energies_{mode_tag}.csv"

        calc = self._make_calc(device, dtype, head, model, dispersion)

        rows = []
        failed = []

        for i, cif_path in enumerate(cif_files, start=1):
            try:
                atoms = cast("Atoms", read(str(cif_path)))
                atoms.calc = calc

                e_ev = float(atoms.get_potential_energy())

                z, L = _parse_z_L_from_stem(cif_path.stem)

                rows.append(
                    {
                        "structure": cif_path.stem,
                        "z": z,
                        "L": L,
                        "energy_eV": e_ev,
                    }
                )
            except Exception as e:
                failed.append((str(cif_path), repr(e)))

            if i % 25 == 0 or i == len(cif_files):
                print(
                    f"[single-point] {i}/{len(cif_files)} done | ok={len(rows)} | failed={len(failed)}"
                )

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

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        input_folder: str | None = None,
        output_csv_dir: str | None = None,
    ) -> None:
        """Run MACE single-point energies for one or more stacking modes.

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
            self._run_folder(
                input_folder=input_folder, output_csv_dir=output_csv_dir
            )
            return

        for folder in get_mode_folders(cof_name, mode):
            self._run_folder(
                input_folder=folder, output_csv_dir=output_csv_dir
            )


class OptMACE(Mace):
    """Geometry optimization using MACE."""

    def __init__(
        self,
        fmax: float = 0.05,
        dtype: str = "float64",
        head: str = "omol",
        model: str | None = None,
        device: str = "cpu",
        fix_z: bool = True,
        dispersion: bool | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            device=device,
            dtype=dtype,
            head=head,
            model=model,
            dispersion=dispersion,
            verbose=verbose,
        )
        self.fmax = fmax
        self.fix_z = fix_z
        _, _, _, model_used, dispersion_used = self._resolve_params()
        self.calc = self._make_calc(
            device=self.device,
            dtype=self.dtype,
            head=self.head,
            model=model_used,
            dispersion=dispersion_used,
        )

    def _apply_constraints(self, atoms: Atoms) -> None:
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
        atoms = cast("Atoms", read(input_path))
        self._apply_constraints(atoms)
        atoms.calc = self.calc

        ucf = UnitCellFilter(atoms)
        dyn = LBFGS(cast("Any", ucf))
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
        self.process_cifs(
            input_folder=input_folder, output_folder=output_folder
        )

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
                If the provided path already starts with `cof_name`, it is
                used as-is.
            input_base: Optional base folder containing per-mode input subfolders.
        """
        from .ild_ils_utils import get_mode_folders

        if input_base is None:
            input_base = f"{cof_name}/3_{cof_name}_landscape/selection"
        if output_base is None:
            output_base = f"4_{cof_name}_optimization"

        output_base_path = Path(output_base)
        if not output_base_path.is_absolute() and (
            not output_base_path.parts or output_base_path.parts[0] != cof_name
        ):
            output_base_path = Path(cof_name) / output_base_path

        for folder in get_mode_folders(cof_name, mode):
            mode_tag = os.path.basename(folder)
            self.run(
                input_folder=f"{input_base}/{mode_tag}",
                output_folder=str(output_base_path / mode_tag),
            )


class MacePreopt(OptMACE):
    """Pre-optimize a single-layer CIF before ILD×ILS matrix generation.

    Assumes input is {cof_name}/1_{cof_name}_single_layer/{cof_name}_unopt.cif
    and writes {cof_name}/1_{cof_name}_single_layer/{cof_name}_preopt.cif.

    Defaults:
        fmax=0.01, dtype="float64", head="omol", model="mace-mh-1", device="cpu",
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
        head: str = "omol",
        model: str | None = None,
        device: str = "cpu",
        fix_z: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            fmax=fmax,
            dtype=dtype,
            head=head,
            model=model,
            device=device,
            fix_z=fix_z,
            dispersion=False,
            verbose=verbose,
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
        output_path = os.path.join(
            output_folder_used, f"{cof_name}_preopt.cif"
        )
        self.optimize_cof(input_path, output_path)


class MaceFullOpt(OptMACE):
    """Full MACE geometry optimization of selected stacking structures.

    Optimized CIFs are written to
    {cof_name}/4_{cof_name}_optimization/{serr|incl} by default, and a
    combined per-layer energy CSV is written to
    {cof_name}/4_{cof_name}_optimization/{cof_name}_opt_energies_per_layer.csv.
    """

    def __init__(
        self,
        fmax: float = 0.01,
        dtype: str = "float64",
        head: str = "omol",
        model: str | None = None,
        device: str = "cpu",
        dispersion: bool | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__(
            fmax=fmax,
            dtype=dtype,
            head=head,
            model=model,
            device=device,
            fix_z=False,
            dispersion=dispersion,
            verbose=verbose,
        )

    def _write_optimized_energy_csv(
        self,
        cof_name: str,
        mode_output_folders: dict[str, Path],
        output_base_path: Path,
    ) -> Path:
        rows: list[dict[str, str | float]] = []
        failed: list[tuple[str, str]] = []

        for mode_tag, folder in mode_output_folders.items():
            cif_files = sorted(folder.glob("*.cif"))
            for cif_path in cif_files:
                try:
                    atoms = cast("Atoms", read(str(cif_path)))
                    atoms.calc = self.calc
                    energy_ev = float(atoms.get_potential_energy())
                    # Serrated mode stores a bilayer; report per-layer energies.
                    energy_ev_per_layer = (
                        energy_ev / 2.0 if mode_tag == "serr" else energy_ev
                    )
                    rows.append(
                        {
                            "structure": cif_path.stem,
                            "stacking_mode": mode_tag,
                            "energy_eV_per_layer": energy_ev_per_layer,
                        }
                    )
                except Exception as exc:
                    failed.append((str(cif_path), repr(exc)))

        if not rows:
            raise RuntimeError(
                "No optimized energies could be evaluated for the generated CIFs."
            )

        df = (
            pd.DataFrame(rows)
            .sort_values(["stacking_mode", "structure"])
            .reset_index(drop=True)
        )
        min_e = float(df["energy_eV_per_layer"].min())
        df["energy_rel_eV_per_layer"] = df["energy_eV_per_layer"] - min_e

        output_base_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_base_path / f"{cof_name}_opt_energies_per_layer.csv"
        df.to_csv(csv_path, index=False)

        if failed:
            print("\nFailed energy evaluations (first 10):")
            for p, err in failed[:10]:
                print(" -", p)
                print("   ", err)
            print("Total failed:", len(failed))

        return csv_path

    def run_mode(
        self,
        cof_name: str,
        mode: str,
        output_base: str | None = None,
        input_base: str | None = None,
    ) -> None:
        """Run full MACE relaxations by mode and write a combined energy CSV.

        Args:
            cof_name: COF name used for folder naming.
            mode: "incl", "serr", or "both".
            output_base: Base folder for optimized CIF outputs.
                Defaults to 4_{cof_name}_optimization under cof_name.
            input_base: Optional base folder containing per-mode input subfolders.
                Defaults to {cof_name}/3_{cof_name}_landscape/selection.

        Notes:
            Serrated structures are bilayers; reported energies are divided by 2
            to produce per-layer values before relative energies are computed.
        """
        from .ild_ils_utils import get_mode_folders

        if input_base is None:
            input_base = f"{cof_name}/3_{cof_name}_landscape/selection"
        if output_base is None:
            output_base = f"4_{cof_name}_optimization"

        output_base_path = Path(output_base)
        if not output_base_path.is_absolute() and (
            not output_base_path.parts or output_base_path.parts[0] != cof_name
        ):
            output_base_path = Path(cof_name) / output_base_path

        mode_output_folders: dict[str, Path] = {}
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = os.path.basename(folder)
            target_output = output_base_path / mode_tag
            self.run(
                input_folder=f"{input_base}/{mode_tag}",
                output_folder=str(target_output),
            )
            mode_output_folders[mode_tag] = target_output

        csv_path = self._write_optimized_energy_csv(
            cof_name=cof_name,
            mode_output_folders=mode_output_folders,
            output_base_path=output_base_path,
        )
        print(f"Saved optimized energies: {csv_path}")
