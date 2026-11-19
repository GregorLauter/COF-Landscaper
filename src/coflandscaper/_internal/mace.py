"""Run MACE single-point and geometry-optimization workflows for COF structures.

This module provides utilities to configure MACE calculators, evaluate
single-point energies for stacked-structure grids, and optimize selected
structures while preserving workflow-compatible file and CSV outputs.
"""

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
from ase.cell import Cell
from ase.constraints import FixCartesian
from ase.filters import Filter, FrechetCellFilter
from ase.io import read
from ase.optimize import LBFGS
from mace.calculators import mace_mp
from mace.modules.models import ScaleShiftMACE

if TYPE_CHECKING:
    from ase.atoms import Atoms


_PREOPT_CELL_ANGLES = {
    "hcb": (90.0, 90.0, 120.0),
    "hcb_ab": (90.0, 90.0, 120.0),
    "kgm": (90.0, 90.0, 120.0),
    "sql": (90.0, 90.0, 90.0),
    "sql_ab": (90.0, 90.0, 90.0),
}


def _prepare_preopt_cell(atoms: Atoms, topo: str) -> None:
    """Set the ideal 2D cell angles and 15 Angstrom layer spacing in place."""
    if topo not in _PREOPT_CELL_ANGLES:
        raise ValueError(
            f"Unsupported preoptimization topology {topo!r}; "
            f"expected one of {sorted(_PREOPT_CELL_ANGLES)}"
        )
    a, b = atoms.cell.lengths()[:2]
    if (
        len(atoms) == 0
        or not np.all(np.isfinite(atoms.cell))
        or abs(np.linalg.det(atoms.cell)) < 1e-10
        or not np.all(np.isfinite(atoms.positions))
    ):
        raise ValueError(
            "Preoptimization requires a finite, nondegenerate cell and atoms"
        )
    atoms.set_cell(
        Cell.fromcellpar([a, b, 15.0, *_PREOPT_CELL_ANGLES[topo]]),
        scale_atoms=True,
    )


class _FixedAngleCellFilter(Filter):
    """Relax atoms and independent in-plane lengths at fixed angles and height.

    The extra optimizer row holds two scaled logarithmic length changes. Atomic
    coordinates refer to the undeformed cell. Cell forces are the stress
    derivatives of these length variables, including for nonorthogonal cells.
    For a row-vector cell H, stress tensor sigma, volume V, and optimizer
    coordinate q_i = N log(L_i/L_i0), the cell force is
    -(V/N) [H sigma H^-1]_ii. Cartesian stress diagonals alone would omit the
    shear contribution for a nonorthogonal cell.
    """

    def __init__(self, atoms: Atoms) -> None:
        """Keep the prepared lattice directions and c vector fixed."""
        super().__init__(atoms, indices=range(len(atoms)))
        self._reference_cell = np.asarray(atoms.cell).copy()
        self._cell_factor = float(len(atoms))

    def __len__(self) -> int:
        """Include the extra row of cell coordinates."""
        return len(self.atoms) + 1

    def get_positions(self) -> np.ndarray:
        """Return undeformed atomic positions and logarithmic length changes."""
        deformation = np.linalg.solve(self._reference_cell, self.atoms.cell)
        positions = np.zeros((len(self), 3))
        positions[:-1] = np.linalg.solve(
            deformation.T, self.atoms.positions.T
        ).T
        ratios = self.atoms.cell.lengths()[:2] / np.linalg.norm(
            self._reference_cell[:2], axis=1
        )
        positions[-1, :2] = self._cell_factor * np.log(ratios)
        return positions

    def set_positions(self, positions: np.ndarray, **kwargs: Any) -> None:
        """Change a and b without changing angles or the out-of-plane vector."""
        scales = np.ones(3)
        scales[:2] = np.exp(positions[-1, :2] / self._cell_factor)
        cell = self._reference_cell * scales[:, None]
        deformation = np.linalg.solve(self._reference_cell, cell)
        self.atoms.set_cell(cell, scale_atoms=False)
        self.atoms.set_positions(positions[:-1] @ deformation, **kwargs)

    def get_forces(self, **kwargs: Any) -> np.ndarray:
        """Return energy-consistent atomic forces and in-plane cell forces."""
        cell = np.asarray(self.atoms.cell)
        deformation = np.linalg.solve(self._reference_cell, cell)
        forces = np.zeros((len(self), 3))
        forces[:-1] = self.atoms.get_forces(**kwargs) @ deformation.T
        stress = self.atoms.get_stress(voigt=False)
        # Scaling cell row i gives dE/dlog(L_i) = V [H sigma H^-1]_ii.
        length_stress = np.diag(cell @ stress @ np.linalg.inv(cell))
        forces[-1, :2] = (
            -self.atoms.get_volume() * length_stress[:2] / self._cell_factor
        )
        return forces


def _parse_z_L_from_stem(stem: str) -> tuple[float, float]:
    """Parse ILD/ILS-like values encoded as `_z` and `_L` in a filename stem.

    Args:
        stem: Structure filename stem (without extension).

    Returns:
        Tuple `(z, L)` as floats, or `(nan, nan)` when parsing fails.
    """
    mz = re.search(r"_z(\d+)", stem)
    mL = re.search(r"_L(\d+)", stem)
    if not (mz and mL):
        return np.nan, np.nan
    z = float(mz.group(1)) / 10.0
    L = float(mL.group(1)) / 10.0
    return z, L


def calculator_settings_for_head(head: str) -> dict[str, Any]:
    """Map a MACE head name to fixed calculator settings.

    Args:
        head: Requested MACE head name.

    Returns:
        Calculator keyword settings compatible with `mace_mp`.

    Raises:
        ValueError: If `head` is not one of the supported presets.
    """
    head_key = (head or "").lower()
    if head_key == "omat_pbe":
        return {
            "head": "omat_pbe",
            "dispersion": True,
            "dispersion_xc": "pbe",
            "dispersion_cutoff": 21.167088422553647,
        }
    if head_key == "matpes_r2scan":
        return {
            "head": "matpes_r2scan",
            "dispersion": True,
            "dispersion_xc": "r2scan",
            "dispersion_cutoff": 40.0,
        }
    if head_key == "omol":
        return {
            "head": "omol",
            "dispersion": False,
        }
    if head_key in {"spice_wb97m", "spice"}:
        return {
            "head": "spice_wB97M",
            "dispersion": False,
        }
    supported = ["omat_pbe", "matpes_r2scan", "omol", "spice_wB97M"]
    raise ValueError(
        f"Unsupported MACE head '{head}'. Supported heads: {supported}."
    )


class Mace:
    """Base helper for constructing MACE calculators with normalized settings.

    Returns:
        None.
    """

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        head: str = "spice_wB97M",
        model: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Store default MACE runtime options for downstream operations.

        Args:
            device: Torch device string. Defaults to `"cpu"`.
            dtype: Floating-point precision for MACE. Defaults to `"float64"`.
            head: MACE head preset. Defaults to `"spice_wB97M"`.
            model: Optional MACE model identifier. Defaults to `None`
                (resolved later to `"mh-1"`).
            verbose: Whether to emit calculator initialization logs.
                Defaults to `True`.
        """
        self._device = device
        self._dtype = dtype
        self._head = head
        self._model = model
        self._verbose = verbose

    def _resolve_params(
        self,
        device: str | None = None,
        dtype: str | None = None,
        head: str | None = None,
        model: str | None = None,
    ) -> tuple[str, str, str, dict[str, Any]]:
        """Resolve runtime parameters using call-time overrides and defaults.

        Args:
            device: Optional device override. Defaults to `None`.
            dtype: Optional dtype override. Defaults to `None`.
            head: Optional head override. Defaults to `None`.
            model: Optional model override. Defaults to `None`.

        Returns:
            Tuple `(device, dtype, model, calc_settings)`.
        """
        resolved_head = head or self._head
        resolved_model = model or self._model or "mh-1"
        calc_settings = calculator_settings_for_head(resolved_head)
        return (
            device or self._device,
            dtype or self._dtype,
            resolved_model,
            calc_settings,
        )

    def _make_calc(
        self,
        device: str,
        dtype: str,
        model: str,
        calc_settings: dict[str, Any],
    ):
        """Create a MACE calculator while silencing known load-time warnings.

        Args:
            device: Torch device string.
            dtype: Floating-point precision string.
            model: MACE model identifier.
            calc_settings: Head-specific calculator keyword settings.

        Returns:
            Configured MACE calculator instance.
        """
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
                device=device,
                dtype=dtype,
                model=model,
                calc_settings=calc_settings,
            )

    def _make_calc_inner(
        self,
        device: str,
        dtype: str,
        model: str,
        calc_settings: dict[str, Any],
    ):
        """Instantiate the underlying MACE calculator and optional log file.

        Args:
            device: Torch device string.
            dtype: Floating-point precision string.
            model: MACE model identifier.
            calc_settings: Head-specific calculator keyword settings.

        Returns:
            Configured MACE calculator instance.
        """
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([ScaleShiftMACE])

        if self._verbose:
            mace_stdout = StringIO()
            with redirect_stdout(mace_stdout):
                calc = mace_mp(
                    model=model,
                    default_dtype=dtype,
                    device=device,
                    **calc_settings,
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
                    f"requested_calc_settings={calc_settings}\n"
                )
        else:
            calc = mace_mp(
                model=model,
                default_dtype=dtype,
                device=device,
                **calc_settings,
            )
        return calc


class MaceSP(Mace):
    """Evaluate single-point energies for CIF structures using MACE.

    This class computes absolute and relative energies for CIF files and writes
    CSV outputs compatible with downstream landscape analysis.

    Returns:
        None.
    """

    def _run_folder(
        self,
        input_folder: str,
        output_csv_dir: str | None = None,
    ) -> Path:
        """Run single-point evaluation for all CIF files in one folder.

        Args:
            input_folder: Folder containing `.cif` files.
            output_csv_dir: Optional CSV output directory. Defaults to `None`
                (uses `{cof_name}/3_{cof_name}_landscape`).

        Returns:
            Path to the generated single-point energy CSV.

        Raises:
            FileNotFoundError: If no `.cif` files are found in `input_folder`.
        """
        input_path = Path(input_folder)
        folder_tag = input_path.name
        mode_tag = folder_tag

        cof_name = folder_tag
        if input_path.parent.name.endswith("_matrix"):
            cof_name = input_path.parents[1].name
        elif input_path.parent.name:
            cof_name = input_path.parent.name

        cif_files = sorted(input_path.glob("*.cif"))
        if not cif_files:
            raise FileNotFoundError(
                f"No .cif files found in: {input_path.resolve()}"
            )

        csv_dir = Path(output_csv_dir or f"{cof_name}/3_{cof_name}_landscape")
        csv_dir.mkdir(parents=True, exist_ok=True)

        device, dtype, model, calc_settings = self._resolve_params()
        energies_csv_path = csv_dir / f"{cof_name}_sp_energies_{mode_tag}.csv"
        calc = self._make_calc(device, dtype, model, calc_settings)

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
            except Exception as exc:
                failed.append((str(cif_path), repr(exc)))

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

    def run(
        self,
        cof_name: str,
        mode: str,
        input_folder: str | None = None,
        output_csv_dir: str | None = None,
    ) -> None:
        """Run MACE single-point calculations for selected stacking mode(s).

        By default, CIF structures are read from the mode-specific matrix folders
        under ``{cof_name}/2_{cof_name}_matrix``. The method supports ``"serr"``,
        ``"incl"``, and ``"both"`` and writes mode-specific single-point energy CSV
        files for subsequent landscape analysis.

        When ``input_folder`` is provided, only that explicit folder is processed.

        Args:
            cof_name: COF identifier used for default folder resolution.
            mode: Stacking mode selector: ``"serr"``, ``"incl"``, or ``"both"``.
            input_folder: Optional explicit folder containing CIF files. Defaults to
                ``None``, which uses the mode-specific matrix folders.
            output_csv_dir: Optional CSV output directory. Defaults to ``None``,
                which uses ``{cof_name}/3_{cof_name}_landscape``.
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


class MaceOpt(Mace):
    """Optimize COF structures with MACE.

    The class provides the normal full-cell optimization workflow used for
    selected stacking structures, single-layer preoptimization, and fixed-cell
    relaxation of PXRD-scaled structures.

    Normal optimization relaxes atomic coordinates and the unit cell using
    ``FrechetCellFilter``. Fixed-cell optimization relaxes atomic coordinates
    while preserving the complete unit cell.
    Single-layer preoptimization with an explicit topology fixes its ideal
    angles and 15 Angstrom height while relaxing a and b independently.

    Returns:
        None.
    """

    def __init__(
        self,
        fmax: float = 0.01,
        dtype: str = "float64",
        head: str = "spice_wB97M",
        model: str | None = None,
        device: str = "cpu",
        fix_z: bool = False,
        max_steps: int = 2000,
        verbose: bool = True,
    ) -> None:
        """Initialize optimization settings and prebuild a calculator instance.

        Args:
            fmax: Force convergence threshold in eV/Angstrom. Defaults to `0.01`.
            dtype: Floating-point precision for MACE. Defaults to `"float64"`.
            head: MACE head preset. Defaults to `"spice_wB97M"`.
            model: Optional MACE model identifier. Defaults to `None`
                (resolved to `"mh-1"`).
            device: Torch device string. Defaults to `"cpu"`.
            fix_z: Whether to constrain atomic z motion. Defaults to `False`.
            max_steps: Maximum optimizer steps. Defaults to `2000`.
            verbose: Whether to emit calculator initialization logs.
                Defaults to `True`.
        """
        super().__init__(
            device=device,
            dtype=dtype,
            head=head,
            model=model,
            verbose=verbose,
        )
        self._fmax = fmax
        self._fix_z = fix_z
        self._max_steps = max_steps
        _, _, model_used, calc_settings = self._resolve_params()
        self.calc = self._make_calc(
            device=self._device,
            dtype=self._dtype,
            model=model_used,
            calc_settings=calc_settings,
        )

    def _apply_constraints(self, atoms: Atoms) -> None:
        """Apply optional Cartesian z-axis constraints to all atoms.

        Args:
            atoms: ASE atoms object to constrain in place.
        """
        if self._fix_z:
            indices = range(len(atoms))
            con = FixCartesian(indices, mask=[False, False, True])
            atoms.set_constraint(con)

    def _optimize_cof(
        self,
        input_path: str,
        output_path: str,
        fixed_cell: bool = False,
        preopt_topo: str | None = None,
    ) -> bool:
        """Optimize one CIF structure and write the resulting structure.

        Args:
            input_path: Input CIF file path.
            output_path: Output CIF file path.
            fixed_cell: If ``False``, relax atomic coordinates and the unit cell using
                ``FrechetCellFilter``. If ``True``, keep the complete unit cell fixed
                and optimize atomic coordinates directly with ``LBFGS``. Defaults to
                ``False``.
            preopt_topo: Optional 2D topology. Sets its ideal angles and a 15
                Angstrom height, then relaxes only atoms and the a/b lengths.
                Cannot be combined with ``fixed_cell=True``.

        Returns:
            ``True`` if the optimization converged within ``max_steps``; otherwise
            ``False``. The current structure is written in either case.

        Raises:
            ValueError: If both cell modes are requested, or the topology or
                single-layer geometry is invalid.
        """
        if fixed_cell and preopt_topo is not None:
            raise ValueError("fixed_cell and preopt_topo cannot be combined")
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="ase.io.cif",
        )
        atoms = cast("Atoms", read(input_path))
        if preopt_topo is not None:
            _prepare_preopt_cell(atoms, preopt_topo)
        self._apply_constraints(atoms)
        atoms.calc = self.calc

        if fixed_cell:
            dyn = LBFGS(atoms)
        elif preopt_topo is not None:
            length_filter = _FixedAngleCellFilter(atoms)
            dyn = LBFGS(cast("Any", length_filter))
        else:
            fcf = FrechetCellFilter(atoms)
            dyn = LBFGS(cast("Any", fcf))
        converged = dyn.run(fmax=self._fmax, steps=self._max_steps)
        if not converged:
            warnings.warn(
                (
                    "MACE optimization did not converge within "
                    f"{self._max_steps} steps for {input_path}. "
                    "Writing current structure and continuing."
                ),
                category=UserWarning,
                stacklevel=2,
            )
        atoms.write(output_path)
        return bool(converged)

    def optimize_cof(self, input_path: str, output_path: str) -> bool:
        """Optimize one CIF with full atomic and unit-cell relaxation.

        This is the low-level public interface for optimizing a single explicitly
        specified structure. Atomic coordinates and the complete unit cell are
        relaxed using MACE with ``FrechetCellFilter``.
        Cell angles and height remain unrestricted even after a topology-specific
        :meth:`run_preopt` call on the same optimizer instance.

        Args:
            input_path: Input CIF file path.
            output_path: Output CIF file path.

        Returns:
            ``True`` if the optimization converged within ``max_steps``; otherwise
            ``False``.
        """
        return self._optimize_cof(input_path, output_path)

    def run_preopt(
        self,
        cof_name: str,
        input_path: str | None = None,
        output_path: str | None = None,
        fix_z: bool = True,
        *,
        topo: str | None = None,
    ) -> bool:
        """Pre-optimize the assembled single-layer COF structure.

        By default, atomic motion along the Cartesian z direction is constrained
        during this step. This preserves the approximately planar single-layer
        geometry before generation of the stacked structure matrix.
        When ``topo`` is supplied, its ideal cell angles are set before relaxation
        and held fixed with a 15 Angstrom out-of-plane spacing. The a and b
        lengths relax independently. The input CIF is left unchanged.

        Args:
            cof_name: COF name used for default path construction.
            input_path: Optional input CIF path. Defaults to
                ``{cof_name}/1_{cof_name}_single_layer/{cof_name}_unopt.cif``.
            output_path: Optional output CIF path. Defaults to
                ``{cof_name}/1_{cof_name}_single_layer/{cof_name}_preopt.cif``.
            fix_z: Whether to constrain atomic motion along the Cartesian z direction
                during this preoptimization. Defaults to ``True``.
            topo: Optional 2D topology: ``hcb``, ``hcb_ab``, and ``kgm`` use
                90/90/120 degree angles; ``sql`` and ``sql_ab`` use 90/90/90.
                Pass the same topology used for construction. ``None`` retains
                the legacy preoptimization with unrestricted cell angles and
                height. This option affects only this call; subsequent full-cell
                optimization is unchanged.

        Returns:
            ``True`` if the optimization converged; otherwise ``False``.

        Raises:
            ValueError: If ``topo`` is unsupported or the input geometry is invalid.
        """
        if topo is not None and topo not in _PREOPT_CELL_ANGLES:
            raise ValueError(
                f"Unsupported preoptimization topology {topo!r}; "
                f"expected one of {sorted(_PREOPT_CELL_ANGLES)}"
            )
        default_dir = Path(cof_name) / f"1_{cof_name}_single_layer"
        resolved_input = Path(
            input_path or default_dir / f"{cof_name}_unopt.cif"
        )
        resolved_output = Path(
            output_path or default_dir / f"{cof_name}_preopt.cif"
        )
        resolved_output.parent.mkdir(parents=True, exist_ok=True)

        prev_fix_z = self._fix_z
        self._fix_z = fix_z
        try:
            if topo is None:
                converged = self.optimize_cof(
                    str(resolved_input), str(resolved_output)
                )
            else:
                converged = self._optimize_cof(
                    str(resolved_input), str(resolved_output), preopt_topo=topo
                )
            print("Converged" if converged else "Not converged")
            return converged
        finally:
            self._fix_z = prev_fix_z

    def process_cifs(
        self, input_folder: str, output_folder: str
    ) -> dict[str, bool]:
        """Optimize all CIF files in one folder and return convergence flags.

        Args:
            input_folder: Folder containing input CIF files.
            output_folder: Folder where optimized CIF files are written.

        Returns:
            Mapping from structure stem to convergence status.
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        convergence_by_structure: dict[str, bool] = {}
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".cif"):
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, file_name)
                converged = self.optimize_cof(input_path, output_path)
                convergence_by_structure[Path(file_name).stem] = converged
        return convergence_by_structure

    def run_fixed(
        self,
        cof_name: str,
        mode: str,
        input_folder: str | None = None,
        output_folder: str | None = None,
        source_cif: str | None = None,
    ) -> dict[str, bool]:
        """Relax PXRD-scaled structures while keeping the complete unit cell fixed.

        This method is intended for the post-scaling refinement stage. Atomic
        coordinates are optimized directly with ``LBFGS(atoms)`` while the lattice
        vectors and cell parameters remain unchanged.

        Atomic z coordinates are not constrained by this method itself. They remain
        free unless ``fix_z=True`` was explicitly set when constructing ``MaceOpt``.

        Args:
            cof_name: COF name used for default path construction.
            mode: Stacking mode selector: ``"serr"``, ``"incl"``, or ``"both"``.
            input_folder: Optional explicit scaling-stage input folder. Defaults to
                the mode-specific folder under
                ``{cof_name}/6_{cof_name}_scaling/scaling``.
            output_folder: Optional explicit postoptimization output folder. Defaults
                to the mode-specific folder under
                ``{cof_name}/6_{cof_name}_scaling/postopt``.
            source_cif: Optional input CIF filename. If omitted, the input folder must
                contain exactly one CIF file.

        Returns:
            Mapping from ``"mode/structure"`` to convergence status for each
            processed CIF.

        Raises:
            FileNotFoundError: If the requested input folder or CIF is missing.
            ValueError: If explicit folders are used with ``mode="both"`` or the input
                folder contains multiple CIF files without ``source_cif``.
        """
        from .ild_ils_utils import get_mode_folders

        selected_modes = [
            Path(folder).name for folder in get_mode_folders(cof_name, mode)
        ]
        if len(selected_modes) > 1 and (
            input_folder is not None or output_folder is not None
        ):
            raise ValueError(
                "input_folder and output_folder cannot be used with mode='both'; "
                "use the default mode-routed folders."
            )

        convergence_by_structure: dict[str, bool] = {}
        for selected_mode in selected_modes:
            input_dir = Path(
                input_folder
                or f"{cof_name}/6_{cof_name}_scaling/scaling/{selected_mode}"
            )
            output_dir = Path(
                output_folder
                or f"{cof_name}/6_{cof_name}_scaling/postopt/{selected_mode}"
            )
            if source_cif is None:
                cif_files = sorted(input_dir.glob("*.cif"))
                if not cif_files:
                    raise FileNotFoundError(
                        f"No CIF files found in: {input_dir}"
                    )
                if len(cif_files) > 1:
                    raise ValueError(
                        f"Multiple CIF files found in {input_dir}; "
                        "specify source_cif."
                    )
                input_path = cif_files[0]
            else:
                input_path = input_dir / source_cif
                if not input_path.is_file():
                    raise FileNotFoundError(
                        f"Source CIF not found: {input_path}"
                    )

            output_dir.mkdir(parents=True, exist_ok=True)
            convergence_by_structure[f"{selected_mode}/{input_path.stem}"] = (
                self._optimize_cof(
                    str(input_path),
                    str(output_dir / input_path.name),
                    fixed_cell=True,
                )
            )

        return convergence_by_structure

    def _merge_with_existing_energy_csv(
        self,
        new_df: pd.DataFrame,
        csv_path: Path,
    ) -> pd.DataFrame:
        """Merge fresh optimization energies with an existing per-layer CSV.

        Args:
            new_df: Newly evaluated optimization-energy dataframe.
            csv_path: Existing CSV path, if present.

        Returns:
            Combined dataframe with recomputed relative energies.
        """
        columns = [
            "structure",
            "stacking_mode",
            "energy_eV_per_layer",
            "energy_rel_eV_per_layer",
            "stopped_due_to_max_steps",
        ]
        for col in columns:
            if col not in new_df.columns:
                new_df[col] = np.nan

        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            for col in columns:
                if col not in existing_df.columns:
                    existing_df[col] = np.nan
            existing_df = existing_df[columns]
            existing_df["energy_eV_per_layer"] = pd.to_numeric(
                existing_df["energy_eV_per_layer"], errors="coerce"
            )
            existing_df["stopped_due_to_max_steps"] = (
                existing_df["stopped_due_to_max_steps"]
                .fillna(value=False)
                .astype(bool)
            )
            modes_to_replace = set(new_df["stacking_mode"].astype(str))
            existing_df = existing_df[
                ~existing_df["stacking_mode"]
                .astype(str)
                .isin(modes_to_replace)
            ]
        else:
            existing_df = pd.DataFrame(columns=columns)

        combined = pd.concat(
            [existing_df, new_df[columns]],
            ignore_index=True,
        )
        combined = combined.dropna(
            subset=["structure", "stacking_mode", "energy_eV_per_layer"]
        )
        combined = combined.drop_duplicates(
            subset=["stacking_mode", "structure"], keep="last"
        )
        combined = combined.sort_values(["stacking_mode", "structure"])
        combined = combined.reset_index(drop=True)
        combined["stopped_due_to_max_steps"] = (
            combined["stopped_due_to_max_steps"]
            .fillna(value=False)
            .astype(bool)
        )

        min_e = float(combined["energy_eV_per_layer"].min())
        combined["energy_rel_eV_per_layer"] = (
            combined["energy_eV_per_layer"] - min_e
        )
        return combined

    def _write_optimized_energy_csv(
        self,
        cof_name: str,
        mode_output_folders: dict[str, Path],
        output_base_path: Path,
        convergence_map: dict[tuple[str, str], bool] | None = None,
    ) -> Path:
        """Evaluate optimized CIF energies and write/update summary CSV.

        Args:
            cof_name: COF identifier for output CSV naming.
            mode_output_folders: Mapping from mode tag to optimized CIF folder.
            output_base_path: Base output directory for CSV persistence.
            convergence_map: Optional convergence status map. Defaults to `None`.

        Returns:
            Path to the written optimized-energy CSV.

        Raises:
            RuntimeError: If no optimized energies can be evaluated.
        """
        rows: list[dict[str, str | float]] = []
        failed: list[tuple[str, str]] = []

        for mode_tag, folder in mode_output_folders.items():
            cif_files = sorted(folder.glob("*.cif"))
            for cif_path in cif_files:
                try:
                    atoms = cast("Atoms", read(str(cif_path)))
                    atoms.calc = self.calc
                    energy_ev = float(atoms.get_potential_energy())
                    # Serrated mode stores a bilayer; report per-layer energy.
                    energy_ev_per_layer = (
                        energy_ev / 2.0 if mode_tag == "serr" else energy_ev
                    )
                    rows.append(
                        {
                            "structure": cif_path.stem,
                            "stacking_mode": mode_tag,
                            "energy_eV_per_layer": energy_ev_per_layer,
                            "stopped_due_to_max_steps": (
                                not convergence_map.get(
                                    (mode_tag, cif_path.stem), True
                                )
                                if convergence_map is not None
                                else False
                            ),
                        }
                    )
                except Exception as exc:
                    failed.append((str(cif_path), repr(exc)))

        if not rows:
            raise RuntimeError(
                "No optimized energies could be evaluated for the generated CIFs."
            )

        new_df = (
            pd.DataFrame(rows)
            .sort_values(["stacking_mode", "structure"])
            .reset_index(drop=True)
        )

        output_base_path.mkdir(parents=True, exist_ok=True)
        csv_path = output_base_path / f"{cof_name}_opt_energies_per_layer.csv"
        df = self._merge_with_existing_energy_csv(
            new_df=new_df, csv_path=csv_path
        )
        df.to_csv(csv_path, index=False)

        if failed:
            print("\nFailed energy evaluations (first 10):")
            for p, err in failed[:10]:
                print(" -", p)
                print("   ", err)
            print("Total failed:", len(failed))

        return csv_path

    def run(
        self,
        cof_name: str,
        mode: str,
        output_base: str | None = None,
        input_base: str | None = None,
        save_opt_energies_csv: bool = True,
    ) -> None:
        """Run the normal full-cell optimization workflow for selected structure(s).

        The method processes the selected serrated and/or inclined structures,
        relaxing both atomic coordinates and the complete unit cell using MACE and
        ``FrechetCellFilter``. Optimized structures are written to the standard
        optimization folders and, by default, a consolidated per-layer energy CSV
        is generated.

        Default folder behavior:

        - input: ``{cof_name}/3_{cof_name}_landscape/selection/{mode}``
        - output: ``{cof_name}/4_{cof_name}_optimization/{mode}``

        Args:
            cof_name: COF identifier used for default path construction.
            mode: Stacking mode selector: ``"serr"``, ``"incl"``, or ``"both"``.
            output_base: Optional optimization output base folder. Defaults to
                ``{cof_name}/4_{cof_name}_optimization``.
            input_base: Optional selection input base folder. Defaults to
                ``{cof_name}/3_{cof_name}_landscape/selection``.
            save_opt_energies_csv: Whether to write the consolidated per-layer
                optimized-energy CSV. Defaults to ``True``.
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
        convergence_map: dict[tuple[str, str], bool] = {}
        for folder in get_mode_folders(cof_name, mode):
            mode_tag = os.path.basename(folder)
            target_output = output_base_path / mode_tag
            mode_convergence = self.process_cifs(
                input_folder=f"{input_base}/{mode_tag}",
                output_folder=str(target_output),
            )
            for structure_name, converged in mode_convergence.items():
                convergence_map[(mode_tag, structure_name)] = converged
            mode_output_folders[mode_tag] = target_output

        if save_opt_energies_csv:
            csv_path = self._write_optimized_energy_csv(
                cof_name=cof_name,
                mode_output_folders=mode_output_folders,
                output_base_path=output_base_path,
                convergence_map=convergence_map,
            )
            print(f"Saved optimized energies: {csv_path}")
