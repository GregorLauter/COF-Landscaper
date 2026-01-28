import os
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import numpy as np
from ase.constraints import FixCartesian
from ase.filters import UnitCellFilter
from ase.io import read
from ase.optimize import LBFGS

def _make_calc(default_dtype: str, device: str, head: str):
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected*",
        )
        warnings.filterwarnings("ignore", category=UserWarning, module="e3nn")
        with open(os.devnull, "w") as _devnull, redirect_stdout(_devnull), redirect_stderr(
            _devnull
        ), _silence_fds():
            try:
                import torch
                from mace.modules.models import ScaleShiftMACE

                if hasattr(torch.serialization, "add_safe_globals"):
                    torch.serialization.add_safe_globals([ScaleShiftMACE])
            except Exception:
                pass

            from mace.calculators import mace_mp
            return mace_mp(default_dtype=default_dtype, device=device, head=head)

@contextmanager
def _silence_fds():
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out = os.dup(1)
    old_err = os.dup(2)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        os.close(devnull)


class OptMACE:
    def __init__(
        self,
        fmax: float = 0.05,
        default_dtype: str = "float64",
        head: str = "spice_wB97M",
        device: str = "cpu",
        fix_z: bool = True,
    ):
        self.fmax = fmax
        self.fix_z = fix_z
        self.calc = _make_calc(default_dtype, device, head)

    def _apply_constraints(self, atoms):
        if self.fix_z:
            indices = range(len(atoms))
            con = FixCartesian(indices, mask=[False, False, True])
            atoms.set_constraint(con)

    def optimize_cof(self, input_path: str, output_path: str):
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

        with open(os.devnull, "w") as _devnull, redirect_stdout(_devnull), redirect_stderr(
            _devnull
        ), _silence_fds():
            ucf = UnitCellFilter(atoms)
            dyn = LBFGS(ucf)
            dyn.attach(_print_step_info, interval=1)
            dyn.run(fmax=self.fmax)
        atoms.write(output_path)

    def process_cifs(self, input_folder: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".cif"):
                input_path = os.path.join(input_folder, file_name)
                output_path = os.path.join(output_folder, file_name)
                self.optimize_cof(input_path, output_path)


def opt_mace(
    input_folder: str,
    output_folder: str,
    fmax: float = 0.05,
    default_dtype: str = "float64",
    head: str = "spice_wB97M",
    device: str = "cpu",
    fix_z: bool = True,
):
    return OptMACE(
        fmax=fmax,
        default_dtype=default_dtype,
        head=head,
        device=device,
        fix_z=fix_z,
    ).process_cifs(input_folder, output_folder)
