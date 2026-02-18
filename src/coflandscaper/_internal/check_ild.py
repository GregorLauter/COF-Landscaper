import os
import numpy as np
from pymatgen.core import Structure
from .change_2d_utils import list_cifs, _calculate_ild, _unwrap_fractional_z


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
                f"[CHECK_ILD] {os.path.basename(input_file)}: "
                f"ILD = {ild:.6f} Å, Layer Thickness = {thickness:.6f} Å"
            )

    __call__ = run
