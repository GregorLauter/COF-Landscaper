import os
import math
import numpy as np
from pymatgen.core import Structure
from .change_2d_utils import list_cifs, _generate_values, _slug


class IlsSerr:
    def run(
        self,
        input_folder: str,
        output_folder: str,
        shift_length_start: float,
        shift_length_end: float,
        shift_length_step: float,
        shift_angle: float,
    ):
        os.makedirs(output_folder, exist_ok=True)
        shift_lengths = _generate_values(shift_length_start, shift_length_end, shift_length_step)

        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            for slen in shift_lengths:
                tag = f"L{_slug(slen)}"
                outname = f"{base}_ser_{tag}.cif"
                outpath = os.path.join(output_folder, outname)
                self._shift_serrated(input_file, outpath, slen, shift_angle)

    def _shift_serrated(self, input_file, output_file, shift_length, shift_angle_deg):
        struct = Structure.from_file(input_file)
        supercell = struct * (1, 1, 2)

        angle_rad = math.radians(shift_angle_deg)
        shift_cart = np.array(
            [shift_length * math.cos(angle_rad), shift_length * math.sin(angle_rad), 0.0]
        )
        fx, fy, _ = supercell.lattice.get_fractional_coords(shift_cart)

        mid_z = 0.5
        new_frac = []
        for site in supercell.sites:
            f = np.array(site.frac_coords, dtype=float)
            if f[2] > mid_z:
                f[0] += fx
                f[1] += fy
            new_frac.append(np.mod(f, 1.0))

        out = Structure(
            lattice=supercell.lattice,
            species=supercell.species,
            coords=new_frac,
            coords_are_cartesian=False,
        )
        out.to(filename=output_file)

    __call__ = run
