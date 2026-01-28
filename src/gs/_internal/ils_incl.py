import os
import math
from pymatgen.core import Structure, Lattice
from .change_2d_utils import list_cifs, _generate_values, _slug


class IlsIncl:
    def run(
        self,
        input_folder: str,
        output_folder: str,
        incl_length_start: float,
        incl_length_end: float,
        incl_length_step: float,
        incl_angle: float,
    ):
        os.makedirs(output_folder, exist_ok=True)
        incl_lengths = _generate_values(incl_length_start, incl_length_end, incl_length_step)

        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            for ilen in incl_lengths:
                outname = f"{base}_inc_L{_slug(ilen)}.cif"
                outpath = os.path.join(output_folder, outname)
                self._inclined_shift(input_file, outpath, ilen, incl_angle)

    def _inclined_shift(self, input_file, output_file, shift_length, shift_angle_deg):
        struct = Structure.from_file(input_file)

        a_vec, b_vec, c_vec = struct.lattice.matrix
        c_len = struct.lattice.c

        angle_rad = math.radians(shift_angle_deg)
        x_shift = shift_length * math.cos(angle_rad)
        y_shift = shift_length * math.sin(angle_rad)

        new_c_vec = [x_shift, y_shift, c_len]
        new_lattice = Lattice([a_vec, b_vec, new_c_vec])

        cart_coords = struct.cart_coords
        new_frac = new_lattice.get_fractional_coords(cart_coords)

        new_struct = Structure(
            lattice=new_lattice,
            species=struct.species,
            coords=new_frac,
            coords_are_cartesian=False,
        )
        new_struct.to(filename=output_file)

    __call__ = run
