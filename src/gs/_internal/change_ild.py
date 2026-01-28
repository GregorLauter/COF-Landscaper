import os
import numpy as np
from pymatgen.core import Structure, Lattice
from .change_2d_utils import list_cifs, _calculate_ild, _unwrap_fractional_z, _generate_values, _z_tag


class ChangeIld:
    def run(self, input_folder: str, output_folder: str, new_z_start: float, new_z_end: float, new_z_step: float):
        os.makedirs(output_folder, exist_ok=True)
        z_values = _generate_values(new_z_start, new_z_end, new_z_step)

        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            for new_z in z_values:
                outname = f"{base}_{_z_tag(new_z)}.cif"
                outpath = os.path.join(output_folder, outname)
                self._change_interlayer_distance(input_file, outpath, float(new_z))

    def _change_interlayer_distance(self, input_file, output_file, new_z):
        struct = Structure.from_file(input_file)
        lat_old = struct.lattice
        a_vec, b_vec, c_vec_old = lat_old.matrix

        z_len_old = _calculate_ild(lat_old)

        fz = struct.frac_coords[:, 2]
        z0 = _unwrap_fractional_z(fz)
        fz_unwrapped = np.mod(fz - z0, 1.0)
        thickness = (np.max(fz_unwrapped) - np.min(fz_unwrapped)) * z_len_old

        if new_z < thickness:
            raise ValueError(f"New ILD {new_z:.4f} Å < slab thickness {thickness:.4f} Å; cannot fit.")

        scale_factor = new_z / z_len_old
        new_c_vec = c_vec_old * scale_factor
        lat_new = Lattice([a_vec, b_vec, new_c_vec])

        frac_raw = lat_new.get_fractional_coords(struct.cart_coords)
        fz_new = frac_raw[:, 2]
        zmin_f = np.min(fz_new)
        zmax_f = np.max(fz_new)
        z_mid_f = (zmin_f + zmax_f) / 2
        delta_f = 0.5 - z_mid_f
        fz_centered = fz_new + delta_f

        fx = np.mod(frac_raw[:, 0], 1.0)
        fy = np.mod(frac_raw[:, 1], 1.0)
        frac_final = np.column_stack([fx, fy, fz_centered])

        new_struct = Structure(
            lattice=lat_new,
            species=struct.species,
            coords=frac_final,
            coords_are_cartesian=False,
        )
        new_struct.to(filename=output_file)

    __call__ = run
