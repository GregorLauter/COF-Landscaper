import os
import numpy as np
from pymatgen.core import Structure
from .change_2d_utils import list_cifs, _unwrap_fractional_z


class CenterZ:
    def run(self, input_folder: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            outname = f"{base}_centered.cif"
            outpath = os.path.join(output_folder, outname)
            self._center_slab_z(input_file, outpath)

    def _center_slab_z(self, input_file, output_file):
        struct = Structure.from_file(input_file)
        lat = struct.lattice

        frac = np.array([s.frac_coords for s in struct.sites], dtype=float)
        fz = frac[:, 2]

        z0 = _unwrap_fractional_z(fz)
        fz_unwrapped = np.mod(fz - z0, 1.0)

        zmin = float(np.min(fz_unwrapped))
        zmax = float(np.max(fz_unwrapped))
        z_mid = 0.5 * (zmin + zmax)

        dz_frac = 0.5 - z_mid

        fz_centered = np.mod(fz_unwrapped + dz_frac, 1.0)
        frac_centered = frac.copy()
        frac_centered[:, 2] = fz_centered

        out = Structure(lattice=lat, species=struct.species, coords=frac_centered, coords_are_cartesian=False)
        out.to(filename=output_file)

    __call__ = run
