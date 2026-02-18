import os
import numpy as np
from pymatgen.core import Structure, Lattice
from .change_2d_utils import list_cifs


class SetVacuum:
    def run(self, input_folder: str, output_folder: str, vacuum_top: float):
        os.makedirs(output_folder, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            outname = f"{base}_vacuum_{int(round(vacuum_top*10)):03d}.cif"
            outpath = os.path.join(output_folder, outname)
            self._set_vacuum_on_top(input_file, outpath, vacuum_top)

    def _set_vacuum_on_top(self, input_file, output_file, vacuum_top):
        struct = Structure.from_file(input_file)
        a_vec, b_vec, c_vec = struct.lattice.matrix
        c_len = np.linalg.norm(c_vec)
        c_hat = c_vec / c_len

        z_along_c = np.array([np.dot(site.coords, c_hat) for site in struct.sites])
        zmin = float(np.min(z_along_c))
        zmax = float(np.max(z_along_c))
        thickness = zmax - zmin

        new_c_len = thickness + float(vacuum_top)
        new_c_vec = c_hat * new_c_len
        new_lat = Lattice([a_vec, b_vec, new_c_vec])

        new_cart = [site.coords - zmin * c_hat for site in struct.sites]
        out = Structure(lattice=new_lat, species=struct.species, coords=new_cart, coords_are_cartesian=True)
        out.to(filename=output_file)

    __call__ = run
