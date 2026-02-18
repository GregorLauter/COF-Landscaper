import os
import numpy as np
from pymatgen.core import Structure
from .change_2d_utils import list_cifs, _periodic_delta_frac, _slug


class RemoveLayer:
    def run(self, input_folder: str, output_folder: str, remove_z: float, remove_tol: float, mode: str = "frac"):
        os.makedirs(output_folder, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            if mode == "frac":
                tag = f"rm_z{int(round(remove_z*1000))}_t{int(round(remove_tol*1000))}"
            else:
                tag = f"rm_Z{_slug(remove_z)}_t{_slug(remove_tol)}"
            outname = f"{base}_{tag}.cif"
            outpath = os.path.join(output_folder, outname)
            self._remove_layer(input_file, outpath, remove_z, remove_tol, mode=mode)

    def _remove_layer(self, input_file: str, output_file: str, z_value: float, tol: float, mode: str = "frac"):
        struct = Structure.from_file(input_file)
        lat = struct.lattice
        c_vec = lat.matrix[2]
        c_len = float(np.linalg.norm(c_vec))
        c_hat = c_vec / c_len

        keep_species = []
        keep_frac = []

        removed = 0
        if mode == "frac":
            z0 = float(z_value) % 1.0
            tol_f = float(tol)
            for s in struct.sites:
                fz = float(s.frac_coords[2]) % 1.0
                dz = _periodic_delta_frac(fz, z0)
                if dz > tol_f:
                    keep_species.append(s.species)
                    keep_frac.append(s.frac_coords)
                else:
                    removed += 1
        elif mode == "cart":
            z0a = float(z_value)
            tol_a = float(tol)
            for s in struct.sites:
                zcart = float(np.dot(s.coords, c_hat))
                if abs(zcart - z0a) > tol_a:
                    keep_species.append(s.species)
                    keep_frac.append(s.frac_coords)
                else:
                    removed += 1
        else:
            raise ValueError("mode must be 'frac' or 'cart'")

        if len(keep_species) == 0:
            raise RuntimeError("All atoms would be removed; aborting to avoid empty structure.")

        out = Structure(lattice=lat, species=keep_species, coords=keep_frac, coords_are_cartesian=False)
        out.to(filename=output_file)

    __call__ = run
