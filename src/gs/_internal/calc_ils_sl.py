import math
import os
from pymatgen.core import Structure
from .change_2d_utils import list_cifs


class CalcIlsSl:
    def run(self, input_folder: str):
        for input_file in list_cifs(input_folder):
            struct = Structure.from_file(input_file)
            lat = struct.lattice
            a, b, c = lat.abc
            alpha_deg, beta_deg, gamma_deg = lat.angles
            ar = math.radians(alpha_deg)
            br = math.radians(beta_deg)
            gr = math.radians(gamma_deg)

            cx = c * math.cos(br)
            cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / math.sin(gr)
            slip = math.sqrt(cx**2 + cy**2)

            print(f"[CALC_ILS] {os.path.basename(input_file)}: ILS = {slip:.2f} Å")

    __call__ = run
