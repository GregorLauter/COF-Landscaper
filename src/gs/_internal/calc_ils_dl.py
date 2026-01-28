import os
import numpy as np
from pymatgen.core import Structure
from .change_2d_utils import list_cifs, pick_lower_left_pair_from_lines, parse_xyz_from_atom_line, wrap01


class CalcIlsDl:
    def run(self, input_folder: str):
        for input_file in list_cifs(input_folder):
            struct = Structure.from_file(input_file)

            with open(input_file, "r") as f:
                lines = f.readlines()

            atom_lines = []
            in_atom_loop = False
            saw_atom_headers = False

            for line in lines:
                ls = line.strip()
                if ls.startswith("loop_"):
                    in_atom_loop = False
                    saw_atom_headers = False
                    continue

                if ls.startswith("_atom_site_"):
                    saw_atom_headers = True
                    in_atom_loop = True
                    continue

                if in_atom_loop and saw_atom_headers:
                    if not ls or ls.startswith("_") or ls.startswith("loop_"):
                        break
                    if ls and ls[0].isalpha():
                        atom_lines.append(ls)

            if not atom_lines:
                raise ValueError("No atom lines found in CIF")

            pair_idx, (lower_line, upper_line), (xl, yl, zl), key = pick_lower_left_pair_from_lines(atom_lines)
            xu, yu, zu = parse_xyz_from_atom_line(upper_line)
            xu_w, yu_w = wrap01(xu), wrap01(yu)

            dxu = xu_w - xl
            if dxu < 0.0:
                dxu += 1.0
            dyu = yu_w - yl
            if dyu < 0.0:
                dyu += 1.0

            a_vec, b_vec, _ = struct.lattice.matrix
            a_xy = np.array([a_vec[0], a_vec[1]])
            b_xy = np.array([b_vec[0], b_vec[1]])

            slip_vec_xy = dxu * a_xy + dyu * b_xy
            slip_mag = float(np.linalg.norm(slip_vec_xy))

            print(f"[CALC_ILS_DL] {os.path.basename(input_file)}: Pair {pair_idx}, ILS = {slip_mag:.2f} Å")

    __call__ = run
