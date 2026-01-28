import os
from pymatgen.core import Structure
from .change_2d_utils import list_cifs


class Supercell:
    def run(self, input_folder: str, output_folder: str, supercell_size=(2, 2, 2)):
        os.makedirs(output_folder, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            outname = f"{base}_supercell_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.cif"
            outpath = os.path.join(output_folder, outname)
            struct = Structure.from_file(input_file)
            supercell = struct * supercell_size
            supercell.to(filename=outpath)

    __call__ = run
