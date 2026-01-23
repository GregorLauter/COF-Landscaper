#!/Users/gregorlauter/tools/anaconda3/envs/tools/bin/python

import os
from CRYSTALpytools.crystal_io import Crystal_output

for file_name in os.listdir("."):
    if file_name.endswith(".out"):
        output_file = file_name
        output_cif = file_name.replace(".out", "_opt.cif")
        crystal_output = Crystal_output(output_file)
        geometry = crystal_output.get_geometry(initial=False)
        geometry.to(fmt="cif", filename=output_cif)
        print(f"Extracted final geometry and saved to {output_cif}")