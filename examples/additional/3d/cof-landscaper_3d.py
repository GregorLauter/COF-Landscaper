#!/usr/bin/env python3
"""Build, optimise and simulate PXRD for a 3D COF on a compute node.

Run python cof-landscaper_3d.py from the directory holding the building blocks.
Use --params to select another JSON file. Input and output paths are relative
to the working directory. NODES maps PORMAKE node types to He-marked XYZ files.
LINKERS maps edge types such as "0,1" to linker XYZ files. An empty mapping
uses direct node-to-node connections.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib as mpl

import coflandscaper as cl

DEFAULT_PARAMS_FILE = Path(__file__).with_name("cof-landscaper_3d.params.json")


def run_workflow(params: dict[str, Any]) -> None:
    """Build one framework, relax the full cell and save simulated PXRD."""
    mpl.use("Agg")
    cof_name = params["COF_NAME"]
    nodes = {int(key): path for key, path in params["NODES"].items()}
    linkers: dict[tuple[int, int], str | None] = {}
    for key, path in params["LINKERS"].items():
        first_type, second_type = map(int, key.split(","))
        linkers[first_type, second_type] = path

    export_hkl = params["EXPORT_HKL"]
    if not isinstance(export_hkl, bool):
        raise TypeError("EXPORT_HKL must be a JSON boolean (true or false).")

    cl.BuildCOF3D().build(
        topo=params["TOPOLOGY"],
        cof_name=cof_name,
        input_nodes=nodes,
        input_linkers=linkers,
    )
    optimizer = cl.GeometryOpt3D(
        **cl.utilities.get_calculator_options(params),
        fmax=cl.utilities.get_float_param(params, "OPT_FMAX", 0.01),
        max_steps=cl.utilities.get_int_param(params, "MAX_STEPS"),
    )
    if not optimizer.run(cof_name):
        raise RuntimeError(
            f"Geometry optimisation of {cof_name} did not converge. "
            "The intermediate CIF was saved."
        )

    angle_min, angle_max = params["PXRD_TWO_THETA_RANGE"]
    pxrd = cl.PXRD3D(
        wavelength=params["PXRD_WAVELENGTH"],
        two_theta_range=(angle_min, angle_max),
    )
    pxrd.run(
        cof_name,
        export_hkl=export_hkl,
        max_reflections=cl.utilities.get_int_param(params, "MAX_REFLECTIONS"),
    )
    pxrd.extract_peaks(cof_name)
    pxrd.plot_sim(cof_name, show=False, save=True)


def main() -> None:
    """Read the configuration and run the workflow in the working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        "-p",
        type=Path,
        default=DEFAULT_PARAMS_FILE,
        help=f"Workflow JSON (default: {DEFAULT_PARAMS_FILE.name} beside this script).",
    )
    run_workflow(cl.utilities.load_params(parser.parse_args().params))


if __name__ == "__main__":
    main()
