#!/usr/bin/env python3
"""Run the 2D COF workflow without a display.

From the directory containing 0_node/, 0_linker/ and experimental_pxrd/, run
    python cof-landscaper_2d.py
or select another configuration with
    python cof-landscaper_2d.py --params other.params.json

Paths to inputs and numbered outputs are relative to the working directory.
CALCULATOR and CALCULATOR_KWARGS select the ASE calculator and its own options.
For a different calculator, replace the MACE-specific options in the JSON.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib as mpl

import coflandscaper as cl

DEFAULT_PARAMS_FILE = Path(__file__).with_name("cof-landscaper_2d.params.json")


def run_workflow(params: dict[str, Any]) -> None:
    """Build, screen and optimise the selected stacks, then compare their PXRD."""
    mpl.use("Agg")
    cof_name = params["COF_NAME"]
    topology = params["TOPOLOGY"]
    mode = params["MODE"]
    minima_mode = params["MINIMA_MODE"]
    options = cl.utilities.get_calculator_options(params)

    for key in (
        "FIX_Z",
        "FIX_CELL_ANGLES",
        "INCLUDE_AUTOSELECT",
        "SHOW_MINIMA_MARKERS",
        "SHOW_TITLE_BLOCK",
        "EXPORT_HKL",
    ):
        if not isinstance(params[key], bool):
            raise TypeError(f"{key} must be a JSON boolean (true or false).")

    cl.BuildCOF2D().build(
        topo=topology,
        cof_name=cof_name,
        input_nodes=cl.utilities.get_optional_path_list(params, "NODES"),
        input_linkers=cl.utilities.get_optional_path_list(params, "LINKERS"),
    )

    optimizer = cl.GeometryOpt(
        **options,
        fmax=params["OPT_FMAX"],
        max_steps=cl.utilities.get_int_param(params, "MAX_STEPS"),
    )
    optimizer.run_preopt(
        cof_name,
        topo=topology if params["FIX_CELL_ANGLES"] else None,
        fix_z=params["FIX_Z"],
        fmax=params["PREOPT_FMAX"],
    )

    # A null end uses the exact AB endpoint. A null path uses the topology default.
    matrix = cl.CreateMatrix(
        ild_start=cl.utilities.get_float_param(params, "ILD_START", 3.0),
        ild_end=cl.utilities.get_float_param(params, "ILD_END", 4.0),
        ild_step=cl.utilities.get_float_param(params, "ILD_STEP", 0.1),
        ils_length_start=params["ILS_LENGTH_START"],
        ils_length_end=params["ILS_LENGTH_END"],
        ils_length_step=params["ILS_LENGTH_STEP"],
        ils_angle=params["ILS_ANGLE"],
        slip_path=params["SLIP_PATH"],
    )
    matrix.run(cof_name, topo=topology, mode=mode)

    cl.SinglePoint(calculator=optimizer.calc).run(cof_name, mode=mode)
    cl.Landscape().run(
        cof_name,
        mode=mode,
        minima_mode=minima_mode,
        show=False,
        show_title_block=params["SHOW_TITLE_BLOCK"],
        show_minima_markers=params["SHOW_MINIMA_MARKERS"],
        method_label=params["METHOD_LABEL"],
    )

    # Each manual selection is an [ILD, ILS] pair in angstroms.
    cl.SelectCofs().run(
        cof_name,
        mode=mode,
        autoselect_minima=minima_mode,
        include_autoselect=params["INCLUDE_AUTOSELECT"],
        selections_serr=[(d, s) for d, s in (params["EXTRA_SERR"] or [])],
        selections_incl=[(d, s) for d, s in (params["EXTRA_INCL"] or [])],
    )
    optimizer.run(cof_name, mode=mode)
    cl.AnalyzeStacking().analyze(cof_name, mode=mode)

    angle_min, angle_max = params["PXRD_TWO_THETA_RANGE"]
    pxrd = cl.PXRD(
        wavelength=params["PXRD_WAVELENGTH"],
        two_theta_range=(angle_min, angle_max),
    )
    pxrd.run(
        cof_name,
        mode=mode,
        export_hkl=params["EXPORT_HKL"],
        max_reflections=cl.utilities.get_int_param(params, "MAX_REFLECTIONS"),
    )
    pxrd.extract_peaks(cof_name, mode=mode)
    plot_min, plot_max = params["PXRD_XLIM"]
    pxrd.plot_sim_vs_exp(
        cof_name, mode=mode, xlim=(plot_min, plot_max), show=False, save=True
    )


def main() -> None:
    """Read the configuration and run the workflow in the working directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        "-p",
        type=Path,
        default=DEFAULT_PARAMS_FILE,
        help="Workflow JSON (default: cof-landscaper_2d.params.json beside this script).",
    )
    run_workflow(cl.utilities.load_params(parser.parse_args().params))


if __name__ == "__main__":
    main()
