#!/usr/bin/env python3
"""End-to-end COF-Landscaper workflow runner for HPC preparation.

Run from project folder:
    python cof-landscaper.py
    python cof-landscaper.py --params /path/to/params.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coflandscaper as cl

DEFAULT_PARAMS_FILE = Path(__file__).with_name("cof-landscaper.params.json")


def run_workflow(params: dict[str, object]) -> None:
    """Run the full benchmark workflow using parameters from JSON."""
    topology = str(params["TOPOLOGY"])
    cof_name = str(params["COF_NAME"])
    mode = str(params["MODE"])
    mace_head = str(params["MACE_HEAD"])
    device = str(params["DEVICE"])

    mace_opt_max_steps = cl.utilities.get_int_param(params, "MAX_STEPS")

    ild_start = cl.utilities.get_float_param(params, "ILD_START", 3.0)
    ild_end = cl.utilities.get_float_param(params, "ILD_END", 4.0)
    ild_step = cl.utilities.get_float_param(params, "ILD_STEP", 0.1)

    minima_mode = str(params.get("MINIMA_MODE", "global"))
    show_landscape = bool(params.get("SHOW_LANDSCAPE", False))
    show_title_block = bool(params.get("SHOW_TITLE_BLOCK", False))
    show_minima_markers = bool(params.get("SHOW_MINIMA_MARKERS", True))
    fix_z = params.get("FIX_Z", True)
    if not isinstance(fix_z, bool):
        raise TypeError("FIX_Z must be a boolean (true/false) in JSON.")

    input_nodes = cl.utilities.get_optional_path_list(params, "NODES")
    input_linkers = cl.utilities.get_optional_path_list(params, "LINKERS")

    extra_serr = cl.utilities._parse_extra_points(params.get("EXTRA_SERR"))
    extra_incl = cl.utilities._parse_extra_points(params.get("EXTRA_INCL"))

    builder = cl.BuildCOF2D()
    builder.build(
        topo=topology,
        cof_name=cof_name,
        input_nodes=input_nodes,
        input_linkers=input_linkers,
    )

    preopt = cl.MaceOpt(head=mace_head, device=device)
    preopt.run_preopt(cof_name=cof_name, fix_z=fix_z)

    matrix = cl.CreateMatrix(
        ild_start=ild_start,
        ild_end=ild_end,
        ild_step=ild_step,
    )
    matrix.run(cof_name=cof_name, topo=topology, mode=mode)

    sp = cl.MaceSP(head=mace_head, device=device)
    sp.run_mode(cof_name=cof_name, mode=mode)

    landscape = cl.Landscape()
    landscape.run_mode(
        cof_name=cof_name,
        mode=mode,
        minima_mode=minima_mode,
        show=show_landscape,
        show_title_block=show_title_block,
        show_minima_markers=show_minima_markers,
    )

    selector = cl.SelectCofs()
    selector.run_mode(
        cof_name=cof_name,
        mode=mode,
        selections_serr=extra_serr,
        selections_incl=extra_incl,
        include_autoselect=True,
        autoselect_minima=minima_mode,
    )

    mace_opt = cl.MaceOpt(
        head=mace_head,
        device=device,
        max_steps=mace_opt_max_steps,
    )
    mace_opt.run_mode(cof_name=cof_name, mode=mode)

    analyzer = cl.AnalyzeStacking()
    analyzer.analyze(cof_name=cof_name, mode=mode)

    pxrd = cl.PXRD(wavelength="CuKa", two_theta_range=(1.5, 60.0))
    pxrd.run(cof_name=cof_name, mode=mode)
    pxrd.extract_peaks(cof_name=cof_name, mode=mode)


def main() -> None:
    """Entrypoint for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run the COF-Landscaper hybrid workflow.",
    )
    parser.add_argument(
        "--params",
        "-p",
        type=Path,
        default=DEFAULT_PARAMS_FILE,
        help=(
            "Path to the workflow JSON parameters file. "
            f"Defaults to {DEFAULT_PARAMS_FILE.name} next to this script."
        ),
    )
    args = parser.parse_args()

    params = cl.utilities.load_params(args.params)
    run_workflow(params)


if __name__ == "__main__":
    main()
