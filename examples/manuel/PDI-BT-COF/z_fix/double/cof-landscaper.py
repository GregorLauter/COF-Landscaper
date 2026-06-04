#!/usr/bin/env python3
"""End-to-end asymmetric-bilayer COF-Landscaper workflow runner.

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
    """Run the full asymmetric explicit-bilayer workflow from JSON parameters."""
    topology = str(params["TOPOLOGY"])
    cof_name = str(params["COF_NAME"])
    mode = str(params["MODE"])
    mace_head = str(params["MACE_HEAD"])
    device = str(params["DEVICE"])

    mace_opt_max_steps = cl.utilities.get_int_param(params, "MAX_STEPS")

    ild_start = cl.utilities.get_float_param(params, "ILD_START", 5.5)
    ild_end = cl.utilities.get_float_param(params, "ILD_END", 6.5)
    ild_step = cl.utilities.get_float_param(params, "ILD_STEP", 0.1)

    ils_length_end_raw = params.get("ILS_END")
    if ils_length_end_raw is None:
        ils_length_end = None
    elif isinstance(ils_length_end_raw, (int, float, str)):
        ils_length_end = float(ils_length_end_raw)
    else:
        raise TypeError("ILS_END must be a number, numeric string, or null in JSON.")

    ils_length_step = cl.utilities.get_float_param(params, "ILS_STEP", 1.0)
    force_invalid_ild = bool(params.get("FORCE_INVALID_ILD", False))

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

    preopt_interlayer_distance = cl.utilities.get_float_param(
        params,
        "PREOPT_INTERLAYER_DISTANCE",
        15.0,
    )

    builder = cl.BuildCOF2D()
    outputs = builder.build_same_and_flip(
        topo=topology,
        cof_name=cof_name,
        input_nodes=input_nodes,
        input_linkers=input_linkers,
    )

    single_layer_dir = Path(cof_name) / f"1_{cof_name}_single_layer"
    double_unopt_cif = single_layer_dir / f"{cof_name}_double_unopt.cif"
    double_preopt_cif = single_layer_dir / f"{cof_name}_preopt.cif"

    cl.combine_single_layers(
        layer_a_cif=outputs["same_unopt"],
        layer_b_cif=outputs["flip_unopt"],
        output_cif=double_unopt_cif,
        interlayer_distance=preopt_interlayer_distance,
    )

    cl.MaceOpt(
        head=mace_head,
        device=device,
    ).run_preopt(
        cof_name=cof_name,
        fix_z=fix_z,
        input_path=str(double_unopt_cif),
        output_path=str(double_preopt_cif),
    )

    cl.CreateMatrix(
        ild_start=ild_start,
        ild_end=ild_end,
        ild_step=ild_step,
        ils_length_end=ils_length_end,
        ils_length_step=ils_length_step,
        force_invalid_ild=force_invalid_ild,
        explicit_bilayer=True,
    ).run(
        cof_name=cof_name,
        topo=topology,
        mode=mode,
        input_cif=str(double_preopt_cif),
    )

    cl.MaceSP(
        head=mace_head,
        device=device,
    ).run_mode(
        cof_name=cof_name,
        mode=mode,
    )

    cl.Landscape().run_mode(
        cof_name=cof_name,
        mode=mode,
        minima_mode=minima_mode,
        show=show_landscape,
        show_title_block=show_title_block,
        show_minima_markers=show_minima_markers,
    )

    cl.SelectCofs().run_mode(
        cof_name=cof_name,
        mode=mode,
        selections_serr=extra_serr,
        selections_incl=extra_incl,
        include_autoselect=True,
        autoselect_minima=minima_mode,
    )

    cl.MaceOpt(
        head=mace_head,
        device=device,
        max_steps=mace_opt_max_steps,
    ).run_mode(
        cof_name=cof_name,
        mode=mode,
    )

    pxrd = cl.PXRD(wavelength="CuKa", two_theta_range=(1.5, 60.0))
    pxrd.run(cof_name=cof_name, mode=mode)
    pxrd.extract_peaks(cof_name=cof_name, mode=mode)


def main() -> None:
    """Entrypoint for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run the COF-Landscaper asymmetric-bilayer workflow.",
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