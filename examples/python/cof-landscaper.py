#!/usr/bin/env python3
"""End-to-end COF-Landscaper workflow runner for HPC preparation.

Run from project folder:
    python cof-landscaper.py
"""

from __future__ import annotations

import json
from pathlib import Path

import coflandscaper as cl

PARAMS_FILE = Path(__file__).with_name("cof-landscaper.params.json")


def load_params() -> dict[str, object]:
    """Load workflow parameters from the sidecar JSON file."""
    if not PARAMS_FILE.exists():
        raise FileNotFoundError(
            f"Missing parameter file: {PARAMS_FILE}. Create it next to this script."
        )
    return json.loads(PARAMS_FILE.read_text(encoding="utf-8"))


def run_workflow(params: dict[str, object]) -> None:
    """Run the full benchmark workflow using parameters from JSON."""
    topology = str(params["TOPOLOGY"])
    bond_type = str(params["BOND_TYPE"])
    cof_name = str(params["COF_NAME"])
    mode = str(params["MODE"])
    mace_head = str(params["MACE_HEAD"])
    device = str(params["DEVICE"])
    mace_opt_max_steps = int(params["MAX_STEPS"])
    show_landscape = bool(params.get("SHOW_LANDSCAPE", False))
    minima_mode = str(params.get("MINIMA_MODE", "global"))

    builder = cl.BuildCOF2D()
    builder.build(topo=topology, bond_type=bond_type, cof_name=cof_name)

    preopt = cl.MaceOpt(head=mace_head, device=device, fix_z=True)
    preopt.run_preopt(cof_name=cof_name)

    matrix = cl.CreateMatrix()
    matrix.run(cof_name=cof_name, topo=topology, mode=mode)

    sp = cl.MaceSP(head=mace_head, device=device)
    sp.run_mode(cof_name=cof_name, mode=mode)

    crystal_sp = cl.CrystalSP()
    crystal_sp.generate_input(cof_name=cof_name, mode=mode)

    landscape = cl.Landscape()
    landscape.run_mode(
        cof_name=cof_name,
        mode=mode,
        minima_mode=minima_mode,
        show=show_landscape,
    )

    selector = cl.SelectCofs()
    selector.run_mode(
        cof_name=cof_name,
        mode=mode,
        include_autoselect=True,
        autoselect_minima=minima_mode,
    )

    mace_opt = cl.MaceOpt(
        head=mace_head, device=device, max_steps=mace_opt_max_steps
    )
    mace_opt.run_mode(cof_name=cof_name, mode=mode)

    crystal_opt = cl.CrystalOpt()
    crystal_opt.generate_input(cof_name=cof_name, mode=mode)

    cl.analyze(cof_name=cof_name, mode=mode)

    pxrd = cl.Pxrd(wavelength="CuKa", two_theta_range=(1.5, 60.0))
    pxrd.run(cof_name=cof_name, mode=mode)


def main() -> None:
    """Entrypoint for command-line execution."""
    params = load_params()
    run_workflow(params)


if __name__ == "__main__":
    main()
