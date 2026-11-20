"""Private factory for the project's MACE-MH-1 calculator preset."""

from __future__ import annotations

import os
import warnings
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any


def _default_mace(**kwargs: Any):
    """Build the MH-1 preset, applying explicit calculator settings last."""
    import torch

    head = kwargs.pop("head", "spice_wB97M")
    canonical = {
        "spice": "spice_wB97M",
        "spice_wb97m": "spice_wB97M",
        "omat_pbe": "omat_pbe",
        "matpes_r2scan": "matpes_r2scan",
        "omol": "omol",
    }
    head = canonical.get(head.lower(), head) if isinstance(head, str) else head
    settings: dict[str, Any] = {
        "model": "mh-1",
        "head": head,
        "device": "cpu",
        "default_dtype": "float64",
        "dispersion": False,
    }
    if head in ("omat_pbe", "matpes_r2scan"):
        settings.update(
            dispersion=True,
            dispersion_xc="pbe" if head == "omat_pbe" else "r2scan",
            dispersion_cutoff=21.167088422553647
            if head == "omat_pbe"
            else 40.0,
        )
    settings.update(kwargs)

    captured = StringIO()
    env_key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(env_key)
    os.environ[env_key] = "1"
    try:
        with warnings.catch_warnings(), redirect_stdout(captured):
            warnings.filterwarnings(
                "ignore",
                message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
                category=UserWarning,
            )
            from mace.calculators import mace_mp
            from mace.modules.models import ScaleShiftMACE

            safe_globals = getattr(torch.serialization, "safe_globals", None)
            with (
                safe_globals([ScaleShiftMACE])
                if safe_globals
                else nullcontext()
            ):
                calc = mace_mp(**settings)
    finally:
        if previous is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous

    raw_mace_output = captured.getvalue()
    log_path = Path("mace_calculator.log")
    with log_path.open("w", encoding="utf-8") as f:
        if raw_mace_output:
            f.write(raw_mace_output)
            if not raw_mace_output.endswith("\n"):
                f.write("\n")

        metadata_source = calc
        mixer = getattr(calc, "mixer", None)
        mixer_calcs = getattr(mixer, "calcs", None)
        if isinstance(mixer_calcs, list):
            for wrapped_calc in mixer_calcs:
                if hasattr(wrapped_calc, "available_heads"):
                    metadata_source = wrapped_calc
                    break

        effective_head = getattr(metadata_source, "head", "unknown")
        if settings["model"] == "mh-1" and effective_head != "unknown":
            dispersion = " + D3" if settings["dispersion"] else ""
            calc._coflandscaper_method_label = (
                f"MACE-MH-1 ({effective_head}){dispersion}"
            )
        available_heads = getattr(metadata_source, "available_heads", None)
        effective_device = getattr(metadata_source, "device", "unknown")
        d3_detected = "Using TorchDFTD3Calculator" in raw_mace_output
        f.write(
            "[runtime]\n"
            f"head={effective_head}\n"
            f"available_heads={available_heads}\n"
            f"device={effective_device}\n"
            f"d3_detected={d3_detected}\n"
            f"requested_calc_settings={settings}\n"
        )
    return calc
