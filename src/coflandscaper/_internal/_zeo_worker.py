"""Run pyzeo for one prepared P1 cell in a private working directory."""

import json
import sys
from itertools import pairwise
from pathlib import Path


def _fields(report: bytes) -> dict[str, float]:
    """Read the labelled scalar quantities on Zeo++'s summary line."""
    tokens = report.decode().splitlines()[0].split()
    return {
        key.removesuffix(":"): float(value)
        for key, value in pairwise(tokens)
        if key.endswith(":")
    }


def _calculate(
    channel: float,
    probe: float,
    surface_samples: int,
    volume_samples: int,
    atom_count: int,
) -> dict[str, float]:
    """Calculate accessible surface/volume and probe-independent pore diameters."""
    try:
        from pyzeo import (
            AtomNetwork,
            high_accuracy_atomnet,
            surface_area,
            volume,
        )
    except ImportError as error:
        raise ImportError(
            'Cannot load pyzeo. Install a compatible build with pip install "cof-landscaper[porosity]" on macOS or Linux.'
        ) from error

    network = AtomNetwork.read_from_CIF("structure.cif")
    if network.no_atoms != atom_count:
        raise ValueError(
            "Zeo++ read a different number of atoms from the prepared cell."
        )
    refined = network.copy()
    high_accuracy_atomnet(refined, "DEF")
    settings = {
        "high_accuracy": True,
        "high_accuracy_atmnet": refined,
        "exclude_pockets": True,
    }
    area = _fields(
        surface_area(network, channel, probe, surface_samples, **settings)
    )
    void = _fields(volume(network, channel, probe, volume_samples, **settings))
    refined.calculate_free_sphere_parameters("diameters.res")
    _, lcd, pld, along_path = Path("diameters.res").read_text().split()
    return {
        "accessible_surface_area_m2_g": area["ASA_m^2/g"],
        "accessible_surface_area_m2_cm3": area["ASA_m^2/cm^3"],
        "accessible_pore_volume_cm3_g": void["AV_cm^3/g"],
        "accessible_void_fraction_percent": 100 * void["AV_Volume_fraction"],
        "lcd_angstrom": float(lcd),
        "pld_angstrom": float(pld),
        "largest_included_diameter_along_free_path_angstrom": float(
            along_path
        ),
    }


if __name__ == "__main__":
    results = _calculate(
        float(sys.argv[1]),
        float(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
    )
    Path("result.json").write_text(json.dumps(results, allow_nan=False))
