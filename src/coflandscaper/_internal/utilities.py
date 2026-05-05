"""Shared utility helpers for COF-Landscaper workflows."""

from __future__ import annotations

import json
from pathlib import Path


def load_params(params_file: str | Path) -> dict[str, object]:
    """Load workflow parameters from a JSON file.

    Args:
        params_file: Path to the JSON parameter file.

    Returns:
        Parsed JSON data as a dictionary.
    """
    params_path = Path(params_file)
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameter file: {params_path}.")
    return json.loads(params_path.read_text(encoding="utf-8"))


def get_int_param(params: dict[str, object], key: str) -> int:
    """Read an integer-like parameter from a JSON payload.

    Args:
        params: Parsed JSON parameters.
        key: Parameter key to retrieve.

    Returns:
        Integer value for the requested key.
    """
    value = params.get(key)
    if isinstance(value, (int, float, str)):
        return int(value)
    raise TypeError(
        f"Parameter '{key}' must be an int, float, or numeric string, got {type(value).__name__}."
    )


def read_cif_atom_lines(input_file: str | Path) -> list[str]:
    """Read atom-site lines from a CIF file.

    Args:
        input_file: CIF file path.

    Returns:
        List of atom-site lines in the CIF.

    Raises:
        ValueError: If no atom lines are found.
    """
    lines = Path(input_file).read_text(encoding="utf-8").splitlines()

    atom_lines: list[str] = []
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
            if not ls or ls.startswith(("_", "loop_")):
                break
            if ls[0].isalpha():
                atom_lines.append(ls)

    if not atom_lines:
        raise ValueError("No atom lines found in CIF")

    return atom_lines


__all__ = [
    "get_int_param",
    "load_params",
    "read_cif_atom_lines",
]
