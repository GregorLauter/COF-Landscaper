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


__all__ = [
    "get_int_param",
    "load_params",
]
