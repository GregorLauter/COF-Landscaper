"""Utility helpers for ILD/ILS value generation, parsing, and mode routing.

This module contains shared low-level functions used by ILD/ILS matrix
generation and related workflows, including CIF file discovery, default shift
derivation, and small geometry/value conversion helpers.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core import Lattice, Structure

if TYPE_CHECKING:
    from collections.abc import Iterable


def list_cifs(input_folder: str) -> list[str]:
    """List CIF files in a folder.

    Args:
        input_folder: Folder containing CIF files.

    Returns:
        Sorted list of CIF file paths.

    Raises:
        FileNotFoundError: If no CIF files are found.
    """
    files = sorted(
        f
        for f in (
            os.path.join(input_folder, n) for n in os.listdir(input_folder)
        )
        if f.endswith(".cif")
    )
    if not files:
        raise FileNotFoundError(f"No .cif files found in '{input_folder}'")
    return files


def _calculate_ild(lat: Lattice) -> float:
    """Compute interlayer distance (ILD) from lattice metrics.

    Args:
        lat: Pymatgen lattice.

    Returns:
        Interlayer distance in Å.
    """
    a, b, c = lat.abc
    alpha_deg, beta_deg, gamma_deg = lat.angles
    alpha_r = np.radians(alpha_deg)
    beta_r = np.radians(beta_deg)
    gamma_r = np.radians(gamma_deg)
    V = (
        a
        * b
        * c
        * np.sqrt(
            1
            + 2 * np.cos(alpha_r) * np.cos(beta_r) * np.cos(gamma_r)
            - np.cos(alpha_r) ** 2
            - np.cos(beta_r) ** 2
            - np.cos(gamma_r) ** 2
        )
    )
    return V / (a * b * np.sin(gamma_r))


def _unwrap_fractional_z(frac_z: np.ndarray) -> float:
    """Choose a fractional-z reference that avoids periodic discontinuities.

    Args:
        frac_z: Fractional $z$ coordinates.

    Returns:
        Reference fractional $z$ in [0, 1).
    """
    z = np.mod(frac_z, 1.0)
    idx = np.argsort(z)
    z_sorted = z[idx]
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    cut = int(np.argmax(gaps))
    start = (cut + 1) % len(z_sorted)
    return float(z_sorted[start])


def _periodic_delta_frac(z: float, z0: float) -> float:
    """Compute minimal periodic distance between two fractional coordinates.

    Args:
        z: Fractional coordinate.
        z0: Reference fractional coordinate.

    Returns:
        Minimal periodic distance in [0, 0.5].
    """
    dz = abs((z - z0) % 1.0)
    return min(dz, 1.0 - dz)


def _z_tag(val: float) -> str:
    """Encode a $z$ value (Å) to a short tag in 0.1 Å units.

    Args:
        val: Value in Å.

    Returns:
        Tag like "z34" for 3.4 Å.
    """
    return f"z{round(float(val) * 10.0)}"


def _slug(val: float) -> str:
    """Encode a length to a zero‑padded 0.1 Å slug.

    Args:
        val: Value in Å.

    Returns:
        Slug like "034" for 3.4 Å.
    """
    val_tenths = round(float(val) * 10)
    return f"{val_tenths:03d}"


def _generate_values(start: float, end: float, step: float) -> list[float]:
    """Generate a monotonic list of values with inclusive end.

    Args:
        start: Start value.
        end: End value.
        step: Step size.

    Returns:
        Sorted unique values including both endpoints.

    Raises:
        ValueError: If step is not positive.
    """
    if step <= 0:
        raise ValueError("Step must be positive.")
    eps = 1e-10 * max(1.0, abs(end))
    values = [float(start)]
    v = float(start)
    while v + step <= end + eps:
        v = v + step
        values.append(round(v, 10))
    if abs(values[-1] - end) > eps:
        values.append(float(end))
    return sorted(set(values))


def wrap01(u: float) -> float:
    """Wrap a value into [0, 1).

    Args:
        u: Input value.

    Returns:
        Wrapped value.
    """
    return u % 1.0


def parse_xyz_from_atom_line(line: str) -> tuple[float, float, float] | None:
    """Parse XYZ coordinates from a CIF atom line.

    Args:
        line: CIF atom line with x, y, z in columns 4‑6.

    Returns:
        (x, y, z) tuple if parsing succeeds, otherwise None.
    """
    parts = line.split()
    if len(parts) < 6:
        return None
    try:
        x = float(parts[3])
        y = float(parts[4])
        z = float(parts[5])
        return x, y, z
    except ValueError:
        return None


def pick_lower_left_pair_from_lines(
    atom_lines: Iterable[str],
) -> tuple[
    int, tuple[str, str], tuple[float, float, float], tuple[float, float]
]:
    """Pick the lower‑left atom pair from consecutive atom lines.

    Each pair is assumed to be consecutive lines representing a lower and
    upper atom. The "best" pair is chosen by the smallest wrapped (x, y).

    Args:
        atom_lines: Iterable of atom lines in consecutive pairs.

    Returns:
        Tuple of (pair_index, (lower_line, upper_line), (x, y, z), (xw, yw)).

    Raises:
        ValueError: If the line count is odd or coordinates cannot be parsed.
    """
    atom_lines = list(atom_lines)
    if len(atom_lines) % 2 != 0:
        raise ValueError(
            "Expected an even number of atom lines (consecutive pairs)."
        )

    best_key = None
    best_pair_idx = None
    best_lower_upper = None
    best_lower_xyz = None

    for i in range(0, len(atom_lines), 2):
        l1 = atom_lines[i]
        l2 = atom_lines[i + 1]

        xyz1 = parse_xyz_from_atom_line(l1)
        xyz2 = parse_xyz_from_atom_line(l2)
        if xyz1 is None or xyz2 is None:
            raise ValueError(f"Could not parse xyz from pair:\n{l1!r}\n{l2!r}")

        if xyz1[2] <= xyz2[2]:
            lower_line, upper_line = l1, l2
            x, y, z = xyz1
        else:
            lower_line, upper_line = l2, l1
            x, y, z = xyz2

        xw, yw = wrap01(x), wrap01(y)
        key = (xw, yw)

        if best_key is None or key < best_key:
            best_key = key
            best_pair_idx = i // 2
            best_lower_upper = (lower_line, upper_line)
            best_lower_xyz = (xw, yw, z)

    if (
        best_pair_idx is None
        or best_lower_upper is None
        or best_lower_xyz is None
        or best_key is None
    ):
        raise ValueError("No valid atom pairs found.")

    return best_pair_idx, best_lower_upper, best_lower_xyz, best_key


def get_mode_folders(cof_name: str, mode: str) -> list[str]:
    """Return output folders for the selected stacking mode(s).

    Args:
        cof_name: COF name used for folder naming.
        mode: Mode selector. Allowed values are `"incl"`, `"serr"`, or
            `"both"`.

    Returns:
        List of folder paths to process.

    Raises:
        ValueError: If `mode` is not one of "incl", "serr", or "both".
    """
    mode = mode.lower()
    if mode not in {"incl", "serr", "both"}:
        raise ValueError("mode must be 'incl', 'serr', or 'both'.")

    incl = f"{cof_name}/2_{cof_name}_matrix/incl"
    serr = f"{cof_name}/2_{cof_name}_matrix/serr"

    if mode == "incl":
        return [incl]
    if mode == "serr":
        return [serr]
    return [serr, incl]


def ab_half_diagonal_from_cif(input_file: str) -> tuple[float, float]:
    """Compute half the $a+b$ diagonal length and angle from a CIF.

    Args:
        input_file: Path to the CIF file.

    Returns:
        (length, angle_deg) tuple.
    """
    struct = Structure.from_file(input_file)
    a_vec, b_vec, _ = struct.lattice.matrix
    vec = 0.5 * (a_vec + b_vec)
    vec_xy = (vec[0], vec[1])
    length = float(np.linalg.norm(vec_xy))
    angle = float(math.degrees(math.atan2(vec_xy[1], vec_xy[0])))
    return length, angle


def default_shift_from_cif(
    input_file: str,
    topo: str,
    print_shift: bool = False,
) -> tuple[float, float]:
    """Compute the default slip length and angle for AB stacking.

    For both ``hcb`` and ``kgm``, the default AB shift length uses
    ``(2/sqrt(3)) * ||0.5*(a+b)||`` and the angle is fixed to ``90°``.

    Args:
        input_file: Path to the CIF file.
        topo: Topology string. Allowed values are `"sql"`, `"hcb"`, or `"kgm"`.
        print_shift: If `True`, print computed default shift values.
            Defaults to `False`.

    Returns:
        (length, angle_deg) tuple.

    Raises:
        ValueError: If `topo` is not "sql", "hcb", or "kgm".
    """
    if topo not in ("sql", "hcb", "kgm"):
        raise ValueError("topo must be 'sql', 'hcb', or 'kgm'")
    struct = Structure.from_file(input_file)
    a_vec, b_vec, _ = struct.lattice.matrix

    vec = 0.5 * (a_vec + b_vec)
    vec_xy = np.array([vec[0], vec[1]], dtype=float)
    sql_len = float(np.linalg.norm(vec_xy))

    if topo in {"hcb", "kgm"}:
        length = (2.0 / math.sqrt(3.0)) * sql_len
        angle = 90.0
        if print_shift:
            print(
                f"[DEFAULT_SHIFT_VALUES] Length={length:.2f}Å Angle={angle:.2f}"
            )
        return length, angle

    length = sql_len
    angle = float(math.degrees(math.atan2(vec_xy[1], vec_xy[0])))
    if print_shift:
        print(f"[DEFAULT_SHIFT_VALUES] Length={length:.2f}Å Angle={angle:.2f}")
    return length, angle
