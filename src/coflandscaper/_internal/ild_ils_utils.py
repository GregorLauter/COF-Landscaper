"""Shared geometry and routing utilities for ILD/ILS workflows.

This module contains low-level helpers used throughout COF-Landscaper for
interlayer distance (ILD) and interlayer slipping (ILS) calculations, CIF-file
discovery, stacking-mode routing, periodic-coordinate handling, filename
encoding, and determination of topology-specific default AB-stacking shifts.
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
    """Return all CIF files in a folder in deterministic order.

    Args:
        input_folder: Folder containing CIF structures.

    Returns:
        Sorted list of CIF file paths.

    Raises:
        FileNotFoundError: If no CIF files are found in ``input_folder``.
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
    """Wrap a fractional coordinate into the interval ``[0, 1)``.

    Args:
        u: Fractional coordinate or other periodic scalar value.

    Returns:
        Wrapped value in the interval ``[0, 1)``.
    """
    return u % 1.0


def parse_xyz_from_atom_line(line: str) -> tuple[float, float, float] | None:
    """Parse fractional coordinates from a CIF atom-site line.

    The helper expects the x, y, and z coordinates in columns four through six of
    the whitespace-separated atom-site row.

    Args:
        line: CIF atom-site line.

    Returns:
        ``(x, y, z)`` tuple when parsing succeeds, otherwise ``None``.
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
    """Select a deterministic lower-layer reference atom pair from CIF atom lines.

    Atom-site lines are interpreted as consecutive lower/upper-layer pairs. For
    each pair, the atom with the smaller fractional z coordinate is treated as the
    lower-layer atom. The reference pair is then selected using the smallest wrapped
    fractional ``(x, y)`` coordinates.

    This helper is used for registry-based interlayer-slipping analysis of serrated
    structures.

    Args:
        atom_lines: CIF atom-site lines arranged as consecutive atom pairs.

    Returns:
        Tuple containing the selected pair index, the lower/upper atom lines, the
        lower-atom coordinates, and its wrapped ``(x, y)`` coordinates.

    Raises:
        ValueError: If the number of atom lines is odd, coordinates cannot be
            parsed, or no valid pair can be identified.
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
    """Return matrix folders for selected stacking mode(s).

    The helper resolves ``"serr"``, ``"incl"``, or ``"both"`` into the
    corresponding mode-specific folders below
    ``{cof_name}/2_{cof_name}_matrix``.

    Args:
        cof_name: COF name used for workflow folder naming.
        mode: Stacking mode selector: ``"incl"``, ``"serr"``, or ``"both"``.

    Returns:
        List of matrix-folder paths. ``mode="both"`` returns serrated first,
        followed by inclined.

    Raises:
        ValueError: If ``mode`` is unsupported.
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
    """Calculate half of the in-plane ``a + b`` lattice diagonal.

    Args:
        input_file: CIF structure path.

    Returns:
        Tuple ``(length, angle_deg)`` containing the in-plane half-diagonal length
        in Å and its angle in degrees.
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
    """Determine the default AB-stacking interlayer-slipping vector.

        The default interlayer slipping (ILS) magnitude and direction are derived from
        the in-plane lattice vectors of the supplied structure.

        For ``sql``, the shift corresponds to half of the ``a + b`` diagonal.

        For ``hcb`` and ``kgm``, the shift magnitude is calculated as
        ``(2 / sqrt(3)) * ||0.5 * (a + b)||`` and the direction is fixed at 90 degrees.

        The returned values are used as the default upper limit and direction of the
        ILS scan when explicit values are not supplied.

    Args:
        input_file: CIF structure path.
        topo: Topology selector: ``"sql"`, ``"hcb"`, or ``"kgm"``.
        print_shift: Whether to print the calculated default shift.
            Defaults to ``False``.

    Returns:
        Tuple ``(length, angle_deg)`` containing the default ILS magnitude in Å
        and direction in degrees.

    Raises:
        ValueError: If ``topo`` is unsupported.
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
