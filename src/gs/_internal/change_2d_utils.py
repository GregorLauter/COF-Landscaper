import os
import math
import numpy as np
from typing import List, Union
from pymatgen.core import Structure, Lattice


def list_cifs(input_folder: str) -> list[str]:
    files = sorted(
        f for f in (os.path.join(input_folder, n) for n in os.listdir(input_folder))
        if f.endswith(".cif")
    )
    if not files:
        raise FileNotFoundError(f"No .cif files found in '{input_folder}'")
    return files


def _calculate_ild(lat):
    a, b, c = lat.abc
    alpha_deg, beta_deg, gamma_deg = lat.angles
    alpha_r = np.radians(alpha_deg)
    beta_r = np.radians(beta_deg)
    gamma_r = np.radians(gamma_deg)
    V = a * b * c * np.sqrt(
        1
        + 2 * np.cos(alpha_r) * np.cos(beta_r) * np.cos(gamma_r)
        - np.cos(alpha_r) ** 2
        - np.cos(beta_r) ** 2
        - np.cos(gamma_r) ** 2
    )
    ild = V / (a * b * np.sin(gamma_r))
    return ild


def _unwrap_fractional_z(frac_z: np.ndarray) -> float:
    z = np.mod(frac_z, 1.0)
    idx = np.argsort(z)
    z_sorted = z[idx]
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    cut = int(np.argmax(gaps))
    start = (cut + 1) % len(z_sorted)
    z0 = float(z_sorted[start])
    return z0


def _periodic_delta_frac(z: float, z0: float) -> float:
    dz = abs((z - z0) % 1.0)
    return min(dz, 1.0 - dz)


def _z_tag(val: Union[float, int]) -> str:
    return f"z{int(round(float(val) * 10.0))}"


def _slug(val: Union[float, int]) -> str:
    val_tenths = int(round(float(val) * 10))
    return f"{val_tenths:03d}"


def _generate_values(start: float, end: float, step: float) -> List[float]:
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
    values = sorted(set(values))
    return values


def wrap01(u: float) -> float:
    return u % 1.0


def parse_xyz_from_atom_line(line: str):
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


def pick_lower_left_pair_from_lines(atom_lines):
    if len(atom_lines) % 2 != 0:
        raise ValueError("Expected an even number of atom lines (consecutive pairs).")

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

    return best_pair_idx, best_lower_upper, best_lower_xyz, best_key
