#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import math
import numpy as np
from typing import List, Union
from pymatgen.core import Structure, Lattice

# =========================
# Core operations
# =========================
def _calculate_ild(lat):
    """
    Calculate interlayer distance (ILD = d_001) from lattice parameters.
    Returns the ILD in Ångströms.
    """
    a, b, c = lat.abc
    alpha_deg, beta_deg, gamma_deg = lat.angles
    alpha_r = np.radians(alpha_deg)
    beta_r = np.radians(beta_deg)
    gamma_r = np.radians(gamma_deg)
    
    # Calculate unit cell volume
    V = a * b * c * np.sqrt(
        1
        + 2 * np.cos(alpha_r) * np.cos(beta_r) * np.cos(gamma_r)
        - np.cos(alpha_r)**2
        - np.cos(beta_r)**2
        - np.cos(gamma_r)**2
    )
    
    # Calculate true interlayer spacing (ILD = d_001)
    ild = V / (a * b * np.sin(gamma_r))
    return ild

def check_interlayer_distance(input_file):
    """
    Read out the interlayer distance (ILD) and layer thickness from CIF.
    Prints the values to the terminal.
    """
    struct = Structure.from_file(input_file)
    ild = _calculate_ild(struct.lattice)
    
    # Calculate slab thickness from unwrapped fractional z
    fz = struct.frac_coords[:, 2]
    z0 = _unwrap_fractional_z(fz)
    fz_unwrapped = np.mod(fz - z0, 1.0)
    thickness = (np.max(fz_unwrapped) - np.min(fz_unwrapped)) * ild
    
    print(f"[CHECK_ILD] {os.path.basename(input_file)}: ILD = {ild:.6f} Å, Layer Thickness = {thickness:.6f} Å")

def make_supercell(input_file, output_file, scale_matrix):
    struct = Structure.from_file(input_file)
    supercell = struct * scale_matrix
    supercell.to(filename=output_file)
    print(f"[SUPERCELL] {os.path.basename(input_file)} -> {output_file}")

def interlayer_shift_serrated(input_file, output_file, shift_length, shift_angle_deg):
    """
    2-layer supercell in z, shift the top layer in-plane by (shift_length, shift_angle_deg).
    """
    struct = Structure.from_file(input_file)

    # 1) Make 2-layer supercell
    supercell = struct * (1, 1, 2)

    # 2) In-plane shift (Cartesian) -> fractional (handles γ≠90°)
    angle_rad = math.radians(shift_angle_deg)
    shift_cart = np.array([shift_length * math.cos(angle_rad),
                           shift_length * math.sin(angle_rad), 0.0])
    fx, fy, _ = supercell.lattice.get_fractional_coords(shift_cart)

    print(f"[SERRATED] Shift (Å): ({shift_cart[0]:.6f}, {shift_cart[1]:.6f}) | (frac): ({fx:.6f}, {fy:.6f})")

    # 3) Apply shift to top layer (z>0.5)
    mid_z = 0.5
    new_frac = []
    top_mask = []
    for site in supercell.sites:
        f = np.array(site.frac_coords, dtype=float)
        if f[2] > mid_z:
            f[0] += fx
            f[1] += fy
            top_mask.append(True)
        else:
            top_mask.append(False)
        new_frac.append(np.mod(f, 1.0))

    out = Structure(
        lattice=supercell.lattice,
        species=supercell.species,
        coords=new_frac,
        coords_are_cartesian=False,
    )
    out.to(filename=output_file)
    print(f"[SERRATED] {os.path.basename(input_file)} -> {output_file}")
    
def inclined_shift_rigid(input_file, output_file, shift_length, shift_angle_deg):
    # 1. Read the original structure
    struct = Structure.from_file(input_file)

    # 2. Extract original lattice vectors (as Cartesian vectors)
    a_vec, b_vec, c_vec = struct.lattice.matrix
    c_len = struct.lattice.c  # length of c-axis

    # 3. Define the desired in-plane shift per +1c translation
    angle_rad = math.radians(shift_angle_deg)
    x_shift = shift_length * math.cos(angle_rad)
    y_shift = shift_length * math.sin(angle_rad)

    # 4. Build the new tilted c vector (same length along z as before)
    new_c_vec = [x_shift, y_shift, c_len]
    new_lattice = Lattice([a_vec, b_vec, new_c_vec])

    # 5. Get original *Cartesian* coordinates (Angstroms!)
    cart_coords = struct.cart_coords

    # 6. Express those Cartesian coords as fractional coords in the new lattice
    new_frac = new_lattice.get_fractional_coords(cart_coords)

    # 7. Build new Structure with:
    #    - new lattice
    #    - SAME species
    #    - fractional coords = new_frac
    new_struct = Structure(
        lattice=new_lattice,
        species=struct.species,
        coords=new_frac,
        coords_are_cartesian=False,  # <-- FRACTIONAL coords here
    )

    new_struct.to(filename=output_file)
    print(
        f"[INCLINED-RIGID] (Δx,Δy)=({x_shift:.6f},{y_shift:.6f}) Å | "
        f"{os.path.basename(input_file)} -> {output_file}"
    )

def set_vacuum_on_top(input_file, output_file, vacuum_top):
    """
    Place a specified vacuum thickness above the slab, independent of slab thickness.
    Final c = slab_thickness + vacuum_top. Bottom aligned to ~0, vacuum on top.
    """
    struct = Structure.from_file(input_file)
    a_vec, b_vec, c_vec = struct.lattice.matrix
    c_len = np.linalg.norm(c_vec)
    c_hat = c_vec / c_len

    z_along_c = np.array([np.dot(site.coords, c_hat) for site in struct.sites])
    zmin = float(np.min(z_along_c))
    zmax = float(np.max(z_along_c))
    thickness = zmax - zmin

    new_c_len = thickness + float(vacuum_top)
    new_c_vec = c_hat * new_c_len
    new_lat = Lattice([a_vec, b_vec, new_c_vec])

    new_cart = [site.coords - zmin * c_hat for site in struct.sites]
    out = Structure(lattice=new_lat, species=struct.species, coords=new_cart, coords_are_cartesian=True)
    out.to(filename=output_file)

    print(f"[SET_VACUUM] slab={thickness:.6f} Å, vacuum={vacuum_top:.6f} Å, c={new_c_len:.6f} Å | {os.path.basename(input_file)} -> {output_file}")

def center_slab_z(input_file, output_file):
    """
    Center the slab along the cell's c-direction even if atoms straddle the PBC.
    Logic:
      1) Work in fractional z.
      2) Unwrap by cutting at the largest gap (so the slab becomes a single segment).
      3) Compute segment midpoint and shift to 0.5 in fractional z.
      4) Wrap back to [0,1).
    This is equivalent to a pure translation along +c and is valid for tilted cells.
    """
    struct = Structure.from_file(input_file)
    lat = struct.lattice

    # 1) fractional coords
    frac = np.array([s.frac_coords for s in struct.sites], dtype=float)
    fz = frac[:, 2]

    # 2) unwrap: find z0 so that (fz - z0) % 1 is one continuous segment
    z0 = _unwrap_fractional_z(fz)
    fz_unwrapped = np.mod(fz - z0, 1.0)

    # 3) midpoint of the slab segment (now entirely within [0,1))
    zmin = float(np.min(fz_unwrapped))
    zmax = float(np.max(fz_unwrapped))
    z_mid = 0.5 * (zmin + zmax)

    # shift so midpoint -> 0.5
    dz_frac = 0.5 - z_mid

    # 4) apply shift in fractional z and wrap
    fz_centered = np.mod(fz_unwrapped + dz_frac, 1.0)
    frac_centered = frac.copy()
    frac_centered[:, 2] = fz_centered

    out = Structure(lattice=lat, species=struct.species, coords=frac_centered, coords_are_cartesian=False)
    out.to(filename=output_file)

    # report useful diagnostics
    print(f"[CENTER_Z] unwrap offset z0={z0:.6f}, dz_frac={dz_frac:.6f} -> centered to z=0.5 (frac). "
          f"{os.path.basename(input_file)} -> {output_file}")

def remove_layer(input_file: str, output_file: str, z_value: float, tol: float, mode: str = "frac"):
    """
    Remove all atoms whose z-coordinate is within a tolerance of the target layer.
    - mode="frac": z_value and tol are fractional (0..1) along c (PBC-aware).
    - mode="cart": z_value and tol are in Å along c (projection on c-hat; no wrap).
    """
    struct = Structure.from_file(input_file)
    lat = struct.lattice
    c_vec = lat.matrix[2]
    c_len = float(np.linalg.norm(c_vec))
    c_hat = c_vec / c_len

    keep_species = []
    keep_frac = []

    removed = 0
    if mode == "frac":
        z0 = float(z_value) % 1.0
        tol_f = float(tol)
        for s in struct.sites:
            fz = float(s.frac_coords[2]) % 1.0
            dz = _periodic_delta_frac(fz, z0)
            if dz > tol_f:
                keep_species.append(s.species)
                keep_frac.append(s.frac_coords)
            else:
                removed += 1
        tag = f"frac z0={z0:.4f} tol={tol_f:.4f}"
    elif mode == "cart":
        z0a = float(z_value)  # Å
        tol_a = float(tol)    # Å
        for s in struct.sites:
            zcart = float(np.dot(s.coords, c_hat))  # projection along c in Å
            if abs(zcart - z0a) > tol_a:
                keep_species.append(s.species)
                keep_frac.append(s.frac_coords)
            else:
                removed += 1
        tag = f"cart z0={z0a:.3f}Å tol={tol_a:.3f}Å"
    else:
        raise ValueError("mode must be 'frac' or 'cart'")

    if len(keep_species) == 0:
        raise RuntimeError("All atoms would be removed; aborting to avoid empty structure.")

    out = Structure(lattice=lat, species=keep_species, coords=keep_frac, coords_are_cartesian=False)
    out.to(filename=output_file)

    print(f"[REMOVE_LAYER] Removed {removed} atom(s) using {tag}. {os.path.basename(input_file)} -> {output_file}")

def change_interlayer_distance(input_file, output_file, new_z):
    struct = Structure.from_file(input_file)
    lat_old = struct.lattice
    a_vec, b_vec, c_vec_old = lat_old.matrix
    
    # Calculate true interlayer spacing using helper function
    z_len_old = _calculate_ild(lat_old)

    # Calculate slab thickness from unwrapped fractional z, scaled to Cartesian z-extent
    fz = struct.frac_coords[:, 2]
    z0 = _unwrap_fractional_z(fz)
    fz_unwrapped = np.mod(fz - z0, 1.0)
    thickness = (np.max(fz_unwrapped) - np.min(fz_unwrapped)) * z_len_old

    if new_z < thickness:
        raise ValueError(f"New ILD {new_z:.4f} Å < slab thickness {thickness:.4f} Å; cannot fit.")

    # Calculate new c-vector: scale c_vec_old so ILD = new_z
    scale_factor = new_z / z_len_old
    new_c_vec = c_vec_old * scale_factor
    lat_new = Lattice([a_vec, b_vec, new_c_vec])

    frac_raw = lat_new.get_fractional_coords(struct.cart_coords)
    fz_new = frac_raw[:, 2]
    zmin_f = np.min(fz_new)
    zmax_f = np.max(fz_new)
    z_mid_f = (zmin_f + zmax_f) / 2
    delta_f = 0.5 - z_mid_f
    fz_centered = fz_new + delta_f

    fx = np.mod(frac_raw[:, 0], 1.0)
    fy = np.mod(frac_raw[:, 1], 1.0)
    frac_final = np.column_stack([fx, fy, fz_centered])

    new_struct = Structure(
        lattice=lat_new,
        species=struct.species,
        coords=frac_final,
        coords_are_cartesian=False,
    )
    new_struct.to(filename=output_file)

    print(
        f"[ILD] Old ILD = {z_len_old:.6f} Å -> New ILD = {new_z:.6f} Å, "
        f"Layer Thickness={thickness:.6f} Å | "
        f" -> {output_file}"
    )

def calculate_interlayer_slip(input_file):
    """
    Calculate the in-plane interlayer slip (ILS) from lattice parameters.
    ILS is the projection of the c-vector onto the ab-plane.
    """
    struct = Structure.from_file(input_file)
    lat = struct.lattice
    a, b, c = lat.abc
    alpha_deg, beta_deg, gamma_deg = lat.angles
    ar = math.radians(alpha_deg)
    br = math.radians(beta_deg)
    gr = math.radians(gamma_deg)
    
    # Components of c-vector in ab-plane
    cx = c * math.cos(br)
    cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / math.sin(gr)
    
    # In-plane slip magnitude
    slip = math.sqrt(cx**2 + cy**2)
    
    print(f"[CALC_ILS] {os.path.basename(input_file)}: ILS = {slip:.2f} Å")

def calculate_interlayer_slip_double_layer(input_file):
    """
    Calculate the in-plane interlayer slip of a serrated double layer.
    Reads CIF, identifies atom pairs, picks the lower-left pair,
    and calculates the slip vector between lower and upper atoms.
    """
    struct = Structure.from_file(input_file)
    
    # Read raw CIF to extract atom lines
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Find the atom-site loop block
    atom_lines = []
    in_atom_loop = False
    saw_atom_headers = False

    for i, line in enumerate(lines):
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
            # End of atom block on new header/loop or blank
            if not ls or ls.startswith("_") or ls.startswith("loop_"):
                break
            # Keep only data lines
            if ls and ls[0].isalpha():
                atom_lines.append(ls)

    if not atom_lines:
        raise ValueError("No atom lines found in CIF")

    # Pick lower-left pair
    pair_idx, (lower_line, upper_line), (xl, yl, zl), key = pick_lower_left_pair_from_lines(atom_lines)
    
    # Parse upper atom coords
    xu, yu, zu = parse_xyz_from_atom_line(upper_line)
    xu_w, yu_w = wrap01(xu), wrap01(yu)
    
   # Calculate slip vector (use positive fractional delta without minimum-image wrap)
    dxu = xu_w - xl
    if dxu < 0.0:
        dxu += 1.0
    dyu = yu_w - yl
    if dyu < 0.0:
        dyu += 1.0
    
    # Get lattice vectors and use XY components for in-plane slip
    a_vec, b_vec, _ = struct.lattice.matrix
    a_xy = np.array([a_vec[0], a_vec[1]])
    b_xy = np.array([b_vec[0], b_vec[1]])
    
    slip_vec_xy = dxu * a_xy + dyu * b_xy
    slip_mag = float(np.linalg.norm(slip_vec_xy))
    
    print(f"[CALC_ILS_DL] {os.path.basename(input_file)}: Pair {pair_idx}, ILS = {slip_mag:.2f} Å")

# ========================
# Batch settings (editable)
# ========================

MODE = "REMOVE_LAYER"      # CHECK_ILD / CHANGE_ILD / SUPERCELL / ILS_SERR / ILS_INCL / SET_VACUUM / CENTER_Z / REMOVE_LAYER / CALC_ILS_SL / CALC_ILS_DL
INPUT_FOLDER = "test"
OUTPUT_FOLDER = "test"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# CHANGE_ILD : start, end, step (Å)
NEW_Z_START = 3.0
NEW_Z_END   = 4.5
NEW_Z_STEP  = 0.1

# SUPERCELL
SUPERCELL_SIZE = (2, 2, 2)  # (na, nb, nc)

# SERRATED
SHIFT_LENGTH_START = 9.0   # Å
SHIFT_LENGTH_END   = 10.6   # Å
SHIFT_LENGTH_STEP  = 1.0   # Å
SHIFT_ANGLE        = 90.0  # deg (Cartesian; for γ=120°, b ≈120° from a) (for hcb = ~90°, for sql = ~45°)

# INCLINED
INCL_LENGTH_START = 12.0    # Å
INCL_LENGTH_END   = 15.8    # Å
INCL_LENGTH_STEP  = 1.0    # Å
INCL_ANGLE        = 45.0   # deg (for hcb = ~90°, for sql = ~45°)

# SET_VACUUM
VACUUM_TOP = 10.0        # Å

# REMOVE_LAYER settings
REMOVE_MODE = "frac"     # "frac" or "cart"
REMOVE_Z    = 0.75       # if frac: 0..1; if cart: Å along c from cell bottom
REMOVE_TOL  = 0.02       # if frac: fraction; if cart: Å

# ========================
# Helpers
# ========================
def _z_tag(val: Union[float, int]) -> str:
    """
    Encode a length in Å as zXY where XY is tenths of an Å.
    e.g., 4.0 Å -> z40, 6.5 Å -> z65, 10.0 Å -> z100
    """
    return f"z{int(round(float(val) * 10.0))}"

def _slug(val: Union[float, int]) -> str:
    """
    Create a numeric tag like 040 or 127 for 4.0 and 12.7 respectively.
    Always encodes tenths of an Å and pads to 3 digits minimum.
    """
    val_tenths = int(round(float(val) * 10))
    return f"{val_tenths:03d}"

def _generate_values(start: float, end: float, step: float) -> List[float]:
    """
    Robust float range including endpoint when exact.
    Uses a tiny epsilon to avoid missing the end due to FP error.
    """
    if step <= 0:
        raise ValueError("Step must be positive.")
    eps = 1e-10 * max(1.0, abs(end))
    values = [float(start)]
    v = float(start)
    # Step forward while still before end
    while v + step <= end + eps:
        v = v + step
        values.append(round(v, 10))
    # Ensure exact end is included
    if abs(values[-1] - end) > eps:
        values.append(float(end))
    # Deduplicate and sort
    values = sorted(set(values))
    return values

def _unwrap_fractional_z(frac_z: np.ndarray) -> float:
    """
    Find an offset z0 so that (frac_z - z0) % 1 becomes a single continuous segment
    (i.e., we cut at the largest gap on the unit circle).
    Returns z0 (in [0,1)).
    """
    z = np.mod(frac_z, 1.0)
    idx = np.argsort(z)
    z_sorted = z[idx]
    # gaps including wrap-around
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    cut = int(np.argmax(gaps))  # cut *after* z_sorted[cut]
    start = (cut + 1) % len(z_sorted)
    z0 = float(z_sorted[start])  # shift so this becomes ~0
    return z0

def _periodic_delta_frac(z: float, z0: float) -> float:
    """Smallest periodic distance on the unit circle between two fractional z in [0,1)."""
    dz = abs((z - z0) % 1.0)
    return min(dz, 1.0 - dz)

def wrap01(u: float) -> float:
    """Wrap a value into [0, 1)."""
    return u % 1.0

def parse_xyz_from_atom_line(line: str):
    """
    Parses a line like:
      H  H0  1  0.07219570  0.13953946  0.34603277  1
    Returns (x, y, z) as floats, or None if it doesn't parse.
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

def pick_lower_left_pair_from_lines(atom_lines):
    """
    atom_lines: list[str] of atom rows in CIF order.
    Assumes consecutive pairing: (0,1), (2,3), ...
    Selection key uses the LOWER-LAYER atom in each pair (smaller z), then picks
    the pair with smallest (x,y) (lexicographic) after wrapping x,y into [0,1).

    Returns:
        pair_index (0-based),
        (lower_line, upper_line)  # ordered by z
        (x, y, z) of the lower atom (wrapped x,y),
        key (x, y)
    """
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

        # Choose lower-layer atom by smaller z (robust even if file order ever changes)
        if xyz1[2] <= xyz2[2]:
            lower_line, upper_line = l1, l2
            x, y, z = xyz1
        else:
            lower_line, upper_line = l2, l1
            x, y, z = xyz2

        xw, yw = wrap01(x), wrap01(y)
        key = (xw, yw)  # lower-left: smallest x, then smallest y

        if best_key is None or key < best_key:
            best_key = key
            best_pair_idx = i // 2
            best_lower_upper = (lower_line, upper_line)
            best_lower_xyz = (xw, yw, z)

    return best_pair_idx, best_lower_upper, best_lower_xyz, best_key

# ========================
# Batch runner
# ========================

def main():
    cif_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, "*.cif")))
    if not cif_files:
        raise FileNotFoundError(f"No .cif files found in '{INPUT_FOLDER}'")

    print(f"[INFO] Found {len(cif_files)} CIF(s):")
    for f in cif_files:
        print("   •", os.path.basename(f))

    if MODE == "CHANGE_ILD":
        z_values = _generate_values(NEW_Z_START, NEW_Z_END, NEW_Z_STEP)
        print(f"[INFO] ILD sweep values: {z_values}")
        for inpath in cif_files:
            base = os.path.splitext(os.path.basename(inpath))[0]
            for cval in z_values:
                outname = f"{base}_{_z_tag(cval)}.cif"
                outpath = os.path.join(OUTPUT_FOLDER, outname)
                change_interlayer_distance(inpath, outpath, float(cval))

    elif MODE == "SUPERCELL":
        for inpath in cif_files:
            base = os.path.splitext(os.path.basename(inpath))[0]
            outname = f"{base}_supercell_{SUPERCELL_SIZE[0]}x{SUPERCELL_SIZE[1]}x{SUPERCELL_SIZE[2]}.cif"
            outpath = os.path.join(OUTPUT_FOLDER, outname)
            make_supercell(inpath, outpath, SUPERCELL_SIZE)
    elif MODE == "ILS_SERR":
        shift_lengths = _generate_values(SHIFT_LENGTH_START, SHIFT_LENGTH_END, SHIFT_LENGTH_STEP)
        print(f"[INFO] SERRATED sweep lengths: {shift_lengths}")
        for inpath in cif_files:
            base = os.path.splitext(os.path.basename(inpath))[0]
            for slen in shift_lengths:
                tag = f"L{_slug(slen)}"
                outname = f"{base}_ser_{tag}.cif"
                outpath = os.path.join(OUTPUT_FOLDER, outname)
                interlayer_shift_serrated(inpath, outpath, slen, SHIFT_ANGLE)

    elif MODE == "ILS_INCL":
        incl_lengths = _generate_values(INCL_LENGTH_START, INCL_LENGTH_END, INCL_LENGTH_STEP)
        print(f"[INFO] INCLINED sweep lengths: {incl_lengths}")
        for inpath in cif_files:
            base = os.path.splitext(os.path.basename(inpath))[0]
            for ilen in incl_lengths:
                outname = f"{base}_inc_L{_slug(ilen)}.cif"
                outpath = os.path.join(OUTPUT_FOLDER, outname)
                inclined_shift_rigid(inpath, outpath, ilen, INCL_ANGLE)

    elif MODE == "SET_VACUUM":
        for inpath in cif_files:
            base = os.path.splitext(os.path.basename(inpath))[0]
            outname = f"{base}_vacuum_{_slug(VACUUM_TOP)}.cif"
            outpath = os.path.join(OUTPUT_FOLDER, outname)
            set_vacuum_on_top(inpath, outpath, VACUUM_TOP)

    elif MODE == "CENTER_Z":
        for inpath in cif_files:
            base = os.path.splitext(os.path.basename(inpath))[0]
            outname = f"{base}_centered.cif"
            outpath = os.path.join(OUTPUT_FOLDER, outname)
            center_slab_z(inpath, outpath)

    elif MODE == "REMOVE_LAYER":
        for inpath in cif_files:
            base = os.path.splitext(os.path.basename(inpath))[0]
            # filename tag based on mode
            if REMOVE_MODE == "frac":
                tag = f"rm_z{int(round(REMOVE_Z*1000))}_t{int(round(REMOVE_TOL*1000))}"
            else:
                tag = f"rm_Z{_slug(REMOVE_Z)}_t{_slug(REMOVE_TOL)}"
            outname = f"{base}_{tag}.cif"
            outpath = os.path.join(OUTPUT_FOLDER, outname)
            remove_layer(inpath, outpath, REMOVE_Z, REMOVE_TOL, mode=REMOVE_MODE)

    elif MODE == "CHECK_ILD":
        for inpath in cif_files:
            check_interlayer_distance(inpath)

    elif MODE == "CALC_ILS_SL":
        for inpath in cif_files:
            calculate_interlayer_slip(inpath)

    elif MODE == "CALC_ILS_DL":
        for inpath in cif_files:
            calculate_interlayer_slip_double_layer(inpath)

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

if __name__ == "__main__":
    main()