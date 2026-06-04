"""Generate ILD and ILS structure matrices from a preoptimized COF layer.

This module provides classes to scan interlayer distance (ILD), apply lateral
interlayer slipping (ILS) in serrated or inclined form, and combine both
dimensions into a matrix of output CIF structures for downstream screening.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from .ild_ils_utils import (
    _calculate_ild,
    _generate_values,
    _slug,
    _unwrap_fractional_z,
    _z_tag,
    default_shift_from_cif,
    list_cifs,
)


def _compact_unwrap_fractional_1d(fz_values: np.ndarray) -> tuple[np.ndarray, float]:
    """Compactly unwrap 1D fractional z coordinates using the largest gap.

    Returns the unwrapped coordinates in [0, 1) relative to the coordinate
    after the largest periodic gap, and that local start coordinate.
    """
    fz_mod = np.mod(np.asarray(fz_values, dtype=float), 1.0)
    if fz_mod.size == 0:
        raise ValueError("Cannot unwrap an empty coordinate set.")
    if fz_mod.size == 1:
        return np.zeros(1, dtype=float), float(fz_mod[0])

    fz_sorted = np.sort(fz_mod)
    gaps = np.diff(fz_sorted)
    wrap_gap = (fz_sorted[0] + 1.0) - fz_sorted[-1]
    all_gaps = np.concatenate([gaps, np.array([wrap_gap])])
    largest_gap_idx = int(np.argmax(all_gaps))
    start = float(fz_sorted[(largest_gap_idx + 1) % fz_sorted.size])
    z_rel_frac = np.mod(fz_mod - start, 1.0)
    return z_rel_frac, start


def _explicit_bilayer_cluster_masks(
    struct: Structure,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect bottom/top layer masks from periodic z clustering.

    Layers are split by the two largest periodic gaps in sorted fractional z,
    then ordered as bottom/top by compact layer centers in a global unwrapped
    frame.
    """
    fz = np.mod(np.asarray(struct.frac_coords[:, 2], dtype=float), 1.0)
    n_atoms = fz.size
    if n_atoms < 2:
        raise ValueError("Explicit bilayer requires at least two atoms.")

    order = np.argsort(fz)
    fz_sorted = fz[order]
    gaps = np.diff(fz_sorted)
    wrap_gap = (fz_sorted[0] + 1.0) - fz_sorted[-1]
    all_gaps = np.concatenate([gaps, np.array([wrap_gap])])

    largest_two = np.argsort(all_gaps)[-2:]
    cut_a = int((int(largest_two[0]) + 1) % n_atoms)
    cut_b = int((int(largest_two[1]) + 1) % n_atoms)
    if cut_a == cut_b:
        raise ValueError("Could not identify two distinct layer boundaries.")

    def _segment(start: int, end: int) -> list[int]:
        out: list[int] = []
        i = start
        while i != end:
            out.append(i)
            i = (i + 1) % n_atoms
        return out

    seg1_pos = _segment(cut_a, cut_b)
    seg2_pos = _segment(cut_b, cut_a)
    if not seg1_pos or not seg2_pos:
        raise ValueError("Degenerate explicit-bilayer clustering; empty layer detected.")

    mask1 = np.zeros(n_atoms, dtype=bool)
    mask2 = np.zeros(n_atoms, dtype=bool)
    mask1[order[np.array(seg1_pos, dtype=int)]] = True
    mask2[order[np.array(seg2_pos, dtype=int)]] = True

    _all_rel, global_start = _compact_unwrap_fractional_1d(fz)

    def _layer_center_rel(mask: np.ndarray) -> float:
        z_rel_frac, layer_start = _compact_unwrap_fractional_1d(fz[mask])
        z_mid_rel = 0.5 * (float(np.min(z_rel_frac)) + float(np.max(z_rel_frac)))
        center_abs = (layer_start + z_mid_rel) % 1.0
        return float((center_abs - global_start) % 1.0)

    c1 = _layer_center_rel(mask1)
    c2 = _layer_center_rel(mask2)

    if c1 <= c2:
        return mask1, mask2
    return mask2, mask1


def _compact_layer_offsets_from_fractional_z(
    fz_layer: np.ndarray,
    z_len_old: float,
) -> tuple[np.ndarray, float]:
    """Return compact cartesian offsets and thickness for one layer.

    The compact frame is defined by finding the largest periodic gap on the
    modulo-wrapped fractional z coordinates and taking the coordinate just
    after that gap as the local origin.
    """
    if np.asarray(fz_layer, dtype=float).size == 0:
        raise ValueError("Layer contains no atoms.")
    if np.asarray(fz_layer, dtype=float).size == 1:
        return np.zeros(1, dtype=float), 0.0

    z_rel_frac, _start = _compact_unwrap_fractional_1d(fz_layer)
    z_rel_cart = z_rel_frac * z_len_old
    z_min = float(np.min(z_rel_cart))
    z_max = float(np.max(z_rel_cart))
    thickness = z_max - z_min
    z_mid = 0.5 * (z_min + z_max)
    dz = z_rel_cart - z_mid
    return dz, thickness


class ChangeIld:
    """Generate ILD variations by rescaling the layer separation along $z$.

    This class scans the interlayer distance (ILD) by rescaling the lattice
    vector along $z$ while keeping the in‑plane lattice vectors and atomic
    positions consistent in fractional coordinates. The layer thickness is
    preserved and the slab is re‑centered in the new unit cell.
    """

    def run(
        self,
        input_folder: str,
        output_folder: str,
        ild_start: float = 3.0,
        ild_end: float = 4.5,
        ild_step: float = 0.1,
        force_invalid_ild: bool = False,
        explicit_bilayer: bool = False,
    ) -> None:
        """Scan interlayer distances and write updated CIFs.

        Args:
            input_folder: Folder containing input CIF files.
            output_folder: Destination folder for ILD‑modified CIFs.
            ild_start: Minimum ILD in Å. Defaults to `3.0`.
            ild_end: Maximum ILD in Å. Defaults to `4.5`.
            ild_step: Step size in Å. Defaults to `0.1`.
            force_invalid_ild: If `True`, continue even when the requested
                ILD is smaller than the slab thickness, emitting a warning.
                Defaults to `False`.
            explicit_bilayer: If `True`, treat input as an explicit bilayer
                and define layer identity from periodic z clustering.
                Requested ILD is the layer reference-plane spacing (not free
                vacuum gap), so target
                c-length is `2 * requested_ild`.
                Defaults to `False`.

        Raises:
            ValueError: If a requested ILD is smaller than the slab thickness
                and `force_invalid_ild` is `False`.
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        z_values = _generate_values(ild_start, ild_end, ild_step)

        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            for new_z in z_values:
                outname = f"{base}_{_z_tag(new_z)}.cif"
                outpath = os.path.join(output_folder, outname)
                self._change_interlayer_distance(
                    input_file,
                    outpath,
                    float(new_z),
                    force_invalid_ild=force_invalid_ild,
                    explicit_bilayer=explicit_bilayer,
                )

    def _change_interlayer_distance(
        self,
        input_file: str,
        output_file: str,
        new_z: float,
        force_invalid_ild: bool = False,
        explicit_bilayer: bool = False,
    ) -> None:
        """Write one CIF with a rescaled $z$ lattice vector.

        Args:
            input_file: Path to the source CIF.
            output_file: Output CIF path.
            new_z: Target interlayer distance in Å.
            force_invalid_ild: If `True`, continue even when the requested
                ILD is smaller than the slab thickness, emitting a warning.
                Defaults to `False`.
            explicit_bilayer: If `True`, treat `new_z` as the layer-to-layer
                repeat distance for an already explicit bilayer, so target
                c-length is `2 * new_z` with layers repacked around fractional
                z of 0.25 and 0.75. Layer identity is obtained from periodic
                z clustering. In this mode `new_z`
                represents reference-plane spacing, not free gap. Layer
                thickness/offsets are computed directly from fractional z
                within each detected layer.

        Raises:
            ValueError: If the requested ILD cannot accommodate the slab and
                `force_invalid_ild` is `False`.
        """
        struct = Structure.from_file(input_file)
        lat_old = struct.lattice
        a_vec, b_vec, c_vec_old = lat_old.matrix

        z_len_old = _calculate_ild(lat_old)

        fz = struct.frac_coords[:, 2]
        z0 = _unwrap_fractional_z(fz)
        fz_unwrapped = np.mod(fz - z0, 1.0)
        thickness = (np.max(fz_unwrapped) - np.min(fz_unwrapped)) * z_len_old

        if explicit_bilayer:
            target_z = 2.0 * new_z
        else:
            if new_z < thickness:
                message = (
                    f"Requested ILD {new_z:.4f} Å is smaller than slab thickness "
                    f"{thickness:.4f} Å; periodic layers may overlap."
                )
                if force_invalid_ild:
                    warnings.warn(message, UserWarning, stacklevel=2)
                else:
                    raise ValueError(
                        f"New ILD {new_z:.4f} Å < slab thickness {thickness:.4f} Å; cannot fit."
                    )
            target_z = new_z

        scale_factor = target_z / z_len_old
        new_c_vec = c_vec_old * scale_factor
        lat_new = Lattice([a_vec, b_vec, new_c_vec])

        if explicit_bilayer:
            if target_z <= 0.0:
                raise ValueError("Target ILD/c length must be positive.")

            frac_old = np.array(struct.frac_coords, dtype=float)
            bottom_mask, top_mask = _explicit_bilayer_cluster_masks(struct)

            dz_bottom, thickness_bottom = _compact_layer_offsets_from_fractional_z(
                frac_old[bottom_mask, 2], z_len_old
            )
            dz_top, thickness_top = _compact_layer_offsets_from_fractional_z(
                frac_old[top_mask, 2], z_len_old
            )
            required_ild = max(thickness_bottom, thickness_top)
            if new_z < required_ild:
                message = (
                    f"New ILD {new_z:.4f} Å < slab thickness {required_ild:.4f} Å; cannot fit."
                )
                if force_invalid_ild:
                    warnings.warn(message, UserWarning, stacklevel=2)
                else:
                    raise ValueError(message)

            z_final = np.empty(frac_old.shape[0], dtype=float)
            z_final[bottom_mask] = 0.25 * target_z + dz_bottom
            z_final[top_mask] = 0.75 * target_z + dz_top

            fz_final = z_final / target_z

            fx = np.mod(frac_old[:, 0], 1.0)
            fy = np.mod(frac_old[:, 1], 1.0)
            frac_final = np.column_stack([fx, fy, fz_final])

            new_struct = Structure(
                lattice=lat_new,
                species=struct.species,
                coords=frac_final.tolist(),
                coords_are_cartesian=False,
            )
            CifWriter(new_struct).write_file(output_file, mode="wt")
            return

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
            coords=frac_final.tolist(),
            coords_are_cartesian=False,
        )
        CifWriter(new_struct).write_file(output_file, mode="wt")


class IlsSerr:
    r"""Generate serrated ILS structures by shifting the top layer in a bilayer.

    A $2\times$ supercell is built along $z$ and the upper layer is shifted in
    the $ab$ plane. The shift length and angle can be scanned; the default
    shift corresponds to the AB stacking derived from the parent cell.
    """

    def run(
        self,
        input_folder: str,
        output_folder: str,
        topo: str,
        cof_name: str | None = None,
        ils_length_step: float = 1.0,
        ils_length_start: float = 0.0,
        ils_length_end: float | None = None,
        ils_angle: float | None = None,
        print_shift: bool = False,
        explicit_bilayer: bool = False,
    ) -> None:
        """Generate serrated ILS variants for each input CIF.

        Args:
            input_folder: Folder containing ILD‑modified CIFs.
            output_folder: Destination folder for serrated structures.
            topo: Topology string used for defaults. Allowed values are
                `"hcb"`, `"sql"`, `"hcb_ab"`, and `"kgm"`.
            cof_name: Optional name used for output file naming. Defaults to
                `None`.
            ils_length_step: Step size for slip length in Å. Defaults to `1.0`.
            ils_length_start: Minimum slip length in Å. Defaults to `0.0`.
            ils_length_end: Maximum slip length in Å. Defaults to `None`
                (auto-computed from AB shift).
            ils_angle: Slip direction angle in degrees. Defaults to `None`
                (auto-computed from AB shift).
            print_shift: If `True`, print auto-computed default shift values.
                Defaults to `False`.
            explicit_bilayer: If `True`, treat inputs as already-explicit
                bilayers and shift only the existing top layer without
                creating a z supercell. `ChangeIld` canonicalizes explicit
                bilayers to lower/upper layer centers near 0.25/0.75
                fractional z, so ILS selects the top layer with `z > 0.5`.
                Defaults to `False`.

        Raises:
            ValueError: If `topo` is not "hcb", "sql", "hcb_ab", or "kgm".
        """
        if topo not in {"hcb", "sql", "hcb_ab", "kgm"}:
            raise ValueError("topo must be 'hcb', 'sql', 'hcb_ab', or 'kgm'.")
        topo_used = "hcb" if topo == "hcb_ab" else topo
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        cif_files = list_cifs(input_folder)
        if ils_length_end is None or ils_angle is None:
            auto_len, auto_ang = default_shift_from_cif(
                cif_files[0], topo_used, print_shift=print_shift
            )
            if ils_length_end is None:
                ils_length_end = auto_len
            if ils_angle is None:
                ils_angle = auto_ang
        ils_lengths = _generate_values(
            ils_length_start, ils_length_end, ils_length_step
        )

        for input_file in cif_files:
            base = os.path.splitext(os.path.basename(input_file))[0]
            z_tag = None
            match = re.search(r"_z\d+", base)
            if match:
                z_tag = match.group(0).lstrip("_")
            for slen in ils_lengths:
                tag = f"L{_slug(slen)}"
                if cof_name and z_tag:
                    outname = f"{cof_name}_{z_tag}_{tag}_serr.cif"
                elif cof_name:
                    outname = f"{cof_name}_{tag}_serr.cif"
                else:
                    outname = f"{base}_ser_{tag}.cif"
                outpath = os.path.join(output_folder, outname)
                self._shift_serrated(
                    input_file,
                    outpath,
                    slen,
                    ils_angle,
                    explicit_bilayer=explicit_bilayer,
                )

    def _shift_serrated(
        self,
        input_file: str,
        output_file: str,
        ils_length: float,
        ils_angle_deg: float,
        explicit_bilayer: bool = False,
    ) -> None:
        """Write a serrated bilayer CIF with a shifted upper layer.

        Args:
            input_file: Path to the source CIF.
            output_file: Output CIF path.
            ils_length: Slip length in Å.
            ils_angle_deg: Slip direction angle in degrees.
            explicit_bilayer: If `True`, do not create a z supercell and
                shift only the top layer in the existing bilayer. Layer
                identity uses canonical split `fractional z > 0.5` after
                explicit-bilayer `ChangeIld` canonicalization.
        """
        struct = Structure.from_file(input_file)

        angle_rad = math.radians(ils_angle_deg)
        shift_cart = np.array(
            [
                ils_length * math.cos(angle_rad),
                ils_length * math.sin(angle_rad),
                0.0,
            ]
        )
        if explicit_bilayer:
            frac_z = np.mod(np.array(struct.frac_coords[:, 2], dtype=float), 1.0)
            top_mask = frac_z > 0.5
            fx, fy, _ = struct.lattice.get_fractional_coords(shift_cart)

            new_frac = []
            for idx, site in enumerate(struct.sites):
                f = np.array(site.frac_coords, dtype=float)
                if top_mask[idx]:
                    f[0] += fx
                    f[1] += fy
                f[0] = np.mod(f[0], 1.0)
                f[1] = np.mod(f[1], 1.0)
                new_frac.append(f)

            out = Structure(
                lattice=struct.lattice,
                species=struct.species,
                coords=new_frac,
                coords_are_cartesian=False,
            )
            if len(out) != len(struct):
                raise ValueError(
                    "Explicit bilayer serrated output atom count mismatch."
                )
        else:
            supercell = struct * (1, 1, 2)
            fx, fy, _ = supercell.lattice.get_fractional_coords(shift_cart)

            mid_z = 0.5
            new_frac = []
            for site in supercell.sites:
                f = np.array(site.frac_coords, dtype=float)
                if f[2] > mid_z:
                    f[0] += fx
                    f[1] += fy
                new_frac.append(np.mod(f, 1.0))

            out = Structure(
                lattice=supercell.lattice,
                species=supercell.species,
                coords=new_frac,
                coords_are_cartesian=False,
            )
        CifWriter(out).write_file(output_file, mode="wt")


class IlsIncl:
    """Generate inclined ILS structures by tilting the $c$ vector.

    The in‑plane shift is encoded in the $c$ lattice vector, producing a
    continuous lateral offset between layers along a fixed direction. The
    default shift length and angle correspond to the AB stacking derived from
    the parent cell.
    """

    def run(
        self,
        input_folder: str,
        output_folder: str,
        topo: str,
        cof_name: str | None = None,
        ils_length_start: float = 0.0,
        ils_length_end: float | None = None,
        ils_length_step: float = 1.0,
        ils_angle: float | None = None,
        print_shift: bool = False,
        explicit_bilayer: bool = False,
    ) -> None:
        """Generate inclined ILS variants for each input CIF.

        Args:
            input_folder: Folder containing ILD‑modified CIFs.
            output_folder: Destination folder for inclined structures.
            topo: Topology string used for defaults. Allowed values are
                `"hcb"`, `"sql"`, `"hcb_ab"`, and `"kgm"`.
            cof_name: Optional name used for output file naming. Defaults to
                `None`.
            ils_length_start: Minimum slip length in Å. Defaults to `0.0`.
            ils_length_end: Maximum slip length in Å. Defaults to `None`
                (auto-computed from AB shift).
            ils_length_step: Step size for slip length in Å. Defaults to `1.0`.
            ils_angle: Slip direction angle in degrees. Defaults to `None`
                (auto-computed from AB shift).
            print_shift: If `True`, print auto-computed default shift values.
                Defaults to `False`.
            explicit_bilayer: If `True`, treat inputs as already-explicit
                bilayers and shift only the existing top layer while applying
                the inclined c-vector tilt. `ChangeIld` canonicalizes explicit
                bilayers to lower/upper layer centers near 0.25/0.75
                fractional z, so ILS selects the top layer with `z > 0.5`.
                Defaults to `False`.

        Raises:
            ValueError: If `topo` is not "hcb", "sql", "hcb_ab", or "kgm".
        """
        if topo not in {"hcb", "sql", "hcb_ab", "kgm"}:
            raise ValueError("topo must be 'hcb', 'sql', 'hcb_ab', or 'kgm'.")
        topo_used = "hcb" if topo == "hcb_ab" else topo
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        cif_files = list_cifs(input_folder)
        if ils_length_end is None or ils_angle is None:
            auto_len, auto_ang = default_shift_from_cif(
                cif_files[0], topo_used, print_shift=print_shift
            )
            if ils_length_end is None:
                ils_length_end = auto_len
            if ils_angle is None:
                ils_angle = auto_ang
        incl_lengths = _generate_values(
            ils_length_start, ils_length_end, ils_length_step
        )

        for input_file in cif_files:
            base = os.path.splitext(os.path.basename(input_file))[0]
            z_tag = None
            match = re.search(r"_z\d+", base)
            if match:
                z_tag = match.group(0).lstrip("_")
            for ilen in incl_lengths:
                tag = f"L{_slug(ilen)}"
                if cof_name and z_tag:
                    outname = f"{cof_name}_{z_tag}_{tag}_incl.cif"
                elif cof_name:
                    outname = f"{cof_name}_{tag}_incl.cif"
                else:
                    outname = f"{base}_inc_{tag}.cif"
                outpath = os.path.join(output_folder, outname)
                self._inclined_shift(
                    input_file,
                    outpath,
                    ilen,
                    ils_angle,
                    explicit_bilayer=explicit_bilayer,
                )

    def _inclined_shift(
        self,
        input_file: str,
        output_file: str,
        ils_length: float,
        ils_angle_deg: float,
        explicit_bilayer: bool = False,
    ) -> None:
        """Write an inclined CIF by tilting the $c$ lattice vector.

        Args:
            input_file: Path to the source CIF.
            output_file: Output CIF path.
            ils_length: Slip length in Å.
            ils_angle_deg: Slip direction angle in degrees.
            explicit_bilayer: If `True`, shift only top-layer atoms in the
                existing bilayer before applying the inclined c-vector tilt.
                Layer identity uses canonical split `fractional z > 0.5`
                after explicit-bilayer `ChangeIld` canonicalization.
        """
        struct = Structure.from_file(input_file)

        a_vec, b_vec, _c_vec = struct.lattice.matrix
        c_len = struct.lattice.c

        angle_rad = math.radians(ils_angle_deg)
        x_shift = ils_length * math.cos(angle_rad)
        y_shift = ils_length * math.sin(angle_rad)

        new_c_vec = [x_shift, y_shift, c_len]
        new_lattice = Lattice([a_vec, b_vec, new_c_vec])

        cart_coords = np.array(struct.cart_coords, dtype=float)
        if explicit_bilayer:
            frac_z = np.mod(np.array(struct.frac_coords[:, 2], dtype=float), 1.0)
            top_mask = frac_z > 0.5
            cart_coords[top_mask, 0] += x_shift
            cart_coords[top_mask, 1] += y_shift
        new_frac = new_lattice.get_fractional_coords(cart_coords)

        new_struct = Structure(
            lattice=new_lattice,
            species=struct.species,
            coords=new_frac.tolist(),
            coords_are_cartesian=False,
        )
        if explicit_bilayer and len(new_struct) != len(struct):
            raise ValueError(
                "Explicit bilayer inclined output atom count mismatch."
            )
        CifWriter(new_struct).write_file(output_file, mode="wt")


class CreateMatrix:
    """Create an ILD×ILS matrix of stacking variants for a fixed COF layer.

    The layer itself is kept unchanged while (1) the interlayer distance (ILD)
    is varied by rescaling the $z$ lattice vector and (2) interlayer slipping
    (ILS) is applied either as a serrated bilayer shift or as an inclined
    lattice tilt. Both serrated and inclined modes converge to the AB stacking
    limit; the corresponding default shift length and angle are computed
    automatically when unset and can be printed via `print_shift`. The default
    ILD range is 3.0–4.0 Å in 0.1 Å steps.

    Users may override the slip angle, minimum/maximum slip length, and step
    size to scan a specific region or alternative slip pathway. Outputs are
    written to COF_NAME/2_{COF_NAME}_matrix/{serr|incl} by default and are
    intended for subsequent single‑point energy evaluations (e.g., MACE or DFT).
    """

    def __init__(
        self,
        ild_start: float = 3.0,
        ild_end: float = 4.0,
        ild_step: float = 0.1,
        ils_length_start: float = 0.0,
        ils_length_end: float | None = None,
        ils_length_step: float = 1.0,
        ils_angle: float | None = None,
        print_shift: bool = False,
        force_invalid_ild: bool = False,
        explicit_bilayer: bool = False,
    ) -> None:
        """Configure the ILD×ILS scan parameters.

        Args:
            ild_start: Minimum ILD in Å. Defaults to `3.0`.
            ild_end: Maximum ILD in Å. Defaults to `4.0`.
            ild_step: ILD step size in Å. Defaults to `0.1`.
            ils_length_start: Minimum slip length in Å. Defaults to `0.0`.
            ils_length_end: Maximum slip length in Å. Defaults to `None`
                (auto-computed from AB shift).
            ils_length_step: Slip length step size in Å. Defaults to `1.0`.
            ils_angle: Slip direction angle in degrees. Defaults to `None`
                (auto-computed from AB shift).
            print_shift: If `True`, print auto-computed default shift values.
                Defaults to `False`.
            force_invalid_ild: If `True`, continue even when the requested ILD
                is smaller than the slab thickness, emitting a warning.
                Defaults to `False`.
            explicit_bilayer: If `True`, treat inputs as already-explicit
                bilayers for ILS generation (no z supercell in serrated mode;
                top-layer-only shifts). Layer identity uses canonical split
                `fractional z > 0.5` after explicit-bilayer `ChangeIld`
                canonicalization.
                Defaults to `False`.
        """
        self._ild_start = ild_start
        self._ild_end = ild_end
        self._ild_step = ild_step
        self._ils_length_start = ils_length_start
        self._ils_length_end = ils_length_end
        self._ils_length_step = ils_length_step
        self._ils_angle = ils_angle
        self._print_shift = print_shift
        self._force_invalid_ild = force_invalid_ild
        self._explicit_bilayer = explicit_bilayer

    def run(
        self,
        cof_name: str,
        topo: str,
        mode: str,
        input_cif: str | None = None,
        output_base_folder: str | None = None,
    ) -> None:
        """Create the ILD×ILS matrix for a given COF.

        Args:
            cof_name: COF name used for input/output folder naming.
            topo: Topology string used for defaults. Allowed values are
                `"hcb"`, `"sql"`, `"hcb_ab"`, and `"kgm"`.
            mode: ILS mode selector. Allowed values are `"incl"`, `"serr"`,
                or `"both"`.
            input_cif: Optional path to a pre-optimized CIF file.
                Defaults to {cof_name}/1_{cof_name}_single_layer/{cof_name}_preopt.cif.
            output_base_folder: Optional base folder for outputs (relative to cof_name).
                Defaults to 2_{cof_name}_matrix, which yields
                {cof_name}/2_{cof_name}_matrix/{serr|incl}.

        Raises:
            ValueError: If `mode` is not one of "incl", "serr", or "both".
            FileNotFoundError: If the resolved input CIF does not exist.
        """
        mode = mode.lower()
        if mode not in {"incl", "serr", "both"}:
            raise ValueError("mode must be 'incl', 'serr', or 'both'.")

        input_preopt = input_cif or os.path.join(
            cof_name,
            f"1_{cof_name}_single_layer",
            f"{cof_name}_preopt.cif",
        )
        if not os.path.exists(input_preopt):
            raise FileNotFoundError(f"Missing input CIF: {input_preopt}")

        output_base_folder_used = output_base_folder or f"2_{cof_name}_matrix"
        output_base_path = Path(output_base_folder_used)
        if not output_base_path.is_absolute() and (
            not output_base_path.parts or output_base_path.parts[0] != cof_name
        ):
            output_base_path = Path(cof_name) / output_base_path

        with tempfile.TemporaryDirectory() as tmp_ild:
            tmp_input_dir = os.path.join(tmp_ild, "input")
            Path(tmp_input_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                input_preopt,
                os.path.join(tmp_input_dir, os.path.basename(input_preopt)),
            )
            ChangeIld().run(
                input_folder=tmp_input_dir,
                output_folder=tmp_ild,
                ild_start=self._ild_start,
                ild_end=self._ild_end,
                ild_step=self._ild_step,
                force_invalid_ild=self._force_invalid_ild,
                explicit_bilayer=self._explicit_bilayer,
            )

            if mode in {"incl", "both"}:
                out_incl = str(output_base_path / "incl")
                IlsIncl().run(
                    input_folder=tmp_ild,
                    output_folder=out_incl,
                    topo=topo,
                    cof_name=cof_name,
                    ils_length_start=self._ils_length_start,
                    ils_length_end=self._ils_length_end,
                    ils_length_step=self._ils_length_step,
                    ils_angle=self._ils_angle,
                    print_shift=self._print_shift,
                    explicit_bilayer=self._explicit_bilayer,
                )

            if mode in {"serr", "both"}:
                out_serr = str(output_base_path / "serr")
                IlsSerr().run(
                    input_folder=tmp_ild,
                    output_folder=out_serr,
                    topo=topo,
                    cof_name=cof_name,
                    ils_length_start=self._ils_length_start,
                    ils_length_end=self._ils_length_end,
                    ils_length_step=self._ils_length_step,
                    ils_angle=self._ils_angle,
                    print_shift=self._print_shift,
                    explicit_bilayer=self._explicit_bilayer,
                )
