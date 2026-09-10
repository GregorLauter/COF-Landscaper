"""Generate ILD/ILS structure matrices from a preoptimized COF layer.

This module provides the structure-generation step used to construct the
reduced-dimensional stacking matrix of COF-Landscaper. Interlayer distance
(ILD) is scanned over a user-defined range, while interlayer slipping (ILS) is
introduced either as a serrated bilayer displacement or as an inclined lattice
tilt.

The generated CIF structures are used for subsequent single-point energy
evaluation and construction of the simplified stacking potential energy
landscape.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
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


class ChangeIld:
    """Generate structures with systematically varied interlayer distance.

    The interlayer distance (ILD) is varied by rescaling the lattice vector
    associated with the layer separation while preserving the in-plane lattice
    vectors. Atomic positions are transformed consistently and the layer is
    recentered in the modified unit cell.

    This class is used internally during construction of the ILD/ILS matrix.
    """

    def run(
        self,
        input_folder: str,
        output_folder: str,
        ild_start: float = 3.0,
        ild_end: float = 4.5,
        ild_step: float = 0.1,
    ) -> None:
        """Generate a scan over interlayer distance values.

        For every CIF structure in ``input_folder``, structures are generated from
        ``ild_start`` to ``ild_end`` using ``ild_step``.

        Args:
            input_folder: Folder containing input CIF structures.
            output_folder: Destination folder for ILD-modified CIF structures.
            ild_start: Minimum interlayer distance in Å. Defaults to ``3.0``.
            ild_end: Maximum interlayer distance in Å. Defaults to ``4.5``.
            ild_step: Interlayer-distance step size in Å. Defaults to ``0.1``.

        Raises:
            ValueError: If a requested interlayer distance is smaller than the layer
                thickness and therefore cannot contain the structure.
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        z_values = _generate_values(ild_start, ild_end, ild_step)

        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            for new_z in z_values:
                outname = f"{base}_{_z_tag(new_z)}.cif"
                outpath = os.path.join(output_folder, outname)
                self._change_interlayer_distance(
                    input_file, outpath, float(new_z)
                )

    def _change_interlayer_distance(
        self, input_file: str, output_file: str, new_z: float
    ) -> None:
        """Write one CIF with a rescaled $z$ lattice vector.

        Args:
            input_file: Path to the source CIF.
            output_file: Output CIF path.
            new_z: Target interlayer distance in Å.

        Raises:
            ValueError: If the requested ILD cannot accommodate the slab.
        """
        struct = Structure.from_file(input_file)
        lat_old = struct.lattice
        a_vec, b_vec, c_vec_old = lat_old.matrix

        z_len_old = _calculate_ild(lat_old)

        fz = struct.frac_coords[:, 2]
        z0 = _unwrap_fractional_z(fz)
        fz_unwrapped = np.mod(fz - z0, 1.0)
        thickness = (np.max(fz_unwrapped) - np.min(fz_unwrapped)) * z_len_old

        if new_z < thickness:
            raise ValueError(
                f"New ILD {new_z:.4f} Å < slab thickness {thickness:.4f} Å; cannot fit."
            )

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
            coords=frac_final.tolist(),
            coords_are_cartesian=False,
        )
        CifWriter(new_struct).write_file(output_file, mode="wt")


class IlsSerr:
    r"""Generate serrated interlayer-slipping structures.

    For serrated stacking, a $2\times$ supercell is generated along the stacking
    direction and one layer is displaced laterally relative to the other. This
    preserves a bilayer representation while scanning the interlayer slipping
    (ILS) magnitude along a fixed in-plane direction.

    If the maximum ILS value or slip direction is not supplied explicitly, the
    default values are derived automatically from the corresponding AB-stacking
    shift of the parent unit cell.
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
    ) -> None:
        """Generate serrated ILS variants for all input structures.

        The lateral displacement is scanned from ``ils_length_start`` to
        ``ils_length_end`` using ``ils_length_step``. The shift direction is defined by
        ``ils_angle``.

        When ``ils_length_end`` or ``ils_angle`` is omitted, the corresponding
        AB-stacking shift is derived automatically from the first input structure.

        Args:
            input_folder: Folder containing ILD-modified CIF structures.
            output_folder: Destination folder for serrated structures.
            topo: Topology selector used to determine the default AB shift. Allowed
                values are ``"hcb"``, ``"sql"``, ``"hcb_ab"``, and ``"kgm"``.
            cof_name: Optional COF name used for standardized output filenames.
                Defaults to ``None``.
            ils_length_step: Interlayer-slipping step size in Å. Defaults to ``1.0``.
            ils_length_start: Minimum interlayer slipping in Å. Defaults to ``0.0``.
            ils_length_end: Maximum interlayer slipping in Å. Defaults to ``None``,
                which uses the automatically determined AB-stacking shift.
            ils_angle: In-plane slip direction in degrees. Defaults to ``None``, which
                uses the automatically determined AB-stacking direction.
            print_shift: Whether to print automatically determined shift parameters.
                Defaults to ``False``.

        Raises:
            ValueError: If ``topo`` is unsupported.
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
                self._shift_serrated(input_file, outpath, slen, ils_angle)

    def _shift_serrated(
        self,
        input_file: str,
        output_file: str,
        ils_length: float,
        ils_angle_deg: float,
    ) -> None:
        """Write a serrated bilayer CIF with a shifted upper layer.

        Args:
            input_file: Path to the source CIF.
            output_file: Output CIF path.
            ils_length: Slip length in Å.
            ils_angle_deg: Slip direction angle in degrees.
        """
        struct = Structure.from_file(input_file)
        supercell = struct * (1, 1, 2)

        angle_rad = math.radians(ils_angle_deg)
        shift_cart = np.array(
            [
                ils_length * math.cos(angle_rad),
                ils_length * math.sin(angle_rad),
                0.0,
            ]
        )
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
    """Generate inclined interlayer-slipping structures.

    For inclined stacking, interlayer slipping (ILS) is encoded directly in the
    lattice geometry by adding in-plane components to the stacking lattice vector.
    This produces a continuous lateral offset between periodically repeated layers
    without explicitly constructing a bilayer supercell.

    If the maximum ILS value or slip direction is not supplied explicitly, the
    default values are derived automatically from the corresponding AB-stacking
    shift of the parent unit cell.
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
    ) -> None:
        """Generate inclined ILS variants for all input structures.

        The lateral displacement is scanned from ``ils_length_start`` to
        ``ils_length_end`` using ``ils_length_step``. The shift direction is defined by
        ``ils_angle``.

        When ``ils_length_end`` or ``ils_angle`` is omitted, the corresponding
        AB-stacking shift is derived automatically from the first input structure.

        Args:
            input_folder: Folder containing ILD-modified CIF structures.
            output_folder: Destination folder for inclined structures.
            topo: Topology selector used to determine the default AB shift. Allowed
                values are ``"hcb"``, ``"sql"``, ``"hcb_ab"``, and ``"kgm"``.
            cof_name: Optional COF name used for standardized output filenames.
                Defaults to ``None``.
            ils_length_start: Minimum interlayer slipping in Å. Defaults to ``0.0``.
            ils_length_end: Maximum interlayer slipping in Å. Defaults to ``None``,
                which uses the automatically determined AB-stacking shift.
            ils_length_step: Interlayer-slipping step size in Å. Defaults to ``1.0``.
            ils_angle: In-plane slip direction in degrees. Defaults to ``None``, which
                uses the automatically determined AB-stacking direction.
            print_shift: Whether to print automatically determined shift parameters.
                Defaults to ``False``.

        Raises:
            ValueError: If ``topo`` is unsupported.
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
                self._inclined_shift(input_file, outpath, ilen, ils_angle)

    def _inclined_shift(
        self,
        input_file: str,
        output_file: str,
        ils_length: float,
        ils_angle_deg: float,
    ) -> None:
        """Write an inclined CIF by tilting the $c$ lattice vector.

        Args:
            input_file: Path to the source CIF.
            output_file: Output CIF path.
            ils_length: Slip length in Å.
            ils_angle_deg: Slip direction angle in degrees.
        """
        struct = Structure.from_file(input_file)

        a_vec, b_vec, _c_vec = struct.lattice.matrix
        c_len = struct.lattice.c

        angle_rad = math.radians(ils_angle_deg)
        x_shift = ils_length * math.cos(angle_rad)
        y_shift = ils_length * math.sin(angle_rad)

        new_c_vec = [x_shift, y_shift, c_len]
        new_lattice = Lattice([a_vec, b_vec, new_c_vec])

        cart_coords = struct.cart_coords
        new_frac = new_lattice.get_fractional_coords(cart_coords)

        new_struct = Structure(
            lattice=new_lattice,
            species=struct.species,
            coords=new_frac.tolist(),
            coords_are_cartesian=False,
        )
        CifWriter(new_struct).write_file(output_file, mode="wt")


class CreateMatrix:
    """Generate the ILD/ILS stacking matrix used for energy-landscape screening.

    ``CreateMatrix`` combines a systematic interlayer-distance (ILD) scan with an
    interlayer-slipping (ILS) scan to generate the reduced-dimensional structure
    matrix used by COF-Landscaper.

    Two stacking representations are supported:

    - ``"serr"``: a serrated bilayer in which one layer is displaced laterally;
    - ``"incl"``: an inclined unit cell in which the lateral offset is encoded in
        the stacking lattice vector.

    ``mode="both"`` generates both representations.

    The default ILD scan spans 3.0–4.0 Å in 0.1 Å steps. The ILS scan begins at
    0 Å and, unless overridden, extends to the automatically determined AB-stacking
    shift for the selected topology.

    Generated structures are written to
    ``{cof_name}/2_{cof_name}_matrix/{serr|incl}`` and are intended for subsequent
    single-point energy evaluation.
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
    ) -> None:
        """Configure the ILD/ILS stacking scan.

        Args:
            ild_start: Minimum interlayer distance in Å. Defaults to ``3.0``.
            ild_end: Maximum interlayer distance in Å. Defaults to ``4.0``.
            ild_step: Interlayer-distance step size in Å. Defaults to ``0.1``.
            ils_length_start: Minimum interlayer slipping in Å. Defaults to ``0.0``.
            ils_length_end: Maximum interlayer slipping in Å. Defaults to ``None``,
                which uses the automatically determined AB-stacking shift.
            ils_length_step: Interlayer-slipping step size in Å. Defaults to ``1.0``.
            ils_angle: In-plane slip direction in degrees. Defaults to ``None``, which
                uses the automatically determined AB-stacking direction.
            print_shift: Whether to print automatically determined AB-shift parameters.
                Defaults to ``False``.
        """
        self._ild_start = ild_start
        self._ild_end = ild_end
        self._ild_step = ild_step
        self._ils_length_start = ils_length_start
        self._ils_length_end = ils_length_end
        self._ils_length_step = ils_length_step
        self._ils_angle = ils_angle
        self._print_shift = print_shift

    def run(
        self,
        cof_name: str,
        topo: str,
        mode: str,
        input_cif: str | None = None,
        output_base_folder: str | None = None,
    ) -> None:
        """Generate the ILD/ILS matrix for a COF.

        The method starts from the preoptimized single-layer structure, generates the
        requested interlayer-distance scan, and then applies serrated and/or inclined
        interlayer slipping according to ``mode``.

        By default, the input structure is read from
        ``{cof_name}/1_{cof_name}_single_layer/{cof_name}_preopt.cif`` and generated
        structures are written under
        ``{cof_name}/2_{cof_name}_matrix/{serr|incl}``.

        Args:
            cof_name: COF name used for default workflow folder and file naming.
            topo: Topology selector used to determine the default AB-stacking shift.
                Allowed values are ``"hcb"``, ``"sql"``, ``"hcb_ab"``, and ``"kgm"``.
            mode: Stacking-mode selector: ``"incl"``, ``"serr"``, or ``"both"``.
            input_cif: Optional preoptimized input CIF path. Defaults to
                ``{cof_name}/1_{cof_name}_single_layer/{cof_name}_preopt.cif``.
            output_base_folder: Optional matrix output-base override. Defaults to
                ``{cof_name}/2_{cof_name}_matrix``.

        Raises:
            ValueError: If ``mode`` is invalid.
            FileNotFoundError: If the resolved preoptimized input CIF is missing.
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
                )
