"""ILD/ILS utilities and matrix generator."""

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
    ) -> None:
        """Scan interlayer distances and write updated CIFs.

        Args:
            input_folder: Folder containing input CIF files.
            output_folder: Destination folder for ILD‑modified CIFs.
            ild_start: Minimum ILD in Å.
            ild_end: Maximum ILD in Å.
            ild_step: Step size in Å.

        Raises:
            ValueError: If a requested ILD is smaller than the slab thickness.
        """
        os.makedirs(output_folder, exist_ok=True)
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
        """Write a CIF with a rescaled $z$ lattice vector.

        Args:
            input_file: Path to the source CIF.
            output_file: Output CIF path.
            new_z: Target interlayer distance in Å.

        Raises:
            ValueError: If the requested ILD cannot accommodate the slab.

        Returns:
            None.
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
    """Generate serrated ILS structures by shifting the top layer in a bilayer.

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
    ) -> None:
        """Generate serrated ILS variants for each input CIF.

        Args:
            input_folder: Folder containing ILD‑modified CIFs.
            output_folder: Destination folder for serrated structures.
            topo: Topology string ("hcb" or "sql") used for defaults.
            cof_name: Optional name used for output file naming.
            ils_length_step: Step size for slip length in Å.
            ils_length_start: Minimum slip length in Å.
            ils_length_end: Maximum slip length in Å. If None, auto‑computed.
            ils_angle: Slip direction angle in degrees. If None, auto‑computed.
            print_shift: If True, print the auto‑computed default shift values.

        Raises:
            ValueError: If `topo` is not "hcb" or "sql".
        """
        os.makedirs(output_folder, exist_ok=True)
        cif_files = list_cifs(input_folder)
        if ils_length_end is None or ils_angle is None:
            auto_len, auto_ang = default_shift_from_cif(
                cif_files[0], topo, print_shift=print_shift
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

        Returns:
            None.
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
    ) -> None:
        """Generate inclined ILS variants for each input CIF.

        Args:
            input_folder: Folder containing ILD‑modified CIFs.
            output_folder: Destination folder for inclined structures.
            topo: Topology string ("hcb" or "sql") used for defaults.
            cof_name: Optional name used for output file naming.
            ils_length_start: Minimum slip length in Å.
            ils_length_end: Maximum slip length in Å. If None, auto‑computed.
            ils_length_step: Step size for slip length in Å.
            ils_angle: Slip direction angle in degrees. If None, auto‑computed.
            print_shift: If True, print the auto‑computed default shift values.

        Raises:
            ValueError: If `topo` is not "hcb" or "sql".
        """
        os.makedirs(output_folder, exist_ok=True)
        cif_files = list_cifs(input_folder)
        if ils_length_end is None or ils_angle is None:
            auto_len, auto_ang = default_shift_from_cif(
                cif_files[0], topo, print_shift=print_shift
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

        Returns:
            None.
        """
        struct = Structure.from_file(input_file)

        a_vec, b_vec, c_vec = struct.lattice.matrix
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
    """Create an ILD×ILS matrix of stacking variants for a fixed COF layer.

    The layer itself is kept unchanged while (1) the interlayer distance (ILD)
    is varied by rescaling the $z$ lattice vector and (2) interlayer slipping
    (ILS) is applied either as a serrated bilayer shift or as an inclined
    lattice tilt. Both serrated and inclined modes converge to the AB stacking
    limit; the corresponding default shift length and angle are computed
    automatically and can be printed via `print_shift`.

    Users may override the slip angle, minimum/maximum slip length, and step
    size to scan a specific region or alternative slip pathway. Outputs are
    written to COF_NAME/2_{COF_NAME}_matrix/{serr|incl} by default and are
    intended for subsequent single‑point energy evaluations (e.g., MACE or DFT).
    """

    def __init__(
        self,
        ild_start: float = 3.0,
        ild_end: float = 4.5,
        ild_step: float = 0.1,
        ils_length_start: float = 0.0,
        ils_length_end: float | None = None,
        ils_length_step: float = 1.0,
        ils_angle: float | None = None,
        print_shift: bool = False,
    ) -> None:
        """Configure the ILD×ILS scan parameters.

        Args:
            ild_start: Minimum ILD in Å.
            ild_end: Maximum ILD in Å.
            ild_step: ILD step size in Å.
            ils_length_start: Minimum slip length in Å.
            ils_length_end: Maximum slip length in Å. If None, auto‑computed.
            ils_length_step: Slip length step size in Å.
            ils_angle: Slip direction angle in degrees. If None, auto‑computed.
            print_shift: If True, print the auto‑computed default shift values.
        """
        self.ild_start = ild_start
        self.ild_end = ild_end
        self.ild_step = ild_step
        self.ils_length_start = ils_length_start
        self.ils_length_end = ils_length_end
        self.ils_length_step = ils_length_step
        self.ils_angle = ils_angle
        self.print_shift = print_shift

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
            topo: Topology string ("hcb" or "sql") used for defaults.
            mode: "incl", "serr", or "both" to select ILS mode(s).
            input_cif: Optional path to a pre-optimized CIF file.
                Defaults to {cof_name}/1_{cof_name}_single_layer/{cof_name}_preopt.cif.
            output_base_folder: Optional base folder for outputs (relative to cof_name).
                Defaults to 2_{cof_name}_matrix, which yields
                {cof_name}/2_{cof_name}_matrix/{serr|incl}.

        Raises:
            ValueError: If `mode` is not one of "incl", "serr", or "both".
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
            not output_base_path.parts
            or output_base_path.parts[0] != cof_name
        ):
            output_base_path = Path(cof_name) / output_base_path

        with tempfile.TemporaryDirectory() as tmp_ild:
            tmp_input_dir = os.path.join(tmp_ild, "input")
            os.makedirs(tmp_input_dir, exist_ok=True)
            shutil.copy2(
                input_preopt,
                os.path.join(tmp_input_dir, os.path.basename(input_preopt)),
            )
            ChangeIld().run(
                input_folder=tmp_input_dir,
                output_folder=tmp_ild,
                ild_start=self.ild_start,
                ild_end=self.ild_end,
                ild_step=self.ild_step,
            )

            if mode in {"incl", "both"}:
                out_incl = str(output_base_path / "incl")
                IlsIncl().run(
                    input_folder=tmp_ild,
                    output_folder=out_incl,
                    topo=topo,
                    cof_name=cof_name,
                    ils_length_start=self.ils_length_start,
                    ils_length_end=self.ils_length_end,
                    ils_length_step=self.ils_length_step,
                    ils_angle=self.ils_angle,
                    print_shift=self.print_shift,
                )

            if mode in {"serr", "both"}:
                out_serr = str(output_base_path / "serr")
                IlsSerr().run(
                    input_folder=tmp_ild,
                    output_folder=out_serr,
                    topo=topo,
                    cof_name=cof_name,
                    ils_length_start=self.ils_length_start,
                    ils_length_end=self.ils_length_end,
                    ils_length_step=self.ils_length_step,
                    ils_angle=self.ils_angle,
                    print_shift=self.print_shift,
                )

