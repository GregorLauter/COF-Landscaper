"""Helper transformations for layered COF structures."""

import os
from typing import Any, cast

import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from .ild_ils_utils import (
    _periodic_delta_frac,
    _slug,
    _unwrap_fractional_z,
    list_cifs,
)


class Supercell:
    """Build a supercell $a\times b\times c$ from each input unit cell.

    Physically, this replicates the periodic unit cell in-plane and along $c$
    to create a larger slab for visualization or downstream calculations.
    """

    def run(
        self,
        input_folder: str,
        output_folder: str,
        supercell_size: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            outname = f"{base}_supercell_{supercell_size[0]}x{supercell_size[1]}x{supercell_size[2]}.cif"
            outpath = os.path.join(output_folder, outname)
            struct = Structure.from_file(input_file)
            supercell = struct * supercell_size
            CifWriter(supercell).write_file(outpath, mode="wt")

class SetVacuum:
    """Place the layer at the bottom of the cell and add vacuum above it.

    This preserves the in-plane lattice vectors and extends $c$ so that a
    user-defined vacuum thickness is added on top of the layer.
    """

    def run(
        self, input_folder: str, output_folder: str, vacuum_top: float
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            outname = f"{base}_vacuum_{int(round(vacuum_top * 10)):03d}.cif"
            outpath = os.path.join(output_folder, outname)
            self._set_vacuum_on_top(input_file, outpath, vacuum_top)

    def _set_vacuum_on_top(
        self, input_file: str, output_file: str, vacuum_top: float
    ) -> None:
        struct = Structure.from_file(input_file)
        a_vec, b_vec, c_vec = struct.lattice.matrix
        c_len = np.linalg.norm(c_vec)
        c_hat = c_vec / c_len

        z_along_c = np.array(
            [np.dot(site.coords, c_hat) for site in struct.sites]
        )
        zmin = float(np.min(z_along_c))
        zmax = float(np.max(z_along_c))
        thickness = zmax - zmin

        new_c_len = thickness + float(vacuum_top)
        new_c_vec = c_hat * new_c_len
        new_lat = Lattice([a_vec, b_vec, new_c_vec])

        new_cart = [site.coords - zmin * c_hat for site in struct.sites]
        out = Structure(
            lattice=new_lat,
            species=struct.species,
            coords=new_cart,
            coords_are_cartesian=True,
        )
        out.to(filename=output_file)

class CenterZ:
    """Center the layer along $z$ so vacuum is symmetric above and below.

    This shifts fractional $z$ coordinates so the slab midpoint is at 0.5,
    giving equal vacuum thickness on both sides of the layer.
    """

    def run(self, input_folder: str, output_folder: str) -> None:
        os.makedirs(output_folder, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            outname = f"{base}_centered.cif"
            outpath = os.path.join(output_folder, outname)
            self._center_slab_z(input_file, outpath)

    def _center_slab_z(self, input_file: str, output_file: str) -> None:
        struct = Structure.from_file(input_file)
        lat = struct.lattice

        frac = np.array([s.frac_coords for s in struct.sites], dtype=float)
        fz = frac[:, 2]

        z0 = _unwrap_fractional_z(fz)
        fz_unwrapped = np.mod(fz - z0, 1.0)

        zmin = float(np.min(fz_unwrapped))
        zmax = float(np.max(fz_unwrapped))
        z_mid = 0.5 * (zmin + zmax)

        dz_frac = 0.5 - z_mid
        fz_centered = np.mod(fz_unwrapped + dz_frac, 1.0)
        frac_centered = frac.copy()
        frac_centered[:, 2] = fz_centered

        out = Structure(
            lattice=lat,
            species=struct.species,
            coords=frac_centered.tolist(),
            coords_are_cartesian=False,
        )
        out.to(filename=output_file)

class RemoveLayer:
    """Remove atoms near a target $z$ with a tolerance.

    In fractional mode, all atoms within a periodic tolerance of $z$ are
    removed. In Cartesian mode, atoms within a real-space band around $z$
    (projected along $c$) are removed.
    """

    def run(
        self,
        input_folder: str,
        output_folder: str,
        remove_z: float,
        remove_tol: float,
        mode: str = "frac",
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)
        for input_file in list_cifs(input_folder):
            base = os.path.splitext(os.path.basename(input_file))[0]
            if mode == "frac":
                tag = f"rm_z{int(round(remove_z * 1000))}_t{int(round(remove_tol * 1000))}"
            else:
                tag = f"rm_Z{_slug(remove_z)}_t{_slug(remove_tol)}"
            outname = f"{base}_{tag}.cif"
            outpath = os.path.join(output_folder, outname)
            self._remove_layer(
                input_file, outpath, remove_z, remove_tol, mode=mode
            )

    def _remove_layer(
        self,
        input_file: str,
        output_file: str,
        z_value: float,
        tol: float,
        mode: str = "frac",
    ) -> None:
        struct = Structure.from_file(input_file)
        lat = struct.lattice
        c_vec = lat.matrix[2]
        c_len = float(np.linalg.norm(c_vec))
        c_hat = c_vec / c_len

        keep_species: list[Any] = []
        keep_frac: list[np.ndarray] = []

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
        elif mode == "cart":
            z0a = float(z_value)
            tol_a = float(tol)
            for s in struct.sites:
                zcart = float(np.dot(s.coords, c_hat))
                if abs(zcart - z0a) > tol_a:
                    keep_species.append(s.species)
                    keep_frac.append(s.frac_coords)
                else:
                    removed += 1
        else:
            raise ValueError("mode must be 'frac' or 'cart'")

        if len(keep_species) == 0:
            raise RuntimeError(
                "All atoms would be removed; aborting to avoid empty structure."
            )

        out = Structure(
            lattice=lat,
            species=cast(list[Any], keep_species),
            coords=keep_frac,
            coords_are_cartesian=False,
        )
        out.to(filename=output_file)

