"""Construct single-layer 2D COF structures from node and linker XYZ inputs.

This module prepares pormake-compatible building blocks, assembles a framework
for supported topologies, and writes an unoptimized CIF intended for downstream
energy and optimization workflows.
"""

import glob
import os
import shutil
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any, cast

import ase.io
import numpy as np
import pormake as pm
from ase.atoms import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from pormake.building_block import BuildingBlock
from pormake.framework import Framework
from pormake.topology import Topology
from pymatgen.core import Structure

from .ild_ils_matrix import ChangeIld
from .ild_ils_utils import _unwrap_fractional_z

DEFAULT_X_SCALE = 0.8
TOPOLOGY_INPUT_COUNTS = {
    "hcb": {"nodes": 1, "linkers": 1},
    "sql": {"nodes": 1, "linkers": 1},
    "hcb_ab": {"nodes": 2, "linkers": 0},
    "kgm": {"nodes": 1, "linkers": 1},
}


def _package_topology_dir() -> Path:
    """Return the package-local topology directory path.

    Returns:
        Path to the directory containing bundled topology files.
    """
    return Path(__file__).resolve().parent.parent / "database" / "topologies"


def _topology_cache_dir() -> Path:
    """Return the first writable topology cache directory path."""
    roots: list[Path] = []
    cache_root = os.environ.get("XDG_CACHE_HOME")
    if cache_root:
        roots.append(Path(cache_root))
    roots.append(Path.home() / ".cache")
    roots.append(Path(tempfile.gettempdir()))

    attempted: list[Path] = []
    for root in roots:
        candidate = root / "coflandscaper" / "topologies"
        attempted.append(candidate)
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            marker = candidate / ".write_check"
            marker.write_text("ok", encoding="utf-8")
            marker.unlink()
        except OSError:
            continue
        return candidate

    attempted_text = ", ".join(str(path) for path in attempted)
    raise PermissionError(
        "Unable to create a writable topology cache directory. Tried: "
        f"{attempted_text}"
    )


def _sync_topology_cache(cache_dir: Path, source_dir: Path) -> None:
    """Sync packaged topology files into a writable cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("*.cgd", "*.pickle")
    for pattern in patterns:
        for src in source_dir.glob(pattern):
            dst = cache_dir / src.name
            should_copy = True
            if dst.exists():
                try:
                    if (
                        src.stat().st_mtime <= dst.stat().st_mtime
                        and src.read_bytes() == dst.read_bytes()
                    ):
                        should_copy = False
                except OSError:
                    should_copy = True
            if should_copy:
                shutil.copy2(src, dst)


class PackageDatabase(pm.Database):
    """Initialize a pormake database that defaults to packaged topologies.

    Args:
        topo_dir: Optional topology directory override. Defaults to `None`
            (uses bundled package topologies).
        bb_dir: Optional building-block directory override. Defaults to `None`.

    Returns:
        None.
    """

    def __init__(
        self,
        topo_dir: str | os.PathLike[str] | None = None,
        bb_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        if topo_dir is None:
            cache_dir = _topology_cache_dir()
            _sync_topology_cache(cache_dir, _package_topology_dir())
            default_topo_dir = cache_dir
        else:
            default_topo_dir = topo_dir
        super().__init__(topo_dir=default_topo_dir, bb_dir=bb_dir)


class CofLandscaperBuilder(pm.Builder):
    """Build frameworks and apply a post-step linker plane alignment."""

    def build(
        self,
        topology: Topology,
        bbs: Sequence[BuildingBlock | None],
        permutations: Mapping[int, Sequence[int]] | None = None,
        **kwargs: Any,
    ) -> Framework:
        """Build a framework and align linker planes after assembly.

        Args:
            topology: pormake topology instance used for construction.
            bbs: Mapping of topology types to building blocks.
            permutations: Optional permutation data for connection matching.
                Defaults to `None`.
            **kwargs: Additional keyword arguments passed to `pm.Builder.build`.

        Returns:
            Built framework with post-aligned linker geometry.
        """
        framework = super().build(
            topology=topology,
            bbs=bbs,
            permutations=permutations,
            **kwargs,
        )
        self._post_align_linker_planes(framework)
        return framework

    @staticmethod
    def _plane_normal(points: np.ndarray) -> np.ndarray | None:
        """Estimate a best-fit plane normal from Cartesian point coordinates.

        Args:
            points: Array of shape `(n, 3)` containing 3D points.

        Returns:
            Unit normal vector, or `None` if a stable normal cannot be computed.
        """
        if points.shape[0] < 3:
            return None
        centered = points - points.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        n = vt[-1]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            return None
        return n / n_norm

    @staticmethod
    def _rotate_about_axis(
        positions: np.ndarray,
        axis_point: np.ndarray,
        axis_dir: np.ndarray,
        angle: float,
    ) -> np.ndarray:
        """Rotate positions around an axis using Rodrigues' rotation formula.

        Args:
            positions: Array of Cartesian positions to rotate.
            axis_point: A point on the rotation axis.
            axis_dir: Rotation axis direction vector.
            angle: Rotation angle in radians.

        Returns:
            Rotated Cartesian positions.
        """
        axis = axis_dir / np.linalg.norm(axis_dir)
        x, y, z = axis
        k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        ident = np.eye(3)
        rot = ident + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
        return axis_point + (positions - axis_point) @ rot.T

    @classmethod
    def _align_edge_to_plane(
        cls,
        edge_positions: np.ndarray,
        r1: np.ndarray,
        r2: np.ndarray,
        target_normal: np.ndarray | None,
    ) -> np.ndarray:
        """Rotate one edge building block so its plane matches a target normal.

        Args:
            edge_positions: Cartesian atom positions of the edge building block.
            r1: First point defining the alignment axis.
            r2: Second point defining the alignment axis.
            target_normal: Target plane normal for alignment.

        Returns:
            Aligned edge positions.
        """
        if target_normal is None:
            return edge_positions

        axis = r2 - r1
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            return edge_positions
        axis_u = axis / axis_norm

        edge_normal = cls._plane_normal(edge_positions)
        if edge_normal is None:
            return edge_positions

        def _proj(v: np.ndarray) -> np.ndarray:
            return v - np.dot(v, axis_u) * axis_u

        a = _proj(edge_normal)
        b = _proj(target_normal)
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm < 1e-8 or b_norm < 1e-8:
            return edge_positions

        a /= a_norm
        b /= b_norm
        cosang = np.clip(np.dot(a, b), -1.0, 1.0)
        angle = np.arccos(cosang)
        if angle < 1e-8:
            return edge_positions

        sign = np.sign(np.dot(axis_u, np.cross(a, b)))
        if sign == 0:
            sign = 1.0
        angle *= sign

        return cls._rotate_about_axis(edge_positions, r1, axis_u, angle)

    @staticmethod
    def _topology_plane_normal(cell: np.ndarray) -> np.ndarray | None:
        """Compute a unit normal from the topology a and b lattice vectors.

        Args:
            cell: Lattice matrix with row vectors `(a, b, c)`.

        Returns:
            Unit normal to the a-b plane, or `None` if degenerate.
        """
        a = cell[0]
        b = cell[1]
        n = np.cross(a, b)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            return None
        return n / n_norm

    @staticmethod
    def _find_matched_atom_indices(
        topology: Topology,
        located_bbs: Sequence[BuildingBlock | None],
        permutations: Sequence[np.ndarray | None],
        e: int,
    ) -> tuple[int, int]:
        """Find the two connection atom indices matched to a topology edge."""
        n1, n2 = topology.neighbor_list[e]
        i1 = n1.index
        i2 = n2.index

        bb1 = located_bbs[i1]
        bb2 = located_bbs[i2]
        if bb1 is None or bb2 is None:
            raise ValueError(f"Missing building block for topology edge {e}.")

        a1: int | None = None
        a2: int | None = None

        for o, n in enumerate(topology.neighbor_list[i1]):
            s = n.distance_vector + n1.distance_vector
            if np.linalg.norm(s) < 0.01:
                perm = permutations[i1]
                if perm is None:
                    raise ValueError(
                        f"Missing permutation for topology slot {i1}."
                    )
                a1 = int(bb1.connection_point_indices[perm][o])
                break

        for o, n in enumerate(topology.neighbor_list[i2]):
            s = n.distance_vector + n2.distance_vector
            if np.linalg.norm(s) < 0.01:
                perm = permutations[i2]
                if perm is None:
                    raise ValueError(
                        f"Missing permutation for topology slot {i2}."
                    )
                a2 = int(bb2.connection_point_indices[perm][o])
                break

        if a1 is None or a2 is None:
            raise ValueError(
                f"Could not find matched atom indices for edge {e}."
            )

        return a1, a2

    @staticmethod
    def _calc_image(topology, ni, nj, invc: np.ndarray) -> np.ndarray:
        """Calculate the periodic image vector between two topology neighbors.

        Args:
            topology: Topology object containing atomic positions.
            ni: First neighbor descriptor.
            nj: Second neighbor descriptor.
            invc: Inverse cell matrix.

        Returns:
            Fractional image shift vector from `ni` to `nj`.
        """
        i = ni.index
        j = nj.index

        d = nj.distance_vector - ni.distance_vector

        ri = topology.atoms.positions[i]
        rj = topology.atoms.positions[j]

        return (d - (rj - ri)) @ invc

    def _post_align_linker_planes(self, framework) -> None:
        """Apply post-build linker plane alignment directly on framework atoms.

        Args:
            framework: Built framework object with pormake metadata in `info`.
        """
        info = framework.info
        topology = info["topology"]
        located_bbs = info["located_bbs"]
        permutations = info["permutations"]

        topo_normal = self._topology_plane_normal(topology.atoms.cell)
        if topo_normal is None:
            return

        cell = topology.atoms.cell
        invc = np.linalg.inv(cell)

        for e in topology.edge_indices:
            edge_bb = located_bbs[e]
            if edge_bb is None:
                continue

            n1, n2 = topology.neighbor_list[e]
            i1 = n1.index
            i2 = n2.index

            bb1 = located_bbs[i1]
            bb2 = located_bbs[i2]

            a1, a2 = self._find_matched_atom_indices(
                topology, located_bbs, permutations, e
            )
            r1 = bb1.atoms.positions[a1]
            r2 = bb2.atoms.positions[a2]

            image = self._calc_image(topology, n1, n2, invc)
            d = r2 - r1 + image @ cell

            edge_bb.atoms.positions[:] = self._align_edge_to_plane(
                edge_bb.atoms.positions,
                r1,
                r1 + d,
                topo_normal,
            )

        bb_atoms_list = [v.atoms for v in located_bbs if v is not None]
        if not bb_atoms_list:
            return

        updated_atoms = bb_atoms_list[0].copy()
        for bb_atoms in bb_atoms_list[1:]:
            updated_atoms += bb_atoms
        updated_atoms.set_pbc(True)
        updated_atoms.set_cell(topology.atoms.cell)
        del updated_atoms[[a.symbol == "X" for a in updated_atoms]]

        if len(updated_atoms) != len(framework.atoms):
            raise ValueError(
                "Aligned atom count mismatch after removing connection atoms."
            )

        framework.atoms.positions[:] = updated_atoms.positions


def _disable_pormake_file_logging() -> None:
    """Disable pormake file logging and remove `runtime.log` when present."""
    try:
        from pormake import log as pmlog

        with suppress(Exception):
            pmlog.logger.removeHandler(pmlog.file_log_handler)
        with suppress(Exception):
            pmlog.file_log_handler.close()
    except Exception:
        return

    try:
        if os.path.exists("runtime.log"):
            os.remove("runtime.log")
    except Exception:
        pass


_disable_pormake_file_logging()


def _replace_atoms(
    atoms: Atoms, from_symbol: str, to_symbol: str
) -> tuple[Atoms, list[int]]:
    """Replace one element symbol and return updated atoms plus indices.

    Args:
        atoms: ASE atoms object to read.
        from_symbol: Element symbol to replace.
        to_symbol: Replacement element symbol.

    Returns:
        Tuple of updated atoms and replaced atom indices.
    """
    x_indices: list[int] = []
    updated = atoms.copy()

    for i, atom in enumerate(updated):
        if atom.symbol == from_symbol:
            atom.symbol = to_symbol
            x_indices.append(i)

    return updated, x_indices


def _infer_connectivity(
    atoms: Atoms,
    mult: float = 1.1,
) -> list[tuple[int, int]]:
    """Infer undirected atom connectivity using ASE covalent-radius cutoffs."""
    cutoffs = natural_cutoffs(atoms, mult=mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    bonds: set[tuple[int, int]] = set()
    for i in range(len(atoms)):
        indices, _offsets = nl.get_neighbors(i)
        for j in indices:
            a, b = sorted((i, int(j)))
            if a != b:
                bonds.add((a, b))
    return sorted(bonds)


def _write_xyz_with_bonds(
    atoms: Atoms,
    x_indices: list[int],
    bonds: list[tuple[int, int]],
    out_path: str,
) -> None:
    """Write a pormake-style XYZ file with bond records.

    Args:
        atoms: ASE atoms containing coordinates and symbols.
        x_indices: Atom indices to mark as connection sites.
        bonds: List of bonded atom index pairs.
        out_path: Destination XYZ file path.
    """
    x_index_set = set(x_indices)
    neighbor_map: dict[int, list[int]] = {i: [] for i in x_indices}
    for i, j in bonds:
        if i in neighbor_map:
            neighbor_map[i].append(j)
        if j in neighbor_map:
            neighbor_map[j].append(i)

    positions = atoms.get_positions()

    with open(out_path, "w") as xyz_file:
        xyz_file.write(f"{len(atoms)}\n")
        xyz_file.write("Molecule\n")

        for i, atom in enumerate(atoms):
            pos = positions[i].copy()
            if i in x_index_set:
                symbol = "X"
                neighbors = neighbor_map.get(i)
                if neighbors:
                    connected_idx = neighbors[0]
                    connected_pos = positions[connected_idx]
                    vector = pos - connected_pos
                    pos = connected_pos + DEFAULT_X_SCALE * vector
            else:
                symbol = atom.symbol

            xyz_file.write(f"{symbol} {pos[0]} {pos[1]} {pos[2]}\n")

        xyz_file.writelines(
            f"{atom1}    {atom2}    S\n" for atom1, atom2 in bonds
        )


def _prepare_xyz(input_folder: str, output_folder: str) -> list[str]:
    """Prepare all XYZ files in a folder for pormake input.

    Args:
        input_folder: Folder with raw XYZ files.
        output_folder: Folder to write prepared XYZ files.

    Returns:
        List of input XYZ file paths processed.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    xyz_files = sorted(glob.glob(os.path.join(input_folder, "*.xyz")))

    return _prepare_xyz_files(xyz_files, output_folder)


def _prepare_xyz_files(
    xyz_files: list[str],
    output_folder: str,
) -> list[str]:
    """Prepare explicit XYZ file paths into pormake-compatible XYZ files.

    Args:
        xyz_files: Absolute or relative paths to input .xyz files.
        output_folder: Folder to write prepared XYZ files.

    Returns:
        The list of processed XYZ input file paths.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for path in xyz_files:
        atoms = cast("Atoms", ase.io.read(path))
        updated_atoms, x_indices = _replace_atoms(atoms, "He", "H")
        bonds = _infer_connectivity(updated_atoms)

        base_filename = os.path.basename(os.path.splitext(path)[0] + ".xyz")
        out_path = os.path.join(output_folder, base_filename)
        _write_xyz_with_bonds(updated_atoms, x_indices, bonds, out_path)

    return xyz_files


def _copy_xyz_file_to_folder(input_file: str, output_folder: str) -> str:
    """Copy a single XYZ file into a destination folder.

    Args:
        input_file: Source `.xyz` file path.
        output_folder: Destination directory.

    Returns:
        Path to the copied file in the destination directory.

    Raises:
        FileNotFoundError: If `input_file` does not exist or is not `.xyz`.
    """
    path = Path(input_file)
    if (
        not path.exists()
        or not path.is_file()
        or path.suffix.lower() != ".xyz"
    ):
        raise FileNotFoundError(f"Input xyz file not found: {input_file}")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    target = Path(output_folder) / path.name
    target.write_text(path.read_text())
    return str(target)


def _normalize_edge_pair(pair: Sequence[int] | np.ndarray) -> tuple[int, int]:
    """Normalize one edge pair into a tuple of two integers.

    Args:
        pair: Edge pair as a sequence or numpy array.

    Returns:
        A tuple of two integers.

    Raises:
        ValueError: If the input cannot be coerced into a 2-element pair.
    """
    arr = np.array(pair, dtype=int).ravel()
    if arr.size != 2:
        raise ValueError(f"Unexpected edge type shape: {pair}")
    return (int(arr[0]), int(arr[1]))


def _normalize_edge_types(
    edge_types: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Normalize edge-type data into an `(n, 2)` integer array.

    Args:
        edge_types: Edge type data from a topology.

    Returns:
        A numpy array of shape (n, 2) with integer pairs.
    """
    try:
        arr = np.array(edge_types, dtype=int)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr
    except Exception:
        pass
    return np.array([_normalize_edge_pair(p) for p in edge_types], dtype=int)


def _sanitize_edge_types_inplace(
    edge_types: Sequence[Sequence[int]] | np.ndarray,
) -> Sequence[Sequence[int]] | np.ndarray:
    """Convert list-based edge entries to tuple form in place.

    Args:
        edge_types: Edge type data from a topology.

    Returns:
        The sanitized edge types.
    """
    if isinstance(edge_types, np.ndarray):
        if edge_types.dtype == object:
            for idx, val in np.ndenumerate(edge_types):
                if isinstance(val, list):
                    edge_types[idx] = tuple(val)
        return edge_types
    if isinstance(edge_types, list):
        for i, val in enumerate(edge_types):
            if isinstance(val, list):
                edge_types[i] = tuple(val)
            elif isinstance(val, np.ndarray):
                edge_types[i] = tuple(val.tolist())
        return edge_types
    return edge_types


def _center_structure_slab_z(struct: Structure) -> Structure:
    """Center a slab so its midpoint lies at fractional `z = 0.5`.

    Args:
        struct: Input periodic structure.

    Returns:
        Centered structure with symmetric vacuum around the slab.
    """
    lat = struct.lattice
    frac = np.array([site.frac_coords for site in struct.sites], dtype=float)
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

    return Structure(
        lattice=lat,
        species=struct.species,
        coords=frac_centered.tolist(),
        coords_are_cartesian=False,
    )


def _build_cof(
    topo: str,
    node_paths: Sequence[str],
    linker_paths: Sequence[str],
    output_folder: str,
    cof_name: str | None = None,
) -> str:
    """Build one COF structure and write the unoptimized CIF to disk.

    Args:
        topo: Topology name.
        node_paths: Paths to node XYZ files.
        linker_paths: Paths to linker XYZ files.
        output_folder: Output folder for CIF.
        cof_name: Optional COF name for output filename. Defaults to `None`
            (filename falls back to node/linker stems).

    Returns:
        Path to the written CIF file.
    """
    database = PackageDatabase()
    topology = database.get_topo(topo)
    _sanitize_edge_types_inplace(topology.edge_types)
    builder = CofLandscaperBuilder()
    if topo == "hcb_ab":
        node_a = pm.BuildingBlock(node_paths[0])
        node_b = pm.BuildingBlock(node_paths[1])
        bbs: list[BuildingBlock | None] = [None] * topology.n_slots
        bbs[0] = node_a
        bbs[1] = node_b
        cof = builder.build(topology=topology, bbs=bbs)
    else:
        edgetype_raw = _normalize_edge_types(topology.edge_types)
        filtered_edge = edgetype_raw[(edgetype_raw != -1).any(axis=1)]
        unique_pairs = set()
        edgetype_filtered = []
        for pair in filtered_edge:
            key = _normalize_edge_pair(pair)
            if key not in unique_pairs:
                unique_pairs.add(key)
                edgetype_filtered.append(key)

        node = pm.BuildingBlock(node_paths[0])
        linker = pm.BuildingBlock(linker_paths[0])
        node_types = np.array(topology.node_types, dtype=int).ravel()
        unique_node_types = sorted({int(t) for t in node_types if t >= 0})
        node_bbs = dict.fromkeys(unique_node_types, node)
        edge_bbs = dict.fromkeys(edgetype_filtered, linker)

        cof = builder.build_by_type(
            topology=topology,
            node_bbs=node_bbs,
            edge_bbs=edge_bbs,
        )

    if cof_name:
        filename = f"{cof_name}_unopt.cif"
    elif linker_paths:
        node_name = Path(node_paths[0]).stem
        linker_name = Path(linker_paths[0]).stem
        filename = f"{node_name}_{linker_name}.cif"
    else:
        node_name_a = Path(node_paths[0]).stem
        node_name_b = Path(node_paths[1]).stem
        filename = f"{node_name_a}_{node_name_b}.cif"

    output_filename = os.path.join(output_folder, filename)
    cof.write_cif(output_filename)
    if topo == "sql":
        centered = _center_structure_slab_z(
            Structure.from_file(output_filename)
        )
        centered.to(filename=output_filename)
    return output_filename


class BuildCOF2D:
    """Construct a single-layer 2D COF from node/linker inputs.

    This class wraps preprocessing needed for pormake-based assembly of 2D COFs.
    It supports topology-driven construction for ``hcb``, ``sql``, ``kgm``,
    and ``hcb_ab``. Dummy-atom preprocessing expects ``He`` connection markers.

    The main workflow resolves topology-dependent node/linker inputs,
    optionally preprocesses input XYZ files into pormake-compatible format,
    builds one framework, writes an unoptimized CIF into the single-layer
    output directory, and adjusts the interlayer distance to 15 Å.

    Topology requirements: ``hcb``, ``sql``, and ``kgm`` require one node and
    one linker. ``hcb_ab`` requires two nodes and no linker. Inputs default to
    ``0_node/`` and ``0_linker/`` unless explicit paths are provided.

    Default output location is ``{cof_name}/1_{cof_name}_single_layer``, and the
    default output CIF name is ``{cof_name}_unopt.cif``.
    """

    def _list_xyz(self, folder: str) -> list[tuple[str, str]]:
        """List XYZ files in a directory as `(stem, path)` pairs.

        Args:
            folder: Directory to scan for `.xyz` files.

        Returns:
            Sorted list of `(basename_without_extension, absolute_or_relative_path)`.
        """
        files = sorted(glob.glob(os.path.join(folder, "*.xyz")))
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in files]

    def build(
        self,
        topo: str,
        cof_name: str,
        input_nodes: Sequence[str | os.PathLike[str]] | None = None,
        input_linkers: Sequence[str | os.PathLike[str]] | None = None,
        output_folder: str | None = None,
    ) -> list[str]:
        """Build one unoptimized single-layer COF CIF from node/linker inputs.

        The method expects node/linker counts defined by the topology after
        input resolution. Input files are preprocessed to map dummy atoms and
        inject bond annotations required by pormake. If explicit input files are
        not provided, default source folders are used.

        Default input behavior:
        - Nodes are read from `0_node/*.xyz`.
        - Linkers are read from `0_linker/*.xyz` only when required by topology.
        - Pass `input_linkers=[]` to explicitly provide no linkers.

        Default output behavior:
            The output CIF is written to
            `{cof_name}/1_{cof_name}_single_layer/{cof_name}_unopt.cif` and
            adjusted to an interlayer distance of 15 Å.

        Args:
            topo: Topology key used for construction. Allowed values are
                `"hcb"`, `"sql"`, `"hcb_ab"`, and `"kgm"`.
            cof_name: COF identifier used in output folder and filename patterns.
            input_nodes: Optional explicit node `.xyz` paths. Defaults to `None`
                (reads from `0_node/*.xyz`).
            input_linkers: Optional explicit linker `.xyz` paths. Defaults to
                `None` (reads from `0_linker/*.xyz` when required). Use an empty
                list to explicitly pass no linkers.
            output_folder: Optional output folder override. Defaults to `None`
                (uses `{cof_name}/1_{cof_name}_single_layer`).

        Returns:
            List containing one output CIF path.

        Raises:
            ValueError: If `topo` is not one of `"hcb"`, `"sql"`,
                `"hcb_ab"`, or `"kgm"`.
            ValueError: If input resolution does not match topology input counts.
        """
        _disable_pormake_file_logging()
        if topo not in TOPOLOGY_INPUT_COUNTS:
            raise ValueError("topo must be 'hcb', 'sql', 'hcb_ab', or 'kgm'")

        required_nodes = TOPOLOGY_INPUT_COUNTS[topo]["nodes"]
        required_linkers = TOPOLOGY_INPUT_COUNTS[topo]["linkers"]

        with ExitStack() as stack:
            nodes_dir_used = stack.enter_context(tempfile.TemporaryDirectory())
            if input_nodes is not None:
                node_inputs = [os.fspath(path) for path in input_nodes]
                _prepare_xyz_files(node_inputs, nodes_dir_used)
                nodes = [
                    (
                        Path(path).stem,
                        str(Path(nodes_dir_used) / f"{Path(path).stem}.xyz"),
                    )
                    for path in node_inputs
                ]
            else:
                _prepare_xyz("0_node", nodes_dir_used)
                nodes = self._list_xyz(nodes_dir_used)

            linkers: list[tuple[str, str]] = []
            if input_linkers is not None:
                linker_dir_used = stack.enter_context(
                    tempfile.TemporaryDirectory()
                )
                linker_inputs = [os.fspath(path) for path in input_linkers]
                _prepare_xyz_files(linker_inputs, linker_dir_used)
                linkers = [
                    (
                        Path(path).stem,
                        str(Path(linker_dir_used) / f"{Path(path).stem}.xyz"),
                    )
                    for path in linker_inputs
                ]
            elif required_linkers > 0:
                linker_dir_used = stack.enter_context(
                    tempfile.TemporaryDirectory()
                )
                _prepare_xyz("0_linker", linker_dir_used)
                linkers = self._list_xyz(linker_dir_used)
            else:
                extra_linkers = sorted(Path("0_linker").glob("*.xyz"))
                if extra_linkers:
                    warnings.warn(
                        "Topology 'hcb_ab' ignores linker files in 0_linker/.",
                        stacklevel=2,
                    )

            if (
                len(nodes) != required_nodes
                or len(linkers) != required_linkers
            ):
                raise ValueError(
                    f"Topology '{topo}' requires exactly {required_nodes} node file(s) "
                    f"and {required_linkers} linker file(s). Found {len(nodes)} node "
                    f"file(s) and {len(linkers)} linker file(s). Either keep only "
                    "the required files in 0_node/ and 0_linker/, or pass explicit "
                    "paths via input_nodes=[...] and input_linkers=[...]."
                )

            node_paths = [path for _, path in nodes]
            linker_paths = [path for _, path in linkers]

            output_folder_used = output_folder or os.path.join(
                cof_name, f"1_{cof_name}_single_layer"
            )

            Path(output_folder_used).mkdir(parents=True, exist_ok=True)
            output = _build_cof(
                topo,
                node_paths,
                linker_paths,
                output_folder_used,
                cof_name=cof_name,
            )
            ChangeIld()._change_interlayer_distance(
                input_file=output,
                output_file=output,
                new_z=15.0,
            )

            return [output]
