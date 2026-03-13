"""Build 2D COFs from prepared nodes/linkers using pormake.

This module prepares XYZ inputs for pormake by inserting dummy connection
atoms, then builds a single-layer COF from one node and one linker. The user
must specify the topology (currently "hcb" or "sql"), the bond type ("single"
or "double"), and a COF name for output organization.
"""

import glob
import os
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from io import StringIO
from pathlib import Path

import ase.io
import numpy as np
import pormake as pm
from ase.atoms import Atoms
from pymatgen.core import Structure
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D

DEFAULT_ILD_GUESS = 15.0
DEFAULT_X_SCALE = 0.8


def _package_topology_dir() -> Path:
    """Return the package-local topology directory."""
    return Path(__file__).resolve().parent.parent / "database" / "topologies"


class PackageDatabase(pm.Database):
    """pormake Database using coflandscaper-shipped topologies by default."""

    def __init__(
        self,
        topo_dir: str | os.PathLike[str] | None = None,
        bb_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        default_topo_dir = _package_topology_dir()
        super().__init__(topo_dir=topo_dir or default_topo_dir, bb_dir=bb_dir)


class CofLandscaperBuilder(pm.Builder):
    """Builder subclass with linker-plane alignment from builder_mod."""

    def build(self, topology, bbs, permutations=None, **kwargs):
        """Build with parent implementation, then align linker planes."""
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
        a = cell[0]
        b = cell[1]
        n = np.cross(a, b)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            return None
        return n / n_norm

    @staticmethod
    def _find_matched_atom_indices(topology, located_bbs, permutations, e):
        n1, n2 = topology.neighbor_list[e]
        i1 = n1.index
        i2 = n2.index

        bb1 = located_bbs[i1]
        bb2 = located_bbs[i2]

        for o, n in enumerate(topology.neighbor_list[i1]):
            s = n.distance_vector + n1.distance_vector
            s = np.linalg.norm(s)
            if s < 0.01:
                perm = permutations[i1]
                a1 = bb1.connection_point_indices[perm][o]
                break

        for o, n in enumerate(topology.neighbor_list[i2]):
            s = n.distance_vector + n2.distance_vector
            s = np.linalg.norm(s)
            if s < 0.01:
                perm = permutations[i2]
                a2 = bb2.connection_point_indices[perm][o]
                break

        return a1, a2

    def _post_align_linker_planes(self, framework) -> None:
        info = framework.info
        topology = info["topology"]
        located_bbs = info["located_bbs"]
        permutations = info["permutations"]

        topo_normal = self._topology_plane_normal(topology.atoms.cell)
        if topo_normal is None:
            return

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

            edge_bb.atoms.positions[:] = self._align_edge_to_plane(
                edge_bb.atoms.positions,
                r1,
                r2,
                topo_normal,
            )

        bb_atoms_list = [v.atoms for v in located_bbs if v is not None]
        if not bb_atoms_list:
            return

        updated_atoms = sum(bb_atoms_list[1:], bb_atoms_list[0])
        updated_atoms.set_pbc(True)
        updated_atoms.set_cell(topology.atoms.cell)
        del updated_atoms[[a.symbol == "X" for a in updated_atoms]]

        if len(updated_atoms) != len(framework.atoms):
            raise ValueError(
                "Aligned atom count mismatch after removing connection atoms."
            )

        framework.atoms.positions[:] = updated_atoms.positions


def _disable_pormake_file_logging() -> None:
    """Disable pormake file logging and remove runtime.log if present."""
    try:
        from pormake import log as pmlog

        try:
            pmlog.logger.removeHandler(pmlog.file_log_handler)
        except Exception:
            pass
        try:
            pmlog.file_log_handler.close()
        except Exception:
            pass
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
) -> tuple[str, list[int]]:
    """Replace atom symbols and return XYZ block and replaced indices.

    Args:
        atoms: ASE Atoms object.
        from_symbol: Element symbol to replace.
        to_symbol: Replacement element symbol.

    Returns:
        XYZ block string and list of indices that were replaced.
    """
    x_indices = []
    xyz_file = StringIO()

    for i, atom in enumerate(atoms):
        if atom.symbol == from_symbol:
            atom.symbol = to_symbol
            x_indices.append(i)

    ase.io.write(xyz_file, atoms, format="xyz")
    xyz_block = xyz_file.getvalue()
    xyz_file.close()
    return xyz_block, x_indices


def _build_rdkit_mol(xyz_block: str) -> Mol | None:
    """Build an RDKit Mol from an XYZ block.

    Args:
        xyz_block: XYZ block string.

    Returns:
        RDKit Mol or None if parsing fails.
    """
    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        return None
    rdDetermineBonds.DetermineBonds(mol)
    return mol


def _tag_isotopes(mol: Mol, indices: list[int]) -> None:
    """Tag atoms in the Mol by isotope for downstream processing.

    Args:
        mol: RDKit Mol.
        indices: Atom indices to tag.
    """
    for atom in mol.GetAtoms():
        if atom.GetIdx() in indices:
            atom.SetIsotope(2)


def _write_xyz_with_bonds(mol: Mol, out_path: str) -> None:
    """Write an XYZ file with bond annotations and scaled X positions.

    Args:
        mol: RDKit Mol.
        out_path: Output file path.
    """
    with open(out_path, "w") as xyz_file:
        num_atoms = mol.GetNumAtoms()
        xyz_file.write(f"{num_atoms}\n")
        xyz_file.write("Molecule\n")

        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            if atom.GetIsotope() == 2:
                symbol = "X"
                neighbors = [a.GetIdx() for a in atom.GetNeighbors()]
                if neighbors:
                    connected_idx = neighbors[0]
                    connected_pos = conf.GetAtomPosition(connected_idx)
                    vector = Point3D(
                        pos.x - connected_pos.x,
                        pos.y - connected_pos.y,
                        pos.z - connected_pos.z,
                    )
                    vector.x *= DEFAULT_X_SCALE
                    vector.y *= DEFAULT_X_SCALE
                    vector.z *= DEFAULT_X_SCALE
                    pos = Point3D(
                        connected_pos.x + vector.x,
                        connected_pos.y + vector.y,
                        connected_pos.z + vector.z,
                    )
            else:
                symbol = atom.GetSymbol()

            xyz_file.write(f"{symbol} {pos.x} {pos.y} {pos.z}\n")

        for bond in mol.GetBonds():
            atom1, atom2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()

            if bond_type == 1.0:
                bond_type_str = "S"
            elif bond_type == 1.5 and bond.IsInRing():
                bond_type_str = "A"
            elif (
                bond_type == 1.5 and not bond.IsInRing()
            ) or bond_type == 2.0:
                bond_type_str = "D"
            elif bond_type == 3.0:
                bond_type_str = "T"
            else:
                bond_type_str = "S"

            xyz_file.write(f"{atom1}    {atom2}    {bond_type_str}\n")


def _prepare_xyz(
    input_folder: str,
    output_folder: str,
    bond_type: str,
) -> list[str]:
    """Prepare XYZ files by inserting bond markers for pormake.

    Args:
        input_folder: Folder with raw XYZ files.
        output_folder: Folder to write prepared XYZ files.
        bond_type: Bond type, "single" or "double".

    Returns:
        List of input XYZ file paths processed.

    Raises:
        ValueError: If bond_type is not supported.
    """
    mode_map = {
        "double": "Se",
        "single": "He",
        "Se": "Se",
        "He": "He",
    }
    if bond_type not in mode_map:
        raise ValueError("bond_type must be 'double' or 'single'")
    mode = mode_map[bond_type]

    os.makedirs(output_folder, exist_ok=True)
    xyz_files = sorted(glob.glob(os.path.join(input_folder, "*.xyz")))

    for path in xyz_files:
        atoms = ase.io.read(path)
        if mode == "Se":
            xyz_block, x_indices = _replace_atoms(atoms, "Se", "O")
        else:
            xyz_block, x_indices = _replace_atoms(atoms, "He", "H")

        mol = _build_rdkit_mol(xyz_block)
        if mol is None:
            continue

        _tag_isotopes(mol, x_indices)

        base_filename = os.path.basename(os.path.splitext(path)[0] + ".xyz")
        out_path = os.path.join(output_folder, base_filename)
        _write_xyz_with_bonds(mol, out_path)

    return xyz_files


def _normalize_edge_pair(pair: Sequence[int] | np.ndarray) -> tuple[int, int]:
    """Normalize an edge pair into a 2-tuple of ints.

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
    """Normalize edge types into a 2D numpy array of int pairs.

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
    """Sanitize edge types in-place by converting lists to tuples.

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


def _build_cof(
    topo: str,
    node_name: str,
    node_path: str,
    linker_name: str,
    linker_path: str,
    output_folder: str,
    cof_name: str | None = None,
) -> str:
    """Build one COF and write a CIF to disk.

    Args:
        topo: Topology name.
        node_name: Node name.
        node_path: Path to node XYZ.
        linker_name: Linker name.
        linker_path: Path to linker XYZ.
        output_folder: Output folder for CIF.
        cof_name: Optional COF name for output filename.

    Returns:
        Path to the written CIF file.
    """
    database = PackageDatabase()
    topo_initial = database.get_topo(f"{topo}_initial")
    _sanitize_edge_types_inplace(topo_initial.edge_types)
    builder = CofLandscaperBuilder()
    edgetype_raw = _normalize_edge_types(topo_initial.edge_types)
    filtered_edge = edgetype_raw[(edgetype_raw != -1).any(axis=1)]
    unique_pairs = set()
    edgetype_filtered = []
    for pair in filtered_edge:
        key = _normalize_edge_pair(pair)
        if key not in unique_pairs:
            unique_pairs.add(key)
            edgetype_filtered.append(key)

    node = pm.BuildingBlock(node_path)
    linker = pm.BuildingBlock(linker_path)
    node_bbs = {0: node}
    edge_bbs = dict.fromkeys(edgetype_filtered, linker)

    cof = builder.build_by_type(
        topology=topo_initial,
        node_bbs=node_bbs,
        edge_bbs=edge_bbs,
    )

    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
        tmp_path = tmp.name
    cof.write_cif(tmp_path)
    structure = Structure.from_file(tmp_path)
    a = structure.lattice.a
    os.remove(tmp_path)

    alpha = 1.73205 if topo == "hcb" else 1.0
    gamma = (alpha * DEFAULT_ILD_GUESS) / a

    base = _package_topology_dir()
    pickle_path = base / f"{topo}_modified.pickle"
    cgd_path = base / f"{topo}_modified.cgd"
    if pickle_path.exists():
        pickle_path.unlink()

    with cgd_path.open() as file:
        lines = file.readlines()
    line_parts = lines[3].split()
    line_parts[3] = f"{float(gamma):.5f}"
    lines[3] = "  ".join(line_parts) + "\n"
    with cgd_path.open("w") as file:
        file.writelines(lines)

    topo_modified = database.get_topo(f"{topo}_modified")
    _sanitize_edge_types_inplace(topo_modified.edge_types)
    cof = builder.build_by_type(
        topology=topo_modified,
        node_bbs=node_bbs,
        edge_bbs=edge_bbs,
    )

    if cof_name:
        filename = f"{cof_name}_unopt.cif"
    else:
        filename = f"{node_name}_{linker_name}.cif"

    output_filename = os.path.join(output_folder, filename)
    cof.write_cif(output_filename)
    return output_filename


class BuildCOF2D:
    """Build a 2D COF from user-prepared nodes and linkers.

    The user provides a topology (currently "hcb" or "sql"), a bond type
    ("single" or "double"), and a COF name. Nodes and linkers must contain a
    dummy connection atom. When the bond type is "single", helium (He) is used
    as a placeholder for the dummy atom; when "double", selenium (Se) is used.
    If your COF contains selenium as a real atom, this workflow will not work
    reliably—please report it to the project maintainers.

    The builder first prepares the special XYZ files required by pormake, then
    constructs the COF. The resulting structure is a single layer with a large
    vacuum (~15 Å) and is intended for later post-processing. Each COF is saved
    under a folder named by the user-specified COF name.

    """

    def __init__(self) -> None:
        """Initialize the builder."""

    def _topology_paths(self, topo: str) -> tuple[str, str]:
        """Resolve topology file paths for a given topology name."""
        base = _package_topology_dir()
        pickle_path = base / f"{topo}_modified.pickle"
        cgd_path = base / f"{topo}_modified.cgd"
        return (str(pickle_path), str(cgd_path))

    def _list_xyz(self, folder: str) -> list[tuple[str, str]]:
        """List XYZ files in a folder."""
        files = sorted(glob.glob(os.path.join(folder, "*.xyz")))
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in files]

    def build(
        self,
        topo: str,
        cof_name: str,
        bond_type: str | None = None,
        output_folder: str | None = None,
    ) -> list[str]:
        """Build a 2D COF from one node and one linker.

        Args:
            topo: Topology name (currently "hcb" or "sql").
            bond_type: Bond type, "single" or "double".
            output_folder: Optional base output folder for CIF files.
            cof_name: COF name used for output folder and filename.
                Outputs default to {cof_name}/1_{cof_name}_single_layer.
        """
        _disable_pormake_file_logging()
        with ExitStack() as stack:
            if bond_type:
                nodes_dir_used = stack.enter_context(
                    tempfile.TemporaryDirectory()
                )
                _prepare_xyz("0_node", nodes_dir_used, bond_type)
                linker_dir_used = stack.enter_context(
                    tempfile.TemporaryDirectory()
                )
                _prepare_xyz("0_linker", linker_dir_used, bond_type)
            else:
                nodes_dir_used = "nodes"
                linker_dir_used = "linker"

            nodes = self._list_xyz(nodes_dir_used)
            linkers = self._list_xyz(linker_dir_used)

            if len(nodes) != 1 or len(linkers) != 1:
                raise ValueError(
                    "Expected exactly one node and one linker file."
                )

            node_name, node_path = nodes[0]
            linker_name, linker_path = linkers[0]

            output_folder_used = output_folder or os.path.join(
                cof_name, f"1_{cof_name}_single_layer"
            )

            os.makedirs(output_folder_used, exist_ok=True)
            output = _build_cof(
                topo,
                node_name,
                node_path,
                linker_name,
                linker_path,
                output_folder_used,
                cof_name=cof_name,
            )

            return [output]
