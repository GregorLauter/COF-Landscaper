"""Build 2D COFs from prepared nodes/linkers using pormake.

This module prepares XYZ inputs for pormake by inserting dummy connection
atoms, then builds a single-layer COF from one node and one linker. The user
must specify the topology (currently "hcb" or "sql"), the bond type ("single"
or "double"), and a COF name for output organization.
"""

import os
import tempfile
import glob
from io import StringIO
from typing import Sequence
from pymatgen.core import Structure
from contextlib import ExitStack
import numpy as np
import pormake as pm
import ase.io
from ase.atoms import Atoms
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry import Point3D


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



def _replace_atoms(atoms: Atoms, from_symbol: str, to_symbol: str) -> tuple[str, list[int]]:
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

def _write_xyz_with_bonds(mol: Mol, out_path: str, scaling_factor: float) -> None:
    """Write an XYZ file with bond annotations and scaled X positions.

    Args:
        mol: RDKit Mol.
        out_path: Output file path.
        scaling_factor: Scale applied to X-atom displacement.
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
                    vector.x *= scaling_factor
                    vector.y *= scaling_factor
                    vector.z *= scaling_factor
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
            elif bond_type == 1.5 and not bond.IsInRing():
                bond_type_str = "D"
            elif bond_type == 2.0:
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
    scaling_factor: float = 0.8,
) -> list[str]:
    """Prepare XYZ files by inserting bond markers for pormake.

    Args:
        input_folder: Folder with raw XYZ files.
        output_folder: Folder to write prepared XYZ files.
        bond_type: Bond type, "single" or "double".
        scaling_factor: Scale applied to X-atom displacement.

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
        _write_xyz_with_bonds(mol, out_path, scaling_factor)

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

def _normalize_edge_types(edge_types: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
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
    ild_guess: float,
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
        ild_guess: Interlayer distance guess.
        cof_name: Optional COF name for output filename.

    Returns:
        Path to the written CIF file.
    """
    database = pm.Database()
    topo_initial = database.get_topo(f"{topo}_initial")
    _sanitize_edge_types_inplace(topo_initial.edge_types)
    builder = pm.Builder()
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
    edge_bbs = {pair: linker for pair in edgetype_filtered}

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
    gamma = (alpha * ild_guess) / a

    base = os.path.join(os.path.dirname(pm.__file__), "database", "topologies")
    pickle_path = os.path.join(base, f"{topo}_modified.pickle")
    cgd_path = os.path.join(base, f"{topo}_modified.cgd")
    if os.path.exists(pickle_path):
        os.remove(pickle_path)

    with open(cgd_path, "r") as file:
        lines = file.readlines()
    line_parts = lines[3].split()
    line_parts[3] = f"{float(gamma):.5f}"
    lines[3] = "  ".join(line_parts) + "\n"
    with open(cgd_path, "w") as file:
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

    Attributes:
        ild_guess: Interlayer distance guess used for topology scaling.
    """

    def __init__(self, ild_guess: float = 15.0) -> None:
        """Initialize the builder."""
        self.ild_guess = ild_guess

    def _topology_paths(self, topo: str) -> tuple[str, str]:
        """Resolve topology file paths for a given topology name."""
        base = os.path.join(os.path.dirname(pm.__file__), "database", "topologies")
        pickle_path = os.path.join(base, f"{topo}_modified.pickle")
        cgd_path = os.path.join(base, f"{topo}_modified.cgd")
        return pickle_path, cgd_path

    def _list_xyz(self, folder: str) -> list[tuple[str, str]]:
        """List XYZ files in a folder."""
        files = sorted(glob.glob(os.path.join(folder, "*.xyz")))
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in files]

    def build(
        self,
        topo: str,
        output_folder: str = "cof_raw",
        bond_type: str | None = None,
        scaling_factor: float = 0.8,
        cof_name: str | None = None,
    ) -> list[str]:
        """Build a 2D COF from one node and one linker.

        Args:
            topo: Topology name (currently "hcb" or "sql").
            output_folder: Base output folder for CIF files.
            bond_type: Bond type, "single" or "double".
            scaling_factor: Scale applied to X-atom displacement.
            cof_name: COF name used for output folder and filename.
        """
        _disable_pormake_file_logging()
        with ExitStack() as stack:
            if bond_type:
                nodes_dir_used = stack.enter_context(tempfile.TemporaryDirectory())
                _prepare_xyz("0_node", nodes_dir_used, bond_type, scaling_factor)
                linker_dir_used = stack.enter_context(tempfile.TemporaryDirectory())
                _prepare_xyz("0_linker", linker_dir_used, bond_type, scaling_factor)
            else:
                nodes_dir_used = "nodes"
                linker_dir_used = "linker"

            nodes = self._list_xyz(nodes_dir_used)
            linkers = self._list_xyz(linker_dir_used)

            if len(nodes) != 1 or len(linkers) != 1:
                raise ValueError("Expected exactly one node and one linker file.")

            node_name, node_path = nodes[0]
            linker_name, linker_path = linkers[0]

            output_folder_used = output_folder
            if cof_name:
                output_folder_used = os.path.join(cof_name, f"1_{cof_name}_single_layer")

            os.makedirs(output_folder_used, exist_ok=True)
            output = _build_cof(
                topo,
                node_name,
                node_path,
                linker_name,
                linker_path,
                output_folder_used,
                self.ild_guess,
                cof_name=cof_name,
            )

            return [output]

