from dataclasses import dataclass
from io import StringIO
import glob
import os
import ase.io
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

@dataclass
class PrepareXYZ:
    scaling_factor: float = 0.8

    def _replace_atoms(self, atoms, from_symbol: str, to_symbol: str):
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

    def _build_rdkit_mol(self, xyz_block: str):
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return None
        rdDetermineBonds.DetermineBonds(mol)
        return mol

    def _tag_isotopes(self, mol, indices):
        for atom in mol.GetAtoms():
            if atom.GetIdx() in indices:
                atom.SetIsotope(2)

    def _iter_xyz_files(self, input_folder: str):
        return sorted(glob.glob(os.path.join(input_folder, "*.xyz")))

    def _write_xyz_with_bonds(self, mol, out_path: str):
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
                        vector.x *= self.scaling_factor
                        vector.y *= self.scaling_factor
                        vector.z *= self.scaling_factor
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

    def process(self, input_folder: str, output_folder: str, mode: str):
        mode_map = {
            "double_bond": "Se",
            "single_bond": "He",
            "Se": "Se",
            "He": "He",
        }
        if mode not in mode_map:
            raise ValueError("mode must be 'double_bond' or 'single_bond'")
        mode = mode_map[mode]

        os.makedirs(output_folder, exist_ok=True)
        xyz_files = self._iter_xyz_files(input_folder)

        for path in xyz_files:
            atoms = ase.io.read(path)
            if mode == "Se":
                xyz_block, x_indices = self._replace_atoms(atoms, "Se", "O")
            else:
                xyz_block, x_indices = self._replace_atoms(atoms, "He", "H")

            mol = self._build_rdkit_mol(xyz_block)
            if mol is None:
                continue

            self._tag_isotopes(mol, x_indices)

            base_filename = os.path.basename(os.path.splitext(path)[0] + ".xyz")
            out_path = os.path.join(output_folder, base_filename)
            self._write_xyz_with_bonds(mol, out_path)

        return xyz_files


def prepare_xyz(input_folder: str, output_folder: str, mode: str, scaling_factor: float = 0.8):
    return PrepareXYZ(scaling_factor=scaling_factor).process(input_folder, output_folder, mode)
