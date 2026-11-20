"""Optional imine precursor preparation and checks for PORMAKE assembly."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from ase.data import covalent_radii, vdw_radii
from ase.neighborlist import neighbor_list

from ._pormake import BuildingBlock
from .ild_ils_utils import _unwrap_fractional_z

_HEADER = "COF-Landscaper imine-v1"


def _configuration(path: str | Path) -> str | None:
    """Recognise the explicit bond-table format before manual XYZ processing."""
    with Path(path).open() as stream:
        stream.readline()
        comment = stream.readline().strip()
    if not comment.startswith(_HEADER):
        return None
    config = comment.removeprefix(_HEADER).strip()
    if config not in {"E", "Z"}:
        raise ValueError(f"Invalid imine configuration in {path}.")
    return config


def _assembly_configuration(paths) -> str | None:
    configurations = {_configuration(path) for path in paths}
    if len(configurations) != 1:
        raise ValueError(
            "Use complementary prepared blocks with the same E/Z target, or all manual XYZ inputs."
        )
    return configurations.pop()


def _read_imine_block(path: str | Path, coordination: int) -> BuildingBlock:
    block = BuildingBlock(str(path))
    if block.n_connection_points != coordination:
        raise ValueError(
            f"{path} requires {coordination} imine sites, found {block.n_connection_points}."
        )
    return block


def _unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-6:
        raise ValueError("Degenerate geometry at an imine connection site.")
    return vector / length


def _read_precursor(smiles, input_file):
    """Read a molecular graph and retain or generate one explicit-H conformer."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if input_file is None:
        mol = Chem.MolFromSmiles(smiles or "")
    else:
        source = Path(input_file)
        if source.suffix.lower() not in {".sdf", ".mol"}:
            raise ValueError("Use a single 3D MOL or SDF record.")
        supplier = Chem.SDMolSupplier(str(source), removeHs=False)
        if len(supplier) != 1:
            raise ValueError("Expected one MOL/SDF record.")
        mol = supplier[0]
    if (
        mol is None
        or len(Chem.GetMolFrags(mol)) != 1
        or any(
            a.GetFormalCharge()
            or a.GetNumRadicalElectrons()
            or a.GetIsotope()
            or a.GetAtomicNum() in {0, 2}
            for a in mol.GetAtoms()
        )
    ):
        raise ValueError(
            "Expected one neutral, connected, closed-shell precursor without isotope labels or markers."
        )
    if any(
        str(stereo.type) == "Atom_Tetrahedral"
        for stereo in Chem.FindPotentialStereo(mol)
    ):
        raise ValueError(
            "Chiral precursor centers are unsupported because PORMAKE may reflect a block during placement."
        )
    if any(
        str(stereo.specified) == "Unspecified"
        for stereo in Chem.FindPotentialStereo(mol)
    ):
        raise ValueError(
            "Specify precursor stereochemistry before preparation."
        )
    if input_file is not None and (
        not mol.GetNumConformers() or not mol.GetConformer().Is3D()
    ):
        raise ValueError(
            "Provide 3D coordinates in molecular files, or use SMILES."
        )
    mol = Chem.AddHs(mol, addCoords=input_file is not None)
    if input_file is None:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("RDKit could not generate a conformer.")
        if (
            AllChem.MMFFHasAllMoleculeParams(mol)
            and AllChem.MMFFOptimizeMolecule(mol, maxIters=1000) != 0
        ):
            raise ValueError(
                "Conformer relaxation failed. Supply a 3D MOL/SDF."
            )
    xyz = mol.GetConformer().GetPositions()
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite precursor coordinates.")
    return mol


def _reaction_sites(stk, mol, role, expected_sites):
    """Select stk bonder/deleter IDs and reject non-carbon-substituted sites."""
    block = stk.BuildingBlock.init_from_rdkit_mol(mol)
    groups = {}
    for label, factory in [
        ("aldehyde", stk.AldehydeFactory()),
        ("amine", stk.PrimaryAminoFactory()),
    ]:
        sites = []
        for group in factory.get_functional_groups(block):
            (bonder,) = group.get_bonder_ids()
            atom = mol.GetAtomWithIdx(bonder)
            anchors = [a for a in atom.GetNeighbors() if a.GetAtomicNum() == 6]
            if len(anchors) != 1 or atom.GetDegree() != 3:
                continue
            anchor = anchors[0]
            if label == "amine" and any(
                b.GetBondTypeAsDouble() > 1.5
                and b.GetOtherAtom(anchor).GetAtomicNum() != 6
                for b in anchor.GetBonds()
            ):
                continue  # Amides, amidines and related non-amine nitrogens.
            sites.append(
                (bonder, anchor.GetIdx(), tuple(group.get_deleter_ids()))
            )
        if sites:
            groups[label] = sites
    if role is None:
        if not groups:
            raise ValueError(
                "No supported aldehyde or primary-amine role found."
            )
        if len(groups) != 1:
            raise ValueError(
                "Expected one reactive role. Specify a role for a mixed precursor."
            )
        role = next(iter(groups))
    sites = groups.get(role, [])
    if not sites or (
        expected_sites is not None and len(sites) != expected_sites
    ):
        raise ValueError(
            f"Found {len(sites)} {role} sites; check the role and expected_sites."
        )
    return role, sites


class PrepareBuildingBlock:
    """Prepare neutral aldehyde or primary-amine precursors for imine assembly.

    Uses stk-lite functional-group factories and RDKit. Cite Turcani et al.,
    J. Chem. Phys. 154, 214102 (2021), doi:10.1063/5.0049708.
    """

    def run(
        self,
        output_path: str | Path,
        *,
        configuration: str,
        smiles: str | None = None,
        input_file: str | Path | None = None,
        reaction: str = "imine",
        role: str | None = None,
        expected_sites: int | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write one PORMAKE bond-table XYZ for the existing builder input slots.

        Supply one SMILES or one 3D MOL/SDF record. SMILES uses one reproducible
        ETKDG conformer. File inputs retain the supplied heavy-atom geometry.
        Select configuration 'E' or 'Z' for all new junctions and role
        'aldehyde' or 'amine' when both are present. All sites of the chosen
        role react. expected_sites checks coordination before assembly.
        Pre-existing files require overwrite=True. The returned Path can be passed
        directly to BuildCOF2D or BuildCOF3D. This prepares an initial
        geometry and does not search for stereoisomers or low-energy torsions.
        """
        try:
            import stk
            from rdkit import Chem
        except ImportError as exc:
            raise ImportError(
                'Install the optional chemistry tools with pip install "cof-landscaper[reactions]".'
            ) from exc

        path = Path(output_path)
        if path.suffix.lower() != ".xyz" or (smiles is None) == (
            input_file is None
        ):
            raise ValueError(
                "Supply one SMILES or input_file and a .xyz output path."
            )
        if (
            reaction != "imine"
            or configuration not in {"E", "Z"}
            or role not in {None, "amine", "aldehyde"}
        ):
            raise ValueError(
                "Select imine chemistry, E or Z configuration, and aldehyde or amine role."
            )
        mol = _read_precursor(smiles, input_file)
        xyz = mol.GetConformer().GetPositions()
        role, sites = _reaction_sites(stk, mol, role, expected_sites)
        if role == "aldehyde" and mol.HasSubstructMatch(
            Chem.MolFromSmarts("[OH]c1c([CH]=O)cccc1")
        ):
            raise ValueError(
                "Ortho-hydroxyaryl aldehydes require explicit tautomer selection; use the manual route."
            )
        remove = set()
        markers = []
        for bonder, anchor_id, leaving in sites:
            remove.update(leaving)
            pos = xyz[bonder]
            if role == "amine":
                marker = pos + 0.20 * _unit(xyz[anchor_id] - pos)
            else:
                (oxygen,) = leaving
                axis = _unit(xyz[oxygen] - pos)
                anchor_vec = xyz[anchor_id] - pos
                side = _unit(anchor_vec - axis * (anchor_vec @ axis))
                # The shared marker lies 0.20 Å from the future N toward its carbon substituent.
                toward_anchor = (
                    0.5 * axis
                    + (1 if configuration == "Z" else -1)
                    * np.sqrt(0.75)
                    * side
                )
                marker = pos + 1.28 * axis + 0.20 * toward_anchor
            markers.append((bonder, marker))
        indices = {}
        atoms: list[tuple[str, np.ndarray]] = []
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if idx not in remove:
                indices[idx] = len(atoms)
                atoms.append((atom.GetSymbol(), xyz[idx]))
        types = {1.0: "S", 1.5: "A", 2.0: "D", 3.0: "T"}
        bonds = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in indices and j in indices:
                order = types.get(bond.GetBondTypeAsDouble())
                if order is None:
                    raise ValueError("Unsupported precursor bond order.")
                bonds.append((indices[i], indices[j], order))
        for i, marker in markers:
            bonds.append((indices[i], len(atoms), "S"))
            atoms.append(("X", marker))
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as scratch:
            temporary = Path(scratch.name)
        try:
            with temporary.open("w") as stream:
                stream.write(f"{len(atoms)}\n{_HEADER} {configuration}\n")
                for symbol, (x, y, z) in atoms:
                    stream.write(f"{symbol} {x:.10f} {y:.10f} {z:.10f}\n")
                for i, j, order in bonds:
                    stream.write(f"{i} {j} {order}\n")
            if overwrite:
                temporary.replace(path)
            else:
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def _prepare_layer(atoms) -> None:
    """Place the complete bonded sheet in the standard 15 Å cell before export."""
    a, b, c = atoms.cell.array
    normal = _unit(np.cross(a, b))
    frac_z = atoms.get_scaled_positions(wrap=False)[:, 2]
    start = _unwrap_fractional_z(frac_z)
    positions = atoms.positions - np.floor(frac_z - start + 1e-10)[:, None] * c
    heights = positions @ normal
    if np.ptp(heights) >= 15:
        raise ValueError(
            "Prepared sheet exceeds the single-layer cell height."
        )
    atoms.set_cell([a, b, 15 * normal])
    atoms.positions = (
        positions + (7.5 - (max(heights) + min(heights)) / 2) * normal
    )


def _finish_imines(framework, config: str) -> None:
    """Verify marker consumption, minimum-image geometry and requested C=N stereo."""
    atoms = framework.atoms
    # PORMAKE writes wrapped coordinates but derives CIF bond images from raw
    # positions. Use the same images for both, away from rounding boundaries.
    fractional = atoms.get_scaled_positions()
    for axis in fractional.T:
        if np.any(np.minimum(axis, 1 - axis) < 1e-5):
            ordered = np.sort(np.r_[axis, axis.min() + 1])
            gap = np.argmax(np.diff(ordered))
            origin = (ordered[gap] + ordered[gap + 1]) / 2
            axis[:] = (axis - origin) % 1
    atoms.set_scaled_positions(fractional)
    slots: list[int] = []
    reactive = set()
    for slot, block in enumerate(framework.info["located_bbs"]):
        if block is None:
            continue
        indices = {}
        for atom in block.atoms:
            if atom.symbol != "X":
                indices[atom.index] = len(slots)
                slots.append(slot)
        for marker in block.connection_point_indices:
            attached = [
                indices[j if i == marker else i]
                for i, j in block.bonds
                if marker in (i, j)
            ]
            if len(attached) != 1 or attached[0] in reactive:
                raise ValueError(
                    "Each marker must identify a distinct reacting atom."
                )
            reactive.add(attached[0])
    neighbors: list[list[int]] = [[] for _ in atoms]
    bonds = set()
    for i, j in framework.bonds:
        neighbors[i].append(j)
        neighbors[j].append(i)
        bonds.add((min(i, j), max(i, j)))
    radii = covalent_radii[atoms.numbers]
    for index, (i, j) in enumerate(framework.bonds):
        if slots[i] == slots[j]:
            length = atoms.get_distance(i, j, mic=True)
            if (
                not 0.65 * (radii[i] + radii[j])
                <= length
                <= 1.30 * (radii[i] + radii[j])
            ):
                raise ValueError(
                    f"Unphysical precursor bond {i}-{j}: {length:.3f} Å."
                )
            continue
        if (
            i not in reactive
            or j not in reactive
            or {atoms[i].symbol, atoms[j].symbol} != {"C", "N"}
        ):
            raise ValueError(
                "Every junction requires one aldehyde carbon and one primary-amine nitrogen."
            )
        reactive.difference_update((i, j))
        carbon, nitrogen = (i, j) if atoms[i].symbol == "C" else (j, i)
        rc = [n for n in neighbors[carbon] if atoms[n].symbol == "C"]
        rn = [
            n
            for n in neighbors[nitrogen]
            if atoms[n].symbol == "C" and n != carbon
        ]
        if (
            len(rc) != 1
            or len(rn) != 1
            or len(neighbors[carbon]) != 3
            or len(neighbors[nitrogen]) != 2
        ):
            raise ValueError("Invalid aldehyde or amine product valence.")
        length = atoms.get_distance(carbon, nitrogen, mic=True)
        angles = atoms.get_angles(
            [[rc[0], carbon, nitrogen], [carbon, nitrogen, rn[0]]], mic=True
        )
        torsion = atoms.get_dihedral(rc[0], carbon, nitrogen, rn[0], mic=True)
        target = 180 if config == "E" else 0
        if not (
            1.16 <= length <= 1.46
            and np.all((angles >= 100) & (angles <= 140))
            and abs((torsion - target + 180) % 360 - 180) <= 35
        ):
            raise ValueError(
                f"Unacceptable assembled {config}-imine geometry: C=N {length:.3f} Å, angles {angles.round(1)}, torsion {torsion:.1f}°. Supply another conformer or use manual inputs."
            )
        framework.bond_types[index] = "D"
    if not bonds or reactive:
        raise ValueError("Unconsumed imine sites after assembly.")
    # Look across periodic images, including close images of a bonded pair.
    # Conservative steric screen. ASE lacks van der Waals radii for some elements.
    contact_radii = radii.copy()
    known = atoms.numbers < len(vdw_radii)
    vdw = vdw_radii[atoms.numbers[known]]
    contact_radii[known] = np.where(np.isfinite(vdw), 0.65 * vdw, radii[known])
    i_list, j_list, distances = neighbor_list("ijd", atoms, contact_radii)
    seen = set()
    for i, j, dist in zip(i_list, j_list, distances, strict=True):
        if (i, j) in seen or (min(i, j), max(i, j)) not in bonds or dist < 0.6:
            raise ValueError(
                f"Unphysical periodic contact {i}-{j}: {dist:.3f} Å."
            )
        seen.add((i, j))
