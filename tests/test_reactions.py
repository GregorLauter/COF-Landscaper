"""Chemistry and periodic geometry checks for the optional precursor route."""

from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from pymatgen.io.cif import CifFile

from coflandscaper import BuildCOF2D, BuildCOF3D, PrepareBuildingBlock
from coflandscaper._internal._pormake import BuildingBlock

_TRIALDEHYDE = "O=Cc1cc(C=O)cc(C=O)c1"
_TRIAMINE = "Nc1cc(N)cc(N)c1"
_DIAMINE = "Nc1ccc(N)cc1"
_TETRALDEHYDE = "C(c1ccc(C=O)cc1)(c1ccc(C=O)cc1)(c1ccc(C=O)cc1)c1ccc(C=O)cc1"


@pytest.fixture
def prepare() -> Callable[..., Path]:
    pytest.importorskip("stk")
    return PrepareBuildingBlock().run


def test_optional_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "stk", None)
    with pytest.raises(ImportError, match=r"cof-landscaper\[reactions\]"):
        PrepareBuildingBlock().run(
            tmp_path / "node.xyz", smiles=_TRIALDEHYDE, configuration="E"
        )


def test_reference_product_and_file_coordinates(
    prepare, tmp_path: Path
) -> None:
    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import EmbedMolecule

    precursors = ["O=Cc1ccccc1", "Nc1ccccc1"]
    product = Chem.RWMol()
    ends = []
    for n, smiles in enumerate(precursors):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert EmbedMolecule(mol, randomSeed=95) == 0
        source = tmp_path / ("aldehyde.mol" if n == 0 else "amine.sdf")
        if n == 0:
            Chem.MolToMolFile(mol, str(source))
        else:
            with Chem.SDWriter(str(source)) as writer:
                writer.write(mol)
        out = tmp_path / f"input{n}.xyz"
        prepare(out, input_file=source, configuration="E", expected_sites=1)
        bb = BuildingBlock(str(out))
        retain = {}
        for atom in bb.atoms:
            if atom.symbol != "X":
                retain[atom.index] = product.AddAtom(Chem.Atom(atom.symbol))
                assert (
                    np.min(
                        np.linalg.norm(
                            mol.GetConformer().GetPositions() - atom.position,
                            axis=1,
                        )
                    )
                    < 0.00012
                )
        for (i, j), order in zip(bb.bonds, bb.bond_types, strict=True):
            if i in retain and j in retain:
                product.AddBond(
                    retain[i],
                    retain[j],
                    {"S": Chem.BondType.SINGLE, "A": Chem.BondType.AROMATIC}[
                        order
                    ],
                )
            else:
                ends.append(retain[i] if i in retain else retain[j])
        original = out.read_bytes()
        with pytest.raises(FileExistsError):
            prepare(out, smiles=smiles, configuration="E")
        assert out.read_bytes() == original
    c, n = ends
    product.AddBond(c, n, Chem.BondType.DOUBLE)
    Chem.SanitizeMol(product)
    assert Chem.MolToSmiles(Chem.RemoveHs(product)) == Chem.MolToSmiles(
        Chem.MolFromSmiles("c1ccccc1C=Nc1ccccc1")
    )


def test_reactive_role_validation(prepare, tmp_path: Path) -> None:
    out = tmp_path / "prepared.xyz"
    for smiles in ["CC(=O)C", "CNC", "CC(=O)N", "O=Cc1ccc(N)cc1"]:
        with pytest.raises(ValueError, match="role"):
            prepare(out, smiles=smiles, configuration="E")
    with pytest.raises(ValueError, match="expected_sites"):
        prepare(out, smiles=_TRIALDEHYDE, configuration="E", expected_sites=4)
    assert not out.exists()
    prepare(
        out,
        smiles="O=Cc1ccc(N)cc1",
        configuration="Z",
        role="aldehyde",
        expected_sites=1,
    )
    assert BuildingBlock(str(out)).n_connection_points == 1


def _check_crystal(
    path: Path, dim: int, expected: dict[str, int], links: int
) -> None:
    """Read the exported graph and periodic images independently of PORMAKE."""
    data = next(iter(CifFile.from_file(path).data.values())).data
    labels = data["_atom_site_label"]
    positions = np.column_stack(
        [data[f"_atom_site_fract_{x}"] for x in "xyz"]
    ).astype(float)
    atoms = read(path)
    assert isinstance(atoms, Atoms)
    cell = atoms.cell.array
    assert Counter(data["_atom_site_type_symbol"]) == expected
    graph: dict[str, list] = {label: [] for label in labels}
    by_name = dict(zip(labels, positions, strict=True))
    imines = []
    for a, b, length, sym, order in zip(
        *(
            data[key]
            for key in (
                "_geom_bond_atom_site_label_1",
                "_geom_bond_atom_site_label_2",
                "_geom_bond_distance",
                "_geom_bond_site_symmetry_2",
                "_ccdc_geom_bond_type",
            )
        ),
        strict=True,
    ):
        shift = (
            np.zeros(3, dtype=int)
            if sym == "."
            else np.array(list(sym.split("_")[1]), dtype=int) - 5
        )
        delta = (by_name[b] + shift - by_name[a]) @ cell
        assert np.linalg.norm(delta) == pytest.approx(float(length), abs=0.003)
        graph[a].append((b, shift))
        graph[b].append((a, -shift))
        if order == "D":
            assert {a[0], b[0]} == {"C", "N"}
            assert 1.16 <= float(length) <= 1.46
            imines.append((a, b) if a.startswith("C") else (b, a))
    assert len(imines) == links
    for c, n in imines:
        (ca,) = [label for label, _ in graph[c] if label.startswith("C")]
        (na,) = [
            label
            for label, _ in graph[n]
            if label.startswith("C") and label != c
        ]
        torsion = atoms.get_dihedral(
            labels.index(ca),
            labels.index(c),
            labels.index(n),
            labels.index(na),
            mic=True,
        )
        assert abs(torsion - 180) <= 35
    seen = {labels[0]: np.zeros(3, dtype=int)}
    queue = [labels[0]]
    cycles = []
    while queue:
        a = queue.pop()
        for b, shift in graph[a]:
            image = seen[a] + shift
            if b in seen:
                cycles.append(image - seen[b])
            else:
                seen[b] = image
                queue.append(b)
    assert len(seen) == len(labels)
    assert np.linalg.matrix_rank(cycles) == dim


@pytest.mark.integration
@pytest.mark.parametrize(
    ("topo", "expected", "links"),
    [
        ("hcb", {"C": 36, "H": 24, "N": 6}, 6),
        ("hcb_ab", {"C": 15, "H": 9, "N": 3}, 3),
        ("sql", {"C": 22, "H": 14, "N": 4}, 4),
        ("dia", {"C": 328, "H": 224, "N": 32}, 32),
    ],
)
def test_periodic_imines(
    prepare, tmp_path, monkeypatch, topo, expected, links
) -> None:
    monkeypatch.chdir(tmp_path)
    node = prepare(
        "0_node/n.xyz",
        smiles={"dia": _TETRALDEHYDE, "sql": "O=Cc1c(C=O)cc(C=O)c(C=O)c1"}.get(
            topo, _TRIALDEHYDE
        ),
        configuration="E",
        expected_sites=4 if topo in {"dia", "sql"} else 3,
    )
    edge = prepare(
        "0_linker/l.xyz",
        smiles=_TRIAMINE if topo == "hcb_ab" else _DIAMINE,
        configuration="E",
        expected_sites=3 if topo == "hcb_ab" else 2,
    )
    if topo == "dia":
        paths = BuildCOF3D().build(topo, "sample", {0: node}, {(0, 0): edge})
    elif topo == "hcb_ab":
        paths = BuildCOF2D().build(topo, "sample", [node, edge], [])
    else:
        paths = BuildCOF2D().build(topo, "sample")
    _check_crystal(Path(paths[0]), 3 if topo == "dia" else 2, expected, links)


@pytest.mark.integration
def test_imine_assembly_rejects_incompatible_junctions(
    prepare, tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    node = prepare("node.xyz", smiles=_TRIALDEHYDE, configuration="E")
    with pytest.raises(ValueError, match="one aldehyde carbon"):
        BuildCOF2D().build("hcb_ab", "bad", [node, node], [])
    with pytest.raises(ValueError, match="requires 4 imine sites"):
        BuildCOF2D().build("sql_ab", "bad", [node, node], [])
    z_node = prepare("z-node.xyz", smiles=_TRIALDEHYDE, configuration="Z")
    z_link = prepare("z-link.xyz", smiles=_DIAMINE, configuration="Z")
    with pytest.raises(ValueError, match="Unacceptable assembled Z-imine"):
        BuildCOF2D().build("hcb", "bad", [z_node], [z_link])
    assert not list(tmp_path.rglob("*.cif"))

    tetraamine = prepare(
        "tetraamine.xyz",
        smiles=_TETRALDEHYDE.replace("C=O", "N"),
        configuration="E",
    )
    dialdehyde = prepare(
        "dialdehyde.xyz", smiles="O=Cc1ccc(C=O)cc1", configuration="E"
    )
    with pytest.raises(
        ValueError, match="Unacceptable assembled E-imine geometry"
    ):
        BuildCOF3D().build("dia", "bad", {0: tetraamine}, {(0, 0): dialdehyde})
    assert not list(tmp_path.rglob("*.cif"))


@pytest.mark.integration
def test_manual_hcb_matches_reference(tmp_path: Path) -> None:
    """The original He-marked LZU-1 inputs still reproduce the published example cell."""
    from pymatgen.core import Structure

    example = Path(__file__).resolve().parents[1] / "example"
    [output] = BuildCOF2D().build(
        "hcb",
        "manual",
        [example / "0_node/example_node.xyz"],
        [example / "0_linker/example_linker.xyz"],
        output_folder=str(tmp_path),
    )
    reference = Structure.from_file(
        example / "COF-LZU-1/1_COF-LZU-1_single_layer/COF-LZU-1_unopt.cif"
    )
    assert reference.matches(
        Structure.from_file(output),
        ltol=0.001,
        stol=0.01,
        angle_tol=0.1,
        scale=False,
        primitive_cell=False,
    )
