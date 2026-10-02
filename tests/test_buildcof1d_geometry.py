from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from pormake.framework import Framework
from pormake.neighbor_list import NeighborList
from pormake.topology import Topology

from coflandscaper._internal import build_cof_1d as builder


def test_build_ignores_hidden_xyz_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ignore hidden AppleDouble XYZ files when discovering default inputs."""
    monkeypatch.chdir(tmp_path)
    node_dir = tmp_path / "0_node"
    node_dir.mkdir()
    write(node_dir / "A.xyz", Atoms("C2He2", positions=np.zeros((4, 3))))
    write(node_dir / "N.xyz", Atoms("C4He4", positions=np.zeros((8, 3))))
    (node_dir / "._A.xyz").write_text("metadata", encoding="utf-8")
    (node_dir / "._N.xyz").write_text("metadata", encoding="utf-8")

    seen: list[Path] = []

    class FakeFramework:
        atoms = Atoms("C", positions=[[0, 0, 0]])

        @staticmethod
        def write_cif(path: str) -> None:
            write(path, FakeFramework.atoms, format="cif")

    def fake_build(_config, paths, _inputs, _cgd_path):
        seen.extend(paths)
        return FakeFramework()

    monkeypatch.setattr(builder, "_build_ladder", fake_build)
    builder.BuildCOF1D().build(topo="ladder_1d", cof_name="test")

    assert [path.name for path in seen] == ["A.xyz", "N.xyz"]


@pytest.mark.parametrize("forced", [False, True])
def test_fixed_embedding_scaler(forced: bool) -> None:
    """Fit a synthetic periodic net without changing fractions or c."""
    top = Topology.__new__(Topology)
    top.atoms = Atoms(
        "CO2",
        scaled_positions=[[0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        cell=[1, 1, 15],
        pbc=True,
        tags=[0, -1, -1],
        info={"cn": [4, 2, 2]},
    )
    top.neighbor_list = NeighborList.__new__(NeighborList)
    top.neighbor_list.set_data(
        [
            [
                (1, np.array([0.5, 0, 0])),
                (1, np.array([-0.5, 0, 0])),
                (2, np.array([0, 0.5, 0])),
                (2, np.array([0, -0.5, 0])),
            ],
            [(0, np.array([-0.5, 0, 0])), (0, np.array([0.5, 0, 0]))],
            [(0, np.array([0, -0.5, 0])), (0, np.array([0, 0.5, 0]))],
        ]
    )
    top.calculate_properties()
    block = SimpleNamespace(
        lengths=np.array([2, 2, 3, 3]),
        centroid=np.zeros(3),
        connection_points=np.array(
            [[2, 0, 0], [-2, 0, 0], [0, 3, 0], [0, -3, 0]]
        ),
    )
    scaler = builder._FixedEmbeddingScaler(
        forced_a=5.0 if forced else None,
        forced_b=7.0 if forced else None,
    )
    scaled, result = scaler.scale(
        top, [block, None, None], [np.arange(4)], return_result=True
    )
    expected = np.diag([5, 7, 15] if forced else [4, 6, 15])
    np.testing.assert_allclose(scaled.atoms.cell, expected, atol=1e-10)
    np.testing.assert_allclose(
        scaled.atoms.get_scaled_positions(), top.atoms.get_scaled_positions()
    )
    np.testing.assert_allclose(top.atoms.cell, np.diag([1, 1, 15]))
    for old, new in zip(top.neighbor_list, scaled.neighbor_list, strict=True):
        for before, after in zip(old, new, strict=True):
            assert before.index == after.index
            np.testing.assert_allclose(
                after.distance_vector, before.distance_vector @ expected
            )
    assert scaled.check_validity()
    assert isinstance(result.fun, float)


@pytest.mark.parametrize("rotation", [0, 47, 180, 263])
def test_two_connected_orientation(rotation: float) -> None:
    """A tilted asymmetric backbone faces outward without moving its anchors."""
    atoms = Atoms(
        "X2C3",
        positions=[
            [0, -1, 0],
            [0, 1, 0],
            [-1, 0, 0],
            [-0.5, -0.5, 0],
            [-0.5, 0.5, 0],
        ],
    )
    atoms.rotate(rotation, "y")
    anchors = atoms.positions[:2].copy()
    distances = atoms.get_all_distances()
    block = SimpleNamespace(
        atoms=atoms,
        connection_points=anchors.copy(),
        connection_point_indices=np.array([0, 1]),
    )
    top = SimpleNamespace(
        atoms=Atoms(cell=[10, 12, 15]),
        neighbor_list=[
            [
                SimpleNamespace(distance_vector=np.array([-1, -1, 0])),
                SimpleNamespace(distance_vector=np.array([-1, 1, 0])),
            ]
        ],
    )
    result = builder._orient_two_connected(block, 0, top)
    np.testing.assert_allclose(result.atoms.positions[:2], anchors, atol=1e-8)
    np.testing.assert_allclose(result.atoms.positions[:, 2], 0, atol=1e-8)
    assert result.atoms.positions[2:, 0].mean() > 0.1
    np.testing.assert_allclose(
        result.atoms.get_all_distances(), distances, atol=1e-8
    )


@pytest.fixture(
    scope="module", params=[(1.0, 60.0, 2.0, 4.0), (1.15, 70.0, 2.2, 5.0)]
)
def synthetic_ladder(
    request: pytest.FixtureRequest,
) -> tuple[Framework, builder.BuildCOF1D]:
    """Run the real builder on dimensioned star and bent test fragments."""
    arm, angle, bond, gap = request.param
    corners = np.array(
        [[-1, -1, 0], [-1, 1, 0], [1, -1, 0], [1, 1, 0]], dtype=float
    )
    star = Atoms(
        "C5He4",
        positions=np.vstack(
            [np.zeros(3), arm * corners, (arm + 0.7) * corners]
        ),
    )
    bend = Atoms(
        "C3He2",
        positions=np.vstack(
            [np.zeros(3), arm * corners[2:], (arm + 0.7) * corners[2:]]
        ),
    )
    config = builder.BuildCOF1D(
        target_j_bond=bond,
        target_connector_angle=angle,
        target_interladder_gap=gap,
    )
    cgd = (
        Path(builder.__file__).parents[1] / "database/topologies/ladder_1d.cgd"
    )
    # Centering two connection points always produces opposite directions:
    # rotation about their axis is ambiguous, even for asymmetric fragments.
    # Exact synthetic matches also trigger PORMAKE's unguarded 0/0 RMSD ratio.
    # Expect only these messages during construction; pytest re-emits others.
    with pytest.warns(
        (UserWarning, RuntimeWarning),
        match=(
            r"^(Optimal rotation is not uniquely or poorly defined for the "
            r"given sets of vectors\.|invalid value encountered in scalar divide)$"
        ),
    ):
        framework = builder._build_ladder(
            config, [Path("star.xyz"), Path("bend.xyz")], [star, bend], cgd
        )
    return framework, config


def _fragment_indices(framework):
    groups = []
    offset = 0
    for block in framework.info["located_bbs"]:
        if block is not None:
            count = sum(s != "X" for s in block.atoms.get_chemical_symbols())
            groups.append(np.arange(offset, offset + count))
            offset += count
    assert offset == len(framework.atoms)
    return groups


def test_connector_adjustment(synthetic_ladder) -> None:
    """Every interfragment bond reaches its target, including periodic bonds."""
    framework, config = synthetic_ladder
    owners = np.empty(len(framework.atoms), dtype=int)
    for index, group in enumerate(_fragment_indices(framework)):
        owners[group] = index
    connectors = [
        (int(i), int(j)) for i, j in framework.bonds if owners[i] != owners[j]
    ]
    assert len(connectors) == 8
    crossed = False
    for i, j in connectors:
        vector = framework.atoms.get_distance(i, j, mic=True, vector=True)
        raw = framework.atoms.positions[j] - framework.atoms.positions[i]
        crossed |= not np.allclose(vector, raw)
        assert np.linalg.norm(vector) == pytest.approx(
            config.target_j_bond, abs=config.bond_tolerance
        )
        angle = np.degrees(np.arctan2(abs(vector[1]), abs(vector[0])))
        assert 45 <= angle <= 85
        assert angle == pytest.approx(config.target_connector_angle, abs=1.0)
    assert crossed


def _unwrapped_ladders(atoms, bonds):
    """Recover whole ladder x coordinates from bond connectivity, not extents."""
    adjacency = [[] for _ in atoms]
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)
    remaining = set(range(len(atoms)))
    families = []
    while remaining:
        seed = remaining.pop()
        x = {seed: atoms.positions[seed, 0]}
        queue = [seed]
        while queue:
            i = queue.pop()
            for j in adjacency[i]:
                value = (
                    x[i] + atoms.get_distance(i, j, mic=True, vector=True)[0]
                )
                if j in x:
                    assert value == pytest.approx(x[j], abs=1e-7)
                else:
                    remaining.remove(j)
                    x[j] = value
                    queue.append(j)
        indices = np.array(sorted(x))
        families.append((indices, np.array([x[i] for i in indices])))
    assert len(families) == 2
    return sorted(families, key=lambda item: item[1].mean())


def test_ladder_gaps_preserve_geometry(synthetic_ladder) -> None:
    """Both gaps survive wrapping, while complete rigid fragments stay intact."""
    framework, config = synthetic_ladder
    groups = _fragment_indices(framework)
    blocks = [b for b in framework.info["located_bbs"] if b is not None]
    for group, block in zip(groups, blocks, strict=True):
        real = block.atoms[np.array(block.atoms.get_chemical_symbols()) != "X"]
        np.testing.assert_allclose(
            framework.atoms[group].get_all_distances(mic=True),
            real.get_all_distances(),
            atol=1e-7,
        )
    for translation in (0.0, 0.43):
        atoms = framework.atoms.copy()
        atoms.positions[:, 0] += translation * atoms.cell[0, 0]
        atoms.wrap()
        (left, lx), (right, rx) = _unwrapped_ladders(atoms, framework.bonds)
        a = atoms.cell[0, 0]
        rx += (np.floor((lx.mean() - rx.mean()) / a) + 1) * a
        # These synthetic fragments have matched y rows in both ladders.
        # Measure each occupied row, independently of the production y bins.
        y_rows = np.unique(np.round(atoms.positions[:, 1], 6))
        for y in y_rows:
            ly = np.isclose(atoms.positions[left, 1], y, atol=1e-5)
            ry = np.isclose(atoms.positions[right, 1], y, atol=1e-5)
            assert ly.any()
            assert ry.any()
            middle = rx[ry].min() - lx[ly].max()
            periodic = lx[ly].min() + a - rx[ry].max()
            assert middle == pytest.approx(
                config.target_interladder_gap, abs=config.gap_tolerance
            )
            assert periodic == pytest.approx(
                config.target_interladder_gap, abs=config.gap_tolerance
            )
        assert atoms.cell[2, 2] == pytest.approx(config.c_fixed)
