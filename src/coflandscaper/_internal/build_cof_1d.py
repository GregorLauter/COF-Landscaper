"""Build periodic 1D COF ladders from helium-marked molecular fragments.

The construction follows the validated make_1d_cof notebook: prepare rigid
fragments, fit the fractional embedding, correct real connector vectors through
a/b cell fitting, then pack the complete ladder families. Only ladder_1d is
supported. Every call owns its construction state.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import ase.io
import numpy as np
import pormake as pm
from pormake.neighbor_list import NeighborList
from pormake.topology import Topology, read_cgd
from scipy.optimize import OptimizeResult, lsq_linear

from coflandscaper._internal import build_cof_2d as preparation


@dataclass(frozen=True)
class BuildCOF1D:
    """Construct a single layer of parallel 1D ladders.

    Inputs default to two He-marked XYZ files in ``0_node/``: one 4c and one
    2c fragment. Distances are in angstroms; connector angle is measured from
    the a axis in degrees. ``target_interladder_gap`` is the sampled gap floor,
    not its mean. This builder creates an initial geometry, not a relaxed COF.

    Example::

        builder = cl.BuildCOF1D()
        paths = builder.build(topo="ladder_1d", cof_name="N5-A10")
    """

    target_j_bond: float = 2.0
    target_connector_angle: float = 60.0
    target_interladder_gap: float = 1.0
    c_fixed: float = 15.0
    x_scale: float = 0.3
    bond_tolerance: float = 0.2
    gap_tolerance: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "target_j_bond",
            "target_interladder_gap",
            "c_fixed",
            "x_scale",
            "bond_tolerance",
            "gap_tolerance",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.target_connector_angle)
            or not 45 <= self.target_connector_angle <= 85
        ):
            raise ValueError(
                "target_connector_angle must be between 45 and 85 degrees"
            )

    def build(
        self,
        topo: str,
        cof_name: str,
        input_nodes: Sequence[str | os.PathLike[str]] | None = None,
        output_folder: str | os.PathLike[str] | None = None,
    ) -> list[str]:
        """Build and save ``{cof_name}_unopt.cif``; return its path in a list.

        ``topo`` currently accepts only ``"ladder_1d"``. Explicit ``input_nodes``
        override ``0_node/*.xyz``. Paths are relative to the caller's working
        directory. The default output folder matches BuildCOF2D:
        ``{cof_name}/1_{cof_name}_single_layer``. Temporary PORMAKE inputs are
        removed automatically. An existing output is replaced only after the
        new structure passes validation and CIF export succeeds.
        """
        if topo != "ladder_1d":
            raise ValueError(
                "BuildCOF1D currently supports only topo='ladder_1d'"
            )
        if (
            not cof_name
            or cof_name in (".", "..")
            or Path(cof_name).name != cof_name
        ):
            raise ValueError(
                "cof_name must be a nonempty name without directory components"
            )
        paths = (
            sorted(
                path
                for path in Path("0_node").glob("*.xyz")
                if not path.name.startswith(".")
            )
            if input_nodes is None
            else [Path(p) for p in input_nodes]
        )
        if len(paths) != 2:
            raise ValueError(
                "ladder_1d requires exactly two node XYZ files: one 4c and one 2c"
            )
        for path in paths:
            if not path.is_file() or path.suffix.lower() != ".xyz":
                raise FileNotFoundError(f"Input XYZ file not found: {path}")
        inputs = [cast("ase.Atoms", ase.io.read(path)) for path in paths]
        if sorted(
            sum(s == "He" for s in atoms.get_chemical_symbols())
            for atoms in inputs
        ) != [2, 4]:
            raise ValueError(
                "ladder_1d requires one fragment with 4 He markers and one with 2"
            )
        preparation._disable_pormake_file_logging()
        cgd_path = (
            Path(__file__).resolve().parents[1]
            / "database/topologies/ladder_1d.cgd"
        )
        framework = _build_ladder(self, paths, inputs, cgd_path)
        folder = (
            Path(output_folder)
            if output_folder is not None
            else Path(cof_name) / f"1_{cof_name}_single_layer"
        )
        folder.mkdir(parents=True, exist_ok=True)
        output = folder / f"{cof_name}_unopt.cif"
        with TemporaryDirectory(dir=folder) as temporary:
            trial = Path(temporary) / output.name
            framework.write_cif(str(trial))
            if not trial.is_file():
                raise RuntimeError(
                    "CIF export failed; see PORMAKE output above"
                )
            written = ase.io.read(trial)
            if len(written) != len(framework.atoms):
                raise RuntimeError("CIF export changed the atom count")
            trial.replace(output)
        print(f"Saved: {output}")
        return [str(output)]


def _orient_two_connected(block, slot, topology):
    """Align a 2c backbone with ab and face its bulk away from bonded neighbors."""
    if len(block.connection_point_indices) != 2:
        return block
    geometry = preparation.CofLandscaperBuilder
    points = block.connection_points.copy()
    normal = geometry._topology_plane_normal(np.asarray(topology.atoms.cell))
    positions = geometry._align_edge_to_plane(
        block.atoms.positions, points[0], points[1], normal
    )
    axis = points[1] - points[0]
    axis /= np.linalg.norm(axis)
    # In this ladder, explicit neighbor vectors point toward the chain centerline.
    outward = -np.mean(
        [n.distance_vector for n in topology.neighbor_list[slot]], axis=0
    )
    outward -= np.dot(outward, normal) * normal
    outward -= np.dot(outward, axis) * axis
    backbone = ~np.isin(block.atoms.get_chemical_symbols(), ["X", "H"])
    if backbone.any() and np.linalg.norm(outward) > 1e-8:
        outward /= np.linalg.norm(outward)
        bulk = positions[backbone].mean(axis=0) - points.mean(axis=0)
        # Ambiguous/symmetric backbones keep their plane-aligned orientation.
        if np.dot(bulk, outward) < -1e-4:
            positions = geometry._rotate_about_axis(
                positions, points[0], axis, np.pi
            )
    if not np.allclose(
        positions[block.connection_point_indices], points, atol=1e-6, rtol=0
    ):
        raise ValueError("Orientation changed connection points")
    block.atoms.set_positions(positions)
    return block


class _FixedEmbeddingScaler:
    """PORMAKE-style dot-product fitting with fixed fractions and orthogonal cell.

    Only a and b vary. For this constraint, dot products are linear in
    a**2 and b**2, so bounded least squares solves the same fitting objective.
    """

    def __init__(
        self, length_weight=1.0, forced_a=None, forced_b=None
    ) -> None:
        if length_weight <= 0:
            raise ValueError("length_weight must be positive")
        if forced_a is not None and forced_a <= 0:
            raise ValueError("forced_a must be positive")
        if forced_b is not None and forced_b <= 0:
            raise ValueError("forced_b must be positive")
        self.length_weight = length_weight
        self.forced_a = forced_a
        self.forced_b = forced_b

    def scale(self, topology, bbs, perms, return_result=False):
        cell0 = np.asarray(topology.atoms.cell).copy()
        if not np.allclose(cell0, np.diag(np.diag(cell0)), atol=1e-6):
            raise ValueError(
                "This prototype requires an axis-aligned orthogonal cell"
            )
        if np.any(np.diag(cell0) <= 0):
            raise ValueError("Cell lengths must be positive")
        frac = topology.atoms.get_scaled_positions(wrap=False).copy()
        inv0 = np.linalg.inv(cell0)
        # Each incidence retains its identity and original neighbor ordering.
        fractional_vectors = [
            [np.asarray(n.distance_vector) @ inv0 for n in neighbors]
            for neighbors in topology.neighbor_list
        ]
        actual = {int(i): [] for i in topology.node_indices}
        targets = {int(i): [] for i in topology.node_indices}

        def connection_index(i, e, edge_neighbor):
            matches = [
                k
                for k, n in enumerate(topology.neighbor_list[i])
                if n.index == e
                and np.allclose(
                    n.distance_vector,
                    -edge_neighbor.distance_vector,
                    atol=1e-6,
                    rtol=0,
                )
            ]
            if len(matches) != 1:
                raise ValueError("Ambiguous reciprocal edge incidence")
            return matches[0]

        for e in topology.edge_indices:
            ni, nj = topology.neighbor_list[e]
            i, j = ni.index, nj.index
            ci = connection_index(i, e, ni)
            cj = connection_index(j, e, nj)
            pi, pj = np.asarray(perms[i]), np.asarray(perms[j])
            length = bbs[i].lengths[pi][ci] + bbs[j].lengths[pj][cj]
            if bbs[e] is not None:
                length += 2 * bbs[e].lengths[0]
            vi = bbs[i].connection_points[pi][ci] - bbs[i].centroid
            vj = bbs[j].connection_points[pj][cj] - bbs[j].centroid
            target_i = vi / np.linalg.norm(vi) * length
            target_j = vj / np.linalg.norm(vj) * length
            dfrac = (nj.distance_vector - ni.distance_vector) @ inv0
            image = dfrac - (frac[j] - frac[i])
            if not np.allclose(image, np.rint(image), atol=1e-6):
                raise ValueError("Noninteger periodic image")
            actual[i].append(dfrac)
            actual[j].append(-dfrac)
            targets[i].append(target_i)
            targets[j].append(target_j)

        rows, dots, weights = [], [], []
        for i in topology.node_indices:
            for u, v in product(range(len(actual[i])), repeat=2):
                rows.append(actual[i][u] * actual[i][v])
                dots.append(np.dot(targets[i][u], targets[i][v]))
                weights.append(2 * self.length_weight if u == v else 1.0)
        rows, dots, weights = map(np.asarray, (rows, dots, weights))
        normalization = np.mean(np.abs(dots))
        if not np.isfinite(normalization) or normalization <= 0:
            raise ValueError("Invalid building-block targets")
        # Physical cell lengths throughout: c is never globally rescaled.
        c_fixed = cell0[2, 2]
        factors = np.sqrt(weights) / normalization
        matrix = rows[:, :2] * factors[:, None]
        rhs = (dots - rows[:, 2] * c_fixed**2) * factors
        if np.linalg.matrix_rank(matrix) < 2:
            raise ValueError("The topology does not constrain both a and b")
        fit = lsq_linear(matrix, rhs, bounds=(1e-8, np.inf), tol=1e-12)
        if not fit.success:
            raise RuntimeError(fit.message)
        if np.any(fit.active_mask):
            raise ValueError("Fit collapsed a cell length to its lower bound")
        fitted_a, fitted_b = np.sqrt(fit.x)
        a = (
            float(self.forced_a)
            if self.forced_a is not None
            else float(fitted_a)
        )
        b = (
            float(self.forced_b)
            if self.forced_b is not None
            else float(fitted_b)
        )
        # Return PORMAKE's scalar-objective interface, not least-squares residuals.
        result = OptimizeResult(
            x=np.array([a**2, b**2]),
            fun=float(np.mean(fit.fun**2)),
            success=bool(fit.success),
            message=fit.message,
            nit=fit.nit,
        )
        cell = np.diag([a, b, c_fixed])
        scaled = topology.copy()
        scaled.atoms.set_cell(cell)
        scaled.atoms.set_scaled_positions(frac)
        new_data = [
            [
                (n.index, v @ cell)
                for n, v in zip(neighbors, vectors, strict=False)
            ]
            for neighbors, vectors in zip(
                topology.neighbor_list, fractional_vectors, strict=False
            )
        ]
        scaled.neighbor_list.set_data(new_data)
        if not scaled.check_validity():
            raise ValueError("Scaled topology failed validity checks")
        return (scaled, result) if return_result else scaled


def _build_ladder(config, paths, inputs, cgd_path):
    """Run one isolated construction, preserving the notebook geometry steps."""
    x_scale = config.x_scale
    target_j_bond = config.target_j_bond
    target_connector_angle = config.target_connector_angle
    target_interladder_gap = config.target_interladder_gap
    c_fixed = config.c_fixed
    bond_tolerance = config.bond_tolerance
    gap_tolerance = config.gap_tolerance
    max_embedding_iterations = 3
    embedding_fd_a = 1.0
    embedding_fd_b = 1.0
    min_nonbonded_distance = 0.9
    safe_initial_gap = max(5.0, target_interladder_gap + 4.0)
    n_gap_slices = 10
    max_bond_iterations = 6

    topology = Topology.__new__(Topology)
    topology.atoms = read_cgd(cgd_path)
    topology.atoms.set_cell([1.0, 1.0, c_fixed], scale_atoms=True)
    topology.name = topology.atoms.info["name"]
    topology.spacegroup = topology.atoms.info["spacegroup"]
    topology.local_structure_func = None
    topology.neighbor_list = NeighborList.__new__(NeighborList)
    # Read explicit EDGE endpoints from the CGD
    edge_endpoints = []

    with open(cgd_path) as f:
        for line in f:
            tokens = line.split()
            if tokens and tokens[0] == "EDGE":
                p1 = np.array(tokens[1:4], dtype=float)
                p2 = np.array(tokens[4:7], dtype=float)
                edge_endpoints.append((p1, p2))

    node_frac = topology.atoms.get_scaled_positions()[:6]
    cell = np.asarray(topology.atoms.cell)

    data = [[] for _ in range(topology.n_all_points)]

    for edge_i, (p1, p2) in enumerate(edge_endpoints):
        e = 6 + edge_i

        # Map endpoints, modulo periodic boundaries, onto node slots.
        def find_node(p):
            delta = p[None, :] - node_frac
            image = np.rint(delta).astype(int)
            residual = delta - image

            matches = np.where(np.linalg.norm(residual, axis=1) < 1e-6)[0]
            if len(matches) != 1:
                raise ValueError(f"Could not uniquely map endpoint {p}")

            i = int(matches[0])
            return i, image[i]

        i, image_i = find_node(p1)
        j, image_j = find_node(p2)

        # Actual endpoint positions, including their periodic images.
        ri = (node_frac[i] + image_i) @ cell
        rj = (node_frac[j] + image_j) @ cell

        # EDGE center exactly halfway between its stated endpoints.
        re = 0.5 * (ri + rj)

        # PORMAKE stores the edge-center atom wrapped into the unit cell.
        se = re @ np.linalg.inv(cell)
        se_wrapped = se % 1.0
        re_wrapped = se_wrapped @ cell

        topology.atoms.positions[e] = re_wrapped

        # Displacements must refer to the same physical edge image.
        vi = re - ri
        vj = re - rj

        data[i].append((e, vi))
        data[j].append((e, vj))
        data[e].append((i, -vi))
        data[e].append((j, -vj))

    topology.neighbor_list.set_data(data)
    topology.calculate_properties()

    if not topology.check_validity():
        raise ValueError("Invalid explicit topology")

    def prepare_bbs(marker_scale):
        # Reuse the shared connectivity and writer without patching its globals.
        # Set the X coordinate records directly from the original He vectors to
        # preserve the notebook arithmetic (PORMAKE orientation is sensitive to
        # tiny coordinate changes for symmetric fragments).
        nodes = []
        with TemporaryDirectory() as prepared_dir:
            for index, (path, atoms) in enumerate(
                zip(paths, inputs, strict=False)
            ):
                updated, x_indices = preparation._replace_atoms(
                    atoms, "He", "H"
                )
                bonds = preparation._infer_connectivity(updated)
                folder = Path(prepared_dir) / str(index)
                folder.mkdir()
                prepared_path = folder / path.name
                preparation._write_xyz_with_bonds(
                    updated, x_indices, bonds, str(prepared_path)
                )
                records = prepared_path.read_text().splitlines(keepends=True)
                for x in x_indices:
                    neighbours = [
                        j if i == x else i for i, j in bonds if x in (i, j)
                    ]
                    if len(neighbours) != 1 or neighbours[0] in x_indices:
                        raise ValueError(
                            f"He marker {x} in {path} must have exactly one real neighbour"
                        )
                    origin = atoms.positions[neighbours[0]]
                    position = origin + marker_scale * (
                        atoms.positions[x] - origin
                    )
                    records[x + 2] = (
                        f"X {position[0]} {position[1]} {position[2]}\n"
                    )
                prepared_path.write_text("".join(records))
                nodes.append(pm.BuildingBlock(str(prepared_path)))
        by_coordination = {
            len(node.connection_point_indices): node for node in nodes
        }
        prepared = [None] * topology.n_slots
        for i in topology.node_indices:
            prepared[i] = by_coordination[topology.cn[i]]
        return prepared

    # Connector J is the fourth CGD EDGE entry: slots G...N correspond to 6...13.
    j_edge_slot = 9

    def finish_orientation(framework):
        atom_offset = 0
        for slot, block in enumerate(framework.info["located_bbs"]):
            if block is None:
                continue
            oriented_block = _orient_two_connected(
                block, slot, framework.info["topology"]
            )
            keep = np.array(oriented_block.atoms.get_chemical_symbols()) != "X"
            atom_count = int(keep.sum())
            framework.atoms.positions[
                atom_offset : atom_offset + atom_count
            ] = oriented_block.atoms.positions[keep]
            atom_offset += atom_count
        return framework

    # None means that PORMAKE performs its normal cell fit.  The embedding solve
    # below temporarily supplies a and b after the initial framework is measured.
    embedding_a = None
    embedding_b = None

    def build_framework(marker_scale):
        local_bbs = prepare_bbs(marker_scale)
        builder = pm.Builder(
            scaler=_FixedEmbeddingScaler(
                forced_a=embedding_a, forced_b=embedding_b
            )
        )
        framework = builder.build(topology=topology, bbs=local_bbs, wrap=False)
        return finish_orientation(framework), local_bbs

    def node_atom_maps(framework):
        maps = {}
        start = 0
        for slot, block in enumerate(framework.info["located_bbs"]):
            if block is None:
                continue
            real = [
                i
                for i, symbol in enumerate(block.atoms.get_chemical_symbols())
                if symbol != "X"
            ]
            if slot in framework.info["topology"].node_indices:
                maps[slot] = {local: start + k for k, local in enumerate(real)}
            start += len(real)
        return maps

    def connector_real_atom(framework, node, edge_slot):
        """Return the real atom attached to a node's X marker for one edge."""
        top = framework.info["topology"]
        block = framework.info["located_bbs"][node]
        incidence = next(
            k
            for k, neighbour in enumerate(top.neighbor_list[node])
            if neighbour.index == edge_slot
        )
        permutation = framework.info["permutations"][node]
        x_atom = int(block.connection_point_indices[permutation][incidence])
        bonded = []
        for i, j in block.bonds:
            if x_atom not in (i, j):
                continue
            real = int(j if i == x_atom else i)
            if block.atoms[real].symbol != "X":
                bonded.append(real)
        if len(bonded) != 1:
            raise ValueError(
                f"Node {node} edge {edge_slot} must have one real atom at its X marker"
            )
        return node_atom_maps(framework)[node][bonded[0]]

    def connector_pair(framework, edge_slot):
        """Return (4c node, 2c node, 4c atom, 2c atom) for one topology edge."""
        top = framework.info["topology"]
        neighbours = top.neighbor_list[edge_slot]
        if len(neighbours) != 2:
            raise ValueError(f"Edge {edge_slot} does not have two endpoints")
        nodes = [int(n.index) for n in neighbours]
        four = [node for node in nodes if int(top.cn[node]) == 4]
        two = [node for node in nodes if int(top.cn[node]) == 2]
        if len(four) != 1 or len(two) != 1:
            raise ValueError(f"Edge {edge_slot} is not a 4c--2c connector")
        node4, node2 = four[0], two[0]
        return (
            node4,
            node2,
            connector_real_atom(framework, node4, edge_slot),
            connector_real_atom(framework, node2, edge_slot),
        )

    def connector_records(framework):
        return [
            (int(edge), *connector_pair(framework, int(edge)))
            for edge in framework.info["topology"].edge_indices
        ]

    def edge_direction(framework, edge_slot, node4, node2):
        """Physical topology direction from the 4c node toward the 2c node."""
        top = framework.info["topology"]
        rec4 = next(
            n for n in top.neighbor_list[edge_slot] if n.index == node4
        )
        rec2 = next(
            n for n in top.neighbor_list[edge_slot] if n.index == node2
        )
        return np.asarray(
            rec2.distance_vector - rec4.distance_vector, dtype=float
        )

    def j_real_atoms(framework):
        return connector_pair(framework, j_edge_slot)[2:]

    def connector_vector(framework, edge_slot):
        _, _, atom4, atom2 = connector_pair(framework, edge_slot)
        return np.asarray(
            framework.atoms.get_distance(atom4, atom2, mic=True, vector=True),
            dtype=float,
        )

    def target_connector_vector(framework, edge_slot):
        """Construct the desired real-atom vector from length and topology direction."""
        node4, node2, atom4, atom2 = connector_pair(framework, edge_slot)
        current = np.asarray(
            framework.atoms.get_distance(atom4, atom2, mic=True, vector=True),
            dtype=float,
        )
        direction = edge_direction(framework, edge_slot, node4, node2)
        if np.linalg.norm(direction[:2]) < 1e-8:
            raise ValueError(
                f"Edge {edge_slot} has no in-plane topology direction"
            )
        if target_j_bond <= abs(current[2]):
            raise ValueError(
                "target_j_bond must exceed the connector z component"
            )
        xy_length = np.sqrt(target_j_bond**2 - current[2] ** 2)
        angle = np.radians(target_connector_angle)
        signs = np.sign(direction[:2])
        signs[signs == 0] = 1.0
        return np.array(
            [
                signs[0] * xy_length * np.cos(angle),
                signs[1] * xy_length * np.sin(angle),
                current[2],
            ]
        )

    def connector_summary(framework):
        rows = []
        for edge, node4, node2, _atom4, _atom2 in connector_records(framework):
            vector = connector_vector(framework, edge)
            rows.append(
                {
                    "edge": edge,
                    "node4": node4,
                    "node2": node2,
                    "vector": vector,
                    "length": float(np.linalg.norm(vector)),
                    "angle": float(
                        np.degrees(np.arctan2(abs(vector[1]), abs(vector[0])))
                    ),
                }
            )
        return rows

    def minimum_nonbonded_distance(framework):
        """Return the shortest MIC distance excluding declared framework bonds."""
        bonded = {tuple(sorted(map(int, bond))) for bond in framework.bonds}
        minimum = np.inf
        pair = None
        for i in range(len(framework.atoms)):
            for j in range(i + 1, len(framework.atoms)):
                if (i, j) in bonded:
                    continue
                distance = float(framework.atoms.get_distance(i, j, mic=True))
                if distance < minimum:
                    minimum, pair = distance, (i, j)
        return minimum, pair

    def solve_embedding_cell(framework):
        """Fit a and b so the real J vector reaches the requested length and angle."""
        nonlocal embedding_a, embedding_b
        base_a, base_b = map(float, framework.atoms.cell.lengths()[:2])
        base_vector = connector_vector(framework, j_edge_slot)
        target = target_connector_vector(framework, j_edge_slot)
        print(
            "Initial J vector:",
            np.round(base_vector, 3),
            "target:",
            np.round(target, 3),
        )

        # The finite differences are cheap, remain inside the existing scaler, and
        # account for the fact that a and b affect the real atoms indirectly through
        # PORMAKE placement.  Recompute once more if the first Newton step is not
        # within the requested tolerances.
        candidate_a, candidate_b = base_a, base_b
        for iteration in range(max_embedding_iterations):
            embedding_a = None
            embedding_b = None
            if iteration == 0:
                reference = framework
            else:
                embedding_a, embedding_b = candidate_a, candidate_b
                reference, _ = build_framework(x_scale)
            ref_vector = connector_vector(reference, j_edge_slot)
            ref_a, ref_b = map(float, reference.atoms.cell.lengths()[:2])
            step_a = max(float(embedding_fd_a), 0.02 * ref_a)
            step_b = max(float(embedding_fd_b), 0.02 * ref_b)

            embedding_a, embedding_b = ref_a + step_a, ref_b
            trial_a, _ = build_framework(x_scale)
            va = connector_vector(trial_a, j_edge_slot)
            embedding_a, embedding_b = ref_a, ref_b + step_b
            trial_b, _ = build_framework(x_scale)
            vb = connector_vector(trial_b, j_edge_slot)
            embedding_a = None
            embedding_b = None

            jacobian = np.column_stack(
                (
                    (va[:2] - ref_vector[:2]) / step_a,
                    (vb[:2] - ref_vector[:2]) / step_b,
                )
            )
            residual = target[:2] - ref_vector[:2]
            if np.linalg.matrix_rank(jacobian) < 2:
                raise RuntimeError(
                    "The connector vector does not constrain both a and b"
                )
            delta = np.linalg.solve(jacobian, residual)
            candidate = np.array([ref_a, ref_b]) + delta
            candidate[0] = np.clip(candidate[0], 0.5 * ref_a, 1.5 * ref_a)
            candidate[1] = np.clip(candidate[1], 0.5 * ref_b, 1.5 * ref_b)
            candidate_a, candidate_b = map(float, candidate)
            embedding_a, embedding_b = candidate_a, candidate_b
            fitted, _ = build_framework(x_scale)
            fitted_vector = connector_vector(fitted, j_edge_slot)
            fitted_length = float(np.linalg.norm(fitted_vector))
            fitted_angle = float(
                np.degrees(
                    np.arctan2(abs(fitted_vector[1]), abs(fitted_vector[0]))
                )
            )
            print(
                f"embedding iteration {iteration}: a={candidate_a:.3f}, b={candidate_b:.3f}, "
                f"J={fitted_length:.3f} Å, angle={fitted_angle:.1f}°"
            )
            framework = fitted
            if (
                abs(fitted_length - target_j_bond) <= bond_tolerance
                and 45.0 <= fitted_angle <= 85.0
            ):
                break
        else:
            raise RuntimeError(
                "Cell embedding did not reach the requested connector geometry"
            )
        return framework

    def measure_j_bond(framework):
        i, j = j_real_atoms(framework)
        return float(framework.atoms.get_distance(i, j, mic=True))

    # Tune the marker scale from the actual fused framework bond.
    bond_history = []
    for bond_iteration in range(max_bond_iterations + 1):
        cof, _bbs = build_framework(x_scale)
        measured_bond = measure_j_bond(cof)
        print(
            f"x_scale iteration {bond_iteration}: x_scale={x_scale:.5f}, "
            f"J bond={measured_bond:.4f} Å"
        )
        if abs(measured_bond - target_j_bond) <= bond_tolerance:
            break
        if bond_iteration == max_bond_iterations:
            raise RuntimeError(
                "x_scale did not reach the requested J-bond length"
            )
        proposed = x_scale * target_j_bond / measured_bond
        if len(bond_history) >= 2:
            previous_scale, previous_bond = bond_history[-1]
            older_scale, older_bond = bond_history[-2]
            slope = (previous_bond - older_bond) / (
                previous_scale - older_scale
            )
            if np.isfinite(slope) and slope > 1e-6:
                secant = (
                    previous_scale + (target_j_bond - previous_bond) / slope
                )
                if secant > 0:
                    proposed = secant
        bond_history.append((x_scale, measured_bond))
        x_scale = float(np.clip(proposed, 0.5 * x_scale, 1.5 * x_scale))

    cof = solve_embedding_cell(cof)

    def wrapped_positions(framework):
        wrapped = framework.atoms.copy()
        wrapped.wrap()
        return (
            wrapped.positions.copy(),
            np.asarray(wrapped.cell),
            np.linalg.inv(np.asarray(wrapped.cell)),
        )

    def unit_cell_components(framework):
        # Find within-cell graph components and the periodic bonds between them.
        positions, cell, inv_cell = wrapped_positions(framework)
        n_atoms = len(positions)
        graph = [set() for _ in range(n_atoms)]
        cross_edges = []
        component_bonds = []
        for bond in framework.bonds:
            i, j = map(int, bond)
            fractional_delta = (positions[j] - positions[i]) @ inv_cell
            image = np.rint(fractional_delta).astype(int)
            if np.any(image != 0):
                cross_edges.append((i, j, image))
            else:
                graph[i].add(j)
                graph[j].add(i)
                component_bonds.append((i, j))

        unseen = set(range(n_atoms))
        components = []
        component_of = {}
        while unseen:
            seed = unseen.pop()
            component = {seed}
            queue = [seed]
            while queue:
                i = queue.pop()
                for j in graph[i]:
                    if j in unseen:
                        unseen.remove(j)
                        component.add(j)
                        queue.append(j)
            index = len(components)
            for i in component:
                component_of[i] = index
            components.append(sorted(component))

        component_graph = [set() for _ in components]
        for i, j, _image in cross_edges:
            ci, cj = component_of[i], component_of[j]
            if ci != cj:
                component_graph[ci].add(cj)
                component_graph[cj].add(ci)
        return positions, cell, components, component_of, component_graph

    connector_rows = connector_summary(cof)
    for row in connector_rows:
        print(
            f"connector edge {row['edge']}: "
            f"length={row['length']:.3f} Å, angle={row['angle']:.1f}°"
        )
        if not (45.0 <= row["angle"] <= 85.0):
            raise RuntimeError(
                f"Connector edge {row['edge']} has an invalid angle"
            )
        if abs(row["length"] - target_j_bond) > bond_tolerance:
            raise RuntimeError(
                f"Connector edge {row['edge']} has an invalid length"
            )
    raw_initial_positions = cof.atoms.positions.copy()
    positions, cell, components, component_of, component_graph = (
        unit_cell_components(cof)
    )
    initial_bond_images = {}
    initial_inv_cell = np.linalg.inv(cell)
    for bond in cof.bonds:
        i, j = map(int, bond)
        initial_bond_images[(i, j)] = np.rint(
            (raw_initial_positions[j] - raw_initial_positions[i])
            @ initial_inv_cell
        ).astype(int)
    largest = sorted(
        range(len(components)), key=lambda i: len(components[i]), reverse=True
    )
    if len(largest) < 2:
        raise RuntimeError(
            "Could not identify two large within-cell ladder components"
        )
    left_component, right_component = largest[:2]
    centres = [
        float(np.mean(positions[components[i], 0]))
        for i in (left_component, right_component)
    ]
    if centres[0] > centres[1]:
        left_component, right_component = right_component, left_component
    print("Component sizes:", [len(components[i]) for i in largest[:8]])
    print(
        "Selected left/right components:",
        len(components[left_component]),
        len(components[right_component]),
    )

    def connected_in_component_graph(start, goal):
        seen, queue = {start}, [start]
        while queue:
            current = queue.pop()
            if current == goal:
                return True
            for neighbour in component_graph[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return False

    if connected_in_component_graph(left_component, right_component):
        raise RuntimeError(
            "The two largest components are periodically covalently connected; "
            "they are pieces of one ladder, so spacing adjustment is unsafe."
        )

    def assign_component_families():
        # Assign overhang components to the nearest main component.
        labels = {left_component: "left", right_component: "right"}
        changed = True
        while changed:
            changed = False
            for component in range(len(components)):
                if component in labels:
                    continue
                neighbours = {
                    labels[n]
                    for n in component_graph[component]
                    if n in labels
                }
                if len(neighbours) > 1:
                    raise RuntimeError(
                        "A boundary overhang is bonded to both ladders"
                    )
                if len(neighbours) == 1:
                    labels[component] = next(iter(neighbours))
                    changed = True
        left_x = np.mean(positions[components[left_component], 0])
        right_x = np.mean(positions[components[right_component], 0])
        for component in range(len(components)):
            if component in labels:
                continue
            centre = np.mean(positions[components[component], 0])
            labels[component] = (
                "left"
                if abs(centre - left_x) <= abs(centre - right_x)
                else "right"
            )
        return labels

    component_labels = assign_component_families()
    initial_framework_bond_distances = np.array(
        [
            cof.atoms.get_distance(int(i), int(j), mic=True)
            for i, j in cof.bonds
        ]
    )
    print(
        f"Initial framework bond-distance range: {initial_framework_bond_distances.min():.3f}-{initial_framework_bond_distances.max():.3f} Å"
    )

    def preserve_a_periodic_bonds(positions, framework, delta):
        # Translate split components so changing a leaves every a-crossing bond fixed.
        offsets = {0: 0.0}
        constraints = []
        for bond_i, bond_j in framework.bonds:
            i, j = int(bond_i), int(bond_j)
            image = initial_bond_images[(i, j)]
            ci, cj = component_of[i], component_of[j]
            if ci == cj:
                if image[0] != 0:
                    raise RuntimeError(
                        "A component contains an internal a-periodic bond"
                    )
                continue
            # If the cell grows by delta, a bond with image n keeps its
            # Cartesian MIC vector when component j moves by +n*delta relative
            # to component i.
            constraints.append((ci, cj, float(image[0]) * delta))
        # Solve each connected constraint subgraph from one fixed root. Components
        # without a periodic constraint remain at zero offset.
        for seed in range(len(components)):
            if seed in offsets:
                continue
            offsets[seed] = 0.0
            changed = True
            while changed:
                changed = False
                for ci, cj, difference in constraints:
                    if ci in offsets and cj not in offsets:
                        offsets[cj] = offsets[ci] + difference
                        changed = True
                    elif cj in offsets and ci not in offsets:
                        offsets[ci] = offsets[cj] - difference
                        changed = True
                    elif (
                        ci in offsets
                        and abs(offsets[cj] - offsets[ci] - difference) > 1e-6
                    ):
                        raise RuntimeError(
                            "Periodic a-bond constraints are inconsistent"
                        )
        for component, offset in offsets.items():
            positions[np.asarray(components[component], dtype=int), 0] += (
                offset
            )
        return positions

    def place_safe_components(framework):
        # Rigidly separate both main components before the final gap measurement.
        wrapped_positions, cell, _, _, _ = unit_cell_components(framework)
        positions = framework.atoms.positions.copy()
        left_atoms = np.array(components[left_component], dtype=int)
        right_atoms = np.array(components[right_component], dtype=int)
        left_min, left_max = (
            wrapped_positions[left_atoms, 0].min(),
            wrapped_positions[left_atoms, 0].max(),
        )
        right_min, right_max = (
            wrapped_positions[right_atoms, 0].min(),
            wrapped_positions[right_atoms, 0].max(),
        )
        left_width, right_width = left_max - left_min, right_max - right_min
        new_a = left_width + right_width + 2.0 * safe_initial_gap
        left_target = safe_initial_gap / 2.0
        right_target = left_target + left_width + safe_initial_gap
        left_shift = left_target - left_min
        right_shift = right_target - right_min
        for component, label in component_labels.items():
            atoms = np.array(components[component], dtype=int)
            positions[atoms, 0] += (
                left_shift if label == "left" else right_shift
            )
        new_cell = cell.copy()
        new_cell[0, 0] = new_a
        positions = preserve_a_periodic_bonds(
            positions, framework, new_a - float(cell[0, 0])
        )
        framework.atoms.set_cell(new_cell, scale_atoms=False)
        framework.atoms.set_positions(positions)
        return framework

    cof = place_safe_components(cof)

    def _unwrap_component_x(x, a):
        # Lift one component onto its shortest contiguous periodic x-branch.
        wrapped = np.mod(np.asarray(x, dtype=float), a)
        if len(wrapped) <= 1:
            return wrapped
        ordered = np.sort(wrapped)
        gaps = np.diff(ordered, append=ordered[0] + a)
        cut = (int(np.argmax(gaps)) + 1) % len(ordered)
        start = ordered[cut]
        lifted = wrapped.copy()
        lifted[lifted < start] += a
        span = float(lifted.max() - lifted.min())
        if span >= a - 1e-6:
            raise RuntimeError(
                "A ladder component does not fit inside the trial a cell"
            )
        return lifted

    def measure_ladder_gaps(
        framework, position_override=None, cell_override=None
    ):
        if position_override is None:
            positions, cell, _, _, _ = unit_cell_components(framework)
        else:
            positions = np.asarray(position_override).copy()
            cell = np.asarray(
                framework.atoms.cell
                if cell_override is None
                else cell_override
            )
        a, b = cell[0, 0], cell[1, 1]
        left = np.concatenate(
            [
                np.asarray(components[i], dtype=int)
                for i, label in component_labels.items()
                if label == "left"
            ]
        )
        right = np.concatenate(
            [
                np.asarray(components[i], dtype=int)
                for i, label in component_labels.items()
                if label == "right"
            ]
        )
        left_x_all = _unwrap_component_x(positions[left, 0], a)
        right_x_all = _unwrap_component_x(positions[right, 0], a)
        # Lift the complete ladder objects, including all pieces connected through
        # periodic bonds, onto one contiguous branch before making x-copies.
        left_centre = float(np.mean(left_x_all))
        right_centre = float(np.mean(right_x_all))
        right_shift = np.floor((left_centre - right_centre) / a) + 1.0
        right_x_all = right_x_all + right_shift * a
        middle_values = []
        periodic_values = []
        for sample in (np.arange(n_gap_slices) + 0.5) * b / n_gap_slices:
            width = b / (2.0 * n_gap_slices)
            for _ in range(2):
                dy_left = np.abs(
                    ((positions[left, 1] - sample + 0.5 * b) % b) - 0.5 * b
                )
                dy_right = np.abs(
                    ((positions[right, 1] - sample + 0.5 * b) % b) - 0.5 * b
                )
                left_selected = dy_left <= width
                right_selected = dy_right <= width
                if np.any(left_selected) and np.any(right_selected):
                    left_x = left_x_all[left_selected]
                    right_x = right_x_all[right_selected]
                    left_images = [left_x + k * a for k in (-1, 0, 1)]
                    right_images = [right_x + k * a for k in (-1, 0, 1)]
                    # The chosen branches put the middle pair next to one another;
                    # the second pair is the explicit +a copy of the left object.
                    middle_values.append(
                        right_images[1].min() - left_images[1].max()
                    )
                    periodic_values.append(
                        left_images[2].min() - right_images[1].max()
                    )
                    break
                width *= 2.0
        if len(middle_values) < max(3, n_gap_slices // 2):
            raise RuntimeError(
                "Too few y slices contain atoms from both ladder components"
            )
        return (
            float(np.mean(middle_values)),
            float(np.mean(periodic_values)),
            np.asarray(middle_values),
            np.asarray(periodic_values),
        )

    safe_cell = np.asarray(cof.atoms.cell).copy()
    safe_a = float(safe_cell[0, 0])
    _measured_middle, _measured_periodic, middle_values, periodic_values = (
        measure_ladder_gaps(cof)
    )
    print("Measured middle y-slice gaps:", np.round(middle_values, 3))
    print("Measured periodic y-slice gaps:", np.round(periodic_values, 3))

    positions = cof.atoms.positions.copy()
    left_family = np.concatenate(
        [
            np.asarray(components[i], dtype=int)
            for i, label in component_labels.items()
            if label == "left"
        ]
    )
    right_family = np.concatenate(
        [
            np.asarray(components[i], dtype=int)
            for i, label in component_labels.items()
            if label == "right"
        ]
    )
    middle_delta = target_interladder_gap - float(np.min(middle_values))
    positions[left_family, 0] -= middle_delta / 2.0
    positions[right_family, 0] += middle_delta / 2.0
    (
        shifted_middle,
        shifted_periodic,
        shifted_middle_values,
        shifted_periodic_values,
    ) = measure_ladder_gaps(cof, positions, safe_cell)
    print(
        f"Safe-cell gap floors after symmetric shift: "
        f"middle={shifted_middle_values.min():.3f} Å, "
        f"periodic={shifted_periodic_values.min():.3f} Å"
    )
    if shifted_middle_values.min() < target_interladder_gap - gap_tolerance:
        raise RuntimeError("Could not establish a safe internal ladder gap")
    if shifted_periodic_values.min() < target_interladder_gap - gap_tolerance:
        raise RuntimeError(
            "The provisional cell is already too small at the periodic boundary"
        )

    def trial_a(delta):
        trial_cell = safe_cell.copy()
        trial_cell[0, 0] = safe_a + delta
        if trial_cell[0, 0] <= 0:
            return None
        trial_positions = preserve_a_periodic_bonds(
            positions.copy(), cof, delta
        )
        middle, periodic, middle_slice, periodic_slice = measure_ladder_gaps(
            cof, trial_positions, trial_cell
        )
        return (
            trial_cell,
            trial_positions,
            middle,
            periodic,
            middle_slice,
            periodic_slice,
        )

    # Find the smallest a that keeps every periodic slice above the requested
    # floor. A larger resulting average gap is acceptable and avoids overlap.
    good_delta = 0.0
    good_trial = (
        safe_cell,
        positions.copy(),
        shifted_middle,
        shifted_periodic,
        shifted_middle_values,
        shifted_periodic_values,
    )
    bad_delta = None
    step = max(1.0, 0.05 * safe_a)
    for _ in range(20):
        candidate_delta = good_delta - step
        candidate = trial_a(candidate_delta)
        if candidate is None:
            bad_delta = candidate_delta
            break
        if (
            candidate[5].min() >= target_interladder_gap
            and candidate[4].min() >= target_interladder_gap - gap_tolerance
        ):
            good_delta, good_trial = candidate_delta, candidate
            step *= 2.0
        else:
            bad_delta = candidate_delta
            break
    if bad_delta is None:
        raise RuntimeError(
            "Could not bracket a safe minimum periodic cell length"
        )
    for _ in range(35):
        candidate_delta = 0.5 * (good_delta + bad_delta)
        candidate = trial_a(candidate_delta)
        if candidate is not None and (
            candidate[5].min() >= target_interladder_gap
            and candidate[4].min() >= target_interladder_gap - gap_tolerance
        ):
            good_delta, good_trial = candidate_delta, candidate
        else:
            bad_delta = candidate_delta

    (
        new_cell,
        positions,
        final_middle,
        final_periodic,
        final_middle_values,
        final_periodic_values,
    ) = good_trial
    cof.atoms.set_cell(new_cell, scale_atoms=False)
    cof.atoms.set_positions(positions)
    final_bond = measure_j_bond(cof)
    bond_distances = np.array(
        [
            cof.atoms.get_distance(int(i), int(j), mic=True)
            for i, j in cof.bonds
        ]
    )
    print(
        f"Framework bond-distance range: {bond_distances.min():.3f}-{bond_distances.max():.3f} Å"
    )
    if bond_distances.max() > 5.5:
        bad = int(np.argmax(bond_distances))
        print(
            "Longest bond:",
            tuple(map(int, cof.bonds[bad])),
            bond_distances[bad],
        )
        bad_i, bad_j = map(int, cof.bonds[bad])
        bad_ci, bad_cj = component_of[bad_i], component_of[bad_j]
        print(
            "Longest-bond components:",
            bad_ci,
            bad_cj,
            component_labels.get(bad_ci),
            component_labels.get(bad_cj),
        )
        print(
            "Longest-bond initial image:",
            initial_bond_images.get((bad_i, bad_j)),
        )
    print("Final middle y-slice gaps:", np.round(final_middle_values, 3))
    print("Final periodic y-slice gaps:", np.round(final_periodic_values, 3))
    print(
        f"Final J bond={final_bond:.3f} Å; final gaps: "
        f"middle mean={final_middle:.3f} Å (min={final_middle_values.min():.3f}), "
        f"periodic mean={final_periodic:.3f} Å (min={final_periodic_values.min():.3f})"
    )
    if final_middle_values.min() < target_interladder_gap - gap_tolerance:
        raise RuntimeError(
            "Final middle ladder gap fell below the safety floor"
        )
    if final_periodic_values.min() < target_interladder_gap - gap_tolerance:
        raise RuntimeError(
            "Final periodic ladder gap fell below the safety floor"
        )
    if abs(final_bond - target_j_bond) > bond_tolerance:
        raise RuntimeError("Final J bond missed the requested tolerance")

    final_connector_rows = connector_summary(cof)
    for row in final_connector_rows:
        if not (45.0 <= row["angle"] <= 85.0):
            raise RuntimeError(
                f"Final connector edge {row['edge']} has an invalid angle"
            )
        if abs(row["length"] - target_j_bond) > bond_tolerance:
            raise RuntimeError(
                f"Final connector edge {row['edge']} has an invalid length"
            )
    minimum_distance, minimum_pair = minimum_nonbonded_distance(cof)
    print(
        f"Minimum final nonbonded MIC distance: {minimum_distance:.3f} Å "
        f"(pair={minimum_pair})"
    )
    if minimum_distance < min_nonbonded_distance:
        raise RuntimeError(
            f"Final nonbonded overlap detected: {minimum_distance:.3f} Å "
            f"for pair {minimum_pair}"
        )

    cof.wrap()
    return cof
