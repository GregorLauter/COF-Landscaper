"""Build fully periodic 3D COFs with PORMAKE's standard assembly pipeline."""

import os
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import ase.io
import numpy as np
import pormake as pm
from ase.atoms import Atoms
from pormake.building_block import BuildingBlock
from pormake.topology import Topology
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifFile, str2float

from . import build_cof_2d as preparation


def _translated_bond_image(code: str, wrapping: np.ndarray) -> str:
    """Adjust a P1 bond image for the wrapping of its translated endpoints."""
    if code == ".":
        image = np.zeros(3, dtype=int)
    elif len(code) == 5 and code.startswith("1_") and code[2:].isdigit():
        image = np.array([int(value) - 5 for value in code[2:]])
    else:
        raise ValueError(f"Unsupported P1 bond symmetry code: {code!r}")
    image += wrapping
    if not np.any(image):
        return "."
    digits = image + 5
    if np.any((digits < 0) | (digits > 9)):
        raise ValueError("Translated bond image cannot be represented in CIF")
    return "1_" + "".join(map(str, digits))


def _interpenetrated_cif(
    source: Path, shift: np.ndarray
) -> tuple[str, int, float]:
    """Duplicate a builder P1 CIF, preserving atom properties and periodic bonds."""
    cif = CifFile.from_file(source)
    if len(cif.data) != 1:
        raise ValueError("Interpenetration requires a single P1 framework CIF")
    block = next(iter(cif.data.values()))
    data = block.data
    if str(data.get("_symmetry_Int_Tables_number", "")) != "1":
        raise ValueError("Interpenetration requires a P1 CIF from BuildCOF3D")
    labels = list(data["_atom_site_label"])
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Framework atom labels must be nonempty and unique")
    coordinate_keys = [f"_atom_site_fract_{axis}" for axis in "xyz"]
    positions = np.array(
        [[str2float(value) for value in data[key]] for key in coordinate_keys]
    ).T
    cell_parameters = [
        str2float(data[f"_cell_{key}"])
        for key in (
            "length_a",
            "length_b",
            "length_c",
            "angle_alpha",
            "angle_beta",
            "angle_gamma",
        )
    ]
    lattice = Lattice.from_parameters(*cell_parameters)
    if (
        positions.shape != (len(labels), 3)
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(lattice.matrix))
        or abs(np.linalg.det(lattice.matrix)) < 1e-10
    ):
        raise ValueError("Framework has invalid coordinates or cell geometry")
    wrapping = np.floor(positions + shift).astype(int)
    translated = positions + shift - wrapping
    minimum_distance = min(
        float(
            np.min(
                lattice.get_all_distances(
                    positions[start : start + 256], translated
                )
            )
        )
        for start in range(0, len(labels), 256)
    )
    if minimum_distance < 1e-3:
        raise ValueError(
            "Translation produces coincident atoms between nets "
            f"(minimum distance {minimum_distance:.6f} Angstrom)"
        )
    copied_labels = [f"{label}_copy" for label in labels]
    while set(labels).intersection(copied_labels):
        copied_labels = [f"{label}_copy" for label in copied_labels]
    label_map = dict(zip(labels, copied_labels, strict=True))
    indices = {label: index for index, label in enumerate(labels)}
    for loop in block.loops:
        is_atoms = "_atom_site_label" in loop
        is_bonds = "_geom_bond_atom_site_label_1" in loop
        if not (is_atoms or is_bonds):
            continue
        copied = {key: list(data[key]) for key in loop}
        if is_atoms:
            copied["_atom_site_label"] = copied_labels
            for axis, key in enumerate(coordinate_keys):
                copied[key] = [
                    f"{value:.15g}" for value in translated[:, axis]
                ]
        else:
            first_key = "_geom_bond_atom_site_label_1"
            second_key = "_geom_bond_atom_site_label_2"
            image_key = "_geom_bond_site_symmetry_2"
            first_images = data.get(
                "_geom_bond_site_symmetry_1", ["."] * len(data[first_key])
            )
            if any(code not in (".", "1_555") for code in first_images):
                raise ValueError(
                    "P1 bond first endpoints must use the identity image"
                )
            copied[image_key] = [
                _translated_bond_image(
                    code, wrapping[indices[second]] - wrapping[indices[first]]
                )
                for first, second, code in zip(
                    data[first_key],
                    data[second_key],
                    data[image_key],
                    strict=True,
                )
            ]
            for key in (first_key, second_key):
                copied[key] = [label_map[label] for label in data[key]]
        for key in loop:
            data[key] = list(data[key]) + copied[key]
    comments = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("data_"):
            break
        if line.lstrip().startswith("#"):
            comments.append(line)
    comments.append(
        f"# Added one translated net; fractional shift: {tuple(shift)}"
    )
    cif.comment = "\n".join(comments)
    return str(cif), 2 * len(labels), minimum_distance


def _periodic_dimension(topology: Topology) -> int:
    """Return the lowest periodic dimension of the net's connected components.

    Integer translations accumulated around graph cycles distinguish a truly
    3D network from a buckled layer or several disconnected lower-dimensional
    networks. Cartesian neighbour directions alone cannot make that distinction.
    """
    adjacency: dict[int, list[tuple[int, np.ndarray]]] = {
        int(node): [] for node in topology.node_indices
    }
    positions = topology.atoms.positions
    inverse_cell = np.linalg.inv(np.asarray(topology.atoms.cell))
    for edge in topology.edge_indices:
        first, second = topology.neighbor_list[edge]
        i, j = int(first.index), int(second.index)
        displacement = second.distance_vector - first.distance_vector
        image = np.rint(
            (displacement - (positions[j] - positions[i])) @ inverse_cell
        ).astype(int)
        adjacency[i].append((j, image))
        adjacency[j].append((i, -image))

    images: dict[int, np.ndarray] = {}
    dimensions = []
    for root in adjacency:
        if root in images:
            continue
        images[root] = np.zeros(3, dtype=int)
        pending = [root]
        cycles = []
        while pending:
            node = pending.pop()
            for neighbour, offset in adjacency[node]:
                target = images[node] + offset
                if neighbour not in images:
                    images[neighbour] = target
                    pending.append(neighbour)
                else:
                    cycle = target - images[neighbour]
                    if np.any(cycle):
                        cycles.append(cycle)
        dimension = (
            int(np.linalg.matrix_rank(np.asarray(cycles))) if cycles else 0
        )
        dimensions.append(dimension)
    return min(dimensions, default=0)


def _load_topology(topo: str) -> Topology:
    """Read a bundled PORMAKE topology without writing installation caches."""
    database = pm.Database()
    if topo not in database.topo_list:
        raise ValueError(f"Unknown PORMAKE topology: {topo!r}")
    topology = pm.Topology(database.topo_dir / f"{topo}.cgd")
    dimension = _periodic_dimension(topology)
    if dimension != 3:
        raise ValueError(
            f"Topology {topo!r} has periodic dimension {dimension}; "
            "BuildCOF3D requires a 3D network in every connected component."
        )
    return topology


def _node_requirements(topology: Topology) -> dict[int, int]:
    """Map each PORMAKE node type to its required connection count."""
    return {
        int(topology.node_types[node]): int(topology.cn[node])
        for node in topology.node_indices
    }


def _prepare_block(
    path: str | os.PathLike[str],
    connection_count: int,
    folder: Path,
    label: str,
) -> BuildingBlock:
    """Validate a He-marked fragment and prepare it in its own directory."""
    source = Path(path)
    if not source.is_file() or source.suffix.lower() != ".xyz":
        raise FileNotFoundError(f"Input XYZ file not found: {source}")
    atoms = cast("Atoms", ase.io.read(source))
    if not np.all(np.isfinite(atoms.positions)):
        raise ValueError(f"{label} has nonfinite coordinates: {source}")
    count = atoms.get_chemical_symbols().count("He")
    if count != connection_count:
        raise ValueError(
            f"{label} requires exactly {connection_count} He connection markers; "
            f"found {count} in {source}."
        )
    preparation._prepare_xyz_files([str(source)], str(folder))
    return pm.BuildingBlock(str(folder / f"{source.stem}.xyz"))


class BuildCOF3D:
    """Construct a 3D COF using bundled PORMAKE topologies and standard fitting.

    Node inputs are keyed by PORMAKE node type. Linker inputs are keyed by pairs
    of endpoint node types. Missing or ``None`` linker assignments produce direct
    node-to-node connections, following PORMAKE's convention. Call
    :meth:`describe_topology` to inspect those types before choosing inputs.

    Fragments use the existing COF-Landscaper He marker convention and its 0.8
    marker-distance adjustment. Assembly uses plain ``pormake.Builder`` with
    full 3D cell scaling and molecular orientation. The CIF retains that cell.

    Example::

        builder = cl.BuildCOF3D()
        paths = builder.build(
            topo="dia", cof_name="my_cof",
            input_nodes={0: "tetrahedral_node.xyz"},
            input_linkers={(0, 0): "linker.xyz"},
        )
    """

    def describe_topology(self, topo: str) -> dict[str, Any]:
        """Return node connectivity, edge types, and cell-slot counts for a net.

        Args:
            topo: Name of a 3D topology in the installed PORMAKE database.

        Returns:
            Dictionary with ``topology``, ``dimensionality``, ``node_types``
            (type-to-connectivity mapping), ``edge_types`` (endpoint pairs),
            ``node_count``, and ``edge_count``.

        Raises:
            ValueError: If the topology is unknown or is not fully 3D.
        """
        topology = _load_topology(topo)
        return {
            "topology": topo,
            "dimensionality": 3,
            "node_types": _node_requirements(topology),
            "edge_types": [
                tuple(map(int, pair)) for pair in topology.unique_edge_types
            ],
            "node_count": len(topology.node_indices),
            "edge_count": len(topology.edge_indices),
        }

    def build(
        self,
        topo: str,
        cof_name: str,
        input_nodes: Mapping[int, str | os.PathLike[str]],
        input_linkers: Mapping[tuple[int, int], str | os.PathLike[str] | None]
        | None = None,
        output_folder: str | os.PathLike[str] | None = None,
    ) -> list[str]:
        """Assemble and export one unoptimized framework with its full 3D cell.

        Inputs are explicit type mappings; node assignments must cover exactly
        the topology's node types. Edge pairs are normalized into ascending
        order; assigning both orders of a pair is an error. Each node needs its
        topology's connection count, and each populated linker needs two markers.
        Marker counts alone do not guarantee compatible molecular geometry.
        An existing original backup from :meth:`interpenetrate` is refreshed
        after a successful new build.

        Args:
            topo: Name of a 3D topology in the installed PORMAKE database.
            cof_name: Nonempty framework name without directory components.
            input_nodes: Mapping from node-type integers to He-marked XYZ paths.
            input_linkers: Optional mapping from endpoint-type pairs to linker
                XYZ paths. Missing entries and ``None`` mean direct connections.
            output_folder: Optional output directory. Defaults to
                ``{cof_name}/1_{cof_name}_framework``.

        Returns:
            List containing the path to ``{cof_name}_unopt.cif``.

        Raises:
            ValueError: If the topology, name, mappings, coordinates, or connection
                marker counts are invalid.
            FileNotFoundError: If an input XYZ file cannot be found.
            RuntimeError: If CIF export fails validation. Existing output is
                replaced only after the new export succeeds.
        """
        preparation._disable_pormake_file_logging()
        if (
            not cof_name
            or cof_name in (".", "..")
            or Path(cof_name).name != cof_name
        ):
            raise ValueError(
                "cof_name must be a nonempty name without directory components"
            )
        topology = _load_topology(topo)
        required_nodes = _node_requirements(topology)
        if not isinstance(input_nodes, Mapping) or any(
            not isinstance(key, Integral) or isinstance(key, bool)
            for key in input_nodes
        ):
            raise ValueError(
                "input_nodes must map integer node types to XYZ paths"
            )
        if set(input_nodes) != set(required_nodes):
            raise ValueError(
                f"Node assignments must cover exactly node types {sorted(required_nodes)}; "
                f"received {sorted(input_nodes)}."
            )

        required_edges = {
            tuple(map(int, pair)) for pair in topology.unique_edge_types
        }
        linker_paths: dict[tuple[int, int], str | os.PathLike[str] | None] = {}
        if input_linkers is not None:
            if not isinstance(input_linkers, Mapping):
                raise ValueError(
                    "input_linkers must map endpoint-type pairs to XYZ paths or None"
                )
            for pair, path in input_linkers.items():
                if (
                    not isinstance(pair, tuple)
                    or len(pair) != 2
                    or any(
                        not isinstance(key, Integral) or isinstance(key, bool)
                        for key in pair
                    )
                ):
                    raise ValueError(
                        "Linker keys must be pairs of integer node types"
                    )
                first_type, second_type = sorted(map(int, pair))
                key = (first_type, second_type)
                if key not in required_edges:
                    raise ValueError(
                        f"Unknown edge type {pair}; expected {sorted(required_edges)}"
                    )
                if key in linker_paths:
                    raise ValueError(f"Duplicate edge assignment for {key}")
                linker_paths[key] = path

        with TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            nodes = {
                int(node_type): _prepare_block(
                    path,
                    required_nodes[node_type],
                    temporary_path / f"node_{node_type}",
                    f"Node type {node_type}",
                )
                for node_type, path in input_nodes.items()
            }
            linkers = {
                pair: _prepare_block(
                    path,
                    2,
                    temporary_path / f"edge_{pair[0]}_{pair[1]}",
                    f"Edge type {pair}",
                )
                if path is not None
                else None
                for pair, path in linker_paths.items()
            }
            framework = pm.Builder().build_by_type(
                topology=topology,
                node_bbs=nodes,
                edge_bbs=linkers,
            )
            folder = (
                Path(output_folder)
                if output_folder is not None
                else Path(cof_name) / f"1_{cof_name}_framework"
            )
            folder.mkdir(parents=True, exist_ok=True)
            output = folder / f"{cof_name}_unopt.cif"
            with TemporaryDirectory(dir=folder) as export_directory:
                trial = Path(export_directory) / output.name
                try:
                    framework.write_cif(str(trial))
                except Exception as error:
                    raise RuntimeError("CIF export failed") from error
                if not trial.is_file():
                    raise RuntimeError(
                        "CIF export failed; see PORMAKE output above"
                    )
                try:
                    written = Structure.from_file(trial)
                except Exception as error:
                    raise RuntimeError("CIF export is unreadable") from error
                cell = np.asarray(written.lattice.matrix)
                if (
                    len(written) != len(framework.atoms)
                    or len(written) == 0
                    or not np.all(np.isfinite(written.cart_coords))
                    or not np.all(np.isfinite(cell))
                    or abs(np.linalg.det(cell)) < 1e-10
                ):
                    raise RuntimeError(
                        "CIF export changed the atom count or produced invalid geometry"
                    )
                original = folder / f"{cof_name}_original.cif"
                if original.is_file():
                    snapshot = Path(export_directory) / original.name
                    snapshot.write_bytes(trial.read_bytes())
                    snapshot.replace(original)
                trial.replace(output)
        print(f"Saved: {output}")
        return [str(output)]

    def interpenetrate(
        self,
        cof_name: str,
        shift_vector: tuple[float, float, float] = (0.5, 0.5, 0.5),
        input_cif: str | os.PathLike[str] | None = None,
        output_folder: str | os.PathLike[str] | None = None,
    ) -> list[str]:
        """Add one translated framework copy in the same periodic cell.

        The result replaces the unoptimized CIF consumed by MaceOpt3D. The
        single-net input is preserved as ``{cof_name}_original.cif``. Repeated
        calls start from that original rather than duplicating the previous
        result; a successful new build refreshes an existing original backup.
        Atom properties and existing within-net periodic bonds are retained.
        The minimum inter-net distance is printed for inspection. Successful
        export does not establish chemical suitability or topological threading;
        inspect the arrangement before optimizing, including the default shift.

        Args:
            cof_name: Nonempty framework name without directory components.
            shift_vector: Translation in fractional cell coordinates. Defaults
                to ``(0.5, 0.5, 0.5)``; integer components wrap periodically.
            input_cif: Optional P1 builder CIF used instead of the saved original
                or, on the first call, the current unoptimized framework.
            output_folder: Framework directory. Defaults to
                ``{cof_name}/1_{cof_name}_framework``.

        Returns:
            List containing the path to the replaced ``{cof_name}_unopt.cif``.

        Raises:
            ValueError: If the name, vector, P1 CIF, or bond images are invalid,
                or inter-net atoms are closer than 0.001 Angstrom.
            FileNotFoundError: If the source CIF is missing.
            RuntimeError: If the exported structure fails validation.

        """
        if (
            not cof_name
            or cof_name in (".", "..")
            or Path(cof_name).name != cof_name
        ):
            raise ValueError(
                "cof_name must be a nonempty name without directory components"
            )
        shift = np.asarray(shift_vector, dtype=float)
        if shift.shape != (3,) or not np.all(np.isfinite(shift)):
            raise ValueError("shift_vector must contain three finite numbers")
        shift = np.mod(shift, 1.0)
        folder = (
            Path(output_folder)
            if output_folder is not None
            else Path(cof_name) / f"1_{cof_name}_framework"
        )
        output = folder / f"{cof_name}_unopt.cif"
        original = folder / f"{cof_name}_original.cif"
        source = (
            Path(input_cif)
            if input_cif is not None
            else original
            if original.is_file()
            else output
        )
        if not source.is_file():
            raise FileNotFoundError(f"Missing input CIF: {source}")
        content, expected_count, minimum_distance = _interpenetrated_cif(
            source, shift
        )
        folder.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=folder) as temporary:
            trial = Path(temporary) / output.name
            trial.write_text(content, encoding="utf-8")
            try:
                written = Structure.from_file(trial, frac_tolerance=0)
            except Exception as error:
                raise RuntimeError(
                    "Interpenetrated CIF export is unreadable"
                ) from error
            if len(written) != expected_count or not np.all(
                np.isfinite(written.cart_coords)
            ):
                raise RuntimeError(
                    "Interpenetrated CIF export changed the atom count or geometry"
                )
            if source.resolve() != original.resolve():
                snapshot = Path(temporary) / original.name
                snapshot.write_bytes(source.read_bytes())
                snapshot.replace(original)
            trial.replace(output)
        print(f"Minimum inter-net distance: {minimum_distance:.6f} Angstrom")
        print(f"Original: {original}")
        print(f"Saved: {output}")
        return [str(output)]
