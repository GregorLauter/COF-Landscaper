import importlib
from pathlib import Path
from types import SimpleNamespace

import ase.io
import numpy as np
import pormake as pm
import pytest
from ase import Atoms

import coflandscaper as cl


def _write_fragment(path: Path, symbol: str, marker_count: int = 4) -> None:
    """Write a simple He-marked input without running molecular assembly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    markers = ["He 0.95 0 0", "He -0.95 0 0", "He 0 0.95 0", "He 0 -0.95 0"]
    path.write_text(
        f"{marker_count + 1}\nfragment\n{symbol} 0 0 0\n"
        + "\n".join(markers[:marker_count])
        + "\n",
        encoding="utf-8",
    )


class _FakeFramework:
    def __init__(self) -> None:
        self.atoms = Atoms(
            "CN",
            positions=[[1, 2, 3], [4, 5, 6]],
            cell=[11, 13, 19],
            pbc=True,
        )

    def write_cif(self, filename: str) -> None:
        ase.io.write(filename, self.atoms)


@pytest.fixture
def assembly_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Keep PORMAKE's type expansion, replacing only expensive assembly."""
    calls: list[dict] = []

    def fake_build(self, topology, bbs, **kwargs) -> _FakeFramework:
        _ = self, kwargs
        calls.append(
            {
                "nodes": {
                    int(topology.node_types[index]): bbs[index].atoms[0].symbol
                    for index in topology.node_indices
                },
                "edges": [
                    None if bbs[index] is None else bbs[index].atoms[0].symbol
                    for index in topology.edge_indices
                ],
            }
        )
        return _FakeFramework()

    monkeypatch.setattr(pm.Builder, "build", fake_build)
    return calls


@pytest.mark.unit
def test_buildcof3d_describes_bundled_topology() -> None:
    """Inspect the inexpensive scaffold instead of building a framework."""
    description = cl.BuildCOF3D().describe_topology("pts")
    assert description["topology"] == "pts"
    assert description["dimensionality"] == 3
    assert description["node_types"] == {0: 4, 1: 4}
    assert description["edge_types"] == [(0, 1)]


@pytest.mark.unit
def test_buildcof3d_assigns_types_and_preserves_full_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    assembly_calls: list[dict],
) -> None:
    """Separate preparation directories keep identical basenames distinct."""
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "A" / "node.xyz"
    second = tmp_path / "B" / "node.xyz"
    _write_fragment(first, "C")
    _write_fragment(second, "N")
    original_inputs = [path.read_text() for path in (first, second)]
    outputs = cl.BuildCOF3D().build(
        topo="pts",
        cof_name="framework",
        input_nodes={1: second, 0: first},
    )
    assert assembly_calls[0]["nodes"] == {0: "C", 1: "N"}
    assert all(edge is None for edge in assembly_calls[0]["edges"])
    expected = (
        Path("framework") / "1_framework_framework" / "framework_unopt.cif"
    )
    assert outputs == [str(expected)]
    written = ase.io.read(outputs[0])
    assert isinstance(written, Atoms)
    assert written.get_chemical_symbols() == ["C", "N"]
    assert written.cell.cellpar() == pytest.approx([11, 13, 19, 90, 90, 90])
    assert written.positions == pytest.approx(np.array([[1, 2, 3], [4, 5, 6]]))
    assert [path.read_text() for path in (first, second)] == original_inputs


@pytest.mark.unit
def test_buildcof3d_normalizes_linker_types(
    tmp_path: Path, assembly_calls: list[dict]
) -> None:
    """Reverse edge keys select the same linkers; None gives direct edges."""
    nodes = {index: tmp_path / f"node_{index}.xyz" for index in (0, 1)}
    for path in nodes.values():
        _write_fragment(path, "C")
    linker = tmp_path / "linker.xyz"
    _write_fragment(linker, "N", marker_count=2)
    builder = cl.BuildCOF3D()
    for linker_value in (linker, None):
        builder.build(
            topo="pts",
            cof_name="framework",
            input_nodes=nodes,
            input_linkers={(1, 0): linker_value},
            output_folder=tmp_path / "output",
        )
    assert all(edge == "N" for edge in assembly_calls[0]["edges"])
    assert all(edge is None for edge in assembly_calls[1]["edges"])


@pytest.mark.unit
def test_buildcof3d_rejects_invalid_type_maps(
    tmp_path: Path, assembly_calls: list[dict]
) -> None:
    """Missing, extra, unknown, and duplicate type assignments fail early."""
    node = tmp_path / "node.xyz"
    _write_fragment(node, "C")
    builder = cl.BuildCOF3D()
    with pytest.raises(ValueError, match="node"):
        builder.build(topo="pts", cof_name="framework", input_nodes={0: node})
    with pytest.raises(ValueError, match="node"):
        builder.build(
            topo="pts",
            cof_name="framework",
            input_nodes={0: node, 1: node, 2: node},
        )
    with pytest.raises(ValueError, match=r"edge|linker"):
        builder.build(
            topo="pts",
            cof_name="framework",
            input_nodes={0: node, 1: node},
            input_linkers={(0, 0): None},
        )
    with pytest.raises(ValueError, match=r"duplicate|Duplicate"):
        builder.build(
            topo="pts",
            cof_name="framework",
            input_nodes={0: node, 1: node},
            input_linkers={(0, 1): None, (1, 0): None},
        )
    assert assembly_calls == []


@pytest.mark.unit
def test_buildcof3d_checks_connection_markers(
    tmp_path: Path, assembly_calls: list[dict]
) -> None:
    """Node connectivity and populated two-connected edges are validated."""
    node = tmp_path / "node.xyz"
    _write_fragment(node, "C", marker_count=3)
    builder = cl.BuildCOF3D()
    with pytest.raises(ValueError, match=r"4.*He|4.*marker"):
        builder.build(
            topo="pts", cof_name="framework", input_nodes={0: node, 1: node}
        )
    _write_fragment(node, "C")
    linker = tmp_path / "linker.xyz"
    _write_fragment(linker, "N", marker_count=1)
    with pytest.raises(ValueError, match=r"2.*He|2.*marker"):
        builder.build(
            topo="pts",
            cof_name="framework",
            input_nodes={0: node, 1: node},
            input_linkers={(0, 1): linker},
        )
    assert assembly_calls == []


@pytest.mark.unit
def test_buildcof3d_rejects_missing_and_unsuitable_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assembly_calls: list[dict]
) -> None:
    """Unknown and planar nets, and missing XYZ files, fail before assembly."""
    builder = cl.BuildCOF3D()
    with pytest.raises(ValueError, match=r"topology|Topology"):
        builder.describe_topology("not_a_topology")
    # The stock database contains 3D nets; inject our planar net to check rejection.
    preparation = importlib.import_module(cl.BuildCOF2D.__module__)
    with monkeypatch.context() as context:
        context.setattr(
            pm,
            "Database",
            lambda: SimpleNamespace(
                topo_dir=preparation._package_topology_dir(), topo_list=["sql"]
            ),
        )
        with pytest.raises(ValueError, match=r"periodic dimension 2.*3D"):
            builder.describe_topology("sql")
    node = tmp_path / "node.xyz"
    _write_fragment(node, "C")
    with pytest.raises(FileNotFoundError, match=r"missing\.xyz"):
        builder.build(
            topo="pts",
            cof_name="framework",
            input_nodes={0: tmp_path / "missing.xyz", 1: node},
        )
    assert assembly_calls == []


@pytest.mark.unit
def test_buildcof3d_failed_export_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assembly_calls: list[dict]
) -> None:
    """A failed trial export cannot replace an existing framework CIF."""
    node = tmp_path / "node.xyz"
    _write_fragment(node, "C")
    folder = tmp_path / "output"
    folder.mkdir()
    existing = folder / "framework_unopt.cif"
    existing.write_text("existing framework\n", encoding="utf-8")

    def failed_write(self, filename: str) -> None:
        _ = self, filename

    monkeypatch.setattr(_FakeFramework, "write_cif", failed_write)
    with pytest.raises(RuntimeError, match=r"CIF|export"):
        cl.BuildCOF3D().build(
            topo="pts",
            cof_name="framework",
            input_nodes={0: node, 1: node},
            output_folder=folder,
        )
    assert len(assembly_calls) == 1
    assert existing.read_text(encoding="utf-8") == "existing framework\n"
    assert list(folder.iterdir()) == [existing]
