import importlib
import shutil
from pathlib import Path

import pytest

import coflandscaper as cl


@pytest.mark.unit
def test_buildcof2d_rejects_invalid_topology() -> None:
    """This test ensures unsupported topologies fail fast."""
    with pytest.raises(
        ValueError,
        match="topo must be 'hcb', 'sql', 'sql_ab', 'hcb_ab', or 'kgm'",
    ):
        cl.BuildCOF2D().build(topo="bad", cof_name="cof")


@pytest.mark.unit
def test_buildcof2d_requires_exactly_one_node_and_linker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures multiple node/linker inputs are rejected."""
    monkeypatch.chdir(tmp_path)

    module = importlib.import_module(cl.BuildCOF2D.__module__)

    def fake_prepare_xyz_files(
        xyz_files: list[str],
        output_folder: str,
    ) -> list[str]:
        for path in xyz_files:
            shutil.copy(path, output_folder)
        return xyz_files

    monkeypatch.setattr(module, "_prepare_xyz_files", fake_prepare_xyz_files)

    node_dir = tmp_path / "0_node"
    linker_dir = tmp_path / "0_linker"
    node_dir.mkdir()
    linker_dir.mkdir()

    (node_dir / "node_a.xyz").write_text(
        "2\nnode\nC 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    (node_dir / "node_b.xyz").write_text(
        "2\nnode\nC 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    (linker_dir / "linker.xyz").write_text(
        "2\nlinker\nC 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=(
            r"Topology 'hcb' requires exactly 1 node file\(s\) and 1 linker "
            r"file\(s\)"
        ),
    ):
        cl.BuildCOF2D().build(
            topo="hcb",
            cof_name="cof",
        )


@pytest.mark.unit
def test_buildcof2d_rejects_missing_input_paths(tmp_path: Path) -> None:
    """This test ensures missing explicit input files fail early."""
    with pytest.raises(FileNotFoundError, match=r"missing\.xyz"):
        cl.BuildCOF2D().build(
            topo="hcb",
            cof_name="cof",
            input_nodes=[tmp_path / "missing.xyz"],
            input_linkers=[tmp_path / "missing.xyz"],
        )


@pytest.mark.unit
@pytest.mark.parametrize("topo", ["hcb_ab", "sql_ab"])
def test_buildcof2d_binary_accepts_two_nodes_no_linkers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, topo: str
) -> None:
    """This test ensures binary topologies accept two node files and zero linkers."""
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(cl.BuildCOF2D.__module__)

    def fake_prepare_xyz_files(
        xyz_files: list[str],
        output_folder: str,
    ) -> list[str]:
        for path in xyz_files:
            shutil.copy(path, output_folder)
        return xyz_files

    def fake_build_cof(
        _topo: str,
        _node_paths: list[str],
        _linker_paths: list[str],
        output_folder: str,
        cof_name: str | None = None,
    ) -> str:
        _ = (_topo, _node_paths, _linker_paths, cof_name)
        out = Path(output_folder) / "out.cif"
        out.write_text("data_test\n", encoding="utf-8")
        return str(out)

    def fake_change_ild(
        _self: object,
        input_file: str,
        output_file: str,
        new_z: float,
    ) -> None:
        _ = input_file
        _ = output_file
        _ = new_z

    monkeypatch.setattr(module, "_prepare_xyz_files", fake_prepare_xyz_files)
    monkeypatch.setattr(module, "_build_cof", fake_build_cof)
    monkeypatch.setattr(
        module.ChangeIld,
        "_change_interlayer_distance",
        fake_change_ild,
    )

    node_a = tmp_path / "node_a.xyz"
    node_b = tmp_path / "node_b.xyz"
    node_a.write_text("2\nnode\nC 0 0 0\nH 0 0 1\n", encoding="utf-8")
    node_b.write_text("2\nnode\nC 0 0 0\nH 0 0 1\n", encoding="utf-8")

    cl.BuildCOF2D().build(
        topo=topo,
        cof_name="cof",
        input_nodes=[node_a, node_b],
        input_linkers=[],
    )


@pytest.mark.unit
@pytest.mark.parametrize("topo", ["hcb_ab", "sql_ab"])
def test_buildcof2d_binary_rejects_wrong_node_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, topo: str
) -> None:
    """This test ensures binary topologies enforce node counts."""
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(cl.BuildCOF2D.__module__)

    def fake_prepare_xyz_files(
        xyz_files: list[str],
        output_folder: str,
    ) -> list[str]:
        for path in xyz_files:
            shutil.copy(path, output_folder)
        return xyz_files

    monkeypatch.setattr(module, "_prepare_xyz_files", fake_prepare_xyz_files)

    node_a = tmp_path / "node_a.xyz"
    node_a.write_text("2\nnode\nC 0 0 0\nH 0 0 1\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            rf"Topology '{topo}' requires exactly 2 node file\(s\) and 0 linker "
            r"file\(s\)"
        ),
    ):
        cl.BuildCOF2D().build(
            topo=topo,
            cof_name="cof",
            input_nodes=[node_a],
            input_linkers=[],
        )


@pytest.mark.unit
@pytest.mark.parametrize("topo", ["hcb_ab", "sql_ab"])
def test_buildcof2d_binary_warns_on_linkers_in_default_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, topo: str
) -> None:
    """This test ensures binary topologies ignore default linkers with a warning."""
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(cl.BuildCOF2D.__module__)

    def fake_prepare_xyz_files(
        xyz_files: list[str],
        output_folder: str,
    ) -> list[str]:
        for path in xyz_files:
            shutil.copy(path, output_folder)
        return xyz_files

    def fake_build_cof(
        _topo: str,
        _node_paths: list[str],
        _linker_paths: list[str],
        output_folder: str,
        cof_name: str | None = None,
    ) -> str:
        _ = (_topo, _node_paths, _linker_paths, cof_name)
        out = Path(output_folder) / "out.cif"
        out.write_text("data_test\n", encoding="utf-8")
        return str(out)

    def fake_change_ild(
        _self: object,
        input_file: str,
        output_file: str,
        new_z: float,
    ) -> None:
        _ = input_file
        _ = output_file
        _ = new_z

    monkeypatch.setattr(module, "_prepare_xyz_files", fake_prepare_xyz_files)
    monkeypatch.setattr(module, "_build_cof", fake_build_cof)
    monkeypatch.setattr(
        module.ChangeIld,
        "_change_interlayer_distance",
        fake_change_ild,
    )

    node_dir = tmp_path / "0_node"
    linker_dir = tmp_path / "0_linker"
    node_dir.mkdir()
    linker_dir.mkdir()
    (node_dir / "node_a.xyz").write_text(
        "2\nnode\nC 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    (node_dir / "node_b.xyz").write_text(
        "2\nnode\nC 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )
    (linker_dir / "linker.xyz").write_text(
        "2\nlinker\nC 0 0 0\nH 0 0 1\n",
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match=f"Topology '{topo}' ignores linker"):
        cl.BuildCOF2D().build(
            topo=topo,
            cof_name="cof",
        )


def _write_sql_ab_node(path: Path, symbol: str, marker_count: int = 4) -> None:
    """Write a minimal molecular fragment with square connection directions."""
    markers = ["He 0.95 0 0", "He -0.95 0 0", "He 0 0.95 0", "He 0 -0.95 0"]
    path.write_text(
        f"{marker_count + 1}\n{symbol}\n{symbol} 0 0 0\n"
        + "\n".join(markers[:marker_count])
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.unit
def test_sql_ab_scaffold() -> None:
    """The binary square scaffold has four distinct periodic A-B connections."""
    import numpy as np
    from pormake.topology import Topology

    module = importlib.import_module(cl.BuildCOF2D.__module__)
    topology = Topology(module._package_topology_dir() / "sql_ab.cgd")
    assert len(topology.node_indices) == 2
    assert len(topology.edge_indices) == 4
    assert set(topology.node_types[topology.node_indices]) == {0, 1}
    assert np.all(np.asarray(topology.cn)[topology.node_indices] == 4)
    assert topology.check_validity()
    for edge in topology.edge_indices:
        neighbours = topology.neighbor_list[edge]
        assert {topology.node_types[n.index] for n in neighbours} == {0, 1}
    for node in topology.node_indices:
        directions = np.array(
            [n.distance_vector for n in topology.neighbor_list[node]]
        )
        assert len(np.unique(np.round(directions, 5), axis=0)) == 4
        assert directions[:, 2] == pytest.approx(np.zeros(4), abs=1e-7)


@pytest.mark.unit
def test_sql_ab_assigns_nodes_by_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Input order sets A/B by node type while direct edges remain empty."""
    module = importlib.import_module(cl.BuildCOF2D.__module__)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    build_calls: list[dict[int, str]] = []
    ild_calls: list[float] = []

    class FakeFramework:
        def write_cif(self, filename: str) -> None:
            Path(filename).write_text("data_test\n", encoding="utf-8")

    def fake_build(self, topology, bbs, **kwargs) -> FakeFramework:
        _ = self, kwargs
        build_calls.append(
            {
                int(topology.node_types[node]): bbs[node].atoms[0].symbol
                for node in topology.node_indices
            }
        )
        assert all(bbs[edge] is None for edge in topology.edge_indices)
        return FakeFramework()

    def fake_change_ild(
        _self: object, input_file: str, output_file: str, new_z: float
    ) -> None:
        assert input_file == output_file
        ild_calls.append(new_z)

    monkeypatch.setattr(module.CofLandscaperBuilder, "build", fake_build)
    monkeypatch.setattr(
        module.ChangeIld, "_change_interlayer_distance", fake_change_ild
    )
    # Opposite alphabetical order catches unintended sorting of explicit inputs.
    nodes = [tmp_path / "N.xyz", tmp_path / "C.xyz"]
    for path in nodes:
        _write_sql_ab_node(path, path.stem)
    outputs = cl.BuildCOF2D().build(
        topo="sql_ab", cof_name="binary", input_nodes=nodes, input_linkers=[]
    )
    assert build_calls == [{0: "N", 1: "C"}]
    assert ild_calls == [15.0]
    assert outputs == [
        str(Path("binary") / "1_binary_single_layer" / "binary_unopt.cif")
    ]
    assert Path(outputs[0]).read_text(encoding="utf-8") == "data_test\n"


@pytest.mark.unit
@pytest.mark.parametrize("invalid_node", [0, 1])
def test_sql_ab_rejects_marker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid_node: int
) -> None:
    """Each binary square node needs four connection markers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    nodes = []
    for index, name in enumerate(("A", "B")):
        path = tmp_path / f"{name}.xyz"
        _write_sql_ab_node(path, "C", 3 if index == invalid_node else 4)
        nodes.append(path)
    with pytest.raises(
        ValueError,
        match=f"node {invalid_node + 1} requires exactly 4 connection markers",
    ):
        cl.BuildCOF2D().build(
            topo="sql_ab",
            cof_name="binary",
            input_nodes=nodes,
            input_linkers=[],
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("node_count", "linker_count"), [(0, 0), (3, 0), (2, 1)]
)
def test_sql_ab_rejects_input_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    node_count: int,
    linker_count: int,
) -> None:
    """Extra or missing fragments cannot change the intended binary net."""
    monkeypatch.chdir(tmp_path)
    nodes = [tmp_path / f"node_{index}.xyz" for index in range(node_count)]
    linkers = [
        tmp_path / f"linker_{index}.xyz" for index in range(linker_count)
    ]
    for path in nodes + linkers:
        _write_sql_ab_node(path, "C")
    with pytest.raises(
        ValueError, match=r"exactly 2 node file\(s\) and 0 linker file\(s\)"
    ):
        cl.BuildCOF2D().build(
            topo="sql_ab",
            cof_name="binary",
            input_nodes=nodes,
            input_linkers=linkers,
        )
