import importlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from ase.atoms import Atoms

import coflandscaper as cl


@pytest.mark.unit
def test_buildcof2d_rejects_invalid_topology() -> None:
    """This test ensures unsupported topologies fail fast."""
    with pytest.raises(
        ValueError, match="topo must be 'hcb', 'sql', 'hcb_ab', or 'kgm'"
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
def test_buildcof2d_hcb_ab_accepts_two_nodes_no_linkers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures hcb_ab accepts two node files and zero linkers."""
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
        linker_anchor_local_indices: list[int] | None = None,
        linker_flip: bool = False,
        output_filename: str | None = None,
    ) -> str:
        _ = (
            _topo,
            _node_paths,
            _linker_paths,
            cof_name,
            linker_anchor_local_indices,
            linker_flip,
        )
        out = Path(output_folder) / (output_filename or "out.cif")
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
        topo="hcb_ab",
        cof_name="cof",
        input_nodes=[node_a, node_b],
        input_linkers=[],
    )


@pytest.mark.unit
def test_buildcof2d_hcb_ab_rejects_wrong_node_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures hcb_ab enforces node counts."""
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
    node_b = tmp_path / "node_b.xyz"
    node_c = tmp_path / "node_c.xyz"
    node_a.write_text("2\nnode\nC 0 0 0\nH 0 0 1\n", encoding="utf-8")
    node_b.write_text("2\nnode\nC 0 0 0\nH 0 0 1\n", encoding="utf-8")
    node_c.write_text("2\nnode\nC 0 0 0\nH 0 0 1\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            r"Topology 'hcb_ab' requires exactly 2 node file\(s\) and 0 linker "
            r"file\(s\)"
        ),
    ):
        cl.BuildCOF2D().build(
            topo="hcb_ab",
            cof_name="cof",
            input_nodes=[],
            input_linkers=[],
        )

    with pytest.raises(
        ValueError,
        match=(
            r"Topology 'hcb_ab' requires exactly 2 node file\(s\) and 0 linker "
            r"file\(s\)"
        ),
    ):
        cl.BuildCOF2D().build(
            topo="hcb_ab",
            cof_name="cof",
            input_nodes=[node_a],
            input_linkers=[],
        )

    with pytest.raises(
        ValueError,
        match=(
            r"Topology 'hcb_ab' requires exactly 2 node file\(s\) and 0 linker "
            r"file\(s\)"
        ),
    ):
        cl.BuildCOF2D().build(
            topo="hcb_ab",
            cof_name="cof",
            input_nodes=[node_a, node_b, node_c],
            input_linkers=[],
        )


@pytest.mark.unit
def test_buildcof2d_hcb_ab_warns_on_linkers_in_default_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures hcb_ab ignores default linkers with a warning."""
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
        linker_anchor_local_indices: list[int] | None = None,
        linker_flip: bool = False,
        output_filename: str | None = None,
    ) -> str:
        _ = (
            _topo,
            _node_paths,
            _linker_paths,
            cof_name,
            linker_anchor_local_indices,
            linker_flip,
        )
        out = Path(output_folder) / (output_filename or "out.cif")
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

    with pytest.warns(UserWarning, match="Topology 'hcb_ab' ignores linker"):
        cl.BuildCOF2D().build(
            topo="hcb_ab",
            cof_name="cof",
        )


@pytest.mark.unit
def test_buildcof2d_requires_exactly_one_node_and_linker_for_kgm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This test ensures kgm enforces one node and one linker."""
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
            r"Topology 'kgm' requires exactly 1 node file\(s\) and 1 linker "
            r"file\(s\)"
        ),
    ):
        cl.BuildCOF2D().build(
            topo="kgm",
            cof_name="cof",
        )


@pytest.mark.unit
def test_extract_anchor_local_indices_requires_two_distinct_real_atoms() -> None:
    module = importlib.import_module(cl.BuildCOF2D.__module__)

    anchors = module._extract_anchor_local_indices(
        x_indices=[0, 3],
        bonds=[(0, 1), (1, 2), (2, 3)],
    )
    assert anchors == [1, 2]
    assert len(set(anchors)) == 2


@pytest.mark.unit
def test_flip_linker_building_block_rotates_around_anchor_axis() -> None:
    module = importlib.import_module(cl.BuildCOF2D.__module__)

    class _Edge:
        def __init__(self) -> None:
            self.atoms = Atoms(
                symbols=["C", "C", "H", "H"],
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            )

    edge = _Edge()
    module._flip_linker_building_block(edge, [0, 1])

    pos = edge.atoms.positions
    assert np.allclose(pos[0], [0.0, 0.0, 0.0])
    assert np.allclose(pos[1], [1.0, 0.0, 0.0])
    assert np.allclose(pos[2], [0.0, -1.0, 0.0], atol=1e-8)
    assert np.allclose(pos[3], [0.0, 0.0, -1.0], atol=1e-8)


@pytest.mark.unit
def test_build_same_and_flip_writes_expected_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builder = cl.BuildCOF2D()
    calls: list[dict[str, object]] = []

    def fake_build(self, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        out_dir = Path(kwargs["output_folder"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / str(kwargs["output_filename"])
        out.write_text("data_test\n", encoding="utf-8")
        return [str(out)]

    monkeypatch.setattr(cl.BuildCOF2D, "build", fake_build)

    outputs = builder.build_same_and_flip(
        topo="sql",
        cof_name="cof",
        output_folder=str(tmp_path),
    )

    assert len(calls) == 2
    assert calls[0]["linker_flip"] is False
    assert calls[1]["linker_flip"] is True
    assert calls[0]["output_filename"] == "cof_same_unopt.cif"
    assert calls[1]["output_filename"] == "cof_flip_unopt.cif"
    assert outputs["same_unopt"].endswith("cof_same_unopt.cif")
    assert outputs["flip_unopt"].endswith("cof_flip_unopt.cif")
