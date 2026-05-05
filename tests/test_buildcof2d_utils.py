import importlib
import shutil
from pathlib import Path

import pytest

import coflandscaper as cl


@pytest.mark.unit
def test_buildcof2d_rejects_invalid_topology() -> None:
    """This test ensures unsupported topologies fail fast."""
    with pytest.raises(ValueError, match="topo must be either"):
        cl.BuildCOF2D().build(topo="bad", bond_type="single", cof_name="cof")


@pytest.mark.unit
def test_buildcof2d_rejects_invalid_bond_type() -> None:
    """This test ensures unsupported bond types fail fast."""
    with pytest.raises(ValueError, match="bond_type must be either"):
        cl.BuildCOF2D().build(topo="hcb", bond_type="bad", cof_name="cof")


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
        _mode: str,
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
        ValueError, match="Expected exactly one node and one linker"
    ):
        cl.BuildCOF2D().build(
            topo="hcb",
            bond_type="single",
            cof_name="cof",
        )


@pytest.mark.unit
def test_buildcof2d_rejects_missing_input_paths(tmp_path: Path) -> None:
    """This test ensures missing explicit input files fail early."""
    with pytest.raises(FileNotFoundError, match=r"missing\.xyz"):
        cl.BuildCOF2D().build(
            topo="hcb",
            bond_type="single",
            cof_name="cof",
            input_node=tmp_path / "missing.xyz",
            input_linker=tmp_path / "missing.xyz",
        )
