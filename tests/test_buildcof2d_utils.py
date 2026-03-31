from pathlib import Path
from typing import cast

import numpy as np
import pytest

from coflandscaper._internal.build_cof_2d import (
    _copy_xyz_file_to_folder,
    _normalize_edge_pair,
    _normalize_edge_types,
    _sanitize_edge_types_inplace,
)


@pytest.mark.unit
def test_copy_xyz_file_to_folder_copies_text_and_name(tmp_path: Path) -> None:
    """This test ensures XYZ inputs are copied verbatim into the chosen folder."""
    src = tmp_path / "node.xyz"
    src.write_text("2\nnode\nC 0 0 0\nH 0 0 1\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    copied = _copy_xyz_file_to_folder(str(src), str(out_dir))

    copied_path = Path(copied)
    assert copied_path.exists()
    assert copied_path.name == "node.xyz"
    assert copied_path.read_text(encoding="utf-8") == src.read_text(
        encoding="utf-8"
    )


@pytest.mark.unit
def test_copy_xyz_file_to_folder_rejects_missing_or_non_xyz_inputs(
    tmp_path: Path,
) -> None:
    """This test ensures invalid source paths are rejected before file operations."""
    bad_suffix = tmp_path / "node.txt"
    bad_suffix.write_text("x", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Input xyz file not found"):
        _copy_xyz_file_to_folder(
            str(tmp_path / "missing.xyz"), str(tmp_path / "out")
        )

    with pytest.raises(FileNotFoundError, match="Input xyz file not found"):
        _copy_xyz_file_to_folder(str(bad_suffix), str(tmp_path / "out"))


@pytest.mark.unit
def test_normalize_edge_pair_returns_int_tuple() -> None:
    """This test ensures edge pairs are always normalized to two Python integers."""
    assert _normalize_edge_pair([1, 2]) == (1, 2)
    assert _normalize_edge_pair(np.array([[3, 4]])) == (3, 4)


@pytest.mark.unit
def test_normalize_edge_pair_raises_for_bad_shape() -> None:
    """This test ensures malformed edge pairs fail with a clear shape error."""
    with pytest.raises(ValueError, match="Unexpected edge type shape"):
        _normalize_edge_pair([1, 2, 3])


@pytest.mark.unit
def test_normalize_edge_types_supports_regular_and_fallback_inputs() -> None:
    """This test ensures edge-type normalization works for arrays and heterogeneous lists."""
    regular = np.array([[1, 2], [3, 4]], dtype=int)
    assert np.array_equal(_normalize_edge_types(regular), regular)

    fallback = [[1, 2], np.array([5, 6])]
    normalized = _normalize_edge_types(
        cast("np.ndarray | list[list[int]]", fallback)
    )
    assert normalized.shape == (2, 2)
    assert np.array_equal(normalized, np.array([[1, 2], [5, 6]], dtype=int))


@pytest.mark.unit
def test_sanitize_edge_types_inplace_converts_nested_lists_to_tuples() -> None:
    """This test ensures mutable list-based edge types are converted to hashable tuples in place."""
    edge_types: list[object] = [[1, 2], np.array([3, 4]), (5, 6)]
    result = _sanitize_edge_types_inplace(
        cast("np.ndarray | list[list[int]]", edge_types)
    )

    assert result is edge_types
    assert edge_types[0] == (1, 2)
    assert edge_types[1] == (3, 4)
    assert edge_types[2] == (5, 6)
