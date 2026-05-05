import importlib
from pathlib import Path

import numpy as np
import pytest

import coflandscaper as cl


@pytest.mark.unit
def test_parse_xyz_from_atom_line_valid_and_invalid() -> None:
    """This test ensures atom-line coordinate parsing is robust to malformed input rows."""
    valid = "C1 C 0 0.25 0.50 0.75"
    invalid = "C1 C 0 x 0.50 0.75"
    too_short = "C1 C 0"

    assert cl.parse_xyz_from_atom_line(valid) == (0.25, 0.5, 0.75)
    assert cl.parse_xyz_from_atom_line(invalid) is None
    assert cl.parse_xyz_from_atom_line(too_short) is None


@pytest.mark.unit
def test_pick_lower_left_pair_from_lines_selects_expected_pair() -> None:
    """This test ensures pair selection picks the lower-left reference atom deterministically."""
    atom_lines = [
        "A1 C 0 0.20 0.30 0.40",  # pair 0 lower
        "A2 C 0 0.20 0.30 0.60",  # pair 0 upper
        "B1 C 0 0.10 0.40 0.70",  # pair 1 upper
        "B2 C 0 0.10 0.40 0.20",  # pair 1 lower
    ]

    pair_idx, (lower, upper), (x, y, z), (xw, yw) = (
        cl.pick_lower_left_pair_from_lines(atom_lines)
    )

    assert pair_idx == 1
    assert lower.startswith("B2")
    assert upper.startswith("B1")
    assert (x, y, z) == (0.1, 0.4, 0.2)
    assert (xw, yw) == (0.1, 0.4)


@pytest.mark.unit
def test_pick_lower_left_pair_requires_even_number_of_lines() -> None:
    """This test ensures odd atom-line counts are rejected before pair grouping."""
    with pytest.raises(ValueError, match="even number of atom lines"):
        cl.pick_lower_left_pair_from_lines(["A1 C 0 0.1 0.2 0.3"])


@pytest.mark.unit
def test_pick_lower_left_pair_raises_on_unparseable_line() -> None:
    """This test ensures unparseable coordinate lines fail with a targeted parsing error."""
    lines = [
        "A1 C 0 x 0.1 0.2",
        "A2 C 0 0.1 0.1 0.3",
    ]
    with pytest.raises(ValueError, match="Could not parse xyz"):
        cl.pick_lower_left_pair_from_lines(lines)


@pytest.mark.unit
def test_mode_folder_resolution() -> None:
    """This test ensures mode-to-folder routing stays stable for all supported modes."""
    cof_name = "cof-x"

    assert cl.get_mode_folders(cof_name, "incl") == [
        f"{cof_name}/2_{cof_name}_matrix/incl"
    ]
    assert cl.get_mode_folders(cof_name, "serr") == [
        f"{cof_name}/2_{cof_name}_matrix/serr"
    ]
    assert cl.get_mode_folders(cof_name, "both") == [
        f"{cof_name}/2_{cof_name}_matrix/serr",
        f"{cof_name}/2_{cof_name}_matrix/incl",
    ]


@pytest.mark.unit
def test_mode_folder_invalid_mode() -> None:
    """This test ensures unsupported mode values are rejected in folder resolution."""
    with pytest.raises(ValueError, match="mode must be"):
        cl.get_mode_folders("cof-x", "invalid")


@pytest.mark.unit
def test_list_cifs_sorted_and_empty(tmp_path: Path) -> None:
    """This test ensures CIF discovery is sorted and fails clearly when no inputs exist."""
    folder = tmp_path / "cifs"
    folder.mkdir()
    (folder / "b.cif").write_text("data_b\n", encoding="utf-8")
    (folder / "a.cif").write_text("data_a\n", encoding="utf-8")
    (folder / "note.txt").write_text("ignore\n", encoding="utf-8")

    found = cl.list_cifs(str(folder))
    assert found == [str(folder / "a.cif"), str(folder / "b.cif")]

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"No \.cif files found"):
        cl.list_cifs(str(empty))


@pytest.mark.unit
def test_wrap01_normalizes_fractional_values() -> None:
    """This test ensures wrap01 normalizes values into the unit interval."""
    assert cl.wrap01(1.2) == pytest.approx(0.2)
    assert cl.wrap01(-0.3) == pytest.approx(0.7)


class _FakeLattice:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix


class _FakeStructure:
    def __init__(self, matrix: np.ndarray) -> None:
        self.lattice = _FakeLattice(matrix)


@pytest.mark.unit
def test_ab_half_diagonal_from_cif(monkeypatch: pytest.MonkeyPatch) -> None:
    """This test ensures half-diagonal geometry extraction from CIF lattice data is correct."""
    matrix = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 8.0],
        ],
    )

    def fake_from_file(_input_file: str) -> _FakeStructure:
        return _FakeStructure(matrix)

    module = importlib.import_module(cl.ab_half_diagonal_from_cif.__module__)
    monkeypatch.setattr(module.Structure, "from_file", fake_from_file)
    length, angle = cl.ab_half_diagonal_from_cif("dummy.cif")

    assert length == pytest.approx(np.sqrt(2.0))
    assert angle == pytest.approx(45.0)


@pytest.mark.unit
def test_default_shift_from_cif_sql_and_hcb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures topology-specific default shifts match expected sql and hcb formulas."""
    matrix = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 8.0],
        ],
    )

    def fake_from_file(_input_file: str) -> _FakeStructure:
        return _FakeStructure(matrix)

    module = importlib.import_module(cl.default_shift_from_cif.__module__)
    monkeypatch.setattr(module.Structure, "from_file", fake_from_file)

    sql_length, sql_angle = cl.default_shift_from_cif("dummy.cif", "sql")
    hcb_length, hcb_angle = cl.default_shift_from_cif("dummy.cif", "hcb")

    assert sql_length == pytest.approx(np.sqrt(2.0))
    assert sql_angle == pytest.approx(45.0)
    assert hcb_length == pytest.approx((2.0 / np.sqrt(3.0)) * np.sqrt(2.0))
    assert hcb_angle == pytest.approx(90.0)


@pytest.mark.unit
def test_default_shift_from_cif_rejects_invalid_topology() -> None:
    """This test ensures invalid topology names fail fast in default shift computation."""
    with pytest.raises(ValueError, match="topo must be 'sql' or 'hcb'"):
        cl.default_shift_from_cif("dummy.cif", "kgm")
