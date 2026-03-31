import pytest

from coflandscaper._internal import ild_ils_utils as utils


@pytest.mark.unit
def test_generate_values_includes_end() -> None:
    values = utils._generate_values(0.0, 0.3, 0.2)
    assert values == [0.0, 0.2, 0.3]


@pytest.mark.unit
def test_generate_values_requires_positive_step() -> None:
    with pytest.raises(ValueError, match="Step must be positive"):
        utils._generate_values(0.0, 1.0, 0.0)


@pytest.mark.unit
def test_parse_xyz_from_atom_line_valid_and_invalid() -> None:
    valid = "C1 C 0 0.25 0.50 0.75"
    invalid = "C1 C 0 x 0.50 0.75"
    too_short = "C1 C 0"

    assert utils.parse_xyz_from_atom_line(valid) == (0.25, 0.5, 0.75)
    assert utils.parse_xyz_from_atom_line(invalid) is None
    assert utils.parse_xyz_from_atom_line(too_short) is None


@pytest.mark.unit
def test_pick_lower_left_pair_from_lines_selects_expected_pair() -> None:
    atom_lines = [
        "A1 C 0 0.20 0.30 0.40",  # pair 0 lower
        "A2 C 0 0.20 0.30 0.60",  # pair 0 upper
        "B1 C 0 0.10 0.40 0.70",  # pair 1 upper
        "B2 C 0 0.10 0.40 0.20",  # pair 1 lower
    ]

    pair_idx, (lower, upper), (x, y, z), (xw, yw) = (
        utils.pick_lower_left_pair_from_lines(atom_lines)
    )

    assert pair_idx == 1
    assert lower.startswith("B2")
    assert upper.startswith("B1")
    assert (x, y, z) == (0.1, 0.4, 0.2)
    assert (xw, yw) == (0.1, 0.4)


@pytest.mark.unit
def test_pick_lower_left_pair_requires_even_number_of_lines() -> None:
    with pytest.raises(ValueError, match="even number of atom lines"):
        utils.pick_lower_left_pair_from_lines(["A1 C 0 0.1 0.2 0.3"])


@pytest.mark.unit
def test_mode_folder_resolution() -> None:
    cof_name = "cof-x"

    assert utils.get_mode_folders(cof_name, "incl") == [
        f"{cof_name}/2_{cof_name}_matrix/incl"
    ]
    assert utils.get_mode_folders(cof_name, "serr") == [
        f"{cof_name}/2_{cof_name}_matrix/serr"
    ]
    assert utils.get_mode_folders(cof_name, "both") == [
        f"{cof_name}/2_{cof_name}_matrix/serr",
        f"{cof_name}/2_{cof_name}_matrix/incl",
    ]


@pytest.mark.unit
def test_mode_folder_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        utils.get_mode_folders("cof-x", "invalid")
