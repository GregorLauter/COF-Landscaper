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
def test_default_shift_from_cif_sql_hcb_kgm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures hcb/kgm use their crystallographic shift vector."""
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
    kgm_length, kgm_angle = cl.default_shift_from_cif("dummy.cif", "kgm")

    assert sql_length == pytest.approx(np.sqrt(2.0))
    assert sql_angle == pytest.approx(45.0)
    hcb_vec_xy = hcb_length * np.array(
        [np.cos(np.radians(hcb_angle)), np.sin(np.radians(hcb_angle))]
    )
    assert 3.0 * hcb_vec_xy == pytest.approx(
        matrix[0, :2] + 2.0 * matrix[1, :2]
    )
    assert kgm_length == pytest.approx(hcb_length)
    assert kgm_angle == pytest.approx(hcb_angle)


@pytest.mark.unit
def test_default_shift_from_cif_hcb_and_kgm_match_for_hex_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures hcb and kgm return the same default ILS shift."""
    L = 6.0
    matrix = np.array(
        [
            [L, 0.0, 0.0],
            [-0.5 * L, (np.sqrt(3.0) / 2.0) * L, 0.0],
            [0.0, 0.0, 12.0],
        ],
    )

    def fake_from_file(_input_file: str) -> _FakeStructure:
        return _FakeStructure(matrix)

    module = importlib.import_module(cl.default_shift_from_cif.__module__)
    monkeypatch.setattr(module.Structure, "from_file", fake_from_file)

    hcb_length, hcb_angle = cl.default_shift_from_cif("dummy.cif", "hcb")
    kgm_length, kgm_angle = cl.default_shift_from_cif("dummy.cif", "kgm")

    expected = L / np.sqrt(3.0)
    assert hcb_length == pytest.approx(expected)
    assert kgm_length == pytest.approx(expected)
    assert hcb_angle == pytest.approx(90.0)
    assert kgm_angle == pytest.approx(90.0)


@pytest.mark.unit
def test_default_shift_from_cif_rejects_invalid_topology() -> None:
    """This test ensures invalid topology names fail fast in default shift computation."""
    with pytest.raises(ValueError, match="topo must be"):
        cl.default_shift_from_cif("dummy.cif", "bad")


@pytest.mark.unit
def test_ils_defaults_map_hcb_ab_but_not_kgm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This test ensures hcb_ab maps to hcb while kgm passes through."""
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    (cif_dir / "sample.cif").write_text("data_test\n", encoding="utf-8")

    module = importlib.import_module(cl.IlsIncl.__module__)
    seen: list[str] = []

    def fake_default_shift(
        _input_file: str,
        topo: str,
        print_shift: bool = False,
    ) -> tuple[float, float]:
        _ = print_shift
        seen.append(topo)
        return 1.0, 90.0

    def fake_inclined_shift(
        _self: object,
        _input_file: str,
        _output_file: str,
        _ils_length: float,
        _ils_angle_deg: float,
    ) -> None:
        _ = (_self, _input_file, _output_file, _ils_length, _ils_angle_deg)

    def fake_serrated_shift(
        _self: object,
        _input_file: str,
        _output_file: str,
        _ils_length: float,
        _ils_angle_deg: float,
    ) -> None:
        _ = (_self, _input_file, _output_file, _ils_length, _ils_angle_deg)

    monkeypatch.setattr(module, "default_shift_from_cif", fake_default_shift)
    monkeypatch.setattr(cl.IlsIncl, "_inclined_shift", fake_inclined_shift)
    monkeypatch.setattr(cl.IlsSerr, "_shift_serrated", fake_serrated_shift)

    cl.IlsIncl().run(
        input_folder=str(cif_dir),
        output_folder=str(tmp_path / "incl"),
        topo="hcb_ab",
    )
    cl.IlsSerr().run(
        input_folder=str(cif_dir),
        output_folder=str(tmp_path / "serr"),
        topo="kgm",
    )

    assert seen == ["hcb", "kgm"]


@pytest.mark.parametrize("gamma", [90.0, 87.123456])
@pytest.mark.parametrize("override", ["none", "both", "length", "angle"])
def test_ladder_matrix_actual_shift(
    tmp_path: Path, gamma: float, override: str
) -> None:
    """Both written stacking modes use b/2 and gamma, or explicit overrides."""
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter

    source = tmp_path / "input.cif"
    lattice = Lattice.from_parameters(12.3, 20.2468, 15.0, 90.0, 90.0, gamma)
    CifWriter(Structure(lattice, ["C"], [[0.2, 0.3, 0.5]])).write_file(source)
    length, angle = cl.default_shift_from_cif(str(source), "ladder_1d")
    assert length == pytest.approx(lattice.b / 2, abs=1e-8)
    assert angle == pytest.approx(gamma, abs=1e-8)
    if override in {"both", "length"}:
        length = 1.2345
    if override in {"both", "angle"}:
        angle = 0.0
    output = tmp_path / "matrix"
    cl.CreateMatrix(
        ild_start=3.5,
        ild_end=3.5,
        ils_length_step=30.0,
        ils_length_end=1.2345 if override in {"both", "length"} else None,
        ils_angle=0.0 if override in {"both", "angle"} else None,
    ).run(
        cof_name="test",
        topo="ladder_1d",
        mode="both",
        input_cif=str(source),
        output_base_folder=str(output),
    )

    incl = [Structure.from_file(p) for p in (output / "incl").glob("*.cif")]
    serr = [Structure.from_file(p) for p in (output / "serr").glob("*.cif")]
    assert len(incl) == len(serr) == 2  # zero and exact endpoint
    endpoint = max(incl, key=lambda s: s.lattice.c)
    a, b, c = endpoint.lattice.matrix
    # Dot products remain valid after CIF canonicalizes the Cartesian frame.
    assert np.dot(c, a) == pytest.approx(
        length * lattice.a * np.cos(np.radians(angle)), abs=2e-6
    )
    assert np.dot(c, b) == pytest.approx(
        length * lattice.b * np.cos(np.radians(gamma - angle)), abs=2e-6
    )
    assert endpoint.lattice.c == pytest.approx(np.hypot(3.5, length), abs=1e-7)
    assert endpoint.lattice.gamma == pytest.approx(gamma, abs=1e-7)

    shifts = []
    for structure in serr:
        lower, upper = sorted(structure.frac_coords, key=lambda f: f[2])
        delta = upper - lower
        delta[:2] -= np.round(delta[:2])
        shifts.append(delta[:2])
        assert delta[2] == pytest.approx(0.5, abs=1e-7)
        assert structure.lattice.c == pytest.approx(7.0, abs=1e-7)
    fractional_y = (
        length
        * np.sin(np.radians(angle))
        / (lattice.b * np.sin(np.radians(gamma)))
    )
    fractional_x = (
        length * np.cos(np.radians(angle))
        - fractional_y * lattice.b * np.cos(np.radians(gamma))
    ) / lattice.a
    expected = np.array([fractional_x, fractional_y])
    assert any(np.allclose(s, 0, atol=1e-7) for s in shifts)
    assert any(
        np.allclose(s - expected - np.round(s - expected), 0, atol=1e-7)
        for s in shifts
    )
