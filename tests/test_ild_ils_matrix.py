from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

import coflandscaper as cl


def _write_test_cif(path: Path) -> None:
    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
    struct = Structure(
        lattice=lattice,
        species=["C", "C"],
        coords=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.6]],
        coords_are_cartesian=False,
    )
    CifWriter(struct).write_file(str(path), mode="wt")


@pytest.mark.unit
def test_change_ild_rejects_invalid_ild(tmp_path: Path) -> None:
    input_file = tmp_path / "input.cif"
    output_file = tmp_path / "out.cif"
    _write_test_cif(input_file)

    with pytest.raises(ValueError, match="slab thickness"):
        cl.ChangeIld()._change_interlayer_distance(
            str(input_file),
            str(output_file),
            4.0,
        )


@pytest.mark.unit
def test_change_ild_force_invalid_warns_and_writes(tmp_path: Path) -> None:
    input_file = tmp_path / "input.cif"
    output_file = tmp_path / "out.cif"
    _write_test_cif(input_file)

    with pytest.warns(UserWarning, match="periodic layers may overlap"):
        cl.ChangeIld()._change_interlayer_distance(
            str(input_file),
            str(output_file),
            4.0,
            force_invalid_ild=True,
        )

    assert output_file.exists()
    assert output_file.stat().st_size > 0
