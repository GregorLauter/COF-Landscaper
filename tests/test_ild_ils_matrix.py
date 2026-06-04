from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

import coflandscaper as cl
from coflandscaper._internal.ild_ils_utils import _unwrap_fractional_z


def _write_test_cif(path: Path) -> None:
    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
    struct = Structure(
        lattice=lattice,
        species=["C", "C"],
        coords=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.6]],
        coords_are_cartesian=False,
    )
    CifWriter(struct).write_file(str(path), mode="wt")


def _slab_thickness(struct: Structure) -> float:
    z_len = struct.lattice.c
    fz = struct.frac_coords[:, 2]
    z0 = _unwrap_fractional_z(fz)
    fz_unwrapped = (fz - z0) % 1.0
    return float((fz_unwrapped.max() - fz_unwrapped.min()) * z_len)


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


@pytest.mark.unit
def test_change_ild_explicit_bilayer_uses_gap_semantics(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input.cif"
    output_file = tmp_path / "out.cif"
    _write_test_cif(input_file)

    in_struct = Structure.from_file(str(input_file))
    _ = _slab_thickness(in_struct)
    requested_ild = 3.0

    cl.ChangeIld()._change_interlayer_distance(
        str(input_file),
        str(output_file),
        requested_ild,
        explicit_bilayer=True,
    )

    out_struct = Structure.from_file(str(output_file))
    assert out_struct.lattice.c == pytest.approx(2.0 * requested_ild)


@pytest.mark.unit
def test_change_ild_explicit_bilayer_keeps_slab_contiguous_centered(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input_split.cif"
    output_file = tmp_path / "out_split.cif"

    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
    # Two-layer slab intentionally split across periodic boundary in input.
    struct = Structure(
        lattice=lattice,
        species=["C", "C", "C", "C"],
        coords=[
            [0.0, 0.0, 0.95],
            [0.2, 0.0, 0.98],
            [0.0, 0.0, 0.02],
            [0.2, 0.0, 0.05],
        ],
        coords_are_cartesian=False,
    )
    CifWriter(struct).write_file(str(input_file), mode="wt")

    cl.ChangeIld()._change_interlayer_distance(
        str(input_file),
        str(output_file),
        3.0,
        explicit_bilayer=True,
    )

    out = Structure.from_file(str(output_file))
    fz = np.array(out.frac_coords[:, 2], dtype=float)
    low = float(np.mean(fz[fz < 0.5]))
    high = float(np.mean(fz[fz >= 0.5]))

    assert low == pytest.approx(0.25, abs=6e-2)
    assert high == pytest.approx(0.75, abs=6e-2)


@pytest.mark.unit
def test_change_ild_explicit_bilayer_rejects_when_ild_below_layer_thickness(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input_overlap.cif"
    output_file = tmp_path / "out_overlap.cif"

    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
    struct = Structure(
        lattice=lattice,
        species=["C", "C", "C", "C"],
        coords=[
            [0.0, 0.0, 0.85],
            [0.2, 0.0, 0.99],
            [0.0, 0.0, 0.01],
            [0.2, 0.0, 0.15],
        ],
        coords_are_cartesian=False,
    )
    CifWriter(struct).write_file(str(input_file), mode="wt")

    with pytest.raises(ValueError, match="slab thickness"):
        cl.ChangeIld()._change_interlayer_distance(
            str(input_file),
            str(output_file),
            1.0,
            explicit_bilayer=True,
        )


@pytest.mark.unit
def test_change_ild_explicit_bilayer_warns_when_ild_below_layer_thickness(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input_overlap_warn.cif"
    output_file = tmp_path / "out_overlap_warn.cif"

    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
    struct = Structure(
        lattice=lattice,
        species=["C", "C", "C", "C"],
        coords=[
            [0.0, 0.0, 0.85],
            [0.2, 0.0, 0.99],
            [0.0, 0.0, 0.01],
            [0.2, 0.0, 0.15],
        ],
        coords_are_cartesian=False,
    )
    CifWriter(struct).write_file(str(input_file), mode="wt")

    with pytest.warns(UserWarning, match="slab thickness"):
        cl.ChangeIld()._change_interlayer_distance(
            str(input_file),
            str(output_file),
            1.0,
            explicit_bilayer=True,
            force_invalid_ild=True,
        )


@pytest.mark.unit
def test_change_ild_explicit_bilayer_detects_layers_with_wrapped_z_clusters(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input_wrapped_by_layer.cif"
    output_file = tmp_path / "out_wrapped_by_layer.cif"

    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90, 90, 90)
    # Each half is compact only after periodic unwrapping across the z boundary.
    struct = Structure(
        lattice=lattice,
        species=["C", "C", "C", "C"],
        coords=[
            [0.0, 0.0, 0.96],
            [0.2, 0.0, 0.06],
            [0.0, 0.0, 0.66],
            [0.2, 0.0, 0.76],
        ],
        coords_are_cartesian=False,
    )
    CifWriter(struct).write_file(str(input_file), mode="wt")

    cl.ChangeIld()._change_interlayer_distance(
        str(input_file),
        str(output_file),
        5.5,
        explicit_bilayer=True,
    )

    out = Structure.from_file(str(output_file))
    assert out.lattice.c == pytest.approx(11.0)
    fz = np.array(out.frac_coords[:, 2], dtype=float)
    low = float(np.mean(fz[fz < 0.5]))
    high = float(np.mean(fz[fz >= 0.5]))
    assert low == pytest.approx(0.25, abs=6e-2)
    assert high == pytest.approx(0.75, abs=6e-2)
