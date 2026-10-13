"""Persist and display Miller indices without changing the legacy XY format."""

from __future__ import annotations

import hashlib
import json
import warnings
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from pymatgen.analysis.diffraction.core import DiffractionPattern
    from pymatgen.core import Structure


def save_reflections(
    cif_path: Path,
    xy_path: Path,
    structure: Structure,
    pattern: DiffractionPattern,
    settings: dict[str, Any],
) -> None:
    """Save full-precision reflections and fingerprints of their indexing inputs."""
    peaks = [
        {
            "peak_index": index,
            "two_theta_deg": float(angle),
            "relative_intensity": float(intensity),
            "d_spacing_angstrom": float(spacing),
            "families": [
                {
                    "hkl": [int(v) for v in family["hkl"]],
                    "multiplicity": int(family["multiplicity"]),
                }
                for family in families
            ],
        }
        for index, (angle, intensity, spacing, families) in enumerate(
            zip(
                pattern.x, pattern.y, pattern.d_hkls, pattern.hkls, strict=True
            )
        )
    ]
    metadata = {
        "schema_version": 1,
        "source_cif": {
            "path": str(cif_path.resolve()),
            "name": cif_path.name,
            "sha256": hashlib.sha256(cif_path.read_bytes()).hexdigest(),
        },
        "calculation_settings": settings,
        "unit_cell": {
            "lengths_angstrom": [float(v) for v in structure.lattice.abc],
            "angles_deg": [float(v) for v in structure.lattice.angles],
            "matrix_angstrom": structure.lattice.matrix.tolist(),
        },
        "xy": {
            "name": xy_path.name,
            "sha256": hashlib.sha256(xy_path.read_bytes()).hexdigest(),
            "row_count": len(peaks),
        },
        "peaks": peaks,
    }
    xy_path.with_suffix(".reflections.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_reflections(
    xy_path: Path, angles: np.ndarray, intensities: np.ndarray
) -> list[dict[str, Any]] | None:
    """Validate metadata against XY rows and an available source CIF.

    Missing metadata returns None. Invalid or stale metadata raises ValueError.
    A relocated dataset may be read when its original CIF path no longer exists;
    the recorded CIF fingerprint and indexing basis remain in the sidecar.
    """
    path = xy_path.with_suffix(".reflections.json")
    if not path.exists():
        return None
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata["schema_version"] != 1:
            raise ValueError("unsupported schema version")
        xy = metadata["xy"]
        if (
            xy["name"] != xy_path.name
            or xy["sha256"] != hashlib.sha256(xy_path.read_bytes()).hexdigest()
            or xy["row_count"] != len(angles)
        ):
            raise ValueError("XY fingerprint or row count does not match")
        source = metadata["source_cif"]
        cif_path = Path(source["path"])
        if source["name"] != xy_path.with_suffix(".cif").name:
            raise ValueError("source CIF identity does not match")
        if (
            cif_path.is_file()
            and source["sha256"]
            != hashlib.sha256(cif_path.read_bytes()).hexdigest()
        ):
            raise ValueError("source CIF changed since calculation")
        # Require the indexing basis and settings, even for relocated datasets.
        cell = metadata["unit_cell"]
        if (
            np.asarray(cell["lengths_angstrom"]).shape != (3,)
            or np.asarray(cell["angles_deg"]).shape != (3,)
            or np.asarray(cell["matrix_angstrom"]).shape != (3, 3)
        ):
            raise ValueError("invalid indexing cell")
        for key in ("wavelength", "two_theta_range", "scaled", "symprec"):
            metadata["calculation_settings"][key]
        peaks = metadata["peaks"]
        if len(peaks) != len(angles):
            raise ValueError("reflection row count does not match")
        for index, peak in enumerate(peaks):
            if peak["peak_index"] != index:
                raise ValueError("peak indices are not in XY row order")
            angle, intensity, spacing = (
                float(peak[key])
                for key in (
                    "two_theta_deg",
                    "relative_intensity",
                    "d_spacing_angstrom",
                )
            )
            if (
                not np.isfinite([angle, intensity, spacing]).all()
                or spacing <= 0
                or intensity < 0
                or abs(angle - angles[index]) > 0.0000051
                or abs(intensity - intensities[index]) > 0.0005001
            ):
                raise ValueError(
                    "reflection values do not match exported XY row"
                )
            if not peak["families"]:
                raise ValueError("reflection has no Miller-index families")
            for family in peak["families"]:
                hkl, mult = family["hkl"], family["multiplicity"]
                if (
                    not isinstance(hkl, list)
                    or not hkl
                    or any(type(v) is not int for v in hkl)
                    or type(mult) is not int
                    or mult <= 0
                ):
                    raise ValueError("invalid Miller indices or multiplicity")
    except (OSError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(
            f"Invalid reflection metadata for {xy_path}: {exc}"
        ) from exc
    return peaks


def validate_annotation_options(max_labels: int, min_intensity: float) -> None:
    """Validate the annotation limit and global relative-intensity threshold."""
    if type(max_labels) is not int or max_labels < 0:
        raise ValueError("max_hkl_labels must be a non-negative integer")
    if not np.isfinite(min_intensity) or not 0 <= min_intensity <= 100:
        raise ValueError("min_hkl_intensity must be between 0 and 100")


def format_hkl(indices: Sequence[int]) -> str:
    """Format plane indices as three-index mathtext with signed overbars.

    Valid four-index Miller-Bravais plane notation drops the redundant third
    index. Invalid or unexpected sequences retain their original signed label.
    Metadata and symmetry-family representatives are never modified.
    """
    original = "(" + ", ".join(str(v) for v in indices) + ")"
    valid = len(indices) in {3, 4} and all(
        isinstance(v, Integral) and not isinstance(v, bool) for v in indices
    )
    if valid and len(indices) == 4:
        valid = indices[2] == -(indices[0] + indices[1])
    if not valid:
        warnings.warn(
            f"Unexpected Miller indices {original}; retaining original label.",
            UserWarning,
            stacklevel=2,
        )
        return original
    displayed = (
        (indices[0], indices[1], indices[3])
        if len(indices) == 4
        else tuple(indices)
    )
    components = [
        rf"\overline{{{-int(v)}}}" if v < 0 else str(int(v)) for v in displayed
    ]
    separator = r",\ " if any(abs(v) >= 10 for v in displayed) else ""
    return "$(" + separator.join(components) + ")$"


def annotate_reflections(
    ax: Axes,
    xy_path: Path,
    peaks: list[dict[str, Any]] | None,
    angles: np.ndarray,
    intensities: np.ndarray,
    heights: np.ndarray,
    order: np.ndarray,
    xlim: tuple[float, float],
    max_labels: int,
    min_intensity: float,
) -> None:
    """Label the strongest visible simulated rows, retaining every hkl family.

    Labels occupy separate horizontal slots above the pattern with leader lines.
    This also separates near-coincident reflections without merging their indices.
    """
    if peaks is None:
        warnings.warn(
            f"No reflection metadata for {xy_path.name}; skipping hkl labels.",
            UserWarning,
            stacklevel=2,
        )
        return
    if max_labels == 0 or intensities.size == 0:
        return
    maximum = float(np.max(intensities))
    if maximum <= 0:
        return
    eligible = np.flatnonzero(
        (angles >= xlim[0])
        & (angles <= xlim[1])
        & (100 * intensities / maximum >= min_intensity)
        & (heights > 0)
    )
    selected = eligible[
        np.argsort(-intensities[eligible], kind="stable")[:max_labels]
    ]
    selected = selected[np.argsort(angles[selected], kind="stable")]
    if selected.size == 0:
        return
    # Reserve vertical headroom and spread labels in display coordinates.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_bbox_to_anchor((1.0, 1.3))
    slots = (
        np.linspace(0.05, 0.95, len(selected)) if len(selected) > 1 else [0.5]
    )
    for slot, index in zip(slots, selected, strict=True):
        peak = peaks[int(order[index])]
        label = "\n".join(
            format_hkl(family["hkl"]) for family in peak["families"]
        )
        ax.annotate(
            label,
            xy=(angles[index], heights[index]),
            xytext=(slot, 0.97),
            textcoords="axes fraction",
            ha="center",
            va="top",
            fontsize=8,
            annotation_clip=True,
            arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.6},
        )
