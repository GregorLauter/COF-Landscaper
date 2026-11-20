"""Geometric porosity of rigid periodic structures using the optional pyzeo bindings."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from importlib.util import find_spec
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.io.cif import CIFBlock


def _write_p1(cif: Path, atoms: Atoms) -> None:
    """Pass the original cell parameters and symmetry-expanded sites to Zeo++."""
    # A cellpar -> Cartesian cell -> cellpar round trip can perturb a nearly
    # degenerate Voronoi network, even for the original P1 workflow CIFs.
    coordinates = atoms.get_scaled_positions()
    keys = [f"_atom_site_fract_{axis}" for axis in "xyz"]
    if all(key in atoms.info for key in keys):
        original = np.column_stack([atoms.info[key] for key in keys])
        if original.shape == coordinates.shape:
            delta = coordinates - original
            if np.allclose(delta, np.rint(delta), rtol=0, atol=1e-14):
                coordinates = original

    with cif.open("w") as stream:
        stream.write("data_structure\n_symmetry_Int_Tables_number 1\n")
        for tag in CIFBlock.cell_tags:
            stream.write(f"{tag} {atoms.info[tag]}\n")
        stream.write(
            "loop_\n_atom_site_type_symbol\n" + "\n".join(keys) + "\n"
        )
        for symbol, (x, y, z) in zip(atoms.symbols, coordinates, strict=True):
            stream.write(f"{symbol} {x} {y} {z}\n")


class Porosity:
    """Compute Zeo++ geometric descriptors for fully occupied, ordered periodic CIFs.

    Accessible surface area and accessible probe-centre volume exclude isolated
    pockets. The void percentage is that volume divided by the unit-cell volume.
    The three Voronoi diameters (Di, Df, Dif) describe the static pore network
    independently of the selected probe. They include the largest cavity even
    if the probe cannot reach it. Geometric surface area is distinct from BET area.

    Uses Zeo++ default atomic radii and masses and its DEF high-accuracy
    decomposition. Cite Willems et al. (doi:10.1016/j.micromeso.2011.08.020)
    and Pinheiro et al. (doi:10.1039/C3CE41057A) when publishing results.
    """

    def __init__(
        self,
        probe_radius: float = 1.86,
        channel_radius: float | None = None,
        surface_samples: int = 2000,
        volume_samples: int = 50000,
    ) -> None:
        """Set probe geometry and Monte Carlo sampling budgets.

        Args:
            probe_radius: Sampling probe radius in Å. The default is an N₂-sized probe.
            channel_radius: Probe radius in Å for channel accessibility. Defaults
                to probe_radius so the same molecule tests entry and samples the pores.
            surface_samples: Number of surface samples per original atom.
            volume_samples: Number of volume samples across the whole cell.

        Increase sampling to test convergence. Zeo++ seeds these routines
        internally; pyzeo 0.2 exposes no user seed.
        """
        channel_radius = (
            probe_radius if channel_radius is None else channel_radius
        )
        for radius in (probe_radius, channel_radius):
            if (
                not isinstance(radius, Real)
                or isinstance(radius, bool)
                or not math.isfinite(radius)
                or radius < 0
            ):
                raise ValueError(
                    "Probe and channel radii must be finite and nonnegative."
                )
        for samples in (surface_samples, volume_samples):
            if (
                isinstance(samples, bool)
                or not isinstance(samples, Integral)
                or not 0 < samples <= 2**31 - 1
            ):
                raise ValueError(
                    "Sampling counts must be positive 32-bit integers."
                )
        self.probe_radius = float(probe_radius)
        self.channel_radius = float(channel_radius)
        self.surface_samples = int(surface_samples)
        self.volume_samples = int(volume_samples)

    def _analyse(self, cif: Path) -> dict[str, float]:
        """Expand symmetry into a private P1 copy and isolate native calls."""
        try:
            images = cast(
                "list[Atoms]",
                read(
                    cif,
                    format="cif",
                    index=":",
                    parallel=False,
                    store_tags=True,
                ),
            )
            if len(images) != 1:
                raise ValueError("Expected exactly one structure in the CIF.")
            atoms = images[0]
            if (
                not len(atoms)
                or not atoms.pbc.all()
                or not np.isfinite(atoms.positions).all()
                or not np.isfinite(atoms.cell).all()
                or atoms.get_volume() <= 0
            ):
                raise ValueError("Expected a finite, nonempty periodic cell.")
            for occupancy in atoms.info.get("occupancy", {}).values():
                if len(occupancy) != 1 or next(iter(occupancy.values())) != 1:
                    raise ValueError(
                        "Partial occupancy or site disorder requires an ordered CIF."
                    )
        except Exception as error:
            raise ValueError(f"Invalid CIF {cif}: {error}") from error

        with TemporaryDirectory() as temporary:
            work = Path(temporary)
            _write_p1(work / "structure.cif", atoms)
            # Some native failures call exit(), including an unknown atomic radius.
            command = [
                sys.executable,
                str(Path(__file__).with_name("_zeo_worker.py")),
                str(self.channel_radius),
                str(self.probe_radius),
                str(self.surface_samples),
                str(self.volume_samples),
                str(len(atoms)),
            ]
            log = work / "backend.log"
            with log.open("wb") as stream:
                child = subprocess.run(  # noqa: S603
                    command,
                    cwd=work,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result = work / "result.json"
            if child.returncode or not result.is_file():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 2000))
                    detail = stream.read().decode(errors="replace").strip()
                raise RuntimeError(
                    f"pyzeo failed for {cif} (exit {child.returncode}): {detail}"
                )
            return json.loads(result.read_text())

    def run(
        self,
        cof_name: str | None = None,
        mode: str = "both",
        *,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        save_csv: bool = True,
    ) -> pd.DataFrame:
        """Calculate one row per CIF and optionally save one CSV per input group.

        Args:
            cof_name: Workflow name. Supply either cof_name or input_path.
            mode: For named workflows, "incl", "serr", "both" (2D), or "3d".
                Reads optimised CIFs from stage 4 (2D) or stage 2 (3D).
            input_path: Independent CIF file or directory of CIFs (nonrecursive).
            output_path: Explicit CSV filename for an independent calculation.
                Otherwise write porosity/<input-stem>_porosity.csv beside a CIF
                or porosity/<directory-name>_porosity.csv inside an input directory.
            save_csv: Write CSVs. Named workflows use stage 5 analysis under
                porosity/<mode>/ (2D) or stage 3 analysis under porosity/ (3D),
                with the filename <cof_name>_porosity.csv.

        Returns:
            A DataFrame with structure, source path, mode, calculation settings,
            accessible surface areas (m²/g and m²/cm³), accessible pore volume
            (cm³/g), accessible void fraction (%) and three pore diameters (Å).
        """
        if (cof_name is None) == (input_path is None):
            raise ValueError("Supply exactly one of cof_name and input_path.")
        if find_spec("pyzeo") is None:
            raise ImportError(
                'Install the optional backend with pip install "cof-landscaper[porosity]" on macOS or Linux (including WSL).'
            )

        mode = mode.lower()
        groups: list[tuple[str | None, Path, Path]] = []
        if input_path is not None:
            if mode != "both":
                raise ValueError("mode applies only to named workflows.")
            source = Path(input_path)
            if not source.exists():
                raise FileNotFoundError(source)
            base = source if source.is_dir() else source.parent
            name = source.resolve().name if source.is_dir() else source.stem
            output = (
                Path(output_path)
                if output_path is not None
                else base / "porosity" / f"{name}_porosity.csv"
            )
            if output.suffix.lower() != ".csv":
                raise ValueError("output_path must be a CSV filename.")
            groups.append((None, source, output))
        else:
            if output_path is not None or mode not in {
                "incl",
                "serr",
                "both",
                "3d",
            }:
                raise ValueError(
                    "Named workflows require mode incl, serr, both or 3d and use the analysis folder."
                )
            if (
                not cof_name
                or Path(cof_name).name != cof_name
                or cof_name in {".", ".."}
            ):
                raise ValueError("cof_name must be a nonempty workflow name.")
            root = Path(cof_name)
            filename = f"{cof_name}_porosity.csv"
            if mode == "3d":
                groups.append(
                    (
                        mode,
                        root / f"2_{cof_name}_optimization",
                        root
                        / f"3_{cof_name}_analysis"
                        / "porosity"
                        / filename,
                    )
                )
            else:
                matrix_modes = ["serr", "incl"] if mode == "both" else [mode]
                optimization = root / f"4_{cof_name}_optimization"
                porosity = root / f"5_{cof_name}_analysis" / "porosity"
                groups.extend(
                    (tag, optimization / tag, porosity / tag / filename)
                    for tag in matrix_modes
                )

        frames = []
        for tag, source, output in groups:
            files = (
                sorted(
                    p
                    for p in source.iterdir()
                    if p.suffix.lower() == ".cif" and p.is_file()
                )
                if source.is_dir()
                else [source]
            )
            if not files or not source.exists():
                raise FileNotFoundError(f"No CIF inputs found at {source}")
            rows = []
            for cif in files:
                if cif.suffix.lower() != ".cif":
                    raise ValueError(f"Expected a CIF file: {cif}")
                rows.append(
                    {
                        "structure": cif.stem,
                        "source_cif": str(cif),
                        "mode": tag,
                        "probe_radius_angstrom": self.probe_radius,
                        "channel_radius_angstrom": self.channel_radius,
                        "surface_samples_per_atom": self.surface_samples,
                        "volume_samples_per_cell": self.volume_samples,
                        **self._analyse(cif),
                    }
                )
            frame = pd.DataFrame(rows)
            if save_csv:
                output.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=output.parent) as staging:
                    trial = Path(staging) / output.name
                    frame.to_csv(trial, index=False)
                    trial.replace(output)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)
