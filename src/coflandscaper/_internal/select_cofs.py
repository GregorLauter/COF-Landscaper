import os
import re
import shutil
from pathlib import Path


class SelectCofs:
    def __init__(self, tol: float = 1e-6):
        self.tol = tol

    def _parse_z_L_from_stem(self, stem: str):
        mz = re.search(r"_z(\d+)", stem)
        mL = re.search(r"_L(\d+)", stem)
        if not (mz and mL):
            return None, None
        z = float(mz.group(1)) / 10.0
        L = float(mL.group(1)) / 10.0
        return z, L

    def run(
        self,
        input_folder: str,
        output_folder: str = "select_cofs",
        selections: list[tuple[float, float]] | None = None,
    ):
        if not selections:
            raise ValueError("selections must be a non-empty list of (z, L) tuples")

        in_path = Path(input_folder)
        out_path = Path(output_folder)
        os.makedirs(out_path, exist_ok=True)

        cif_files = sorted(in_path.glob("*.cif"))
        if not cif_files:
            raise FileNotFoundError(f"No .cif files found in: {in_path.resolve()}")

        remaining = set(selections)
        for cif_path in cif_files:
            z, L = self._parse_z_L_from_stem(cif_path.stem)
            if z is None or L is None:
                continue
            for (z_sel, L_sel) in list(remaining):
                if abs(z - z_sel) <= self.tol and abs(L - L_sel) <= self.tol:
                    shutil.copy2(cif_path, out_path / cif_path.name)
                    remaining.discard((z_sel, L_sel))

        if remaining:
            missing = ", ".join([f"(z={z}, L={L})" for z, L in remaining])
            raise FileNotFoundError(f"No matching CIFs found for: {missing}")

    __call__ = run
