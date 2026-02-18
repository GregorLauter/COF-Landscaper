from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from pymatgen.core import Structure

ModelFormat = Literal["cif", "xyz", "pdb", "mol", "mol2", "sdf"]


@dataclass
class VisualizeCOF:
    width: int = 800
    height: int = 600
    background: str = "white"
    style: str = "stick"

    def _find_files(self, path: Path) -> list[Path]:
        candidates: list[Path] = []
        for ext in ("cif", "xyz", "pdb", "mol", "mol2", "sdf"):
            candidates.extend(sorted(path.glob(f"*.{ext}")))
        return candidates

    def _resolve_model(self, source: str | Path | Structure, model_format: ModelFormat | None) -> tuple[str, str]:
        if isinstance(source, Structure):
            fmt = model_format or "cif"
            data = source.to(fmt=fmt)
            return data, fmt

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Structure file or folder not found: {path}")

        if path.is_dir():
            candidates = self._find_files(path)
            if not candidates:
                raise FileNotFoundError(f"No structure files found in folder: {path}")
            path = candidates[0]

        fmt = model_format or path.suffix.lower().lstrip(".")
        if fmt == "":
            raise ValueError("Unable to infer model format. Pass model_format explicitly.")

        data = path.read_text()
        return data, fmt

    def view(
        self,
        folder: str | Path,
        model_format: ModelFormat | None = None,
        add_unit_cell: bool = True,
        style: str | dict[str, Any] | None = None,
        print_names: bool = True,
    ):
        path = Path(folder)
        if not path.exists():
            raise FileNotFoundError(f"Structure folder not found: {path}")
        if not path.is_dir():
            raise ValueError("view_all expects a folder path.")

        files = self._find_files(path)
        if not files:
            raise FileNotFoundError(f"No structure files found in folder: {path}")

        views = []
        for file_path in files:
            if print_names:
                print(file_path.name)
            view = self._view_single(
                source=file_path,
                model_format=model_format,
                add_unit_cell=add_unit_cell,
                style=style,
            )
            view.show()
            views.append(view)
        return views

    def _view_single(
        self,
        source: str | Path | Structure,
        model_format: ModelFormat | None = None,
        add_unit_cell: bool = True,
        style: str | dict[str, Any] | None = None,
    ):
        try:
            import py3Dmol
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "py3Dmol is required for visualization. Install with: pip install py3Dmol"
            ) from exc

        data, fmt = self._resolve_model(source, model_format)
        view = py3Dmol.view(width=self.width, height=self.height)
        view.addModel(data, fmt)
        resolved_style = style or self.style
        if isinstance(resolved_style, dict):
            view.setStyle(resolved_style)
        else:
            view.setStyle({resolved_style: {}})
        if add_unit_cell:
            view.addUnitCell()
        view.setBackgroundColor(self.background)
        view.zoomTo()
        return view


def visualize_cof(
    folder: str | Path,
    model_format: ModelFormat | None = None,
    add_unit_cell: bool = True,
    width: int = 800,
    height: int = 600,
    background: str = "white",
    style: str | dict[str, Any] = "stick",
    print_names: bool = True,
):
    return VisualizeCOF(
        width=width,
        height=height,
        background=background,
        style=style if isinstance(style, str) else "stick",
    ).view(
        folder=folder,
        model_format=model_format,
        add_unit_cell=add_unit_cell,
        style=style,
        print_names=print_names,
    )
