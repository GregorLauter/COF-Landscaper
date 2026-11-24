"""Check the shipped 2D workflows without running their calculations."""

import ast
import inspect
import json
import runpy
from pathlib import Path

import pytest
from ase.calculators.emt import EMT

import coflandscaper as cl

_MAIN_WORKFLOW = Path(__file__).resolve().parents[1] / "examples/main_workflow"
_DEFAULT_EXAMPLE = _MAIN_WORKFLOW / "local/default"


def test_default_notebook_public_api() -> None:
    """Each cell needs only the setup names and calls the installed public API."""
    notebook = json.loads((_DEFAULT_EXAMPLE / "default.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, cell["id"], "exec")
        tree = ast.parse(source)
        names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert names <= {"cl", "COF_NAME", "TOPOLOGY"}, cell["id"]
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            assert isinstance(call.func, ast.Attribute)
            receiver = call.func.value
            owner = cl
            if isinstance(receiver, ast.Call):
                assert isinstance(receiver.func, ast.Attribute)
                owner = getattr(cl, receiver.func.attr)
            signature = inspect.signature(getattr(owner, call.func.attr))
            args = [None] if "self" in signature.parameters else []
            if call.func.attr == "run_preopt":
                assert len(call.args) == 1
                assert not call.keywords
            signature.bind(
                *args,
                *call.args,
                **{kw.arg: kw.value for kw in call.keywords if kw.arg},
            )


def test_default_notebook_manual_inputs() -> None:
    for folder, sites in (("node", 4), ("linker", 2)):
        input_dir = _DEFAULT_EXAMPLE / f"0_{folder}"
        xyz = input_dir / f"ILCOF-1_{folder}.xyz"
        assert list(input_dir.glob("*.xyz")) == [xyz]
        count, _, *atoms = xyz.read_text().splitlines()
        assert len(atoms) == int(count)
        assert sum(atom.split()[0] == "He" for atom in atoms) == sites
    assert (_DEFAULT_EXAMPLE / "experimental_pxrd/ILCOF-1.xy").is_file()


@pytest.mark.unit
@pytest.mark.parametrize("workflow", ["script", "notebook"])
@pytest.mark.parametrize("topo", ["sql", "kgm"])
def test_standard_workflows_pass_preopt_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, topo: str
) -> None:
    """Run each standard preopt entry point without loading a MACE model."""
    seen: dict[str, object] = {}

    class PreoptReachedError(Exception):
        pass

    def record_preopt(self, cof_name: str, **kwargs: object) -> None:
        assert isinstance(self.calc, EMT)
        seen.update(cof_name=cof_name, **kwargs)
        raise PreoptReachedError

    monkeypatch.setattr(cl.GeometryOpt, "run_preopt", record_preopt)
    monkeypatch.setattr(cl.BuildCOF2D, "build", lambda *_args, **_kwargs: [])
    monkeypatch.chdir(tmp_path)
    if workflow == "script":
        hpc = _MAIN_WORKFLOW / "hpc"
        driver = runpy.run_path(str(hpc / "cof-landscaper_2d.py"))
        params = json.loads(
            (hpc / "cof-landscaper_2d.params.json").read_text()
        )
        params.update(
            TOPOLOGY=topo,
            COF_NAME="workflow-test",
            CALCULATOR="emt",
            CALCULATOR_KWARGS={},
            FIX_CELL_ANGLES=True,
        )
        with pytest.raises(PreoptReachedError):
            driver["run_workflow"](params)
    else:
        notebook = json.loads(
            (
                _MAIN_WORKFLOW / "local/configurable/configurable.ipynb"
            ).read_text()
        )
        cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "run_preopt(" in "".join(cell["source"])
        ]
        assert len(cells) == 1
        cell_script = tmp_path / "preopt_cell.py"
        cell_script.write_text(cells[0])
        with pytest.raises(PreoptReachedError):
            runpy.run_path(
                str(cell_script),
                init_globals={
                    "cl": cl,
                    "COF_NAME": "workflow-test",
                    "TOPOLOGY": topo,
                    "calculator": EMT(),
                },
            )
    assert seen["cof_name"] == "workflow-test"
    assert seen["topo"] == topo
