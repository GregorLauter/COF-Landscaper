"""Check the shipped notebook and inputs without running its calculations."""

import ast
import inspect
import json
from pathlib import Path

import coflandscaper as cl

_QUICK_START = (
    Path(__file__).resolve().parents[1]
    / "examples/main_workflow/local/quick_start"
)


def test_quick_start_public_api() -> None:
    """Each cell needs only the setup names and calls the installed public API."""
    notebook = json.loads((_QUICK_START / "quick_start.ipynb").read_text())
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
            signature.bind(
                *args,
                *call.args,
                **{kw.arg: kw.value for kw in call.keywords if kw.arg},
            )


def test_quick_start_manual_inputs() -> None:
    for folder, sites in (("node", 4), ("linker", 2)):
        input_dir = _QUICK_START / f"0_{folder}"
        xyz = input_dir / f"ILCOF-1_{folder}.xyz"
        assert list(input_dir.glob("*.xyz")) == [xyz]
        count, _, *atoms = xyz.read_text().splitlines()
        assert len(atoms) == int(count)
        assert sum(atom.split()[0] == "He" for atom in atoms) == sites
    assert (_QUICK_START / "experimental_pxrd/ILCOF-1.xy").is_file()
