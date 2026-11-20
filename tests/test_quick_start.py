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
    """Compile the cells and bind their public calls without instantiating calculators."""
    notebook = json.loads((_QUICK_START / "quick_start.ipynb").read_text())
    owners = {
        "opt": cl.GeometryOpt,
        "opt.single_point": cl.SinglePoint,
        "pxrd": cl.PXRD,
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, cell["id"], "exec")
        for call in ast.walk(ast.parse(source)):
            if not isinstance(call, ast.Call):
                continue
            path = ast.unparse(call.func).removeprefix("cl.")
            owner, _, name = path.rpartition(".")
            if owner in owners:
                member = getattr(owners[owner], name)
            elif owner:
                member = getattr(getattr(cl, owner.removesuffix("()")), name)
            else:
                member = getattr(cl, name)
            signature = inspect.signature(member)
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
