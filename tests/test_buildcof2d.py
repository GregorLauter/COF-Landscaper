from pathlib import Path

import coflandscaper as cl


def test_buildcof2d(tmp_path: Path) -> None:
    """Build a COF from example inputs and assert output is written."""
    repo_root = Path(__file__).resolve().parents[1]
    node_file = repo_root / "examples/COF-1/0_all/0_node/boronate_ester.xyz"
    linker_file = repo_root / "examples/COF-1/0_all/0_linker/2-Benzene.xyz"

    cof_name = "cof-test"
    output_folder = tmp_path / cof_name / f"1_{cof_name}_single_layer"

    builder = cl.BuildCOF2D()
    outputs = builder.build(
        topo="hcb",
        bond_type="single",
        cof_name=cof_name,
        input_node=node_file,
        input_linker=linker_file,
        output_folder=str(output_folder),
    )

    assert len(outputs) == 1
    assert Path(outputs[0]).exists()
