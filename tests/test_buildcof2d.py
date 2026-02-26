import coflandscaper as cl

from .conftest import CaseData


def test_buildcof2d(case_data: CaseData) -> None:
    """Test class."""
    builder = cl.BuildCOF2D()
    builder.build(
        topo=case_data.topology,
        bond_type=case_data.bond_type,
        cof_name=case_data.cof_name,
        output_folder=case_data.running_directory,
    )
    preopt = cl.MacePreopt()
    preopt.run(cof_name=case_data.cof_name)
