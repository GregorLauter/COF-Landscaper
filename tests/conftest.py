import pathlib
from dataclasses import dataclass

import pytest


@dataclass(slots=True, frozen=True)
class CaseData:
    topology: str
    bond_type: str
    cof_name: str
    running_directory: pathlib.Path


@pytest.fixture(
    scope="session",
    params=(
        lambda: CaseData(
            topology="sql",
            bond_type="stk.PdbWriter()",
            cof_name="",
            running_directory=pathlib.Path(
                "inputs"
            ),  # get the directory of this current script, and add the outputs here
        ),
    ),
)
def case_data(request) -> CaseData:
    return request.param()
