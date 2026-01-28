from ._internal.prepare_xyz import prepare_xyz, PrepareXYZ
import importlib
from ._internal.check_ild import CheckIld
from ._internal.change_ild import ChangeIld
from ._internal.supercell import Supercell
from ._internal.ils_serr import IlsSerr
from ._internal.ils_incl import IlsIncl
from ._internal.set_vacuum import SetVacuum
from ._internal.center_z import CenterZ
from ._internal.remove_layer import RemoveLayer
from ._internal.calc_ils_sl import CalcIlsSl
from ._internal.calc_ils_dl import CalcIlsDl


def __getattr__(name):
    if name in ("build_cof_2d", "BuildCOF2D"):
        mod = importlib.import_module(". _internal.build_cof_2d".replace(" ", ""), __name__)
        return getattr(mod, name)
    if name in ("opt_mace", "OptMACE"):
        mod = importlib.import_module(". _internal.opt_mace".replace(" ", ""), __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "prepare_xyz",
    "PrepareXYZ",
    "build_cof_2d",
    "BuildCOF2D",
    "opt_mace",
    "OptMACE",
    "CheckIld",
    "ChangeIld",
    "Supercell",
    "IlsSerr",
    "IlsIncl",
    "SetVacuum",
    "CenterZ",
    "RemoveLayer",
    "CalcIlsSl",
    "CalcIlsDl",
]
