"""Import PORMAKE without opening its process-relative runtime.log."""

import logging
import os
import sys
from typing import Any
from unittest.mock import patch

# PORMAKE constructs its FileHandler at import in truncating mode. Other
# handlers, including those created by other threads, keep their destination.
_original_init = logging.FileHandler.__init__


def _init_handler(
    self: logging.FileHandler,
    filename: str | os.PathLike[str],
    *args: Any,
    **kwargs: Any,
) -> None:
    if (
        filename == "runtime.log"
        and sys._getframe(1).f_globals.get("__name__") == "pormake.log"
    ):
        filename = os.devnull
    _original_init(self, filename, *args, **kwargs)


with patch.object(logging.FileHandler, "__init__", _init_handler):
    import pormake as pm
    from pormake import log
    from pormake.building_block import BuildingBlock
    from pormake.framework import Framework
    from pormake.topology import Topology

log.logger.removeHandler(log.file_log_handler)
log.file_log_handler.close()

__all__ = ["BuildingBlock", "Framework", "Topology", "pm"]
