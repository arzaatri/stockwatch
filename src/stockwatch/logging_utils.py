"""Shared logging setup, so every module's log lines share one prefix format
and land in the same place: console + logs/YYYY-MM-DD.log.
"""

import logging
from datetime import date
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"

# %(name)s is the dotted module path (however many submodules deep), %(funcName)s
# is filled in automatically by `logging` from the caller's stack frame - together
# they give "module.submodule.function" with no per-call-site bookkeeping needed.
_LOG_FORMAT = "[%(asctime)s][%(name)s.%(funcName)s] %(message)s"


class _DotMsecFormatter(logging.Formatter):
    default_msec_format = "%s.%03d"  # "HH:MM:SS.mmm" instead of stdlib's "HH:MM:SS,mmm"


def get_logger(name: str) -> logging.Logger:
    """Module-level logger with "[YYYY-MM-DD hh:mm:ss.sss][module.submodule.function]"
    formatted output, per project logging convention. Writes to both stderr
    (visible when running a command directly) and logs/YYYY-MM-DD.log (one
    file per calendar day - a process started before midnight keeps writing
    to that day's file rather than rotating live, which is fine for a
    single-user local pipeline).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = _DotMsecFormatter(_LOG_FORMAT)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        LOGS_DIR.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(LOGS_DIR / f"{date.today().isoformat()}.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
