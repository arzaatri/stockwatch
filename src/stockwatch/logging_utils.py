"""Shared logging setup, so every module's log lines share one prefix format."""

import logging

_LOG_FORMAT = "[%(asctime)s][%(name)s] %(message)s"


class _DotMsecFormatter(logging.Formatter):
    default_msec_format = "%s.%03d"  # "HH:MM:SS.mmm" instead of stdlib's "HH:MM:SS,mmm"


def get_logger(name: str) -> logging.Logger:
    """Module-level logger with "[YYYY-MM-DD hh:mm:ss.sss][module.submodule.function]"
    formatted output, per project logging convention.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_DotMsecFormatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
