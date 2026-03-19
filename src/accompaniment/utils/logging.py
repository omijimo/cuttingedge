"""Logging configuration."""

import logging as _logging
import sys
from pathlib import Path


def setup_logging(
    log_dir: str | Path | None = None,
    level: int = _logging.INFO,
) -> _logging.Logger:
    logger = _logging.getLogger("accompaniment")
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = _logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = _logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = _logging.FileHandler(log_dir / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
