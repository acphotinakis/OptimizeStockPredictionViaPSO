"""
src/utils/logger.py
Centralised logging setup using loguru.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_file: str | Path | None = None, level: str = "INFO") -> None:
    """Configure loguru for the project.

    Args:
        log_file: Optional path to write logs to disk.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, rotation="10 MB", retention="7 days")
