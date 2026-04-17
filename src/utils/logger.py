"""
src/utils/logger.py
Centralised logging setup using standard logging library.
"""

import logging
import sys
from pathlib import Path
from enum import Enum


class LogFileMode(str, Enum):
    APPEND = "a"
    OVERWRITE = "w"


def setup_logger(
    log_file: str | Path | None = None,
    level: str = "INFO",
    mode: LogFileMode = LogFileMode.APPEND,
) -> None:
    """Configure logging for the project.

    Args:
        log_file: Optional path to write logs to disk.
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        # file_handler = logging.FileHandler(log_file)
        file_handler = logging.FileHandler(log_file, mode=mode.value)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
