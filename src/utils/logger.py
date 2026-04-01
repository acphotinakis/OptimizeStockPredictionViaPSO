import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str, log_dir: str = "logs", level: int = logging.INFO
) -> logging.getLogger:
    """
    Initializes a dual-handler logger (Console + File).

    Args:
        name: The name of the module or run.
        log_dir: Directory to store timestamped log files.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        A configured logging.Logger instance.
    """
    # Ensure log directory exists
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Create a unique filename for the current run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = path / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if logger is re-initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
