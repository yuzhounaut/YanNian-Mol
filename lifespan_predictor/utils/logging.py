"""Logging utilities for the lifespan predictor."""

import logging
import sys
from pathlib import Path
from typing import Optional


# ANSI color codes for console output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}" f"{record.levelname}" f"{self.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str, log_file: Optional[str] = None, level: int = logging.INFO, use_colors: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name (typically __name__ of the module)
        log_file: Optional path to log file. If provided, logs will be written to file
        level: Logging level (default: logging.INFO)
        use_colors: Whether to use colored output for console (default: True)

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, log_file="training.log")
        >>> logger.info("Training started")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatters
    detailed_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    simple_format = "%(levelname)s - %(message)s"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors and sys.stdout.isatty():
        # Use colored formatter for interactive terminals
        console_formatter = ColoredFormatter(simple_format)
    else:
        # Use plain formatter for non-interactive or when colors disabled
        console_formatter = logging.Formatter(simple_format)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(detailed_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
