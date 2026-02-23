"""
Structured logging setup using loguru.
Provides application-wide logger with file rotation, rich console output,
and structured JSON logging for production environments.
"""
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    serialize: bool = False,
    rotation: str = "100 MB",
    retention: str = "30 days",
    colorize: bool = True,
) -> None:
    """Configure the global loguru logger.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, logs only to console.
        serialize: If True, output JSON logs (for production log aggregators).
        rotation: When to rotate log file (size like "100 MB" or time like "1 day").
        retention: How long to keep old log files.
        colorize: Whether to colorize console output.
    """
    # Remove default handler
    logger.remove()

    # Console handler
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=colorize,
        backtrace=True,
        diagnose=True,
    )

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation=rotation,
            retention=retention,
            serialize=serialize,
            compression="gz",
            backtrace=True,
            diagnose=False,
            enqueue=True,   # Thread-safe async writing
        )

    logger.info(f"Logger initialized | level={log_level} | file={log_file}")


def get_logger(name: str):
    """Get a named logger instance with contextual binding.

    Args:
        name: Module name (typically __name__).

    Returns:
        Loguru logger bound to the given name.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("Starting data ingestion")
    """
    return logger.bind(name=name)


# Pre-configured module logger
log = get_logger(__name__)
