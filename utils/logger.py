import logging
from pathlib import Path


def get_logger(
    name: str = "cslr",
    log_dir: str | Path | None = None,
    log_filename: str = "train.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a logger.

    The logger writes messages to:
        1. Console
        2. Log file (if log_dir is provided)

    Args:
        name: Logger name.
        log_dir: Directory used to save the log file.
        log_filename: Log file name.
        level: Logging level.

    Returns:
        Configured logging.Logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicated handlers when get_logger()
    # is called multiple times.
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --------------------------------------------------
    # Console handler
    # --------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # --------------------------------------------------
    # File handler
    # --------------------------------------------------
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / log_filename

        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    # Prevent messages from being propagated to the root logger.
    logger.propagate = False

    return logger
