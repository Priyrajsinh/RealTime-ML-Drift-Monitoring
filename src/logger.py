"""Centralized JSON logger for B5 project. Always import from here."""

import logging
import sys

from pythonjsonlogger import json as jsonlogger


def get_logger(name: str) -> logging.Logger:
    """Return a JSON-formatted logger with the given name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
