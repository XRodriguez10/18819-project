"""
Utility functions.
"""

import logging
import sys


def setup_logger(name: str, verbose: bool = False) -> logging.Logger:
    """Create and configure a logger given its name."""
    logger = logging.getLogger(name)
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s: %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
