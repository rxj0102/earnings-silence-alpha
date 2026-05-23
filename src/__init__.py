"""
src
===
Earnings Call Silence Mapping via Transcript Topic Omission.

Package initialisation — exposes the top-level package namespace and
configures the project-wide logger so every sub-module can do::

    import logging
    logger = logging.getLogger(__name__)

without additional setup.
"""

import logging
import sys

# ---------------------------------------------------------------------------
# Project-wide logging configuration
# ---------------------------------------------------------------------------
_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_FMT,
    datefmt=_DATEFMT,
    handlers=[logging.StreamHandler(sys.stdout)],
)

__version__ = "0.1.0"
__author__ = "Earnings Silence Alpha Research"
