"""Central logging configuration for the trading platform."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> None:
    """Configure root logging once (idempotent safe for Streamlit reruns).

    Args:
        level: Logging level for the root logger.
        name: Optional logger name to configure instead of root (unused; reserved).
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)
