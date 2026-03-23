from __future__ import annotations

from typing import Any

__all__ = ["build_feature_matrix"]


def __getattr__(name: str) -> Any:
    if name == "build_feature_matrix":
        from trading_platform.data.pipeline import build_feature_matrix

        return build_feature_matrix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
