"""
Proxy target generators for musical aspects in classical music.

This package hosts per-target modules so new generators can land as small,
independent files (see F1 plan). The original track-level scalar generators
remain in :mod:`._legacy` and are re-exported here to preserve the existing
public API (``MusicalAspectTargets``, ``create_target_dataset``, ``main``).
"""

from __future__ import annotations

from ._legacy import (
    MusicalAspectTargets,
    create_target_dataset,
    main,
)
from ._registry import (
    all_names,
    get_generator,
    register,
)

__all__ = [
    "MusicalAspectTargets",
    "create_target_dataset",
    "main",
    "all_names",
    "get_generator",
    "register",
]
