"""
Proxy target generators for musical aspects in classical music.

This package hosts per-target modules so new generators can land as small,
independent files (see F1 plan). The original track-level scalar generators
remain in :mod:`._legacy` and are re-exported here to preserve the existing
public API (``MusicalAspectTargets``, ``create_target_dataset``, ``main``).

Importing this package populates :mod:`._registry` with every Phase-2
generator (T1-T7). Each generator module calls :func:`register` at import
time, so a single ``from mess.probing.targets import all_names`` is enough
to enumerate every available target.
"""

from __future__ import annotations

# Phase-2 generator modules — imported for their ``register()`` side effects.
# Keep the imports one-per-line so conflicts show up cleanly in future PRs.
from . import (  # noqa: F401
    _centroid_trajectory,
    _dynamic_arc,
    _local_tempo,
    _midi_articulation,
    _midi_expressivity,
    _novelty,
    _tis_tension,
)
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
