"""
Per-target generator registry. T1-T8 units add entries here.

Each entry maps ``target_name -> (descriptor, generator_fn)``. The generator
function is intentionally unconstrained at the type level because T1-T8
units have different signatures (audio-only vs audio+MIDI). Callers of
``get_generator`` must know what to pass.
"""

from __future__ import annotations

from collections.abc import Callable

from .._schema import TargetDescriptor

_GENERATORS: dict[str, tuple[TargetDescriptor, Callable]] = {}


def register(name: str, descriptor: TargetDescriptor, fn: Callable) -> None:
    """Register a target generator under ``name``.

    Re-registering an existing name overwrites the previous entry to keep
    iterative development ergonomic.
    """
    _GENERATORS[name] = (descriptor, fn)


def get_generator(name: str) -> tuple[TargetDescriptor, Callable]:
    """Return the ``(descriptor, fn)`` registered for ``name``."""
    return _GENERATORS[name]


def all_names() -> list[str]:
    """Return all registered target names, sorted."""
    return sorted(_GENERATORS)
