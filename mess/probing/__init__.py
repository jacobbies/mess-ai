"""
MERT Layer Discovery & Probing

Discover what musical aspects each MERT layer encodes through linear probing.

Public API:
    LayerDiscoverySystem - Run probing experiments to find layer specializations
    MusicalAspectTargets - Generate proxy targets for probing (audio-derived)
    MidiExpressionTargets - Generate expression targets from MIDI (optional)
    create_target_dataset - Batch generate targets for a dataset
    generate_segment_targets - Per-segment target generation for segment probing
    create_segment_target_dataset - Batch segment target generation
    SEGMENT_TARGETS - Target names viable at 5s segment level
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "LayerDiscoverySystem",
    "inspect_model",
    "trace_activations",
    "ASPECT_REGISTRY",
    "SEGMENT_TARGETS",
    "resolve_aspects",
    "MusicalAspectTargets",
    "MidiExpressionTargets",
    "resolve_midi_path",
    "create_target_dataset",
    "generate_segment_targets",
    "generate_segment_expression_targets",
    "create_segment_target_dataset",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "LayerDiscoverySystem": (".discovery", "LayerDiscoverySystem"),
    "inspect_model": (".discovery", "inspect_model"),
    "trace_activations": (".discovery", "trace_activations"),
    "ASPECT_REGISTRY": (".discovery", "ASPECT_REGISTRY"),
    "SEGMENT_TARGETS": (".discovery", "SEGMENT_TARGETS"),
    "resolve_aspects": (".discovery", "resolve_aspects"),
    "MusicalAspectTargets": (".targets", "MusicalAspectTargets"),
    "create_target_dataset": (".targets", "create_target_dataset"),
    "MidiExpressionTargets": (".midi_targets", "MidiExpressionTargets"),
    "resolve_midi_path": (".midi_targets", "resolve_midi_path"),
    "generate_segment_targets": (".segment_targets", "generate_segment_targets"),
    "generate_segment_expression_targets": (
        ".segment_targets",
        "generate_segment_expression_targets",
    ),
    "create_segment_target_dataset": (".segment_targets", "create_segment_target_dataset"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
