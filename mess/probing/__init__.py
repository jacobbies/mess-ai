"""
MERT Layer Discovery & Probing

Discover what musical aspects each MERT layer encodes through linear probing.

Public API:
    LayerDiscoverySystem - Run probing experiments to find layer specializations
    MusicalAspectTargets - Generate proxy targets for probing (audio-derived)
    MidiExpressionTargets - Generate expression targets from MIDI (optional)
    create_target_dataset - Batch generate targets for a dataset
"""

from .discovery import (
    ASPECT_REGISTRY,
    LayerDiscoverySystem,
    inspect_model,
    resolve_aspects,
    trace_activations,
)
from .midi_targets import MidiExpressionTargets, resolve_midi_path
from .targets import MusicalAspectTargets, create_target_dataset

__all__ = [
    'LayerDiscoverySystem',
    'inspect_model',
    'trace_activations',
    'ASPECT_REGISTRY',
    'resolve_aspects',
    'MusicalAspectTargets',
    'MidiExpressionTargets',
    'resolve_midi_path',
    'create_target_dataset',
]
