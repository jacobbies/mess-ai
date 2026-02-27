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

from .discovery import (
    ASPECT_REGISTRY,
    SEGMENT_TARGETS,
    LayerDiscoverySystem,
    inspect_model,
    resolve_aspects,
    trace_activations,
)
from .midi_targets import MidiExpressionTargets, resolve_midi_path
from .segment_targets import (
    create_segment_target_dataset,
    generate_segment_expression_targets,
    generate_segment_targets,
)
from .targets import MusicalAspectTargets, create_target_dataset

__all__ = [
    'LayerDiscoverySystem',
    'inspect_model',
    'trace_activations',
    'ASPECT_REGISTRY',
    'SEGMENT_TARGETS',
    'resolve_aspects',
    'MusicalAspectTargets',
    'MidiExpressionTargets',
    'resolve_midi_path',
    'create_target_dataset',
    'generate_segment_targets',
    'generate_segment_expression_targets',
    'create_segment_target_dataset',
]
