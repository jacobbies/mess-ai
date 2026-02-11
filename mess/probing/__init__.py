"""
MERT Layer Discovery & Probing

Discover what musical aspects each MERT layer encodes through linear probing.

Public API:
    LayerDiscoverySystem - Run probing experiments to find layer specializations
    MusicalAspectTargets - Generate proxy targets for probing
    create_target_dataset - Batch generate targets for a dataset
"""

from .discovery import LayerDiscoverySystem, inspect_model, trace_activations
from .targets import MusicalAspectTargets, create_target_dataset

__all__ = [
    'LayerDiscoverySystem',
    'inspect_model',
    'trace_activations',
    'MusicalAspectTargets',
    'create_target_dataset',
]
