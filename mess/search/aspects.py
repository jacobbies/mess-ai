"""
Aspect registry: maps user-facing musical aspects to probing targets.

This module is the bridge between layer discovery (which validates that
layer N encodes proxy target X) and the recommender (which lets users
search by musical aspect names like "brightness" or "dynamics").

The registry defines which probing targets back each user-facing aspect.
At runtime, resolve_aspects() loads discovery results and finds which
MERT layer best encodes each aspect.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import mess_config

logger = logging.getLogger(__name__)


# Maps user-facing aspect name → probing target(s) that validate it.
# When multiple targets are listed, the one with highest R² is used.
ASPECT_REGISTRY: Dict[str, Dict[str, Any]] = {
    'brightness': {
        'targets': ['spectral_centroid'],
        'description': 'Timbral brightness or darkness of the sound',
    },
    'texture': {
        'targets': ['spectral_rolloff', 'zero_crossing_rate'],
        'description': 'Surface texture: smooth vs noisy/rough',
    },
    'warmth': {
        'targets': ['spectral_bandwidth', 'spectral_centroid'],
        'description': 'Tonal warmth and fullness',
    },
    'tempo': {
        'targets': ['tempo'],
        'description': 'Speed and BPM similarity',
    },
    'rhythmic_energy': {
        'targets': ['onset_density'],
        'description': 'Note density and rhythmic activity',
    },
    'dynamics': {
        'targets': ['dynamic_range', 'dynamic_variance'],
        'description': 'Loudness variation and dynamic contrast',
    },
    'crescendo': {
        'targets': ['crescendo_strength', 'diminuendo_strength'],
        'description': 'Building or fading intensity',
    },
    'harmonic_richness': {
        'targets': ['harmonic_complexity'],
        'description': 'Harmonic content and tonal complexity',
    },
    'articulation': {
        'targets': ['attack_slopes', 'attack_sharpness'],
        'description': 'Legato vs staccato character',
    },
    'phrasing': {
        'targets': ['phrase_regularity', 'num_phrases'],
        'description': 'Musical sentence structure and regularity',
    },
}


def load_discovery_results(path: Optional[Path] = None) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Load probing results from JSON, returning {layer: {target: {r2_score, ...}}}."""
    results_path = path or mess_config.probing_results_file
    if not results_path.exists():
        return {}

    with open(results_path) as f:
        raw = json.load(f)

    # JSON keys are strings, convert back to int
    return {int(layer): targets for layer, targets in raw.items()}


def resolve_aspects(
    min_r2: float = 0.5,
    results_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve each aspect in the registry to its best validated MERT layer.

    Loads discovery results and for each aspect, finds the probing target
    with the highest R² score, then returns which layer to use.

    Args:
        min_r2: Minimum R² to consider a layer validated for an aspect.
        results_path: Path to discovery results JSON. Defaults to config.

    Returns:
        {aspect_name: {
            'layer': int,
            'target': str,        # which probing target matched
            'r2_score': float,
            'description': str,
            'confidence': str,    # 'high' (>0.8), 'medium' (>0.5), 'low'
        }}
        Only includes aspects that meet the min_r2 threshold.
    """
    results = load_discovery_results(results_path)
    if not results:
        logger.warning("No discovery results found. Run layer discovery first.")
        return {}

    resolved: Dict[str, Dict[str, Any]] = {}

    for aspect_name, aspect_info in ASPECT_REGISTRY.items():
        best_layer = -1
        best_r2 = -999.0
        best_target = ''

        for target_name in aspect_info['targets']:
            for layer, layer_results in results.items():
                if target_name in layer_results:
                    r2 = layer_results[target_name]['r2_score']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_layer = layer
                        best_target = target_name

        if best_r2 >= min_r2:
            if best_r2 > 0.8:
                confidence = 'high'
            elif best_r2 > 0.5:
                confidence = 'medium'
            else:
                confidence = 'low'

            resolved[aspect_name] = {
                'layer': best_layer,
                'target': best_target,
                'r2_score': round(best_r2, 4),
                'description': aspect_info['description'],
                'confidence': confidence,
            }
        else:
            logger.debug(
                f"Aspect '{aspect_name}' not validated: best R²={best_r2:.4f} < {min_r2}"
            )

    logger.info(f"Resolved {len(resolved)}/{len(ASPECT_REGISTRY)} aspects from discovery results")
    return resolved
