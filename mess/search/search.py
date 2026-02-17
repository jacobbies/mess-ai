"""
Simple FAISS-based similarity search for music recommendations.

Uses exact cosine similarity via FAISS IndexFlatIP for fast, scalable search.
Supports layer-specific search using validated MERT layer discoveries.

Usage:
    from mess.search import find_similar, load_features

    # Load features
    features, track_names = load_features("data/embeddings/smd-emb/aggregated")

    # Find similar tracks
    similar = find_similar(
        query_track="Beethoven_Op027No1-01",
        features=features,
        track_names=track_names,
        k=10
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def load_features(
    features_dir: str,
    layer: Optional[int] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load MERT features from directory.

    Args:
        features_dir: Directory containing .npy feature files
        layer: Specific MERT layer to use (0-12), or None for aggregated features

    Returns:
        features: Array of shape (n_tracks, feature_dim)
        track_names: List of track IDs
    """
    features_dir = Path(features_dir)

    if not features_dir.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    feature_files = sorted(features_dir.glob("*.npy"))

    if not feature_files:
        raise ValueError(f"No .npy files found in {features_dir}")

    features_list = []
    track_names = []

    for feature_file in feature_files:
        features = np.load(feature_file)

        # Extract specific layer if requested
        if layer is not None:
            if features.ndim == 2 and features.shape[0] == 13:  # [13, 768] format
                features = features[layer]
            else:
                logger.warning(f"Cannot extract layer {layer} from {feature_file.name}")
                continue

        features_list.append(features)
        track_names.append(feature_file.stem)

    features_array = np.vstack(features_list)

    logger.info(f"Loaded {len(track_names)} tracks with {features_array.shape[1]} features")

    return features_array, track_names


def build_index(features: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build FAISS flat index for exact cosine similarity search.

    Args:
        features: Feature array of shape (n_tracks, feature_dim)

    Returns:
        FAISS index ready for searching
    """
    # Normalize features for cosine similarity
    features = features.astype('float32')
    faiss.normalize_L2(features)

    # Build flat index (exact search)
    dimension = features.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
    index.add(features)

    return index


def find_similar(
    query_track: str,
    features: np.ndarray,
    track_names: List[str],
    k: int = 10,
    exclude_self: bool = True
) -> List[Tuple[str, float]]:
    """
    Find k most similar tracks using cosine similarity.

    Args:
        query_track: Track ID to find similar tracks for
        features: Feature array of shape (n_tracks, feature_dim)
        track_names: List of track IDs corresponding to features
        k: Number of similar tracks to return
        exclude_self: Whether to exclude the query track from results

    Returns:
        List of (track_id, similarity_score) tuples, sorted by descending similarity
    """
    if query_track not in track_names:
        raise ValueError(f"Query track '{query_track}' not found in dataset")

    # Get query features
    query_idx = track_names.index(query_track)
    query_features = features[query_idx:query_idx+1].astype('float32')

    # Build index and search
    index = build_index(features)
    faiss.normalize_L2(query_features)

    # Search for k+1 to account for self if excluding
    search_k = k + 1 if exclude_self else k
    distances, indices = index.search(query_features, search_k)

    # Convert to results
    results = []
    for idx, score in zip(indices[0], distances[0]):
        track_id = track_names[idx]

        # Skip self if requested
        if exclude_self and track_id == query_track:
            continue

        results.append((track_id, float(score)))

        if len(results) >= k:
            break

    return results


def search_by_aspect(
    query_track: str,
    aspect: str,
    features_dir: str,
    k: int = 10
) -> List[Tuple[str, float]]:
    """
    Find similar tracks using a specific musical aspect.

    Requires layer discovery results to map aspects to layers.

    Args:
        query_track: Track ID to find similar tracks for
        aspect: Musical aspect name (e.g., 'brightness', 'texture', 'dynamics')
        features_dir: Directory containing aggregated features
        k: Number of similar tracks to return

    Returns:
        List of (track_id, similarity_score) tuples
    """
    from ..probing import resolve_aspects

    # Resolve aspect to layer
    aspect_mappings = resolve_aspects()

    if not aspect_mappings:
        raise ValueError("No layer discovery results found. Run layer discovery first.")

    if aspect not in aspect_mappings:
        available = ', '.join(aspect_mappings.keys())
        raise ValueError(f"Aspect '{aspect}' not validated. Available: {available}")

    layer = aspect_mappings[aspect]['layer']
    confidence = aspect_mappings[aspect]['confidence']

    logger.info(f"Using layer {layer} for '{aspect}' (confidence: {confidence})")

    # Load features for specific layer
    features, track_names = load_features(features_dir, layer=layer)

    # Search
    return find_similar(query_track, features, track_names, k=k)
