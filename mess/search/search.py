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

import importlib
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

from .._runtime import configure_faiss_runtime
from .contracts import ClipLocation, ClipMetadata, ClipSearchResult

logger = logging.getLogger(__name__)

NUM_LAYERS = 13
EMBEDDING_DIM = 768
DEFAULT_SEGMENT_DURATION = 5.0
DEFAULT_OVERLAP_RATIO = 0.5
DEFAULT_DEDUPE_WINDOW_SECONDS = 5.0


def _require_faiss() -> Any:
    """Import faiss lazily so module import works without search dependencies."""
    try:
        return configure_faiss_runtime(importlib.import_module("faiss"))
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss is required for similarity search. Install it with `pip install faiss-cpu`."
        ) from exc

def _vectorize_track_features(
    raw_features: np.ndarray,
    layer: int | None,
    source_name: str
) -> np.ndarray:
    """
    Convert a per-track embedding array into exactly one 1D vector.

    Supported shapes:
    - [13, 768] (aggregated): layer select or flatten
    - [segments, 13, 768] (segments): mean over segments, then layer select or flatten
    - [segments, 13, time, 768] (raw): mean over segments/time, then layer select or flatten
    - [feature_dim] (already vectorized): used directly when layer is None
    """
    features = np.asarray(raw_features)

    if features.ndim == 1:
        if layer is not None:
            raise ValueError(
                f"Cannot apply layer={layer} to 1D features in {source_name}; "
                "expected a [13, 768]-compatible array."
            )
        return cast(np.ndarray, features.astype("float32"))

    if features.ndim == 2:
        if features.shape == (NUM_LAYERS, EMBEDDING_DIM):
            vector = features[layer] if layer is not None else features.reshape(-1)
            return cast(np.ndarray, vector.astype("float32"))
        raise ValueError(
            f"Unsupported 2D shape {features.shape} in {source_name}; "
            f"expected ({NUM_LAYERS}, {EMBEDDING_DIM})."
        )

    if features.ndim == 3:
        if features.shape[1:] == (NUM_LAYERS, EMBEDDING_DIM):
            pooled = features.mean(axis=0)  # [13, 768]
            vector = pooled[layer] if layer is not None else pooled.reshape(-1)
            return cast(np.ndarray, vector.astype("float32"))
        raise ValueError(
            f"Unsupported 3D shape {features.shape} in {source_name}; "
            f"expected [segments, {NUM_LAYERS}, {EMBEDDING_DIM}]."
        )

    if features.ndim == 4:
        if features.shape[1] == NUM_LAYERS and features.shape[3] == EMBEDDING_DIM:
            pooled = features.mean(axis=(0, 2))  # [13, 768]
            vector = pooled[layer] if layer is not None else pooled.reshape(-1)
            return cast(np.ndarray, vector.astype("float32"))
        raise ValueError(
            f"Unsupported 4D shape {features.shape} in {source_name}; "
            f"expected [segments, {NUM_LAYERS}, time, {EMBEDDING_DIM}]."
        )

    raise ValueError(
        f"Unsupported feature rank {features.ndim} in {source_name}; "
        "expected rank 1-4."
    )


def _vectorize_segment_features(
    raw_features: np.ndarray,
    layer: int | None,
    source_name: str,
) -> np.ndarray:
    """
    Convert per-segment embeddings into one vector per segment.

    Supported shapes:
    - [segments, 13, 768]
    - [segments, 13, time, 768]
    - [13, 768] (treated as one segment)
    """
    features = np.asarray(raw_features)

    if features.ndim == 2:
        if features.shape != (NUM_LAYERS, EMBEDDING_DIM):
            raise ValueError(
                f"Unsupported 2D shape {features.shape} in {source_name}; "
                f"expected ({NUM_LAYERS}, {EMBEDDING_DIM})."
            )
        features = features[None, :, :]

    elif features.ndim == 3:
        if features.shape[1:] != (NUM_LAYERS, EMBEDDING_DIM):
            raise ValueError(
                f"Unsupported 3D shape {features.shape} in {source_name}; "
                f"expected [segments, {NUM_LAYERS}, {EMBEDDING_DIM}]."
            )

    elif features.ndim == 4:
        if features.shape[1] != NUM_LAYERS or features.shape[3] != EMBEDDING_DIM:
            raise ValueError(
                f"Unsupported 4D shape {features.shape} in {source_name}; "
                f"expected [segments, {NUM_LAYERS}, time, {EMBEDDING_DIM}]."
            )
        features = features.mean(axis=2)
    else:
        raise ValueError(
            f"Unsupported feature rank {features.ndim} in {source_name}; "
            "expected rank 2-4."
        )

    if layer is not None:
        return cast(np.ndarray, features[:, layer, :].astype("float32"))
    return cast(np.ndarray, features.reshape(features.shape[0], -1).astype("float32"))


def _segment_hop_seconds(
    segment_duration: float,
    overlap_ratio: float,
) -> float:
    hop = segment_duration * (1.0 - overlap_ratio)
    if hop <= 0:
        raise ValueError("segment_duration and overlap_ratio must produce hop > 0")
    return hop


def _segment_bounds(
    segment_idx: int,
    segment_duration: float,
    overlap_ratio: float,
) -> tuple[float, float]:
    hop_seconds = _segment_hop_seconds(segment_duration, overlap_ratio)
    start_time = segment_idx * hop_seconds
    end_time = start_time + segment_duration
    return start_time, end_time


def _resolve_query_segment_index(
    track_clip_locations: list[ClipLocation],
    clip_start: float,
) -> int:
    if clip_start < 0:
        raise ValueError("clip_start must be >= 0")

    return min(
        range(len(track_clip_locations)),
        key=lambda i: abs(track_clip_locations[i].start_time - clip_start),
    )


def _load_clip_artifact(artifact: str | Path | Any) -> Any:
    if isinstance(artifact, (str, Path)):
        from .faiss_index import load_artifact

        loaded = load_artifact(artifact)
    else:
        loaded = artifact

    if loaded.manifest.kind != "clip":
        raise ValueError(
            f"search_by_clip requires a clip artifact, got kind={loaded.manifest.kind!r}"
        )
    return loaded


def _resolve_artifact_query_index(
    *,
    clip_records: list[ClipMetadata],
    clip_id: str | None,
    track_id: str | None,
    start_sec: float | None,
) -> int:
    if clip_id is not None:
        if track_id is not None or start_sec is not None:
            raise ValueError("Provide either clip_id or track_id/start_sec, not both")
        for idx, record in enumerate(clip_records):
            if record.clip_id == clip_id:
                return idx
        raise ValueError(f"Query clip_id '{clip_id}' not found in artifact")

    if track_id is None or start_sec is None:
        raise ValueError("Provide either clip_id or both track_id and start_sec")
    if start_sec < 0:
        raise ValueError("start_sec must be >= 0")

    track_indices = [
        idx for idx, record in enumerate(clip_records) if record.track_id == track_id
    ]
    if not track_indices:
        raise ValueError(f"Query track '{track_id}' not found in artifact")

    track_locations = [clip_records[idx].to_location() for idx in track_indices]
    relative_query_idx = _resolve_query_segment_index(track_locations, start_sec)
    return track_indices[relative_query_idx]


def _query_vector_from_artifact(artifact: Any, query_idx: int) -> np.ndarray:
    if artifact.vectors is not None:
        return np.array(artifact.vectors[query_idx:query_idx + 1], dtype=np.float32, copy=True)

    if not hasattr(artifact.index, "reconstruct"):
        raise ValueError(
            "Clip artifact is missing persisted vectors required for clip queries. "
            "Rebuild the artifact with stored vectors."
        )

    try:
        reconstructed = artifact.index.reconstruct(int(query_idx))
    except Exception as exc:  # pragma: no cover - FAISS errors vary by backend
        raise ValueError(
            "Clip artifact could not reconstruct the query vector. "
            "Rebuild the artifact with stored vectors."
        ) from exc

    return np.asarray(reconstructed, dtype=np.float32).reshape(1, -1)


def load_segment_features(
    features_dir: str,
    layer: int | None = None,
    segment_duration: float = DEFAULT_SEGMENT_DURATION,
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO,
) -> tuple[np.ndarray, list[ClipLocation]]:
    """
    Load per-segment vectors and clip metadata from embeddings directory.

    Args:
        features_dir: Directory containing segment embeddings as .npy files
        layer: Specific MERT layer to use (0-12), or None for flattened 13x768 vectors
        segment_duration: Segment duration used during extraction
        overlap_ratio: Segment overlap ratio used during extraction

    Returns:
        features: Array of shape (n_segments, feature_dim)
        clip_locations: List of segment metadata aligned to features rows
    """
    if layer is not None and not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"Invalid layer {layer}; expected range [0, {NUM_LAYERS - 1}]")

    features_dir_path = Path(features_dir)
    if not features_dir_path.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir_path}")

    feature_files = sorted(features_dir_path.glob("*.npy"))
    if not feature_files:
        raise ValueError(f"No .npy files found in {features_dir_path}")

    all_vectors: list[np.ndarray] = []
    clip_locations: list[ClipLocation] = []

    for feature_file in feature_files:
        segment_vectors = _vectorize_segment_features(
            np.load(feature_file),
            layer,
            feature_file.name,
        )
        track_id = feature_file.stem
        for segment_idx, segment_vector in enumerate(segment_vectors):
            start_time, end_time = _segment_bounds(
                segment_idx,
                segment_duration=segment_duration,
                overlap_ratio=overlap_ratio,
            )
            all_vectors.append(segment_vector)
            clip_locations.append(
                ClipLocation(
                    track_id=track_id,
                    segment_idx=segment_idx,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

    if not all_vectors:
        raise ValueError(f"No valid segment features found in {features_dir_path}")

    features = np.vstack(all_vectors).astype("float32")
    logger.info(
        "Loaded %d segment clips across %d tracks with %d-D vectors",
        len(clip_locations),
        len({loc.track_id for loc in clip_locations}),
        features.shape[1],
    )
    return features, clip_locations


def load_features(
    features_dir: str,
    layer: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Load MERT features from directory.

    Args:
        features_dir: Directory containing .npy feature files
        layer: Specific MERT layer to use (0-12), or None for aggregated features

    Returns:
        features: Array of shape (n_tracks, feature_dim)
        track_names: List of track IDs
    """
    if layer is not None and not 0 <= layer < NUM_LAYERS:
        raise ValueError(f"Invalid layer {layer}; expected range [0, {NUM_LAYERS - 1}]")

    features_dir_path = Path(features_dir)

    if not features_dir_path.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir_path}")

    feature_files = sorted(features_dir_path.glob("*.npy"))

    if not feature_files:
        raise ValueError(f"No .npy files found in {features_dir_path}")

    features_list = []
    track_names = []

    for feature_file in feature_files:
        features = np.load(feature_file)
        vector = _vectorize_track_features(features, layer, feature_file.name)
        features_list.append(vector)
        track_names.append(feature_file.stem)

    if not features_list:
        raise ValueError(f"No valid feature files found in {features_dir_path}")

    features_array = np.vstack(features_list)

    logger.info(f"Loaded {len(track_names)} tracks with {features_array.shape[1]} features")

    return features_array, track_names


def _load_layer_features(
    features_dir: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Load per-track features and normalize to [n_tracks, 13, 768].

    Supports input files saved as:
    - [13, 768]
    - [segments, 13, 768] (pooled over segments)
    - [segments, 13, time, 768] (pooled over segments/time)
    """
    features_dir_path = Path(features_dir)

    if not features_dir_path.exists():
        raise FileNotFoundError(f"Features directory not found: {features_dir_path}")

    feature_files = sorted(features_dir_path.glob("*.npy"))
    if not feature_files:
        raise ValueError(f"No .npy files found in {features_dir_path}")

    layer_features: list[np.ndarray] = []
    track_names: list[str] = []

    for feature_file in feature_files:
        features = np.asarray(np.load(feature_file))

        if features.ndim == 2 and features.shape == (NUM_LAYERS, EMBEDDING_DIM):
            matrix = features
        elif features.ndim == 3 and features.shape[1:] == (NUM_LAYERS, EMBEDDING_DIM):
            matrix = features.mean(axis=0)
        elif (
            features.ndim == 4
            and features.shape[1] == NUM_LAYERS
            and features.shape[3] == EMBEDDING_DIM
        ):
            matrix = features.mean(axis=(0, 2))
        else:
            raise ValueError(
                f"Unsupported shape {features.shape} in {feature_file.name}; "
                f"expected [13, 768], [segments, 13, 768], or "
                f"[segments, 13, time, 768]."
            )

        layer_features.append(matrix.astype("float32"))
        track_names.append(feature_file.stem)

    return np.stack(layer_features), track_names


def _compose_weighted_vectors(
    layer_features: np.ndarray,
    layer_weights: np.ndarray,
) -> np.ndarray:
    """Compose one vector per track from per-layer vectors and layer weights."""
    if layer_features.ndim != 3 or layer_features.shape[1:] != (NUM_LAYERS, EMBEDDING_DIM):
        raise ValueError(
            f"Expected layer features shape [n_tracks, {NUM_LAYERS}, {EMBEDDING_DIM}], "
            f"got {layer_features.shape}"
        )

    if layer_weights.shape != (NUM_LAYERS,):
        raise ValueError(
            f"Expected layer weights shape ({NUM_LAYERS},), got {layer_weights.shape}"
        )

    weight_sum = float(layer_weights.sum())
    if weight_sum <= 0:
        raise ValueError("Layer weights must sum to > 0")

    normalized_weights = layer_weights / weight_sum

    # Normalize each layer vector before weighted fusion to avoid norm dominance.
    layer_norms = np.linalg.norm(layer_features, axis=2, keepdims=True)
    layer_norms = np.clip(layer_norms, 1e-12, None)
    normalized_layers = layer_features / layer_norms

    fused = (normalized_layers * normalized_weights[None, :, None]).sum(axis=1)
    return cast(np.ndarray, fused.astype("float32"))


def build_index(features: np.ndarray) -> Any:
    """
    Build FAISS flat index for exact cosine similarity search.

    Args:
        features: Feature array of shape (n_tracks, feature_dim)

    Returns:
        FAISS index ready for searching
    """
    faiss = _require_faiss()

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
    track_names: list[str],
    k: int = 10,
    exclude_self: bool = True
) -> list[tuple[str, float]]:
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
    faiss = _require_faiss()

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
    for idx, score in zip(indices[0], distances[0], strict=True):
        track_id = track_names[idx]

        # Skip self if requested
        if exclude_self and track_id == query_track:
            continue

        results.append((track_id, float(score)))

        if len(results) >= k:
            break

    return results


def search_by_clip(
    artifact: str | Path | Any,
    *,
    clip_id: str | None = None,
    track_id: str | None = None,
    start_sec: float | None = None,
    k: int = 10,
    dedupe_window_seconds: float = DEFAULT_DEDUPE_WINDOW_SECONDS,
    exclude_same_segment: bool = True,
    nprobe: int | None = None,
) -> list[ClipSearchResult]:
    """
    Stable public entry point for clip search over a prebuilt clip artifact.

    Args:
        artifact: Loaded clip artifact or artifact directory path
        clip_id: Exact clip identifier to use as the query
        track_id: Track identifier used together with start_sec
        start_sec: Clip start time used to resolve the nearest query clip in track_id
        k: Number of clip results to return
        dedupe_window_seconds: Suppress nearby results within this window per track
        exclude_same_segment: Exclude the exact query segment from results
        nprobe: Optional IVF nprobe override

    Returns:
        List of clip-level results with preserved clip metadata and similarity scores
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    if dedupe_window_seconds < 0:
        raise ValueError("dedupe_window_seconds must be >= 0")

    loaded_artifact = _load_clip_artifact(artifact)
    clip_records = loaded_artifact.clip_records
    if not clip_records:
        raise ValueError(
            "Clip artifact is missing clip metadata. Rebuild it from ClipIndex-backed records."
        )

    query_idx = _resolve_artifact_query_index(
        clip_records=clip_records,
        clip_id=clip_id,
        track_id=track_id,
        start_sec=start_sec,
    )
    query_record = clip_records[query_idx]

    logger.info(
        "Resolved query clip to '%s' [%s %.2fs - %.2fs]",
        query_record.clip_id or f"{query_record.track_id}:{query_record.segment_idx}",
        query_record.track_id,
        query_record.start_time,
        query_record.end_time,
    )

    query_vector = _query_vector_from_artifact(loaded_artifact, query_idx)

    # Over-fetch so we can still return k after filtering + deduping.
    search_k = min(
        len(clip_records),
        max(k * 25, k + 25),
    )
    distances, indices = loaded_artifact.search(query_vector, search_k, nprobe=nprobe)

    accepted_starts_by_track: dict[str, list[float]] = {}
    results: list[ClipSearchResult] = []

    for idx, score in zip(indices[0], distances[0], strict=True):
        if idx < 0:
            continue

        record = clip_records[int(idx)]

        if exclude_same_segment and int(idx) == query_idx:
            continue

        # Avoid near-duplicate regions from the same track.
        existing_starts = accepted_starts_by_track.setdefault(record.track_id, [])
        if dedupe_window_seconds > 0 and any(
            abs(record.start_time - existing_start) < dedupe_window_seconds
            for existing_start in existing_starts
        ):
            continue

        existing_starts.append(record.start_time)
        results.append(ClipSearchResult.from_clip_metadata(record, similarity=float(score)))

        if len(results) >= k:
            break

    return results


def search_by_aspect(
    query_track: str,
    aspect: str,
    features_dir: str,
    k: int = 10
) -> list[tuple[str, float]]:
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


def search_by_aspects(
    query_track: str,
    aspect_weights: dict[str, float],
    features_dir: str,
    k: int = 10,
    min_r2: float = 0.5,
    scale_by_r2: bool = True,
) -> list[tuple[str, float]]:
    """
    Find similar tracks using a weighted combination of multiple aspects.

    This is a multi-intent extension over single-aspect retrieval. Each aspect is
    resolved to a validated layer, then layer vectors are fused with query-time weights.

    Args:
        query_track: Track ID to find similar tracks for
        aspect_weights: Mapping {aspect_name: weight}; weights must be >= 0
        features_dir: Directory containing aggregated/segments/raw features
        k: Number of similar tracks to return
        min_r2: Minimum R² for resolved aspects
        scale_by_r2: Scale user weights by aspect validation R²

    Returns:
        List of (track_id, similarity_score) tuples
    """
    from ..probing import resolve_aspects

    if not aspect_weights:
        raise ValueError("aspect_weights cannot be empty")

    for aspect_name, weight in aspect_weights.items():
        if weight < 0:
            raise ValueError(
                f"Aspect weight for '{aspect_name}' must be >= 0, got {weight}"
            )

    aspect_mappings = resolve_aspects(min_r2=min_r2)
    if not aspect_mappings:
        raise ValueError("No layer discovery results found. Run layer discovery first.")

    unknown_aspects = sorted(set(aspect_weights) - set(aspect_mappings))
    if unknown_aspects:
        available = ", ".join(sorted(aspect_mappings.keys()))
        unknown = ", ".join(unknown_aspects)
        raise ValueError(f"Aspect(s) not validated: {unknown}. Available: {available}")

    layer_weights = np.zeros(NUM_LAYERS, dtype="float32")
    for aspect_name, user_weight in aspect_weights.items():
        mapping = aspect_mappings[aspect_name]
        weight = user_weight * mapping["r2_score"] if scale_by_r2 else user_weight
        layer_weights[mapping["layer"]] += float(weight)

    if layer_weights.sum() <= 0:
        raise ValueError("Combined aspect weights must sum to > 0")

    layer_features, track_names = _load_layer_features(features_dir)
    fused_vectors = _compose_weighted_vectors(layer_features, layer_weights)

    return find_similar(query_track, fused_vectors, track_names, k=k)
