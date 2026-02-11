"""
Layer-Based Music Recommender

Uses empirically validated MERT layer specializations for similarity search.
Layer mappings are loaded dynamically from discovery results — no hardcoded
layer assumptions. Run layer discovery first to populate results.

Search modes:
    recommend_by_aspect()  - Search by musical aspect name (e.g., "brightness")
    recommend_by_layer()   - Search by raw MERT layer number (0-12)
    multi_aspect_recommendation() - Weighted combination of aspects
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..config import mess_config
from .aspects import ASPECT_REGISTRY, resolve_aspects

logger = logging.getLogger(__name__)


class LayerBasedRecommender:
    """
    Music recommender using validated MERT layer specializations.

    Layer mappings are loaded from discovery results at init time.
    If no results exist, no aspects are available — run discovery first.

    Example:
        recommender = LayerBasedRecommender()
        print(recommender.get_available_aspects())
        results = recommender.recommend_by_aspect("track_name", "brightness")
    """

    def __init__(self, dataset_name: str = "smd", min_r2: float = 0.5):
        from mess.datasets.factory import DatasetFactory

        self.dataset = DatasetFactory.get_dataset(dataset_name)
        self.features_dir = self.dataset.aggregated_dir

        # Load validated aspect→layer mappings from discovery results
        self.aspect_mappings = resolve_aspects(min_r2=min_r2)

        # Build layer→aspect reverse lookup
        self.layer_mappings: Dict[int, Dict] = {}
        for aspect_name, info in self.aspect_mappings.items():
            layer = info['layer']
            self.layer_mappings[layer] = {
                'aspect': aspect_name,
                'r2_score': info['r2_score'],
                'description': info['description'],
                'confidence': info['confidence'],
            }

        # Load all track features
        self.track_features: Dict[str, Dict[int, np.ndarray]] = {}
        self.track_names: List[str] = []
        self._load_all_features()

        if self.aspect_mappings:
            logger.info(
                f"Initialized with {len(self.track_names)} tracks, "
                f"{len(self.aspect_mappings)} validated aspects"
            )
        else:
            logger.warning(
                "No validated aspects found. Run layer discovery first: "
                "python research/scripts/run_probing.py"
            )

    def _load_all_features(self):
        """Load aggregated MERT features [13, 768] for all tracks."""
        for feature_file in self.features_dir.glob("*.npy"):
            track_name = feature_file.stem
            try:
                aggregated = np.load(feature_file)  # [13, 768]
                self.track_features[track_name] = {
                    layer: aggregated[layer, :] for layer in range(13)
                }
                self.track_names.append(track_name)
            except Exception as e:
                logger.warning(f"Error loading features for {track_name}: {e}")

        logger.info(f"Loaded features for {len(self.track_names)} tracks")

    def recommend_by_aspect(
        self,
        query_track: str,
        aspect: str,
        n_recommendations: int = 5,
        exclude_query: bool = True,
    ) -> List[Tuple[str, float, str]]:
        """
        Recommend tracks similar in a specific musical aspect.

        The aspect is resolved to its best validated MERT layer via discovery results.

        Args:
            query_track: Reference track name (without .npy extension)
            aspect: Musical aspect name from the registry (e.g., "brightness", "dynamics")
            n_recommendations: Number of results to return
            exclude_query: Exclude the query track from results

        Returns:
            List of (track_name, similarity_score, aspect_info) tuples
        """
        if query_track not in self.track_features:
            raise ValueError(f"Track '{query_track}' not found in dataset")

        if aspect not in self.aspect_mappings:
            available = ', '.join(sorted(self.aspect_mappings.keys())) or '(none — run discovery first)'
            raise ValueError(
                f"Aspect '{aspect}' not available. Valid aspects: {available}"
            )

        mapping = self.aspect_mappings[aspect]
        layer = mapping['layer']
        info_str = f"{aspect} (Layer {layer}, R²={mapping['r2_score']:.3f})"

        return self._similarity_search(
            query_track, layer, info_str, n_recommendations, exclude_query
        )

    def recommend_by_layer(
        self,
        query_track: str,
        layer: int,
        n_recommendations: int = 5,
        exclude_query: bool = True,
    ) -> List[Tuple[str, float, str]]:
        """
        Recommend tracks using a specific MERT layer directly.

        Use this for experimentation or when you know which layer you want.

        Args:
            query_track: Reference track name
            layer: MERT layer number (0-12)
            n_recommendations: Number of results to return
            exclude_query: Exclude the query track from results

        Returns:
            List of (track_name, similarity_score, layer_info) tuples
        """
        if query_track not in self.track_features:
            raise ValueError(f"Track '{query_track}' not found in dataset")

        if layer < 0 or layer > 12:
            raise ValueError(f"Layer must be 0-12, got {layer}")

        if layer in self.layer_mappings:
            info = self.layer_mappings[layer]
            info_str = f"{info['aspect']} (Layer {layer})"
        else:
            info_str = f"Layer {layer} (unvalidated)"

        return self._similarity_search(
            query_track, layer, info_str, n_recommendations, exclude_query
        )

    def _similarity_search(
        self,
        query_track: str,
        layer: int,
        info_str: str,
        n_recommendations: int,
        exclude_query: bool,
    ) -> List[Tuple[str, float, str]]:
        """Core similarity search on a single layer."""
        query_features = self.track_features[query_track][layer]

        similarities = []
        for track_name in self.track_names:
            if exclude_query and track_name == query_track:
                continue
            candidate_features = self.track_features[track_name][layer]
            sim = cosine_similarity([query_features], [candidate_features])[0, 0]
            similarities.append((track_name, float(sim), info_str))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_recommendations]

    def multi_aspect_recommendation(
        self,
        query_track: str,
        aspect_weights: Dict[str, float],
        n_recommendations: int = 5,
        exclude_query: bool = True,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Recommend by combining multiple aspects with custom weights.

        Computes per-aspect similarity independently (preserving layer specialization),
        then combines with weighted average.

        Args:
            query_track: Reference track name
            aspect_weights: {aspect_name: weight} — weights are normalized internally
            n_recommendations: Number of results to return
            exclude_query: Exclude the query track from results

        Returns:
            List of (track_name, combined_score, {aspect: score}) tuples
        """
        if query_track not in self.track_features:
            raise ValueError(f"Track '{query_track}' not found in dataset")

        # Compute per-aspect similarities
        aspect_sims: Dict[str, Dict[str, float]] = {}

        for aspect, weight in aspect_weights.items():
            if weight <= 0:
                continue
            try:
                recs = self.recommend_by_aspect(
                    query_track, aspect,
                    n_recommendations=len(self.track_names),
                    exclude_query=exclude_query,
                )
                aspect_sims[aspect] = {track: sim for track, sim, _ in recs}
            except ValueError as e:
                logger.warning(f"Skipping aspect '{aspect}': {e}")

        if not aspect_sims:
            raise ValueError("No valid aspects in weights")

        # Combine with weighted average
        all_tracks = set()
        for sims in aspect_sims.values():
            all_tracks.update(sims.keys())

        combined: Dict[str, Tuple[float, Dict[str, float]]] = {}
        for track in all_tracks:
            weighted_sum = 0.0
            total_weight = 0.0
            scores: Dict[str, float] = {}

            for aspect, weight in aspect_weights.items():
                if aspect in aspect_sims and track in aspect_sims[aspect]:
                    score = aspect_sims[aspect][track]
                    weighted_sum += weight * score
                    total_weight += weight
                    scores[aspect] = score
                else:
                    scores[aspect] = 0.0

            if total_weight > 0:
                combined[track] = (weighted_sum / total_weight, scores)

        sorted_results = sorted(combined.items(), key=lambda x: x[1][0], reverse=True)

        return [
            (track, score, breakdown)
            for track, (score, breakdown) in sorted_results[:n_recommendations]
        ]

    def get_available_aspects(self) -> List[str]:
        """Get list of validated aspects available for search."""
        return sorted(self.aspect_mappings.keys())

    def get_all_registry_aspects(self) -> Dict[str, str]:
        """Get all aspects in the registry with descriptions (including unvalidated)."""
        return {
            name: info['description']
            for name, info in ASPECT_REGISTRY.items()
        }

    def get_aspect_info(self) -> Dict[str, Dict]:
        """Get detailed info about validated aspects including layer and R² score."""
        return dict(self.aspect_mappings)

    def explain_recommendation(
        self, query_track: str, recommended_track: str, aspect: str
    ) -> str:
        """Explain why a track was recommended for a given aspect."""
        if aspect not in self.aspect_mappings:
            return f"Aspect '{aspect}' is not validated."

        mapping = self.aspect_mappings[aspect]
        return (
            f"'{recommended_track}' was recommended because it has similar "
            f"{aspect} characteristics to '{query_track}'. "
            f"Detected via MERT Layer {mapping['layer']} "
            f"(R²={mapping['r2_score']:.3f} for {mapping['target']})."
        )


def demo():
    """Demo the layer-based recommender."""
    recommender = LayerBasedRecommender()

    if not recommender.track_names:
        print("No tracks loaded. Check your feature files.")
        return

    demo_track = recommender.track_names[0]
    print(f"Demo track: {demo_track}\n")

    aspects = recommender.get_available_aspects()
    if not aspects:
        print("No validated aspects. Run layer discovery first:")
        print("  python research/scripts/run_probing.py")
        print("\nAll registry aspects (need validation):")
        for name, desc in recommender.get_all_registry_aspects().items():
            print(f"  {name}: {desc}")
        return

    print(f"Available aspects: {', '.join(aspects)}\n")

    # Test first available aspect
    aspect = aspects[0]
    print(f"=== {aspect.upper()} SIMILARITY ===")
    recs = recommender.recommend_by_aspect(demo_track, aspect, n_recommendations=5)
    for i, (track, score, info) in enumerate(recs, 1):
        print(f"{i}. {track} ({score:.4f}) - {info}")

    # Test multi-aspect if we have at least 2
    if len(aspects) >= 2:
        print(f"\n=== MULTI-ASPECT: {aspects[0]} + {aspects[1]} ===")
        multi_recs = recommender.multi_aspect_recommendation(
            demo_track,
            aspect_weights={aspects[0]: 0.6, aspects[1]: 0.4},
            n_recommendations=3,
        )
        for i, (track, score, breakdown) in enumerate(multi_recs, 1):
            print(f"{i}. {track} (combined: {score:.4f})")
            for a, s in breakdown.items():
                print(f"   {a}: {s:.4f}")


if __name__ == "__main__":
    demo()
