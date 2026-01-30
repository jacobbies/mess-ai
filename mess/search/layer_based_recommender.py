"""
Layer-Based Music Recommender: Evidence-Based Similarity Search

This module implements a recommendation system that uses empirically validated MERT layer
specializations to provide accurate, interpretable music similarity search.

The Core Innovation:
-------------------
Instead of naively averaging all 13 MERT layers (which destroys layer-specific information),
we use SPECIFIC LAYERS for SPECIFIC MUSICAL ASPECTS based on empirical validation.

**The Problem with Naive Averaging**:
    avg_embedding = features.mean(axis=0)  # Mix all layers
    similarity = cosine(avg_query, avg_candidate)
    ❌ Result: 90%+ similarity between ALL tracks (no discrimination)

**Our Approach (Layer-Based Search)**:
    brightness_layer = features[0]  # R²=0.944 for brightness
    similarity = cosine(brightness_query[0], brightness_candidate[0])
    ✅ Result: Meaningful similarity for brightness aspect

Validated Layer Mappings:
-------------------------
From systematic layer discovery experiments (see layer_discovery.py):

| Layer | Musical Aspect | R² Score | Confidence |
|-------|---------------|----------|------------|
| **0** | Spectral brightness | **0.944** | High |
| **1** | Timbral texture | **0.922** | High |
| **2** | Acoustic structure | **0.933** | High |
| 4 | Temporal patterns | 0.673 | Medium |
| 7 | Musical phrasing | 0.781 | Medium |

High R² (>0.9) = Excellent fit = Use this layer confidently!

Three Search Modes:
------------------
This recommender supports three ways to find similar music:

**1. Single-Layer Search** (recommend_by_layer):
    - Use ONE specific MERT layer directly
    - Example: Layer 0 for brightness-based similarity
    - Fast, simple, interpretable

**2. Aspect-Based Search** (recommend_by_aspect):
    - Use validated layer for a named musical aspect
    - Example: aspect="spectral_brightness" → uses Layer 0
    - User-friendly interface (no need to know layer numbers)

**3. Multi-Aspect Search** (multi_aspect_recommendation):
    - Combine MULTIPLE aspects with custom weights
    - Example: 60% brightness + 40% texture
    - Flexible, powerful, matches complex user preferences

How Multi-Aspect Weighting Works:
---------------------------------
The key insight: **Compare within layers FIRST, then combine similarities**.

**Step 1**: Compute similarity for EACH aspect separately
    brightness_sim = cosine(query[0], candidate[0])  # Layer 0
    texture_sim = cosine(query[1], candidate[1])     # Layer 1

**Step 2**: Combine similarities with user-specified weights
    final_score = 0.6 * brightness_sim + 0.4 * texture_sim

**Why this works better than averaging embeddings**:
- Preserves layer specialization (each layer compared independently)
- User controls which aspects matter (via weights)
- Interpretable results ("similar in brightness AND texture")

Example:
    Query: "Beethoven Op.27 No.1"
    Weights: {'spectral_brightness': 0.6, 'timbral_texture': 0.4}

    Candidate A: brightness=0.92, texture=0.85 → final=0.892
    Candidate B: brightness=0.78, texture=0.91 → final=0.832

    A wins! (0.892 > 0.832)

Similarity Metric (Cosine Similarity):
-------------------------------------
We use cosine similarity, NOT Euclidean distance.

**Why Cosine?**
- MERT embeddings are roughly normalized
- We care about "direction" (semantic meaning), not "magnitude" (arbitrary scale)
- Standard for embeddings (word2vec, BERT, all use cosine)
- Range: -1 to 1 (for embeddings, typically 0 to 1)

**Formula**:
    cos_sim(A, B) = (A · B) / (||A|| × ||B||)

**Interpretation**:
- 1.0 = Identical (same direction)
- 0.0 = Orthogonal (no similarity)
- -1.0 = Opposite (rare for music embeddings)

Data Flow:
---------
1. Load aggregated features [13, 768] for all tracks
2. Build layer-specific feature matrices (extract validated layers)
3. Query: User specifies reference track + aspect/weights
4. Compute cosine similarity using validated layer(s)
5. Rank tracks by similarity score
6. Return top-K recommendations with interpretable info

Typical Usage:
-------------
**Simple brightness-based search**:
    recommender = LayerBasedRecommender()
    results = recommender.recommend_by_aspect(
        "Beethoven_Op027No1-01",
        aspect="spectral_brightness",
        n_recommendations=5
    )

**Multi-aspect search with custom weights**:
    results = recommender.multi_aspect_recommendation(
        "Beethoven_Op027No1-01",
        aspect_weights={
            'spectral_brightness': 0.7,  # 70% importance
            'timbral_texture': 0.3,      # 30% importance
        },
        n_recommendations=5
    )

**Direct layer access** (if you know which layer you want):
    results = recommender.recommend_by_layer(
        "Beethoven_Op027No1-01",
        layer=0,  # Spectral brightness layer
        n_recommendations=5
    )

Return Format:
-------------
All methods return list of tuples:
    [
        (track_name, similarity_score, layer_info),
        ("Bach_BWV849-01_001_20090916-SMD", 0.9910, "Spectral brightness (Layer 0)"),
        ...
    ]

Why This Matters:
----------------
- **Accuracy**: Using validated layers beats naive averaging (empirically tested)
- **Interpretability**: Results explain WHY tracks are similar (brightness, texture, etc.)
- **Flexibility**: Users can prioritize aspects that matter to them
- **Evidence-based**: Layer mappings backed by cross-validated R² scores

See Also:
---------
- mess/probing/layer_discovery.py - How we discovered layer specializations
- docs/CONCEPTS.md - Detailed explanation of embeddings and layer discovery
- scripts/demo_recommendations.py - Example usage
"""

import sys
from pathlib import Path
from ..config import mess_config

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerBasedRecommender:
    """
    Music recommender using empirically validated MERT layer specializations.

    This class implements evidence-based similarity search by using specific MERT layers
    for specific musical aspects, rather than naively averaging all layers.

    Key Features:
    ------------
    - **Validated layers**: Uses layers with R² > 0.9 (excellent fit)
    - **Multiple search modes**: Single-layer, aspect-based, multi-aspect
    - **Interpretable results**: Explains WHY tracks are similar
    - **Fast**: Cosine similarity on pre-computed embeddings (<1ms per query)

    Initialization loads:
    - All track features (aggregated MERT embeddings [13, 768])
    - Validated layer mappings (Layer 0=brightness, Layer 1=texture, etc.)
    - Builds in-memory index for fast similarity search

    Example:
        recommender = LayerBasedRecommender()
        # Recommender loaded with 50 tracks, 3 validated layers

        # Get brightness-based recommendations
        results = recommender.recommend_by_aspect(
            "Beethoven_Op027No1-01",
            aspect="spectral_brightness",
            n_recommendations=5
        )
    """
    
    def __init__(self, dataset_name: str = "smd"):
        from mess.datasets.factory import DatasetFactory
        self.dataset = DatasetFactory.get_dataset(dataset_name)
        self.data_dir = self.dataset.data_root
        self.features_dir = self.dataset.aggregated_dir
        
        # Load validated layer mappings
        self.layer_mappings = self._load_validated_layers()
        
        # Load all track features
        self.track_features = {}
        self.track_names = []
        self._load_all_features()
        
        logger.info(f"Initialized recommender with {len(self.track_names)} tracks")
        logger.info(f"Validated layers: {list(self.layer_mappings.keys())}")
    
    def _load_validated_layers(self) -> Dict:
        """Load empirically validated layer specializations."""
        
        # From our discovery experiments
        validated_layers = {
            0: {
                'aspect': 'spectral_brightness',
                'r2_score': 0.944,
                'description': 'Spectral centroid, instrumental brightness, timbral clarity',
                'confidence': 'high',
                'weight': 1.0
            },
            1: {
                'aspect': 'timbral_texture', 
                'r2_score': 0.922,
                'description': 'Instrumental timbre, texture, acoustic properties',
                'confidence': 'high',
                'weight': 0.95
            },
            2: {
                'aspect': 'acoustic_structure',
                'r2_score': 0.933,
                'description': 'Acoustic structure and resonance patterns',
                'confidence': 'high', 
                'weight': 0.90
            }
        }
        
        # Add promising layers for future validation
        promising_layers = {
            4: {
                'aspect': 'temporal_patterns',
                'description': 'Rhythmic patterns and temporal structure',
                'confidence': 'medium',
                'weight': 0.6
            },
            7: {
                'aspect': 'musical_phrasing',
                'description': 'Musical phrases and structural patterns',
                'confidence': 'medium',
                'weight': 0.5
            }
        }
        
        # Combine validated and promising
        all_layers = {**validated_layers, **promising_layers}
        
        logger.info(f"Loaded {len(validated_layers)} validated and {len(promising_layers)} promising layer mappings")
        return all_layers
    
    def _load_all_features(self):
        """Load MERT features for all tracks."""

        feature_files = list(self.features_dir.glob("*.npy"))

        for feature_file in feature_files:
            track_name = feature_file.stem

            try:
                # Load aggregated features: [layers, features] = [13, 768]
                aggregated_features = np.load(feature_file)

                # Extract layer-specific features
                layer_features = {}
                for layer in range(13):
                    layer_features[layer] = aggregated_features[layer, :]

                self.track_features[track_name] = layer_features
                self.track_names.append(track_name)

            except Exception as e:
                logger.warning(f"Error loading features for {track_name}: {e}")

        logger.info(f"Successfully loaded features for {len(self.track_names)} tracks")
    
    def recommend_by_aspect(
        self,
        query_track: str,
        aspect: str,
        n_recommendations: int = 5,
        exclude_query: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Get music recommendations based on a specific validated musical aspect.

        This method uses empirically validated MERT layers that encode the requested
        musical aspect. Instead of using arbitrary features, it uses layers with
        proven correlation (R² > 0.9) to the musical characteristic you care about.

        How It Works:
        ------------
        1. **Aspect Lookup**: Find which layer(s) encode the requested aspect
           - Example: aspect="spectral_brightness" → finds Layer 0 (R²=0.944)

        2. **Feature Extraction**: Get embeddings from validated layer(s)
           - Query: query_track's Layer 0 features [768 dims]
           - Candidates: all other tracks' Layer 0 features [768 dims]

        3. **Similarity Computation**: Cosine similarity between query and candidates
           - cos_sim = (query · candidate) / (||query|| × ||candidate||)
           - Range: 0.0 (no similarity) to 1.0 (identical)

        4. **Ranking**: Sort by similarity, return top-K

        Validated Aspects:
        -----------------
        - **spectral_brightness**: Timbral brightness/darkness (Layer 0, R²=0.944)
          - Use when: Finding tracks with similar instrumental brightness
          - Example: "Bright piano pieces" or "Dark cello passages"

        - **timbral_texture**: Instrumental timbre and texture (Layer 1, R²=0.922)
          - Use when: Finding tracks with similar surface texture
          - Example: "Smooth sustained notes" or "Percussive staccato"

        - **acoustic_structure**: Resonance patterns and structure (Layer 2, R²=0.933)
          - Use when: Finding tracks with similar acoustic properties
          - Example: "Similar resonance" or "Room acoustics"

        Example Usage:
        -------------
        # Find tracks with similar brightness to a reference
        results = recommender.recommend_by_aspect(
            query_track="Beethoven_Op027No1-01",
            aspect="spectral_brightness",
            n_recommendations=5
        )

        # Results: Top 5 tracks with similar brightness
        for track, similarity, info in results:
            print(f"{track}: {similarity:.4f} - {info}")

        Output Example:
        Bach_BWV849-02_001_20090916-SMD: 0.9910 - Spectral brightness (Layer 0)

        Args:
            query_track: Reference track name (without .npy extension)
            aspect: Musical aspect to match. Must be one of the validated aspects:
                   "spectral_brightness", "timbral_texture", "acoustic_structure"
            n_recommendations: Number of similar tracks to return (default: 5)
            exclude_query: Whether to exclude the query track from results (default: True)

        Returns:
            List of (track_name, similarity_score, layer_info) tuples, sorted by similarity
            Example: [("Bach_BWV849-02_001_20090916-SMD", 0.9910, "Spectral brightness (Layer 0)")]

        Raises:
            ValueError: If query_track not found or aspect not validated
        """
        
        if query_track not in self.track_features:
            raise ValueError(f"Track '{query_track}' not found in dataset")
        
        # Find layers that encode this aspect
        relevant_layers = [
            layer for layer, info in self.layer_mappings.items() 
            if aspect.lower() in info['aspect'].lower()
        ]
        
        if not relevant_layers:
            raise ValueError(f"No validated layers found for aspect '{aspect}'")
        
        # Get query features from relevant layers
        query_features = []
        layer_weights = []
        
        for layer in relevant_layers:
            query_layer_features = self.track_features[query_track][layer]
            query_features.append(query_layer_features)
            layer_weights.append(self.layer_mappings[layer].get('weight', 1.0))
        
        # Combine features from multiple layers (weighted average)
        if len(query_features) > 1:
            weights = np.array(layer_weights)
            weights = weights / np.sum(weights)  # Normalize
            combined_query = np.average(query_features, axis=0, weights=weights)
        else:
            combined_query = query_features[0]
        
        # Calculate similarities to all other tracks
        similarities = []
        
        for track_name in self.track_names:
            if exclude_query and track_name == query_track:
                continue
            
            # Get candidate features from same layers
            candidate_features = []
            for layer in relevant_layers:
                candidate_layer_features = self.track_features[track_name][layer]
                candidate_features.append(candidate_layer_features)
            
            # Combine candidate features
            if len(candidate_features) > 1:
                weights = np.array(layer_weights)
                weights = weights / np.sum(weights)
                combined_candidate = np.average(candidate_features, axis=0, weights=weights)
            else:
                combined_candidate = candidate_features[0]
            
            # Calculate similarity
            similarity = cosine_similarity([combined_query], [combined_candidate])[0, 0]
            
            # Include layer info in result
            layer_info = f"Layers {relevant_layers}"
            similarities.append((track_name, float(similarity), layer_info))
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_recommendations]
    
    def recommend_by_layer(
        self,
        query_track: str,
        layer: int,
        n_recommendations: int = 5,
        exclude_query: bool = True
    ) -> List[Tuple[str, float, str]]:
        """
        Get recommendations using a specific MERT layer directly.

        This method provides low-level access to individual MERT layers for similarity search.
        Use this when you know the specific layer number you want (e.g., from documentation
        or experiments). For most users, recommend_by_aspect() is more user-friendly.

        Validated Layers (from empirical discovery):
        -------------------------------------------
        - **Layer 0**: Spectral brightness (R²=0.944) - Best for timbral brightness
        - **Layer 1**: Timbral texture (R²=0.922) - Best for instrumental characteristics
        - **Layer 2**: Acoustic structure (R²=0.933) - Best for resonance patterns
        - Layer 4: Temporal patterns (R²=0.673) - Experimental
        - Layer 7: Musical phrasing (R²=0.781) - Experimental

        What This Method Does:
        ---------------------
        1. Extract query track's features from specified layer [768 dims]
        2. Extract all candidate tracks' features from same layer [768 dims]
        3. Compute cosine similarity between query and each candidate
        4. Rank by similarity, return top-K

        When to Use This vs recommend_by_aspect():
        ------------------------------------------
        **Use recommend_by_aspect()** when:
        - You want "brightness similarity" (user-friendly)
        - You don't know/care about layer numbers

        **Use recommend_by_layer()** when:
        - You're experimenting with different layers
        - You want to compare Layer 0 vs Layer 1 directly
        - You're testing an unvalidated layer (4-12)

        Example Usage:
        -------------
        # Direct layer access (brightness via Layer 0)
        results = recommender.recommend_by_layer(
            query_track="Beethoven_Op027No1-01",
            layer=0,  # Brightness layer
            n_recommendations=5
        )

        # Experimental: Test Layer 7 (phrasing)
        results = recommender.recommend_by_layer(
            query_track="Beethoven_Op027No1-01",
            layer=7,  # Not fully validated (R²=0.781)
            n_recommendations=5
        )

        Args:
            query_track: Reference track name (without .npy extension)
            layer: MERT layer number (0-12). Use validated layers (0, 1, 2) for best results.
            n_recommendations: Number of similar tracks to return (default: 5)
            exclude_query: Whether to exclude the query track from results (default: True)

        Returns:
            List of (track_name, similarity_score, layer_description) tuples
            Example: [("Bach_BWV849-02_001_20090916-SMD", 0.9910, "Spectral brightness")]

        Raises:
            ValueError: If query_track not found

        Warning:
            Using unvalidated layers (4-12) may give unreliable results. Check R² scores
            in layer_discovery_results.json before trusting results from these layers.
        """
        
        if query_track not in self.track_features:
            raise ValueError(f"Track '{query_track}' not found in dataset")
        
        if layer not in self.layer_mappings:
            logger.warning(f"Layer {layer} not validated, but proceeding anyway")
        
        # Get query features from specified layer
        query_features = self.track_features[query_track][layer]
        
        # Calculate similarities
        similarities = []
        
        for track_name in self.track_names:
            if exclude_query and track_name == query_track:
                continue
            
            candidate_features = self.track_features[track_name][layer]
            similarity = cosine_similarity([query_features], [candidate_features])[0, 0]
            
            # Get layer description if available
            layer_info = self.layer_mappings.get(layer, {}).get('description', f'Layer {layer}')
            similarities.append((track_name, float(similarity), layer_info))
        
        # Sort and return
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n_recommendations]
    
    def multi_aspect_recommendation(
        self,
        query_track: str,
        aspect_weights: Dict[str, float],
        n_recommendations: int = 5,
        exclude_query: bool = True
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Get recommendations by combining multiple musical aspects with custom weights.

        This is the most powerful search method, allowing you to match music across multiple
        dimensions simultaneously (brightness, texture, structure) with user-defined priorities.

        The Key Innovation (vs Naive Averaging):
        ----------------------------------------
        We **compare per-layer FIRST, then combine similarities** (not embeddings):

        **❌ Wrong (naive embedding average)**:
            combined_emb = 0.6 * features[0] + 0.4 * features[1]  # Mix embeddings
            similarity = cosine(combined_query, combined_candidate)
            Problem: Destroys layer specialization before comparison!

        **✅ Correct (weighted similarity combination)**:
            brightness_sim = cosine(query[0], candidate[0])  # Layer 0
            texture_sim = cosine(query[1], candidate[1])     # Layer 1
            final_score = 0.6 * brightness_sim + 0.4 * texture_sim
            Advantage: Preserves layer specialization, user controls priorities!

        How It Works:
        ------------
        1. **For each aspect**: Compute similarity using validated layer
           - brightness: Use Layer 0 (R²=0.944)
           - texture: Use Layer 1 (R²=0.922)
           - structure: Use Layer 2 (R²=0.933)

        2. **Normalize weights**: Ensure weights sum to 1.0
           - Input: {brightness: 0.6, texture: 0.4}
           - Already normalized (0.6 + 0.4 = 1.0)
           - If not normalized: divide each by sum

        3. **Combine similarities**: Weighted average of per-aspect scores
           - combined_score = Σ(weight_i × similarity_i)

        4. **Rank and return**: Sort by combined score

        Example Scenario:
        ----------------
        Query: "Find tracks like Beethoven Op.27 No.1, prioritizing texture over brightness"

        ```python
        results = recommender.multi_aspect_recommendation(
            query_track="Beethoven_Op027No1-01",
            aspect_weights={
                'spectral_brightness': 0.3,  # 30% importance
                'timbral_texture': 0.7,      # 70% importance
            },
            n_recommendations=5
        )
        ```

        **Per-Track Computation** (for each candidate):
        - Brightness similarity (Layer 0): 0.912
        - Texture similarity (Layer 1): 0.935
        - Combined score: 0.3 × 0.912 + 0.7 × 0.935 = 0.928

        **Results Ranked by Combined Score**:
        1. Beethoven_Op027No2-01 (0.921) - High texture match drives score
        2. Mozart_KV331-01 (0.856) - Lower texture, higher brightness
        ...

        Use Cases:
        ---------
        **Equal weighting** (balanced similarity):
        ```python
        aspect_weights = {
            'spectral_brightness': 0.5,
            'timbral_texture': 0.5,
        }
        # Tracks must match well in BOTH aspects
        ```

        **Single aspect dominant** (90% brightness, 10% texture):
        ```python
        aspect_weights = {
            'spectral_brightness': 0.9,
            'timbral_texture': 0.1,
        }
        # Almost pure brightness search, slight texture consideration
        ```

        **Three-way balance**:
        ```python
        aspect_weights = {
            'spectral_brightness': 0.4,
            'timbral_texture': 0.4,
            'acoustic_structure': 0.2,
        }
        # Brightness and texture important, structure less so
        ```

        Args:
            query_track: Reference track name (without .npy extension)
            aspect_weights: Dict mapping aspect names to importance weights
                          - Keys: "spectral_brightness", "timbral_texture", "acoustic_structure"
                          - Values: Weights (will be normalized to sum to 1.0)
                          - Example: {'spectral_brightness': 0.6, 'timbral_texture': 0.4}
            n_recommendations: Number of similar tracks to return (default: 5)
            exclude_query: Whether to exclude the query track from results (default: True)

        Returns:
            List of (track_name, combined_score, aspect_scores_dict) tuples
            - track_name: Recommended track
            - combined_score: Weighted similarity score (0.0 to 1.0)
            - aspect_scores_dict: Individual aspect similarities for transparency
              Example: {'spectral_brightness': 0.912, 'timbral_texture': 0.935}

        Example Return Value:
        ```python
        [
            ("Beethoven_Op027No2-01", 0.921, {
                'spectral_brightness': 0.893,
                'timbral_texture': 0.935
            }),
            ("Mozart_KV331-01", 0.856, {
                'spectral_brightness': 0.912,
                'timbral_texture': 0.831
            }),
            ...
        ]
        ```

        Raises:
            ValueError: If query_track not found or invalid aspect names provided

        See Also:
        --------
        - docs/CONCEPTS.md (Section 5): Detailed explanation of multi-aspect weighting
        - docs/QUICK_REFERENCE.md: When to use each recommendation method
        """
        
        if query_track not in self.track_features:
            raise ValueError(f"Track '{query_track}' not found in dataset")
        
        # Get recommendations for each aspect
        aspect_similarities = {}
        
        for aspect, weight in aspect_weights.items():
            if weight > 0:
                try:
                    recommendations = self.recommend_by_aspect(
                        query_track, aspect, n_recommendations=len(self.track_names), 
                        exclude_query=exclude_query
                    )
                    # Convert to dict for easy lookup
                    aspect_similarities[aspect] = {
                        track: sim for track, sim, _ in recommendations
                    }
                except ValueError as e:
                    logger.warning(f"Could not get recommendations for aspect '{aspect}': {e}")
        
        if not aspect_similarities:
            raise ValueError("No valid aspects provided")
        
        # Combine similarities using weighted average
        combined_scores = {}
        
        # Get all tracks that appear in any aspect
        all_tracks = set()
        for track_sims in aspect_similarities.values():
            all_tracks.update(track_sims.keys())
        
        for track in all_tracks:
            weighted_score = 0
            total_weight = 0
            aspect_scores = {}
            
            for aspect, weight in aspect_weights.items():
                if aspect in aspect_similarities and track in aspect_similarities[aspect]:
                    score = aspect_similarities[aspect][track]
                    weighted_score += weight * score
                    total_weight += weight
                    aspect_scores[aspect] = score
                else:
                    aspect_scores[aspect] = 0.0
            
            if total_weight > 0:
                combined_scores[track] = (weighted_score / total_weight, aspect_scores)
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1][0], 
            reverse=True
        )
        
        # Format results
        results = []
        for track, (score, aspect_breakdown) in sorted_results[:n_recommendations]:
            results.append((track, score, aspect_breakdown))
        
        return results
    
    def get_available_aspects(self) -> List[str]:
        """Get list of available musical aspects for recommendation."""
        aspects = set()
        for layer_info in self.layer_mappings.values():
            aspects.add(layer_info['aspect'])
        return sorted(list(aspects))
    
    def get_validated_layers(self) -> Dict[int, Dict]:
        """Get information about validated layers."""
        return {
            layer: info for layer, info in self.layer_mappings.items()
            if info.get('confidence') == 'high'
        }
    
    def explain_recommendation(self, query_track: str, recommended_track: str, aspect: str) -> str:
        """Explain why a track was recommended."""
        
        relevant_layers = [
            layer for layer, info in self.layer_mappings.items() 
            if aspect.lower() in info['aspect'].lower()
        ]
        
        if not relevant_layers:
            return f"No explanation available for aspect '{aspect}'"
        
        layer_descriptions = [
            self.layer_mappings[layer]['description'] 
            for layer in relevant_layers
        ]
        
        explanation = f"'{recommended_track}' was recommended because it has similar " \
                     f"{aspect} characteristics to '{query_track}'. " \
                     f"This similarity is detected by analyzing MERT layer(s) {relevant_layers}, " \
                     f"which encode: {'; '.join(layer_descriptions)}."
        
        return explanation


def demo():
    """Demo the layer-based recommender."""
    
    recommender = LayerBasedRecommender()
    
    if not recommender.track_names:
        print("No tracks loaded! Check your feature files.")
        return
    
    # Pick a demo track
    demo_track = recommender.track_names[0]
    print(f"Demo track: {demo_track}\\n")
    
    # Show available aspects
    aspects = recommender.get_available_aspects()
    print(f"Available aspects: {', '.join(aspects)}\\n")
    
    # Test different recommendation types
    print("=== TIMBRAL SIMILARITY (Layers 0-2) ===")
    try:
        timbral_recs = recommender.recommend_by_aspect(demo_track, 'spectral', n_recommendations=5)
        for i, (track, score, layer_info) in enumerate(timbral_recs, 1):
            print(f"{i}. {track} (similarity: {score:.4f}) - {layer_info}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\\n=== LAYER 0 SIMILARITY (Spectral Brightness) ===")
    layer_0_recs = recommender.recommend_by_layer(demo_track, 0, n_recommendations=5)
    for i, (track, score, description) in enumerate(layer_0_recs, 1):
        print(f"{i}. {track} (similarity: {score:.4f}) - {description}")
    
    print("\\n=== MULTI-ASPECT RECOMMENDATION ===")
    try:
        multi_recs = recommender.multi_aspect_recommendation(
            demo_track,
            aspect_weights={
                'spectral_brightness': 0.6,
                'timbral_texture': 0.4
            },
            n_recommendations=3
        )
        
        for i, (track, score, breakdown) in enumerate(multi_recs, 1):
            print(f"{i}. {track} (combined: {score:.4f})")
            for aspect, aspect_score in breakdown.items():
                print(f"   {aspect}: {aspect_score:.4f}")
    except Exception as e:
        print(f"Multi-aspect error: {e}")
    
    print(f"\\n✅ Layer-based recommendation demo completed!")


if __name__ == "__main__":
    demo()