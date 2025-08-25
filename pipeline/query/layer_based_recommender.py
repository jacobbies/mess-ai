"""
Layer-Based Music Recommender
Uses empirically validated MERT layers for different types of musical similarity.
"""

import sys
from pathlib import Path

# Add paths
sys.path.append('/Users/jacobbieschke/mess-ai/pipeline')

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerBasedRecommender:
    """Recommender system based on validated MERT layer specializations."""
    
    def __init__(self):
        self.data_dir = Path("/Users/jacobbieschke/mess-ai/data")
        self.features_dir = self.data_dir / "processed" / "features" / "raw"
        
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
                # Load raw features: [segments, layers, time, features]
                raw_features = np.load(feature_file)
                
                # Extract layer-specific features (average over segments and time)
                layer_features = {}
                for layer in range(13):
                    layer_features[layer] = raw_features[:, layer, :, :].mean(axis=(0, 1))
                
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
        """Get recommendations based on a specific musical aspect."""
        
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
        """Get recommendations based on a specific MERT layer."""
        
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
        """Get recommendations combining multiple musical aspects."""
        
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