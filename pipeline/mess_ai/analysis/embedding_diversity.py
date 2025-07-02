"""
Embedding Space Diversity Analysis and Enhanced Recommendation Strategies

This module provides tools to analyze MERT embedding space distribution
and implement diverse recommendation strategies beyond simple cosine similarity.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns


class EmbeddingDiversityAnalyzer:
    """Analyze diversity and distribution of MERT embeddings."""
    
    def __init__(self, features_dir: str = "data/processed/features"):
        self.features_dir = Path(features_dir)
        self.embeddings = {}
        self.track_names = []
        self._load_embeddings()
        
    def _load_embeddings(self):
        """Load all aggregated embeddings into memory."""
        agg_dir = self.features_dir / "aggregated"
        for file_path in sorted(agg_dir.glob("*.npy")):
            track_name = file_path.stem
            embedding = np.load(file_path).flatten()  # Shape: (13*768,)
            self.embeddings[track_name] = embedding
            self.track_names.append(track_name)
        
        logging.info(f"Loaded {len(self.embeddings)} embeddings")
    
    def analyze_global_statistics(self) -> Dict:
        """Compute global statistics of the embedding space."""
        # Convert to matrix
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        
        # Basic statistics
        stats = {
            "mean": np.mean(embedding_matrix),
            "std": np.std(embedding_matrix),
            "min": np.min(embedding_matrix),
            "max": np.max(embedding_matrix),
            "sparsity": np.mean(np.abs(embedding_matrix) < 0.01),  # Percentage near zero
        }
        
        # Compute pairwise distances
        cosine_distances = pairwise_distances(embedding_matrix, metric='cosine')
        euclidean_distances = pairwise_distances(embedding_matrix, metric='euclidean')
        
        stats["cosine_distance_stats"] = {
            "mean": np.mean(cosine_distances[np.triu_indices_from(cosine_distances, k=1)]),
            "std": np.std(cosine_distances[np.triu_indices_from(cosine_distances, k=1)]),
            "min": np.min(cosine_distances[np.triu_indices_from(cosine_distances, k=1)]),
            "max": np.max(cosine_distances[np.triu_indices_from(cosine_distances, k=1)])
        }
        
        stats["euclidean_distance_stats"] = {
            "mean": np.mean(euclidean_distances[np.triu_indices_from(euclidean_distances, k=1)]),
            "std": np.std(euclidean_distances[np.triu_indices_from(euclidean_distances, k=1)]),
            "min": np.min(euclidean_distances[np.triu_indices_from(euclidean_distances, k=1)]),
            "max": np.max(euclidean_distances[np.triu_indices_from(euclidean_distances, k=1)])
        }
        
        # Intrinsic dimensionality estimate (using PCA)
        pca = PCA()
        pca.fit(embedding_matrix)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        stats["intrinsic_dim_90"] = np.argmax(cumsum >= 0.9) + 1
        stats["intrinsic_dim_95"] = np.argmax(cumsum >= 0.95) + 1
        
        return stats
    
    def cluster_embeddings(self, n_clusters: int = 10, method: str = "kmeans") -> Dict[str, int]:
        """Cluster embeddings to identify groups."""
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(embedding_matrix)
        elif method == "dbscan":
            # DBSCAN for density-based clustering
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            labels = clusterer.fit_predict(embedding_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Create mapping
        cluster_mapping = {name: int(label) for name, label in zip(self.track_names, labels)}
        
        # Analyze cluster distribution
        unique_labels = np.unique(labels)
        cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
        
        logging.info(f"Found {len(unique_labels)} clusters with sizes: {cluster_sizes}")
        
        return cluster_mapping
    
    def compute_diversity_metrics(self, track_name: str, recommendations: List[str]) -> Dict[str, float]:
        """Compute diversity metrics for a set of recommendations."""
        # Get embeddings
        query_embedding = self.embeddings[track_name]
        rec_embeddings = np.array([self.embeddings[rec] for rec in recommendations])
        
        # Intra-list diversity (average pairwise distance among recommendations)
        if len(recommendations) > 1:
            pairwise_distances_recs = pairwise_distances(rec_embeddings, metric='cosine')
            intra_diversity = np.mean(pairwise_distances_recs[np.triu_indices_from(pairwise_distances_recs, k=1)])
        else:
            intra_diversity = 0.0
        
        # Coverage diversity (how much of the space is covered)
        all_embeddings = np.array(list(self.embeddings.values()))
        nearest_distances = []
        for rec_emb in rec_embeddings:
            distances = pairwise_distances([rec_emb], all_embeddings, metric='cosine')[0]
            nearest_distances.append(np.min(distances[distances > 0]))
        coverage_diversity = np.mean(nearest_distances)
        
        # Feature diversity (variance in different dimensions)
        feature_variance = np.mean(np.var(rec_embeddings, axis=0))
        
        return {
            "intra_list_diversity": intra_diversity,
            "coverage_diversity": coverage_diversity,
            "feature_variance": feature_variance
        }
    
    def visualize_embedding_space(self, method: str = "pca", n_components: int = 2,
                                  highlight_tracks: Optional[List[str]] = None):
        """Visualize the embedding space using dimensionality reduction."""
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        
        if method == "pca":
            reducer = PCA(n_components=n_components)
            reduced = reducer.fit_transform(embedding_matrix)
        elif method == "tsne":
            reducer = TSNE(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embedding_matrix)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot all points
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50)
        
        # Highlight specific tracks if provided
        if highlight_tracks:
            highlight_indices = [self.track_names.index(track) for track in highlight_tracks 
                               if track in self.track_names]
            highlight_points = reduced[highlight_indices]
            plt.scatter(highlight_points[:, 0], highlight_points[:, 1], 
                       color='red', s=100, marker='*', label='Highlighted')
        
        plt.title(f"Embedding Space Visualization ({method.upper()})")
        plt.xlabel(f"Component 1")
        plt.ylabel(f"Component 2")
        if highlight_tracks:
            plt.legend()
        
        return plt.gcf()


class DiverseRecommender:
    """Enhanced recommender with diverse recommendation strategies."""
    
    def __init__(self, similarity_engine, diversity_analyzer: Optional[EmbeddingDiversityAnalyzer] = None):
        self.similarity_engine = similarity_engine
        self.diversity_analyzer = diversity_analyzer or EmbeddingDiversityAnalyzer()
        self.clusters = None
        
    def get_diverse_recommendations(self, track_name: str, strategy: str = "mmr", 
                                  n_recommendations: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """
        Get recommendations using various diversity strategies.
        
        Strategies:
        - 'mmr': Maximal Marginal Relevance
        - 'cluster': Cluster-based diversity
        - 'novelty': Novelty-based (less popular items)
        - 'serendipity': Unexpected but relevant
        - 'contrast': Deliberately different recommendations
        """
        if strategy == "mmr":
            return self._mmr_recommendations(track_name, n_recommendations, **kwargs)
        elif strategy == "cluster":
            return self._cluster_diverse_recommendations(track_name, n_recommendations)
        elif strategy == "novelty":
            return self._novelty_recommendations(track_name, n_recommendations)
        elif strategy == "serendipity":
            return self._serendipity_recommendations(track_name, n_recommendations)
        elif strategy == "contrast":
            return self._contrast_recommendations(track_name, n_recommendations)
        else:
            # Fallback to standard similarity
            return self.similarity_engine.search(track_name, n_recommendations)
    
    def _mmr_recommendations(self, track_name: str, n_recommendations: int, 
                           lambda_param: float = 0.5) -> List[Tuple[str, float]]:
        """
        Maximal Marginal Relevance: Balance between relevance and diversity.
        
        MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected_doc))
        """
        # Get query embedding
        query_embedding = self.diversity_analyzer.embeddings[track_name]
        
        # Get all candidates (top 50)
        candidates = self.similarity_engine.search(track_name, top_k=50, exclude_self=True)
        candidate_names = [c[0] for c in candidates]
        candidate_embeddings = np.array([self.diversity_analyzer.embeddings[name] for name in candidate_names])
        
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        candidate_norms = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # Compute relevance scores (cosine similarity)
        relevance_scores = np.dot(candidate_norms, query_norm)
        
        # MMR selection
        selected = []
        selected_embeddings = []
        remaining_indices = list(range(len(candidate_names)))
        
        while len(selected) < n_recommendations and remaining_indices:
            mmr_scores = []
            
            for idx in remaining_indices:
                relevance = relevance_scores[idx]
                
                if selected_embeddings:
                    # Compute max similarity to already selected items
                    selected_matrix = np.array(selected_embeddings)
                    similarities = np.dot(selected_matrix, candidate_norms[idx])
                    max_similarity = np.max(similarities)
                else:
                    max_similarity = 0
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
                mmr_scores.append(mmr)
            
            # Select item with highest MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected.append((candidate_names[best_idx], float(relevance_scores[best_idx])))
            selected_embeddings.append(candidate_norms[best_idx])
            remaining_indices.remove(best_idx)
        
        return selected
    
    def _cluster_diverse_recommendations(self, track_name: str, n_recommendations: int) -> List[Tuple[str, float]]:
        """Select recommendations from different clusters."""
        # Ensure clusters are computed
        if self.clusters is None:
            self.clusters = self.diversity_analyzer.cluster_embeddings(n_clusters=10)
        
        # Get query cluster
        query_cluster = self.clusters[track_name]
        
        # Get similar tracks
        all_similar = self.similarity_engine.search(track_name, top_k=50, exclude_self=True)
        
        # Separate by cluster
        same_cluster = []
        different_clusters = {}
        
        for track, score in all_similar:
            track_cluster = self.clusters[track]
            if track_cluster == query_cluster:
                same_cluster.append((track, score))
            else:
                if track_cluster not in different_clusters:
                    different_clusters[track_cluster] = []
                different_clusters[track_cluster].append((track, score))
        
        # Build diverse recommendations
        recommendations = []
        
        # Add best from query cluster
        if same_cluster and len(recommendations) < n_recommendations:
            recommendations.append(same_cluster[0])
        
        # Add best from each different cluster
        for cluster_tracks in different_clusters.values():
            if len(recommendations) >= n_recommendations:
                break
            if cluster_tracks:
                recommendations.append(cluster_tracks[0])
        
        # Fill remaining with most similar
        for track, score in all_similar:
            if len(recommendations) >= n_recommendations:
                break
            if track not in [r[0] for r in recommendations]:
                recommendations.append((track, score))
        
        return recommendations[:n_recommendations]
    
    def _novelty_recommendations(self, track_name: str, n_recommendations: int) -> List[Tuple[str, float]]:
        """Recommend less frequently recommended tracks (assuming uniform prior)."""
        # Get all candidates
        candidates = self.similarity_engine.search(track_name, top_k=30, exclude_self=True)
        
        # Compute average similarity to all other tracks (popularity proxy)
        popularity_scores = {}
        all_embeddings = np.array(list(self.diversity_analyzer.embeddings.values()))
        
        for track, _ in candidates:
            track_embedding = self.diversity_analyzer.embeddings[track]
            # Average similarity to all tracks
            similarities = np.dot(all_embeddings, track_embedding) / (
                np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(track_embedding)
            )
            popularity_scores[track] = np.mean(similarities)
        
        # Sort by relevance but penalize popular items
        novelty_scores = []
        for track, relevance in candidates:
            popularity = popularity_scores[track]
            # Novelty score: high relevance, low popularity
            novelty = relevance * (1 - popularity)
            novelty_scores.append((track, novelty))
        
        # Sort by novelty score
        novelty_scores.sort(key=lambda x: x[1], reverse=True)
        
        return novelty_scores[:n_recommendations]
    
    def _serendipity_recommendations(self, track_name: str, n_recommendations: int) -> List[Tuple[str, float]]:
        """Find unexpected but relevant recommendations."""
        # Get standard recommendations
        expected = self.similarity_engine.search(track_name, top_k=10, exclude_self=True)
        expected_names = [e[0] for e in expected]
        
        # Get broader set of candidates
        all_candidates = self.similarity_engine.search(track_name, top_k=50, exclude_self=True)
        
        # Find tracks that are relevant but not in expected set
        serendipitous = []
        for track, score in all_candidates[10:]:  # Skip the most expected ones
            if track not in expected_names and score > 0.7:  # Still relevant
                serendipitous.append((track, score))
        
        # If not enough serendipitous items, add some with lower similarity
        if len(serendipitous) < n_recommendations:
            for track, score in all_candidates[10:]:
                if track not in [s[0] for s in serendipitous] and score > 0.5:
                    serendipitous.append((track, score))
                if len(serendipitous) >= n_recommendations:
                    break
        
        return serendipitous[:n_recommendations]
    
    def _contrast_recommendations(self, track_name: str, n_recommendations: int) -> List[Tuple[str, float]]:
        """Recommend deliberately different tracks while maintaining musical coherence."""
        query_embedding = self.diversity_analyzer.embeddings[track_name]
        
        # Find tracks with moderate similarity (not too similar, not too different)
        all_tracks = list(self.diversity_analyzer.embeddings.keys())
        contrast_scores = []
        
        for track in all_tracks:
            if track == track_name:
                continue
            
            track_embedding = self.diversity_analyzer.embeddings[track]
            
            # Cosine similarity
            similarity = np.dot(query_embedding, track_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(track_embedding)
            )
            
            # Look for moderate similarity (0.4-0.7 range)
            if 0.4 <= similarity <= 0.7:
                # Boost tracks in the "sweet spot" of different but related
                contrast_score = 1.0 - abs(similarity - 0.55)  # Peak at 0.55
                contrast_scores.append((track, contrast_score))
        
        # Sort by contrast score
        contrast_scores.sort(key=lambda x: x[1], reverse=True)
        
        return contrast_scores[:n_recommendations]
    
    def get_multi_strategy_recommendations(self, track_name: str, n_per_strategy: int = 2) -> Dict[str, List[Tuple[str, float]]]:
        """Get recommendations from multiple strategies for comparison."""
        strategies = ["mmr", "cluster", "novelty", "serendipity", "contrast"]
        results = {}
        
        for strategy in strategies:
            try:
                results[strategy] = self.get_diverse_recommendations(
                    track_name, strategy=strategy, n_recommendations=n_per_strategy
                )
            except Exception as e:
                logging.error(f"Failed to get {strategy} recommendations: {e}")
                results[strategy] = []
        
        return results


class SimilarityMetrics:
    """Alternative similarity metrics beyond cosine similarity."""
    
    @staticmethod
    def manhattan_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Manhattan (L1) distance converted to similarity."""
        distance = np.sum(np.abs(vec1 - vec2))
        # Convert to similarity (0-1 range)
        return 1.0 / (1.0 + distance)
    
    @staticmethod
    def angular_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Angular similarity (normalized angle between vectors)."""
        cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        # Clamp to valid range
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        # Convert to angle and normalize
        angle = np.arccos(cos_sim)
        return 1.0 - (angle / np.pi)
    
    @staticmethod
    def correlation_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Pearson correlation coefficient as similarity."""
        # Center the vectors
        vec1_centered = vec1 - np.mean(vec1)
        vec2_centered = vec2 - np.mean(vec2)
        
        # Compute correlation
        correlation = np.dot(vec1_centered, vec2_centered) / (
            np.linalg.norm(vec1_centered) * np.linalg.norm(vec2_centered)
        )
        
        # Convert to 0-1 range
        return (correlation + 1.0) / 2.0
    
    @staticmethod
    def weighted_similarity(vec1: np.ndarray, vec2: np.ndarray, 
                          feature_weights: Optional[np.ndarray] = None) -> float:
        """Weighted cosine similarity with feature importance."""
        if feature_weights is None:
            # Default: equal weights
            feature_weights = np.ones_like(vec1)
        
        # Apply weights
        weighted_vec1 = vec1 * feature_weights
        weighted_vec2 = vec2 * feature_weights
        
        # Compute weighted cosine similarity
        return np.dot(weighted_vec1, weighted_vec2) / (
            np.linalg.norm(weighted_vec1) * np.linalg.norm(weighted_vec2)
        )
    
    @staticmethod
    def hybrid_similarity(vec1: np.ndarray, vec2: np.ndarray, 
                         weights: Dict[str, float] = None) -> float:
        """Combine multiple similarity metrics."""
        if weights is None:
            weights = {
                "cosine": 0.5,
                "angular": 0.2,
                "correlation": 0.2,
                "manhattan": 0.1
            }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Compute individual similarities
        similarities = {}
        
        if "cosine" in weights:
            cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            similarities["cosine"] = cos_sim
        
        if "angular" in weights:
            similarities["angular"] = SimilarityMetrics.angular_similarity(vec1, vec2)
        
        if "correlation" in weights:
            similarities["correlation"] = SimilarityMetrics.correlation_similarity(vec1, vec2)
        
        if "manhattan" in weights:
            similarities["manhattan"] = SimilarityMetrics.manhattan_similarity(vec1, vec2)
        
        # Weighted combination
        final_similarity = sum(weights[k] * similarities[k] for k in similarities)
        
        return final_similarity


def analyze_recommendation_patterns(similarity_engine, sample_tracks: List[str] = None, 
                                  top_k: int = 10) -> Dict:
    """Analyze patterns in current recommendations."""
    analyzer = EmbeddingDiversityAnalyzer()
    
    if sample_tracks is None:
        # Sample some tracks
        all_tracks = analyzer.track_names
        sample_tracks = np.random.choice(all_tracks, min(10, len(all_tracks)), replace=False)
    
    patterns = {
        "avg_similarity_scores": [],
        "diversity_metrics": [],
        "similarity_distributions": []
    }
    
    for track in sample_tracks:
        # Get recommendations
        recommendations = similarity_engine.search(track, top_k=top_k, exclude_self=True)
        rec_names = [r[0] for r in recommendations]
        rec_scores = [r[1] for r in recommendations]
        
        # Analyze similarity scores
        patterns["avg_similarity_scores"].append(np.mean(rec_scores))
        patterns["similarity_distributions"].append(rec_scores)
        
        # Compute diversity
        diversity = analyzer.compute_diversity_metrics(track, rec_names)
        patterns["diversity_metrics"].append(diversity)
    
    # Aggregate results
    results = {
        "mean_similarity": np.mean(patterns["avg_similarity_scores"]),
        "std_similarity": np.std(patterns["avg_similarity_scores"]),
        "mean_intra_diversity": np.mean([d["intra_list_diversity"] for d in patterns["diversity_metrics"]]),
        "mean_coverage_diversity": np.mean([d["coverage_diversity"] for d in patterns["diversity_metrics"]]),
        "similarity_range": (
            np.min([np.min(dist) for dist in patterns["similarity_distributions"]]),
            np.max([np.max(dist) for dist in patterns["similarity_distributions"]])
        )
    }
    
    return results


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    from pipeline.mess_ai.search.similarity import SimilaritySearchEngine
    
    # Standard similarity engine
    similarity_engine = SimilaritySearchEngine()
    
    # Diversity analyzer
    analyzer = EmbeddingDiversityAnalyzer()
    
    # Analyze global statistics
    print("\n=== Global Embedding Statistics ===")
    stats = analyzer.analyze_global_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Analyze recommendation patterns
    print("\n=== Current Recommendation Patterns ===")
    patterns = analyze_recommendation_patterns(similarity_engine)
    for key, value in patterns.items():
        print(f"{key}: {value}")
    
    # Test diverse recommendations
    print("\n=== Diverse Recommendation Strategies ===")
    diverse_recommender = DiverseRecommender(similarity_engine, analyzer)
    
    test_track = analyzer.track_names[0]
    print(f"\nTesting with track: {test_track}")
    
    # Get recommendations from different strategies
    multi_recs = diverse_recommender.get_multi_strategy_recommendations(test_track)
    
    for strategy, recs in multi_recs.items():
        print(f"\n{strategy.upper()} Strategy:")
        for track, score in recs[:3]:
            print(f"  - {track}: {score:.3f}")