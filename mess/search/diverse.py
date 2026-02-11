"""
Diverse similarity search strategies for more varied music recommendations.

This module implements various similarity metrics and recommendation strategies
beyond simple cosine similarity to provide more interesting and varied results.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict, Optional, Callable
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from scipy.spatial.distance import correlation, chebyshev, minkowski
from scipy.stats import pearsonr, spearmanr
import random


class DiverseSimilaritySearch:
    """Advanced similarity search with multiple metrics and diversity strategies."""
    
    SIMILARITY_METRICS = {
        'cosine': 'Cosine similarity (default)',
        'euclidean': 'Euclidean distance',
        'manhattan': 'Manhattan (L1) distance',
        'correlation': 'Pearson correlation distance',
        'chebyshev': 'Chebyshev (L∞) distance',
        'angular': 'Angular distance',
        'jensen_shannon': 'Jensen-Shannon divergence',
    }
    
    def __init__(self, embeddings: Dict[str, np.ndarray]):
        """
        Initialize with precomputed embeddings.
        
        Args:
            embeddings: Dictionary mapping track names to embedding vectors
        """
        self.embeddings = embeddings
        self.track_names = list(embeddings.keys())
        self.embedding_matrix = np.array([embeddings[name] for name in self.track_names])
        
        # Precompute normalized embeddings for cosine/angular metrics
        self._normalized_embeddings = self._normalize_vectors(self.embedding_matrix)
        
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-8)
    
    def search_similar(self, 
                      track_name: str, 
                      metric: str = 'cosine',
                      n_results: int = 5,
                      temperature: float = 0.0) -> List[Tuple[str, float]]:
        """
        Search for similar tracks using specified metric.
        
        Args:
            track_name: Reference track
            metric: Similarity metric to use
            n_results: Number of results to return
            temperature: Softmax temperature for probabilistic selection (0=deterministic)
            
        Returns:
            List of (track_name, similarity_score) tuples
        """
        if track_name not in self.embeddings:
            raise ValueError(f"Track '{track_name}' not found")
            
        ref_idx = self.track_names.index(track_name)
        ref_embedding = self.embedding_matrix[ref_idx]
        
        # Calculate similarities/distances
        if metric == 'cosine':
            scores = cosine_similarity([ref_embedding], self.embedding_matrix)[0]
            higher_is_better = True
        elif metric == 'euclidean':
            scores = -euclidean_distances([ref_embedding], self.embedding_matrix)[0]
            higher_is_better = True
        elif metric == 'manhattan':
            scores = -manhattan_distances([ref_embedding], self.embedding_matrix)[0]
            higher_is_better = True
        elif metric == 'correlation':
            scores = np.array([1 - correlation(ref_embedding, emb) for emb in self.embedding_matrix])
            higher_is_better = True
        elif metric == 'chebyshev':
            scores = -np.array([chebyshev(ref_embedding, emb) for emb in self.embedding_matrix])
            higher_is_better = True
        elif metric == 'angular':
            # Angular distance = arccos(cosine_similarity) / pi
            cos_sim = cosine_similarity([ref_embedding], self.embedding_matrix)[0]
            scores = 1 - np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
            higher_is_better = True
        elif metric == 'jensen_shannon':
            # Convert to probability distributions (softmax)
            ref_prob = np.exp(ref_embedding) / np.sum(np.exp(ref_embedding))
            scores = []
            for emb in self.embedding_matrix:
                emb_prob = np.exp(emb) / np.sum(np.exp(emb))
                m = 0.5 * (ref_prob + emb_prob)
                js = 0.5 * np.sum(ref_prob * np.log(ref_prob / m + 1e-8)) + \
                     0.5 * np.sum(emb_prob * np.log(emb_prob / m + 1e-8))
                scores.append(-js)  # Negative because lower JS is more similar
            scores = np.array(scores)
            higher_is_better = True
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Apply temperature-based selection if specified
        if temperature > 0:
            return self._probabilistic_selection(scores, ref_idx, n_results, temperature, higher_is_better)
        else:
            return self._deterministic_selection(scores, ref_idx, n_results, higher_is_better)
    
    def _deterministic_selection(self, scores: np.ndarray, ref_idx: int, 
                               n_results: int, higher_is_better: bool) -> List[Tuple[str, float]]:
        """Standard deterministic selection of top results."""
        # Exclude reference track
        scores[ref_idx] = -np.inf if higher_is_better else np.inf
        
        # Get top indices
        if higher_is_better:
            top_indices = np.argsort(scores)[-n_results:][::-1]
        else:
            top_indices = np.argsort(scores)[:n_results]
            
        results = [(self.track_names[idx], float(scores[idx])) for idx in top_indices]
        return results
    
    def _probabilistic_selection(self, scores: np.ndarray, ref_idx: int,
                               n_results: int, temperature: float, 
                               higher_is_better: bool) -> List[Tuple[str, float]]:
        """Probabilistic selection using softmax with temperature."""
        # Exclude reference track
        valid_indices = np.arange(len(scores))
        valid_indices = valid_indices[valid_indices != ref_idx]
        valid_scores = scores[valid_indices]
        
        # Convert to similarities if needed
        if not higher_is_better:
            valid_scores = -valid_scores
            
        # Apply softmax with temperature
        exp_scores = np.exp(valid_scores / temperature)
        probabilities = exp_scores / np.sum(exp_scores)
        
        # Sample without replacement
        selected_indices = np.random.choice(valid_indices, size=min(n_results, len(valid_indices)),
                                          replace=False, p=probabilities)
        
        results = [(self.track_names[idx], float(scores[idx])) for idx in selected_indices]
        return sorted(results, key=lambda x: x[1], reverse=higher_is_better)
    
    def search_diverse(self, 
                      track_name: str,
                      n_results: int = 5,
                      diversity_weight: float = 0.3,
                      metric: str = 'cosine') -> List[Tuple[str, float]]:
        """
        Search with diversity promotion using Maximal Marginal Relevance (MMR).
        
        Args:
            track_name: Reference track
            n_results: Number of results to return
            diversity_weight: Weight for diversity (0=pure similarity, 1=pure diversity)
            metric: Base similarity metric
            
        Returns:
            List of diverse (track_name, score) tuples
        """
        if track_name not in self.embeddings:
            raise ValueError(f"Track '{track_name}' not found")
            
        # Get initial similarity scores
        initial_results = self.search_similar(track_name, metric, n_results * 3)
        candidate_tracks = [t[0] for t in initial_results]
        
        # MMR-based selection
        selected = []
        remaining = candidate_tracks.copy()
        
        # Add most similar as first result
        selected.append(remaining.pop(0))
        
        # Iteratively add tracks that balance similarity and diversity
        ref_embedding = self.embeddings[track_name]
        while len(selected) < n_results and remaining:
            scores = []
            
            for candidate in remaining:
                cand_embedding = self.embeddings[candidate]
                
                # Similarity to query
                sim_to_query = cosine_similarity([cand_embedding], [ref_embedding])[0, 0]
                
                # Maximum similarity to already selected items
                max_sim_to_selected = 0
                for selected_track in selected:
                    sel_embedding = self.embeddings[selected_track]
                    sim = cosine_similarity([cand_embedding], [sel_embedding])[0, 0]
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                # MMR score
                mmr_score = (1 - diversity_weight) * sim_to_query - diversity_weight * max_sim_to_selected
                scores.append((candidate, mmr_score))
            
            # Select track with highest MMR score
            scores.sort(key=lambda x: x[1], reverse=True)
            next_track = scores[0][0]
            selected.append(next_track)
            remaining.remove(next_track)
        
        # Return with similarity scores
        results = []
        for track in selected:
            track_embedding = self.embeddings[track]
            score = cosine_similarity([track_embedding], [ref_embedding])[0, 0]
            results.append((track, float(score)))
            
        return results
    
    def search_contrasting(self,
                          track_name: str,
                          n_results: int = 5,
                          contrast_range: Tuple[float, float] = (0.4, 0.7)) -> List[Tuple[str, float]]:
        """
        Find tracks that are intentionally different but not completely unrelated.
        
        Args:
            track_name: Reference track
            n_results: Number of results
            contrast_range: Target distance range (min, max) for contrasting tracks
            
        Returns:
            List of contrasting (track_name, distance) tuples
        """
        if track_name not in self.embeddings:
            raise ValueError(f"Track '{track_name}' not found")
            
        ref_idx = self.track_names.index(track_name)
        ref_embedding = self.embedding_matrix[ref_idx]
        
        # Calculate cosine distances
        similarities = cosine_similarity([ref_embedding], self.embedding_matrix)[0]
        distances = 1 - similarities
        
        # Find tracks in the contrast range
        candidates = []
        for i, (track, dist) in enumerate(zip(self.track_names, distances)):
            if i != ref_idx and contrast_range[0] <= dist <= contrast_range[1]:
                candidates.append((track, dist))
        
        # Sort by how close they are to the middle of the range
        target_dist = (contrast_range[0] + contrast_range[1]) / 2
        candidates.sort(key=lambda x: abs(x[1] - target_dist))
        
        return candidates[:n_results]
    
    def search_exploratory(self,
                          track_name: str,
                          n_results: int = 5,
                          exploration_clusters: int = 3) -> List[Tuple[str, float]]:
        """
        Exploratory search that samples from different regions of the embedding space.
        
        Args:
            track_name: Reference track  
            n_results: Number of results
            exploration_clusters: Number of different regions to explore
            
        Returns:
            List of exploratory (track_name, score) tuples
        """
        if track_name not in self.embeddings:
            raise ValueError(f"Track '{track_name}' not found")
            
        # Use k-means to identify different regions
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=exploration_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embedding_matrix)
        
        # Find which cluster the reference track belongs to
        ref_idx = self.track_names.index(track_name)
        ref_cluster = cluster_labels[ref_idx]
        ref_embedding = self.embedding_matrix[ref_idx]
        
        # Sample tracks from different clusters
        results = []
        tracks_per_cluster = n_results // exploration_clusters
        extra_tracks = n_results % exploration_clusters
        
        for cluster_id in range(exploration_clusters):
            # Get tracks from this cluster
            cluster_tracks = [self.track_names[i] for i, label in enumerate(cluster_labels) 
                            if label == cluster_id and i != ref_idx]
            
            if not cluster_tracks:
                continue
                
            # Calculate similarities
            cluster_embeddings = np.array([self.embeddings[t] for t in cluster_tracks])
            similarities = cosine_similarity([ref_embedding], cluster_embeddings)[0]
            
            # Determine how many tracks to take from this cluster
            n_from_cluster = tracks_per_cluster
            if cluster_id < extra_tracks:
                n_from_cluster += 1
            
            # Take top similar from this cluster
            top_indices = np.argsort(similarities)[-n_from_cluster:][::-1]
            
            for idx in top_indices:
                results.append((cluster_tracks[idx], float(similarities[idx])))
        
        # Sort by similarity and return top n
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]
    
    def search_complementary(self,
                           track_name: str,
                           n_results: int = 5,
                           feature_weights: Optional[Dict[str, float]] = None) -> List[Tuple[str, float]]:
        """
        Find complementary tracks by focusing on different feature dimensions.
        
        Args:
            track_name: Reference track
            n_results: Number of results  
            feature_weights: Optional weights for different feature dimensions
            
        Returns:
            List of complementary (track_name, score) tuples
        """
        if track_name not in self.embeddings:
            raise ValueError(f"Track '{track_name}' not found")
            
        ref_embedding = self.embeddings[track_name]
        
        # If no weights provided, use inverse variance weighting
        if feature_weights is None:
            feature_variance = np.var(self.embedding_matrix, axis=0)
            feature_weights = 1 / (feature_variance + 1e-8)
            feature_weights = feature_weights / np.sum(feature_weights)
        
        # Find tracks that are similar in low-variance dimensions but different in high-variance
        scores = []
        for i, track in enumerate(self.track_names):
            if track == track_name:
                continue
                
            track_embedding = self.embedding_matrix[i]
            
            # Weighted difference
            diff = np.abs(ref_embedding - track_embedding)
            weighted_diff = diff * feature_weights
            
            # Score based on inverse weighted difference
            score = 1 / (1 + np.sum(weighted_diff))
            scores.append((track, score))
        
        # Sort and return top results
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_results]