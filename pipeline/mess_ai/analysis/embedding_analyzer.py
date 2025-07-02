"""
Embedding space analysis tools for understanding MERT feature distributions.

This module provides tools to analyze the diversity and structure of MERT embeddings,
helping to understand why recommendations might be too similar and how to improve variety.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class EmbeddingAnalyzer:
    """Analyze MERT embedding space for diversity and structure."""
    
    def __init__(self, features_dir: str = "data/processed/features"):
        """Initialize analyzer with features directory."""
        self.features_dir = Path(features_dir)
        self.embeddings = {}
        self.track_names = []
        self._load_embeddings()
        
    def _load_embeddings(self):
        """Load aggregated embeddings for analysis."""
        aggregated_dir = self.features_dir / "aggregated"
        if not aggregated_dir.exists():
            raise FileNotFoundError(f"Aggregated features not found at {aggregated_dir}")
            
        for feature_file in sorted(aggregated_dir.glob("*.npy")):
            track_name = feature_file.stem
            embedding = np.load(feature_file).flatten()  # Flatten to 1D
            self.embeddings[track_name] = embedding
            self.track_names.append(track_name)
            
        logging.info(f"Loaded {len(self.embeddings)} embeddings for analysis")
        
    def analyze_distribution(self) -> Dict:
        """Analyze the overall distribution of embeddings."""
        # Convert to matrix
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        
        # Basic statistics
        stats = {
            'mean': np.mean(embedding_matrix, axis=0),
            'std': np.std(embedding_matrix, axis=0),
            'min': np.min(embedding_matrix),
            'max': np.max(embedding_matrix),
            'sparsity': np.mean(np.abs(embedding_matrix) < 0.01),  # % near zero
        }
        
        # Pairwise distances
        distances = pairwise_distances(embedding_matrix, metric='cosine')
        np.fill_diagonal(distances, np.nan)  # Ignore self-distances
        
        stats['distance_stats'] = {
            'mean_distance': np.nanmean(distances),
            'min_distance': np.nanmin(distances),
            'max_distance': np.nanmax(distances),
            'std_distance': np.nanstd(distances),
        }
        
        # Find tracks that are too similar or too different
        mean_dist_per_track = np.nanmean(distances, axis=1)
        stats['outliers'] = {
            'most_similar': self.track_names[np.argmin(mean_dist_per_track)],
            'most_different': self.track_names[np.argmax(mean_dist_per_track)],
            'similarity_variance': np.var(mean_dist_per_track),
        }
        
        return stats
    
    def cluster_embeddings(self, n_clusters: int = 5, method: str = 'kmeans') -> Dict:
        """Cluster embeddings to understand groupings."""
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(embedding_matrix)
            centers = clusterer.cluster_centers_
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.3, min_samples=2)
            labels = clusterer.fit_predict(embedding_matrix)
            centers = None
        else:
            raise ValueError(f"Unknown clustering method: {method}")
            
        # Analyze clusters
        cluster_info = defaultdict(list)
        for track, label in zip(self.track_names, labels):
            cluster_info[int(label)].append(track)
            
        # Calculate silhouette score if we have valid clusters
        silhouette = None
        if len(set(labels)) > 1 and -1 not in labels:
            silhouette = silhouette_score(embedding_matrix, labels)
            
        return {
            'labels': labels,
            'centers': centers,
            'cluster_members': dict(cluster_info),
            'silhouette_score': silhouette,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        }
    
    def reduce_dimensions(self, method: str = 'pca', n_components: int = 2) -> np.ndarray:
        """Reduce embeddings to 2D/3D for visualization."""
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        
        if method == 'pca':
            reducer = PCA(n_components=n_components)
            reduced = reducer.fit_transform(embedding_matrix)
            # Store explained variance for PCA
            self.explained_variance_ratio = reducer.explained_variance_ratio_
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=15)
            reduced = reducer.fit_transform(embedding_matrix)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
            
        return reduced
    
    def visualize_embedding_space(self, save_path: Optional[str] = None):
        """Create comprehensive visualization of embedding space."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. PCA visualization with clustering
        pca_2d = self.reduce_dimensions(method='pca', n_components=2)
        clusters = self.cluster_embeddings(n_clusters=5)
        
        scatter = axes[0, 0].scatter(pca_2d[:, 0], pca_2d[:, 1], 
                                    c=clusters['labels'], cmap='viridis', alpha=0.7)
        axes[0, 0].set_title('PCA Projection with K-Means Clusters')
        axes[0, 0].set_xlabel(f'PC1 ({self.explained_variance_ratio[0]:.2%} var)')
        axes[0, 0].set_ylabel(f'PC2 ({self.explained_variance_ratio[1]:.2%} var)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # 2. t-SNE visualization
        tsne_2d = self.reduce_dimensions(method='tsne', n_components=2)
        axes[0, 1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], alpha=0.7)
        axes[0, 1].set_title('t-SNE Projection')
        axes[0, 1].set_xlabel('t-SNE 1')
        axes[0, 1].set_ylabel('t-SNE 2')
        
        # 3. Distance distribution
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        distances = pairwise_distances(embedding_matrix, metric='cosine')
        np.fill_diagonal(distances, np.nan)
        
        axes[0, 2].hist(distances[~np.isnan(distances)].flatten(), bins=50, alpha=0.7)
        axes[0, 2].set_title('Pairwise Cosine Distance Distribution')
        axes[0, 2].set_xlabel('Cosine Distance')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.nanmean(distances), color='red', linestyle='--', 
                          label=f'Mean: {np.nanmean(distances):.3f}')
        axes[0, 2].legend()
        
        # 4. Feature variance across dimensions
        feature_std = np.std(embedding_matrix, axis=0)
        axes[1, 0].plot(feature_std[:100], alpha=0.7)  # Show first 100 dims
        axes[1, 0].set_title('Feature Standard Deviation (first 100 dims)')
        axes[1, 0].set_xlabel('Feature Dimension')
        axes[1, 0].set_ylabel('Standard Deviation')
        
        # 5. Cluster sizes
        cluster_sizes = [len(members) for members in clusters['cluster_members'].values()]
        axes[1, 1].bar(range(len(cluster_sizes)), cluster_sizes)
        axes[1, 1].set_title('Cluster Sizes')
        axes[1, 1].set_xlabel('Cluster ID')
        axes[1, 1].set_ylabel('Number of Tracks')
        
        # 6. Nearest neighbor distances
        nn_distances = np.sort(distances, axis=1)[:, 1]  # First non-self neighbor
        axes[1, 2].hist(nn_distances, bins=30, alpha=0.7)
        axes[1, 2].set_title('Nearest Neighbor Distance Distribution')
        axes[1, 2].set_xlabel('Distance to Nearest Neighbor')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].axvline(np.mean(nn_distances), color='red', linestyle='--',
                          label=f'Mean: {np.mean(nn_distances):.3f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved embedding space visualization to {save_path}")
        
        return fig
    
    def find_diverse_tracks(self, reference_track: str, n_tracks: int = 5) -> List[Tuple[str, float]]:
        """Find tracks that are meaningfully different from reference."""
        if reference_track not in self.embeddings:
            raise ValueError(f"Track '{reference_track}' not found")
            
        ref_embedding = self.embeddings[reference_track]
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        
        # Calculate distances
        distances = pairwise_distances([ref_embedding], embedding_matrix, metric='cosine')[0]
        
        # Find tracks in the "sweet spot" - not too similar, not too different
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Target tracks that are 0.5-1.5 standard deviations away
        target_min = mean_dist - 0.5 * std_dist
        target_max = mean_dist + 1.5 * std_dist
        
        candidates = []
        for i, (track, dist) in enumerate(zip(self.track_names, distances)):
            if track != reference_track and target_min <= dist <= target_max:
                candidates.append((track, dist))
                
        # Sort by distance and return top n
        candidates.sort(key=lambda x: x[1])
        return candidates[:n_tracks]
    
    def analyze_track_neighborhood(self, track_name: str, k: int = 10) -> Dict:
        """Analyze the neighborhood structure around a specific track."""
        if track_name not in self.embeddings:
            raise ValueError(f"Track '{track_name}' not found")
            
        embedding_matrix = np.array([self.embeddings[name] for name in self.track_names])
        ref_idx = self.track_names.index(track_name)
        
        # Calculate all distances from reference
        distances = pairwise_distances([embedding_matrix[ref_idx]], embedding_matrix, metric='cosine')[0]
        
        # Get k nearest neighbors
        neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
        neighbor_names = [self.track_names[i] for i in neighbor_indices]
        neighbor_distances = distances[neighbor_indices]
        
        # Analyze diversity within neighborhood
        neighbor_embeddings = embedding_matrix[neighbor_indices]
        internal_distances = pairwise_distances(neighbor_embeddings, metric='cosine')
        
        return {
            'neighbors': list(zip(neighbor_names, neighbor_distances.tolist())),
            'avg_neighbor_distance': np.mean(neighbor_distances),
            'neighbor_diversity': np.mean(internal_distances),
            'most_similar_pair': self._find_most_similar_in_group(neighbor_names, internal_distances),
            'distance_range': (np.min(neighbor_distances), np.max(neighbor_distances)),
        }
    
    def _find_most_similar_in_group(self, names: List[str], distances: np.ndarray) -> Tuple[str, str, float]:
        """Find the most similar pair within a group."""
        min_dist = np.inf
        pair = (None, None)
        
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                if distances[i, j] < min_dist:
                    min_dist = distances[i, j]
                    pair = (names[i], names[j])
                    
        return (*pair, float(min_dist))