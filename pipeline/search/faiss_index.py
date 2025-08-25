import faiss
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Tuple, List


class FAISSIndex:
    """
    FAISS-based similarity index for high-performance vector search.
    
    Uses IndexFlatIP (Inner Product) which equals cosine similarity 
    when vectors are L2-normalized.
    """
    
    def __init__(self, dimension: int = 9984):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Vector dimension (default: 13*768 = 9984 for MERT embeddings)
        """
        self.dimension = dimension
        self.index = None
        self._track_id_to_name = {}
        self._track_name_to_id = {}
        
    def build_index(self, features: np.ndarray, track_names: List[str]) -> None:
        """
        Build FAISS index from feature vectors.
        
        Args:
            features: Feature matrix of shape (n_tracks, dimension)
            track_names: List of track names corresponding to feature rows
        """
        if features.shape[1] != self.dimension:
            raise ValueError(f"Feature dimension {features.shape[1]} != expected {self.dimension}")
        
        if len(track_names) != features.shape[0]:
            raise ValueError(f"Number of track names {len(track_names)} != number of features {features.shape[0]}")
        
        # Normalize features for cosine similarity
        features_normalized = self._normalize_features(features)
        
        # Create FAISS index (Inner Product = cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors to index
        self.index.add(features_normalized.astype(np.float32))
        
        # Build track name mappings
        self._track_id_to_name = {i: name for i, name in enumerate(track_names)}
        self._track_name_to_id = {name: i for i, name in enumerate(track_names)}
        
        logging.info(f"Built FAISS index with {self.index.ntotal} tracks, dimension {self.dimension}")
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector of shape (dimension,)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (similarities, indices) where similarities are cosine similarity scores
            and indices are internal FAISS indices
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Normalize query vector
        query_normalized = self._normalize_features(query_vector.reshape(1, -1))
        
        # Search
        similarities, indices = self.index.search(query_normalized.astype(np.float32), k)
        
        return similarities[0].tolist(), indices[0].tolist()
    
    def search_by_track_name(self, track_name: str, k: int = 5, exclude_self: bool = True) -> Tuple[List[str], List[float]]:
        """
        Search for similar tracks by track name.
        
        Args:
            track_name: Name of reference track
            k: Number of similar tracks to return
            exclude_self: Whether to exclude the reference track from results
            
        Returns:
            Tuple of (track_names, similarity_scores)
        """
        if track_name not in self._track_name_to_id:
            available_tracks = list(self._track_name_to_id.keys())[:5]
            raise ValueError(f"Track '{track_name}' not found. Available tracks include: {available_tracks}")
        
        track_id = self._track_name_to_id[track_name]
        
        # Get the feature vector for this track
        query_vector = self.index.reconstruct(track_id)
        
        # Search for similar tracks (get k+1 if excluding self)
        search_k = k + 1 if exclude_self else k
        similarities, indices = self.search(query_vector, search_k)
        
        # Convert indices to track names and filter results
        result_tracks = []
        result_scores = []
        
        for sim, idx in zip(similarities, indices):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            result_track = self._track_id_to_name[idx]
            
            # Skip self if requested
            if exclude_self and result_track == track_name:
                continue
                
            result_tracks.append(result_track)
            result_scores.append(float(sim))
            
            # Stop when we have enough results
            if len(result_tracks) >= k:
                break
        
        return result_tracks, result_scores
    
    def get_track_names(self) -> List[str]:
        """Get list of all track names in the index."""
        return list(self._track_name_to_id.keys())
    
    def save_index(self, index_path: str) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        
        faiss.write_index(self.index, index_path)
        logging.info(f"Saved FAISS index to {index_path}")
    
    def load_index(self, index_path: str, track_names: List[str]) -> None:
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to saved index file
            track_names: List of track names in the same order as when index was built
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Rebuild track name mappings
        self._track_id_to_name = {i: name for i, name in enumerate(track_names)}
        self._track_name_to_id = {name: i for i, name in enumerate(track_names)}
        
        logging.info(f"Loaded FAISS index from {index_path} with {self.index.ntotal} tracks")
    
    @staticmethod
    def _normalize_features(features: np.ndarray) -> np.ndarray:
        """
        L2-normalize feature vectors for cosine similarity.
        
        Args:
            features: Feature matrix of shape (n_samples, dimension)
            
        Returns:
            L2-normalized features
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return features / norms