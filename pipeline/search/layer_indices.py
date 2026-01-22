"""
Layer-specific FAISS indexing for MERT embeddings.

This module enables per-layer retrieval by creating separate FAISS indices
for each MERT layer. Supports both raw and classical-mean-centered embeddings.
"""

import faiss
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LayerIndexBuilder:
    """Builds and manages per-layer FAISS indices."""

    def __init__(self, embeddings_dir: Path, output_dir: Path):
        """
        Initialize the layer index builder.

        Args:
            embeddings_dir: Directory containing aggregated embeddings [13, 768]
            output_dir: Directory to save FAISS indices
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track metadata
        self.track_ids: List[str] = []
        self.classical_means: Optional[np.ndarray] = None

    def load_embeddings(self, layer: int) -> Tuple[np.ndarray, List[str]]:
        """
        Load embeddings for a specific layer from all tracks.

        Args:
            layer: MERT layer index (0-12)

        Returns:
            embeddings: [n_tracks, 768] array
            track_ids: List of track identifiers
        """
        embeddings = []
        track_ids = []

        # Load all .npy files from aggregated directory
        aggregated_dir = self.embeddings_dir / "aggregated"
        if not aggregated_dir.exists():
            raise ValueError(f"Aggregated embeddings directory not found: {aggregated_dir}")

        for emb_file in sorted(aggregated_dir.glob("*.npy")):
            # Load full embedding [13, 768]
            full_emb = np.load(emb_file)

            if full_emb.shape != (13, 768):
                logger.warning(f"Skipping {emb_file.name}: unexpected shape {full_emb.shape}")
                continue

            # Extract specific layer
            layer_emb = full_emb[layer, :]  # [768,]
            embeddings.append(layer_emb)
            track_ids.append(emb_file.stem)

        embeddings = np.array(embeddings)  # [n_tracks, 768]
        logger.info(f"Loaded {len(embeddings)} tracks for layer {layer}")

        return embeddings, track_ids

    def compute_classical_mean(self, layers: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute mean embedding per layer across all tracks.

        Args:
            layers: List of layers to compute means for (default: all 13 layers)

        Returns:
            means: [n_layers, 768] array of mean vectors
        """
        if layers is None:
            layers = list(range(13))

        means = []
        for layer in layers:
            embeddings, _ = self.load_embeddings(layer)
            layer_mean = np.mean(embeddings, axis=0)  # [768,]
            means.append(layer_mean)
            logger.info(f"Layer {layer} mean computed: shape {layer_mean.shape}")

        self.classical_means = np.array(means)  # [n_layers, 768]
        return self.classical_means

    def save_classical_means(self, filename: str = "classical_means.npy"):
        """Save computed classical means to disk."""
        if self.classical_means is None:
            raise ValueError("Classical means not computed. Call compute_classical_mean() first.")

        save_path = self.output_dir / filename
        np.save(save_path, self.classical_means)
        logger.info(f"Saved classical means to {save_path}")

    def load_classical_means(self, filename: str = "classical_means.npy") -> np.ndarray:
        """Load pre-computed classical means from disk."""
        load_path = self.output_dir / filename
        if not load_path.exists():
            raise ValueError(f"Classical means file not found: {load_path}")

        self.classical_means = np.load(load_path)
        logger.info(f"Loaded classical means from {load_path}: shape {self.classical_means.shape}")
        return self.classical_means

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings for cosine similarity via inner product.

        Args:
            embeddings: [n_tracks, 768] array

        Returns:
            normalized: [n_tracks, 768] L2-normalized array
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        return embeddings / norms

    def build_layer_index(
        self,
        layer: int,
        center: bool = False,
        index_type: str = "FlatIP"
    ) -> Tuple[faiss.Index, List[str]]:
        """
        Build FAISS index for a specific layer.

        Args:
            layer: MERT layer index (0-12)
            center: Whether to subtract classical mean before indexing
            index_type: FAISS index type (default: FlatIP for exact search)

        Returns:
            index: Built FAISS index
            track_ids: List of track IDs corresponding to index positions
        """
        # Load embeddings
        embeddings, track_ids = self.load_embeddings(layer)

        # Apply centering if requested
        if center:
            if self.classical_means is None:
                raise ValueError("Classical means not loaded. Call compute_classical_mean() or load_classical_means() first.")

            if layer >= len(self.classical_means):
                raise ValueError(f"Layer {layer} not in classical means (shape: {self.classical_means.shape})")

            mean_vector = self.classical_means[layer, :]
            embeddings = embeddings - mean_vector[np.newaxis, :]
            logger.info(f"Layer {layer}: Applied classical mean centering")

        # L2-normalize for cosine similarity
        embeddings = self.normalize_embeddings(embeddings)

        # Build FAISS index
        dim = embeddings.shape[1]
        if index_type == "FlatIP":
            index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # Add embeddings
        index.add(embeddings.astype(np.float32))
        logger.info(f"Layer {layer} (centered={center}): Built index with {index.ntotal} vectors")

        return index, track_ids

    def save_index(
        self,
        index: faiss.Index,
        track_ids: List[str],
        layer: int,
        centered: bool = False
    ):
        """Save FAISS index and track ID mapping to disk."""
        # Save FAISS index
        suffix = "_centered" if centered else "_raw"
        index_filename = f"layer_{layer}{suffix}.index"
        index_path = self.output_dir / index_filename
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved index to {index_path}")

        # Save track IDs
        ids_filename = f"layer_{layer}{suffix}_ids.npy"
        ids_path = self.output_dir / ids_filename
        np.save(ids_path, track_ids)
        logger.info(f"Saved track IDs to {ids_path}")

    def load_index(self, layer: int, centered: bool = False) -> Tuple[faiss.Index, List[str]]:
        """Load pre-built FAISS index and track IDs from disk."""
        suffix = "_centered" if centered else "_raw"

        # Load index
        index_filename = f"layer_{layer}{suffix}.index"
        index_path = self.output_dir / index_filename
        if not index_path.exists():
            raise ValueError(f"Index file not found: {index_path}")
        index = faiss.read_index(str(index_path))

        # Load track IDs
        ids_filename = f"layer_{layer}{suffix}_ids.npy"
        ids_path = self.output_dir / ids_filename
        if not ids_path.exists():
            raise ValueError(f"Track IDs file not found: {ids_path}")
        track_ids = np.load(ids_path, allow_pickle=True).tolist()

        logger.info(f"Loaded layer {layer} (centered={centered}): {index.ntotal} vectors")
        return index, track_ids
