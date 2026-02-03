#!/usr/bin/env python3
"""
Evaluate per-layer FAISS indices with sanity checks and distribution analysis.

This script tests:
1. Self-retrieval accuracy (each track should return itself at rank 1)
2. Similarity distribution analysis (raw vs centered, per layer)
3. Comparison plots showing spread improvement

Usage:
    python research/scripts/evaluate_layer_indices.py --indices data/indices/per_layer --embeddings data/embeddings/smd-emb --output results
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from mess.search.layer_indices import LayerIndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LayerIndexEvaluator:
    """Evaluates per-layer FAISS indices."""

    def __init__(self, indices_dir: Path, embeddings_dir: Path):
        """
        Initialize evaluator.

        Args:
            indices_dir: Directory containing FAISS indices
            embeddings_dir: Directory containing embeddings
        """
        self.indices_dir = Path(indices_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.builder = LayerIndexBuilder(embeddings_dir, indices_dir)

        # Load classical means for centering queries
        try:
            self.builder.load_classical_means()
        except ValueError:
            logger.warning("Classical means not found. Centered evaluation will be skipped.")

    def self_retrieval_test(
        self,
        layer: int,
        centered: bool = False,
        k: int = 10
    ) -> Tuple[float, List[str]]:
        """
        Test if each track retrieves itself at rank 1.

        Args:
            layer: Layer index to test
            centered: Whether to use centered index
            k: Number of neighbors to retrieve

        Returns:
            accuracy: Fraction of tracks that retrieve themselves at rank 1
            failures: List of track IDs that failed self-retrieval
        """
        # Load index
        index, track_ids = self.builder.load_index(layer, centered=centered)

        # Load embeddings
        embeddings, _ = self.builder.load_embeddings(layer)

        # Apply centering if needed
        if centered:
            mean_vector = self.builder.classical_means[layer, :]
            embeddings = embeddings - mean_vector[np.newaxis, :]

        # Normalize
        embeddings = self.builder.normalize_embeddings(embeddings)

        # Test each track
        correct = 0
        failures = []

        for i, track_id in enumerate(track_ids):
            query = embeddings[i:i+1, :].astype(np.float32)
            distances, indices = index.search(query, k)

            # Check if rank 1 is itself
            if indices[0, 0] == i:
                correct += 1
            else:
                failures.append(track_id)

        accuracy = correct / len(track_ids)
        return accuracy, failures

    def similarity_distribution_analysis(
        self,
        layer: int,
        centered: bool = False,
        n_samples: int = 50,
        k: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze similarity distribution for a layer.

        Args:
            layer: Layer index to analyze
            centered: Whether to use centered index
            n_samples: Number of random queries to sample
            k: Number of neighbors to retrieve per query

        Returns:
            stats: Dictionary with similarity statistics
        """
        # Load index
        index, track_ids = self.builder.load_index(layer, centered=centered)

        # Load embeddings
        embeddings, _ = self.builder.load_embeddings(layer)

        # Apply centering if needed
        if centered:
            mean_vector = self.builder.classical_means[layer, :]
            embeddings = embeddings - mean_vector[np.newaxis, :]

        # Normalize
        embeddings = self.builder.normalize_embeddings(embeddings)

        # Sample random queries
        n_tracks = len(track_ids)
        sample_indices = np.random.choice(n_tracks, min(n_samples, n_tracks), replace=False)

        # Collect similarities
        all_similarities = []

        for i in sample_indices:
            query = embeddings[i:i+1, :].astype(np.float32)
            distances, indices = index.search(query, k)

            # FAISS IndexFlatIP returns inner products (already cosine similarity for normalized vectors)
            similarities = distances[0, 1:]  # Skip rank 1 (self)
            all_similarities.extend(similarities)

        all_similarities = np.array(all_similarities)

        # Calculate statistics
        stats = {
            'similarities': all_similarities,
            'mean': np.mean(all_similarities),
            'std': np.std(all_similarities),
            'min': np.min(all_similarities),
            'max': np.max(all_similarities),
            'median': np.median(all_similarities),
            'q25': np.percentile(all_similarities, 25),
            'q75': np.percentile(all_similarities, 75)
        }

        return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-layer FAISS indices")
    parser.add_argument(
        "--indices",
        type=str,
        required=True,
        help="Directory containing FAISS indices"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Directory containing embeddings"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 7, 8, 9, 12],
        help="Layers to evaluate (default: 0 1 2 7 8 9 12)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of random queries for distribution analysis (default: 50)"
    )

    args = parser.parse_args()

    indices_dir = Path(args.indices)
    embeddings_dir = Path(args.embeddings)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluating indices for layers: {args.layers}")
    logger.info(f"Indices directory: {indices_dir}")
    logger.info(f"Embeddings directory: {embeddings_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize evaluator
    evaluator = LayerIndexEvaluator(indices_dir, embeddings_dir)

    # Results storage
    self_retrieval_results = defaultdict(dict)
    distribution_stats = defaultdict(dict)

    # Evaluate each layer
    for layer in args.layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Layer {layer}")
        logger.info(f"{'='*60}")

        # Self-retrieval tests
        for centered in [False, True]:
            suffix = "centered" if centered else "raw"
            logger.info(f"\nSelf-retrieval test: Layer {layer} ({suffix})")

            try:
                accuracy, failures = evaluator.self_retrieval_test(layer, centered=centered)
                self_retrieval_results[layer][suffix] = {
                    'accuracy': accuracy,
                    'failures': failures
                }

                logger.info(f"  Accuracy: {accuracy:.2%}")
                if failures:
                    logger.warning(f"  Failed tracks: {failures}")
                else:
                    logger.info("  ✓ All tracks retrieved themselves at rank 1")

            except Exception as e:
                logger.error(f"  Error in self-retrieval test: {e}")

        # Similarity distribution analysis
        for centered in [False, True]:
            suffix = "centered" if centered else "raw"
            logger.info(f"\nDistribution analysis: Layer {layer} ({suffix})")

            try:
                stats = evaluator.similarity_distribution_analysis(
                    layer, centered=centered, n_samples=args.n_samples
                )
                distribution_stats[layer][suffix] = stats

                logger.info(f"  Mean similarity: {stats['mean']:.4f}")
                logger.info(f"  Std: {stats['std']:.4f}")
                logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                logger.info(f"  Quartiles: Q25={stats['q25']:.4f}, Median={stats['median']:.4f}, Q75={stats['q75']:.4f}")

            except Exception as e:
                logger.error(f"  Error in distribution analysis: {e}")

    # Generate report
    logger.info(f"\n{'='*60}")
    logger.info("Generating evaluation report")
    logger.info(f"{'='*60}")

    report_path = output_dir / "phase1_sanity_check.md"
    with open(report_path, 'w') as f:
        f.write("# Phase 1 Evaluation: Layer-Specific Index Sanity Check\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Evaluated layers: {args.layers}\n")
        f.write(f"- Number of tracks: 50 (SMD dataset)\n")
        f.write(f"- Samples per layer: {args.n_samples}\n\n")

        f.write("## Self-Retrieval Accuracy\n\n")
        f.write("| Layer | Raw Accuracy | Centered Accuracy |\n")
        f.write("|-------|--------------|-------------------|\n")
        for layer in args.layers:
            raw_acc = self_retrieval_results[layer].get('raw', {}).get('accuracy', 0.0)
            cent_acc = self_retrieval_results[layer].get('centered', {}).get('accuracy', 0.0)
            f.write(f"| {layer} | {raw_acc:.2%} | {cent_acc:.2%} |\n")

        f.write("\n## Similarity Distribution Statistics\n\n")

        # Raw indices
        f.write("### Raw Indices (Non-Centered)\n\n")
        f.write("| Layer | Mean | Std | Min | Max | Q25 | Median | Q75 |\n")
        f.write("|-------|------|-----|-----|-----|-----|--------|-----|\n")
        for layer in args.layers:
            stats = distribution_stats[layer].get('raw', {})
            if stats:
                f.write(f"| {layer} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                       f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['q25']:.4f} | "
                       f"{stats['median']:.4f} | {stats['q75']:.4f} |\n")

        # Centered indices
        f.write("\n### Centered Indices (Classical Mean Subtraction)\n\n")
        f.write("| Layer | Mean | Std | Min | Max | Q25 | Median | Q75 |\n")
        f.write("|-------|------|-----|-----|-----|-----|--------|-----|\n")
        for layer in args.layers:
            stats = distribution_stats[layer].get('centered', {})
            if stats:
                f.write(f"| {layer} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                       f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['q25']:.4f} | "
                       f"{stats['median']:.4f} | {stats['q75']:.4f} |\n")

        f.write("\n## Key Observations\n\n")

        # Compare raw vs centered spread
        f.write("### Impact of Classical Centering\n\n")
        for layer in args.layers:
            raw_stats = distribution_stats[layer].get('raw', {})
            cent_stats = distribution_stats[layer].get('centered', {})

            if raw_stats and cent_stats:
                raw_std = raw_stats['std']
                cent_std = cent_stats['std']
                raw_range = raw_stats['max'] - raw_stats['min']
                cent_range = cent_stats['max'] - cent_stats['min']

                f.write(f"**Layer {layer}:**\n")
                f.write(f"- Standard deviation: {raw_std:.4f} (raw) → {cent_std:.4f} (centered)\n")
                f.write(f"- Range: {raw_range:.4f} (raw) → {cent_range:.4f} (centered)\n")
                f.write(f"- Spread improvement: {cent_range / raw_range:.2f}x\n\n")

        f.write("\n## Success Criteria\n\n")
        f.write("- ✓ Self-retrieval accuracy should be 100% for all layers\n")
        f.write("- ✓ Centered embeddings should show wider similarity spread\n")
        f.write("- ✓ All indices should be functional and queryable\n\n")

    logger.info(f"Report saved to: {report_path}")

    # Save raw data for plotting
    np.save(output_dir / "self_retrieval_results.npy", dict(self_retrieval_results))
    np.save(output_dir / "distribution_stats.npy", dict(distribution_stats))
    logger.info(f"Raw data saved to: {output_dir}")

    logger.info(f"\n{'='*60}")
    logger.info("Evaluation complete!")
    logger.info(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
