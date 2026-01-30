#!/usr/bin/env python3
"""
Similarity Evaluation Script
Benchmark different similarity metrics and FAISS index configurations.

Usage:
    python scripts/evaluate_similarity.py
"""

import logging
import time
from pathlib import Path

import numpy as np

from mess.config import mess_config
from mess.datasets.factory import DatasetFactory
from mess.search.faiss_index import FAISSIndex
from mess.search.similarity import SimilarityComputer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def benchmark_search_speed(index, n_queries=100):
    """Benchmark search performance."""
    logger.info(f"Running {n_queries} search queries...")

    # Get random query vectors
    query_vectors = []
    for i in range(n_queries):
        idx = np.random.randint(0, index.n_tracks)
        query_vectors.append(index.get_track_vector(idx))

    # Time searches
    start = time.time()
    for query_vec in query_vectors:
        _ = index.search(query_vec, k=10)
    elapsed = time.time() - start

    avg_time_ms = (elapsed / n_queries) * 1000
    logger.info(f"Average search time: {avg_time_ms:.2f}ms")

    return avg_time_ms


def evaluate_similarity_metrics(dataset="smd"):
    """Compare different similarity metrics."""
    logger.info("Evaluating similarity metrics...")

    # Load features
    dataset_obj = DatasetFactory.get_dataset(dataset)
    features_dir = dataset_obj.aggregated_dir

    # Get sample tracks
    feature_files = sorted(features_dir.glob("*.npy"))[:10]

    results = {}

    for metric in ['cosine', 'euclidean', 'dot_product']:
        logger.info(f"\nTesting {metric} similarity...")

        computer = SimilarityComputer(metric=metric)

        # Compute pairwise similarities
        similarities = []
        for i, f1 in enumerate(feature_files):
            for f2 in feature_files[i+1:]:
                feat1 = np.load(f1)
                feat2 = np.load(f2)

                # Use first layer for demo
                sim = computer.compute(feat1[0], feat2[0])
                similarities.append(sim)

        similarities = np.array(similarities)

        results[metric] = {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities))
        }

        print(f"  Mean: {results[metric]['mean']:.4f}")
        print(f"  Std:  {results[metric]['std']:.4f}")
        print(f"  Range: [{results[metric]['min']:.4f}, {results[metric]['max']:.4f}]")

    return results


def main():
    """Run similarity evaluation benchmarks."""
    print("=" * 70)
    print("SIMILARITY EVALUATION BENCHMARK")
    print("=" * 70)

    # Evaluate metrics
    print("\n1. Similarity Metrics Comparison")
    print("-" * 70)
    metric_results = evaluate_similarity_metrics()

    # Benchmark search speed
    print("\n2. Search Speed Benchmark")
    print("-" * 70)
    index = FAISSIndex(dataset="smd")
    index.build()
    avg_time = benchmark_search_speed(index)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best metric by discrimination: cosine (typically)")
    print(f"Average search time: {avg_time:.2f}ms")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
