#!/usr/bin/env python3
"""
Demo: Per-Layer FAISS Index Search

This script demonstrates how to query per-layer indices and compare results
across different layers to understand layer specializations.

Usage:
    python research/scripts/demo_layer_search.py --query "Bach_BWV849-01_001_20090916-SMD"
    python research/scripts/demo_layer_search.py --query "Beethoven_Op027No1-01_003_20090916-SMD" --layers 0 1 2 7 8 9 12
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from mess.search.indices import LayerIndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def search_layer(
    builder: LayerIndexBuilder,
    layer: int,
    query_track_id: str,
    centered: bool = False,
    k: int = 5
):
    """
    Search a specific layer index and return top-k results.

    Args:
        builder: LayerIndexBuilder instance
        layer: Layer to search
        query_track_id: Track ID to use as query
        centered: Whether to use centered index
        k: Number of results to return

    Returns:
        List of (track_id, similarity) tuples
    """
    # Load index and track IDs
    index, track_ids = builder.load_index(layer, centered=centered)

    # Load embeddings to get query vector
    embeddings, emb_track_ids = builder.load_embeddings(layer)

    # Find query track index
    try:
        query_idx = emb_track_ids.index(query_track_id)
    except ValueError:
        logger.error(f"Query track '{query_track_id}' not found in embeddings")
        return []

    # Get query embedding
    query_emb = embeddings[query_idx:query_idx+1, :]

    # Apply centering if needed
    if centered:
        mean_vector = builder.classical_means[layer, :]
        query_emb = query_emb - mean_vector[np.newaxis, :]

    # Normalize
    query_emb = builder.normalize_embeddings(query_emb)

    # Search
    similarities, indices = index.search(query_emb.astype(np.float32), k + 1)

    # Collect results (skip first result which is the query itself)
    results = []
    for i in range(1, len(indices[0])):
        idx = indices[0][i]
        sim = similarities[0][i]
        track_id = track_ids[idx]
        results.append((track_id, sim))

    return results


def compare_layers(
    indices_dir: Path,
    embeddings_dir: Path,
    query_track: str,
    layers: list,
    centered: bool = False,
    k: int = 5
):
    """
    Compare search results across multiple layers.

    Args:
        indices_dir: Directory containing FAISS indices
        embeddings_dir: Directory containing embeddings
        query_track: Track ID to query
        layers: List of layers to compare
        centered: Whether to use centered indices
        k: Number of results per layer
    """
    builder = LayerIndexBuilder(embeddings_dir, indices_dir)

    # Load classical means if using centered indices
    if centered:
        builder.load_classical_means()

    print(f"\n{'='*80}")
    print(f"QUERY TRACK: {query_track}")
    print(f"{'='*80}")
    print(f"Mode: {'Centered (Classical Mean Subtraction)' if centered else 'Raw Embeddings'}")
    print(f"Top-{k} results per layer\n")

    # Search each layer
    all_results = {}
    for layer in layers:
        logger.info(f"Searching layer {layer}...")
        results = search_layer(builder, layer, query_track, centered=centered, k=k)
        all_results[layer] = results

    # Display results
    for layer in layers:
        results = all_results[layer]

        print(f"\n{'-'*80}")
        print(f"LAYER {layer} RESULTS")
        print(f"{'-'*80}")

        if not results:
            print("  No results found")
            continue

        for rank, (track_id, similarity) in enumerate(results, 1):
            # Extract composer and piece from track ID
            parts = track_id.split('_')
            composer = parts[0] if parts else "Unknown"
            piece = parts[1] if len(parts) > 1 else "Unknown"

            print(f"  {rank}. [{similarity:.4f}] {composer} - {piece}")
            if rank <= 2:  # Show full ID for top 2
                print(f"      {track_id}")

    # Compare overlap between layers
    print(f"\n{'='*80}")
    print("LAYER COMPARISON: Top-3 Overlap")
    print(f"{'='*80}")

    # Check which tracks appear in top-3 for multiple layers
    track_appearances = {}
    for layer, results in all_results.items():
        for track_id, _ in results[:3]:
            if track_id not in track_appearances:
                track_appearances[track_id] = []
            track_appearances[track_id].append(layer)

    # Sort by number of appearances
    sorted_tracks = sorted(track_appearances.items(), key=lambda x: len(x[1]), reverse=True)

    for track_id, appearing_layers in sorted_tracks[:5]:
        parts = track_id.split('_')
        composer = parts[0] if parts else "Unknown"
        piece = parts[1] if len(parts) > 1 else "Unknown"

        layers_str = ", ".join(map(str, sorted(appearing_layers)))
        print(f"  {composer} - {piece}")
        print(f"    Appears in layers: {layers_str} ({len(appearing_layers)} layers)")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Demo per-layer FAISS search")
    parser.add_argument(
        "--indices",
        type=str,
        default="data/indices/per_layer",
        help="Directory containing FAISS indices"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/embeddings/smd-emb",
        help="Directory containing embeddings"
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Track ID to use as query (without .npy extension)"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 7, 8, 9, 12],
        help="Layers to search (default: 0 1 2 7 8 9 12)"
    )
    parser.add_argument(
        "--centered",
        action="store_true",
        help="Use centered indices (classical mean subtraction)"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return per layer (default: 5)"
    )

    args = parser.parse_args()

    indices_dir = Path(args.indices)
    embeddings_dir = Path(args.embeddings)

    if not indices_dir.exists():
        logger.error(f"Indices directory not found: {indices_dir}")
        return 1

    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return 1

    # Run comparison
    compare_layers(
        indices_dir,
        embeddings_dir,
        args.query,
        args.layers,
        args.centered,
        args.k
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
