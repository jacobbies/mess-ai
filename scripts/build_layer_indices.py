#!/usr/bin/env python3
"""
Build per-layer FAISS indices for MERT embeddings.

Usage:
    python scripts/build_layer_indices.py --embeddings data/embeddings/smd-emb --output data/indices/per_layer
    python scripts/build_layer_indices.py --embeddings data/embeddings/smd-emb --output data/indices/per_layer --center
"""

import argparse
import logging
from pathlib import Path

from pipeline.search.layer_indices import LayerIndexBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build per-layer FAISS indices")
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Directory containing embeddings (with aggregated/ subdirectory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for FAISS indices"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 7, 8, 9, 12],
        help="Layers to build indices for (default: 0 1 2 7 8 9 12)"
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Build centered indices (subtract classical mean)"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Build raw (non-centered) indices"
    )

    args = parser.parse_args()

    # Default: build both raw and centered if neither specified
    if not args.center and not args.raw:
        args.center = True
        args.raw = True

    embeddings_dir = Path(args.embeddings)
    output_dir = Path(args.output)

    if not embeddings_dir.exists():
        logger.error(f"Embeddings directory not found: {embeddings_dir}")
        return 1

    logger.info(f"Building indices for layers: {args.layers}")
    logger.info(f"Embeddings directory: {embeddings_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Build raw indices: {args.raw}")
    logger.info(f"Build centered indices: {args.center}")

    # Initialize builder
    builder = LayerIndexBuilder(embeddings_dir, output_dir)

    # Compute classical means if centering is requested
    if args.center:
        logger.info("Computing classical means for all layers...")
        builder.compute_classical_mean(layers=list(range(13)))
        builder.save_classical_means()
        logger.info("Classical means saved")

    # Build indices for each layer
    for layer in args.layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Building indices for Layer {layer}")
        logger.info(f"{'='*60}")

        # Build raw index
        if args.raw:
            logger.info(f"Building raw index for layer {layer}...")
            index, track_ids = builder.build_layer_index(layer, center=False)
            builder.save_index(index, track_ids, layer, centered=False)
            logger.info(f"Layer {layer} raw index: {index.ntotal} vectors")

        # Build centered index
        if args.center:
            logger.info(f"Building centered index for layer {layer}...")
            index, track_ids = builder.build_layer_index(layer, center=True)
            builder.save_index(index, track_ids, layer, centered=True)
            logger.info(f"Layer {layer} centered index: {index.ntotal} vectors")

    logger.info(f"\n{'='*60}")
    logger.info("Index building complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Layers processed: {args.layers}")

    # Summary
    logger.info("\nGenerated files:")
    if args.center:
        logger.info(f"  - classical_means.npy (mean vectors for all 13 layers)")
    for layer in args.layers:
        if args.raw:
            logger.info(f"  - layer_{layer}_raw.index")
            logger.info(f"  - layer_{layer}_raw_ids.npy")
        if args.center:
            logger.info(f"  - layer_{layer}_centered.index")
            logger.info(f"  - layer_{layer}_centered_ids.npy")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
