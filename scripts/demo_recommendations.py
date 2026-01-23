#!/usr/bin/env python3
"""
Demo: Get music similarity recommendations using validated MERT layers.

Usage:
    python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"
    python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspect timbral_texture
    python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --n 10
"""

import argparse
import logging
from pipeline.query.layer_based_recommender import LayerBasedRecommender

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main(track_name: str, aspect: str = None, n: int = 5):
    recommender = LayerBasedRecommender()

    if aspect:
        results = recommender.recommend_by_aspect(track_name, aspect, n_recommendations=n)
    else:
        # Default: Layer 0 (spectral brightness)
        results = recommender.recommend_by_layer(track_name, layer=0, n_recommendations=n)

    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS FOR: {track_name}")
    if aspect:
        print(f"Aspect: {aspect}")
    print(f"{'='*70}\n")

    for i, (rec_track, similarity, info) in enumerate(results, 1):
        print(f"{i}. {rec_track}")
        print(f"   Similarity: {similarity:.4f} | {info}\n")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get music similarity recommendations")
    parser.add_argument("--track", required=True, help="Reference track name")
    parser.add_argument("--aspect", choices=["spectral_brightness", "timbral_texture", "acoustic_structure"])
    parser.add_argument("--n", type=int, default=5, help="Number of recommendations")
    args = parser.parse_args()
    raise SystemExit(main(args.track, args.aspect, args.n))
