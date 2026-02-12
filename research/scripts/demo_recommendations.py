#!/usr/bin/env python3
"""
Demo: Get music similarity recommendations using validated MERT layers.

Usage:
    python research/scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"
    python research/scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspect brightness
    python research/scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --k 10
"""

import argparse
import logging
from pathlib import Path

from mess.search.search import find_similar, load_features, search_by_aspect
from mess.probing import resolve_aspects

logging.basicConfig(level=logging.INFO, format='%(message)s')


def main(track_name: str, aspect: str = None, k: int = 5, dataset: str = "smd"):
    """Run similarity search demo."""

    # Construct features directory path
    features_dir = Path(f"data/processed/features/{dataset}/aggregated")

    if not features_dir.exists():
        print(f"Error: Features not found at {features_dir}")
        print("Run feature extraction first: python research/scripts/extract_features.py")
        return 1

    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS FOR: {track_name}")

    if aspect:
        # Search using specific validated musical aspect
        print(f"Musical Aspect: {aspect}")
        print(f"{'='*70}\n")

        try:
            results = search_by_aspect(track_name, aspect, str(features_dir), k=k)
        except ValueError as e:
            print(f"Error: {e}")
            print("\nAvailable aspects:")
            aspect_mappings = resolve_aspects()
            for asp, info in aspect_mappings.items():
                print(f"  - {asp}: {info['description']} (layer {info['layer']}, R²={info['r2_score']})")
            return 1
    else:
        # Search using all features
        print(f"Using: All features (aggregated)")
        print(f"{'='*70}\n")

        features, track_names = load_features(str(features_dir))
        results = find_similar(track_name, features, track_names, k=k)

    # Display results
    for i, (rec_track, similarity) in enumerate(results, 1):
        print(f"{i}. {rec_track}")
        print(f"   Similarity: {similarity:.4f}\n")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get music similarity recommendations")
    parser.add_argument("--track", required=True, help="Reference track name")
    parser.add_argument("--aspect", help="Musical aspect (e.g., brightness, texture, dynamics)")
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--dataset", default="smd", help="Dataset name (default: smd)")
    args = parser.parse_args()
    raise SystemExit(main(args.track, args.aspect, args.k, args.dataset))
