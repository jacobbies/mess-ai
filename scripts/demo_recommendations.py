#!/usr/bin/env python3
"""
Demo: Get music similarity recommendations using validated MERT layers.

Usage:
    python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01"
    python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --aspect brightness
    python scripts/demo_recommendations.py --track "Beethoven_Op027No1-01" --k 10
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

from mess.search.search import (
    find_similar,
    load_features,
    search_by_aspect,
    search_by_aspects,
)
from mess.probing import resolve_aspects

logging.basicConfig(level=logging.INFO, format='%(message)s')


def _parse_aspect_weights(raw: str) -> Dict[str, float]:
    """
    Parse `aspect=weight,aspect=weight` into a dict.

    Example:
        "brightness=0.7,phrasing=0.3" -> {"brightness": 0.7, "phrasing": 0.3}
    """
    if not raw or not raw.strip():
        raise ValueError("Empty --aspects value")

    weights: Dict[str, float] = {}
    entries = [part.strip() for part in raw.split(",") if part.strip()]
    if not entries:
        raise ValueError("No valid aspect entries found")

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid entry '{entry}'. Use format aspect=weight."
            )
        aspect, weight_str = entry.split("=", 1)
        aspect = aspect.strip()
        if not aspect:
            raise ValueError(f"Invalid entry '{entry}': empty aspect name.")

        try:
            weight = float(weight_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"Invalid weight in '{entry}': must be numeric."
            ) from exc

        if weight < 0:
            raise ValueError(
                f"Invalid weight in '{entry}': must be >= 0."
            )
        weights[aspect] = weights.get(aspect, 0.0) + weight

    if sum(weights.values()) <= 0:
        raise ValueError("At least one aspect weight must be > 0")

    return weights


def main(
    track_name: str,
    aspect: Optional[str] = None,
    aspects: Optional[str] = None,
    k: int = 5,
    dataset: str = "smd",
):
    """Run similarity search demo."""

    # Construct features directory path
    features_dir = Path(f"data/embeddings/{dataset}-emb/aggregated")

    if not features_dir.exists():
        print(f"Error: Features not found at {features_dir}")
        print("Run feature extraction first: python scripts/extract_features.py")
        return 1

    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS FOR: {track_name}")

    if aspect and aspects:
        print("Error: use either --aspect or --aspects, not both.")
        return 1

    if aspects:
        print("Using: Weighted multi-aspect search")
        print(f"{'='*70}\n")

        try:
            aspect_weights = _parse_aspect_weights(aspects)
            results = search_by_aspects(
                query_track=track_name,
                aspect_weights=aspect_weights,
                features_dir=str(features_dir),
                k=k,
            )
        except ValueError as e:
            print(f"Error: {e}")
            print("\nAvailable aspects:")
            aspect_mappings = resolve_aspects()
            for asp, info in aspect_mappings.items():
                print(f"  - {asp}: {info['description']} (layer {info['layer']}, R²={info['r2_score']})")
            return 1
    elif aspect:
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
    parser.add_argument(
        "--aspects",
        help='Weighted aspects, e.g. "brightness=0.7,phrasing=0.3"',
    )
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--dataset", default="smd", help="Dataset name (default: smd)")
    args = parser.parse_args()
    raise SystemExit(main(args.track, args.aspect, args.aspects, args.k, args.dataset))
