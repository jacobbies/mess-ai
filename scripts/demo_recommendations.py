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

from mess.probing import resolve_aspects
from mess.search.search import (
    find_similar,
    load_features,
    search_by_aspect,
    search_by_aspects,
    search_by_clip,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def _parse_aspect_weights(raw: str) -> dict[str, float]:
    """
    Parse `aspect=weight,aspect=weight` into a dict.

    Example:
        "brightness=0.7,phrasing=0.3" -> {"brightness": 0.7, "phrasing": 0.3}
    """
    if not raw or not raw.strip():
        raise ValueError("Empty --aspects value")

    weights: dict[str, float] = {}
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
    aspect: str | None = None,
    aspects: str | None = None,
    clip_start: float | None = None,
    clip_duration: float = 5.0,
    dedupe_window: float = 5.0,
    k: int = 5,
    dataset: str = "smd",
):
    """Run similarity search demo."""
    # Construct features directory paths
    aggregated_features_dir = Path(f"data/embeddings/{dataset}-emb/aggregated")
    segment_features_dir = Path(f"data/embeddings/{dataset}-emb/segments")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS FOR: {track_name}")

    if aspect and aspects:
        print("Error: use either --aspect or --aspects, not both.")
        return 1
    if clip_start is not None and (aspect or aspects):
        print("Error: clip mode cannot be combined with --aspect/--aspects.")
        return 1

    if clip_start is not None:
        if not segment_features_dir.exists():
            print(f"Error: Segment features not found at {segment_features_dir}")
            print(
                "Run feature extraction with segments: "
                "python scripts/extract_features.py --feature-level segments"
            )
            return 1
        if clip_duration <= 0:
            print("Error: --clip-duration must be > 0.")
            return 1
        if dedupe_window < 0:
            print("Error: --dedupe-window must be >= 0.")
            return 1

        print(f"Clip Query: start={clip_start:.2f}s, duration={clip_duration:.2f}s")
        print("Using: Clip-level segment search")
        print(f"{'='*70}\n")

        try:
            results = search_by_clip(
                query_track=track_name,
                clip_start=clip_start,
                clip_duration=clip_duration,
                features_dir=str(segment_features_dir),
                k=k,
                dedupe_window_seconds=dedupe_window,
            )
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        for i, rec in enumerate(results, 1):
            print(f"{i}. {rec.track_id} [{rec.start_time:.2f}s - {rec.end_time:.2f}s]")
            print(f"   Similarity: {rec.similarity:.4f}\n")

        return 0

    if aspects:
        if not aggregated_features_dir.exists():
            print(f"Error: Features not found at {aggregated_features_dir}")
            print("Run feature extraction first: python scripts/extract_features.py")
            return 1
        print("Using: Weighted multi-aspect search")
        print(f"{'='*70}\n")

        try:
            aspect_weights = _parse_aspect_weights(aspects)
            results = search_by_aspects(
                query_track=track_name,
                aspect_weights=aspect_weights,
                features_dir=str(aggregated_features_dir),
                k=k,
            )
        except ValueError as e:
            print(f"Error: {e}")
            print("\nAvailable aspects:")
            aspect_mappings = resolve_aspects()
            for asp, info in aspect_mappings.items():
                print(
                    f"  - {asp}: {info['description']} "
                    f"(layer {info['layer']}, R²={info['r2_score']})"
                )
            return 1
    elif aspect:
        if not aggregated_features_dir.exists():
            print(f"Error: Features not found at {aggregated_features_dir}")
            print("Run feature extraction first: python scripts/extract_features.py")
            return 1
        # Search using specific validated musical aspect
        print(f"Musical Aspect: {aspect}")
        print(f"{'='*70}\n")

        try:
            results = search_by_aspect(track_name, aspect, str(aggregated_features_dir), k=k)
        except ValueError as e:
            print(f"Error: {e}")
            print("\nAvailable aspects:")
            aspect_mappings = resolve_aspects()
            for asp, info in aspect_mappings.items():
                print(
                    f"  - {asp}: {info['description']} "
                    f"(layer {info['layer']}, R²={info['r2_score']})"
                )
            return 1
    else:
        if not aggregated_features_dir.exists():
            print(f"Error: Features not found at {aggregated_features_dir}")
            print("Run feature extraction first: python scripts/extract_features.py")
            return 1
        # Search using all features
        print("Using: All features (aggregated)")
        print(f"{'='*70}\n")

        features, track_names = load_features(str(aggregated_features_dir))
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
    parser.add_argument(
        "--clip-start",
        type=float,
        help="Clip query start time in seconds (enables clip-level search)",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=5.0,
        help="Clip query duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--dedupe-window",
        type=float,
        default=5.0,
        help="Per-track timestamp dedupe window in seconds for clip results",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--dataset", default="smd", help="Dataset name (default: smd)")
    args = parser.parse_args()
    raise SystemExit(
        main(
            track_name=args.track,
            aspect=args.aspect,
            aspects=args.aspects,
            clip_start=args.clip_start,
            clip_duration=args.clip_duration,
            dedupe_window=args.dedupe_window,
            k=args.k,
            dataset=args.dataset,
        )
    )
