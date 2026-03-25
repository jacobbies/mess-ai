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

from mess.datasets.metadata_table import DatasetMetadataTable
from mess.probing import resolve_aspects
from mess.search.faiss_index import find_latest_artifact_dir
from mess.search.hybrid import hybrid_search
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


def _parse_filters(raw_filters: list[str] | None) -> dict[str, str]:
    if not raw_filters:
        return {}
    parsed: dict[str, str] = {}
    for entry in raw_filters:
        if "=" not in entry:
            raise ValueError(
                f"Invalid filter '{entry}'. Use format key=value."
            )
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                f"Invalid filter '{entry}'. Both key and value are required."
            )
        parsed[key] = value
    return parsed


def main(
    track_name: str,
    aspect: str | None = None,
    aspects: str | None = None,
    clip_start: float | None = None,
    clip_duration: float = 5.0,
    dedupe_window: float = 5.0,
    keyword: str | None = None,
    filters: list[str] | None = None,
    semantic_weight: float = 1.0,
    keyword_weight: float = 0.3,
    metadata_path: str | None = None,
    k: int = 5,
    dataset: str = "smd",
):
    """Run similarity search demo."""
    # Construct features directory paths
    aggregated_features_dir = Path(f"data/embeddings/{dataset}-emb/aggregated")
    artifact_root = Path("data/indices")

    print(f"\n{'='*70}")
    print(f"RECOMMENDATIONS FOR: {track_name}")

    if aspect and aspects:
        print("Error: use either --aspect or --aspects, not both.")
        return 1
    if clip_start is not None and (aspect or aspects):
        print("Error: clip mode cannot be combined with --aspect/--aspects.")
        return 1
    if (keyword or filters) and (clip_start is not None or aspect or aspects):
        print(
            "Error: --keyword/--filter are currently supported only for baseline semantic mode."
        )
        return 1
    if semantic_weight < 0:
        print("Error: --semantic-weight must be >= 0.")
        return 1
    if keyword_weight < 0:
        print("Error: --keyword-weight must be >= 0.")
        return 1

    if clip_start is not None:
        if clip_duration <= 0:
            print("Error: --clip-duration must be > 0.")
            return 1
        if dedupe_window < 0:
            print("Error: --dedupe-window must be >= 0.")
            return 1

        try:
            clip_artifact_dir = find_latest_artifact_dir(
                artifact_root,
                artifact_name="clip_index",
            )
        except FileNotFoundError:
            print(f"Error: Clip artifact not found under {artifact_root / 'clip_index'}")
            print(
                "Build one first with "
                "python scripts/publish_faiss_index.py --kind clip --clip-index <path>"
            )
            return 1

        print(f"Clip Query: start={clip_start:.2f}s, duration={clip_duration:.2f}s")
        print("Using: Artifact-backed clip search")
        print(f"{'='*70}\n")

        try:
            results = search_by_clip(
                artifact=clip_artifact_dir,
                track_id=track_name,
                start_sec=clip_start,
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
        try:
            metadata_filters = _parse_filters(filters)
        except ValueError as exc:
            print(f"Error: {exc}")
            return 1
        if keyword or metadata_filters:
            metadata_file = (
                Path(metadata_path)
                if metadata_path
                else Path(f"data/metadata/{dataset}_metadata.csv")
            )
            if not metadata_file.exists():
                print(
                    "Error: metadata file not found for hybrid search: "
                    f"{metadata_file}"
                )
                return 1

            features, track_names = load_features(str(aggregated_features_dir))
            metadata = DatasetMetadataTable.from_path(metadata_file)
            results = hybrid_search(
                query_track=track_name,
                features=features,
                track_names=track_names,
                metadata=metadata,
                keyword=keyword,
                filters=metadata_filters,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                k=k,
            )
            print("Using: Hybrid semantic + metadata search")
            if keyword:
                print(f"Keyword: {keyword}")
            if metadata_filters:
                filter_text = ", ".join(
                    f"{key}={value}" for key, value in metadata_filters.items()
                )
                print(f"Filters: {filter_text}")
            print(
                f"Weights: semantic={semantic_weight:.2f}, "
                f"keyword={keyword_weight:.2f}"
            )
        else:
            # Search using all features
            print("Using: All features (aggregated)")
            features, track_names = load_features(str(aggregated_features_dir))
            results = find_similar(track_name, features, track_names, k=k)
        print(f"{'='*70}\n")

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
    parser.add_argument(
        "--keyword",
        help="Optional metadata keyword boost (hybrid mode, baseline semantic search only)",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Metadata hard filter key=value (repeatable, hybrid mode only)",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=1.0,
        help="Semantic score weight in hybrid mode",
    )
    parser.add_argument(
        "--keyword-weight",
        type=float,
        default=0.3,
        help="Keyword score weight in hybrid mode",
    )
    parser.add_argument(
        "--metadata-path",
        help=(
            "Metadata CSV/parquet path for hybrid mode "
            "(default: data/metadata/<dataset>_metadata.csv)"
        ),
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
            keyword=args.keyword,
            filters=args.filter,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
            metadata_path=args.metadata_path,
            k=args.k,
            dataset=args.dataset,
        )
    )
