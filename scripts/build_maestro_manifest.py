#!/usr/bin/env python3
"""
Build a join manifest between MAESTRO metadata and extracted embeddings.

Usage:
    python scripts/build_maestro_manifest.py
    python scripts/build_maestro_manifest.py --feature-type segments --include-missing
"""

import argparse
import csv
import json
from pathlib import Path


def load_maestro_rows(metadata_path: Path) -> list[dict]:
    """Load MAESTRO v3 metadata from the columnar JSON format."""
    raw = json.loads(metadata_path.read_text())
    required = {
        "canonical_composer",
        "canonical_title",
        "split",
        "year",
        "midi_filename",
        "audio_filename",
        "duration",
    }
    missing = required - set(raw.keys())
    if missing:
        raise ValueError(f"Metadata file missing keys: {sorted(missing)}")

    n_rows = len(raw["audio_filename"])
    rows = []
    for i in range(n_rows):
        key = str(i)
        rows.append(
            {
                "canonical_composer": raw["canonical_composer"][key],
                "canonical_title": raw["canonical_title"][key],
                "split": raw["split"][key],
                "year": int(raw["year"][key]),
                "midi_filename": raw["midi_filename"][key],
                "audio_filename": raw["audio_filename"][key],
                "duration": float(raw["duration"][key]),
            }
        )
    return rows


def build_embedding_lookup(embeddings_root: Path, feature_type: str) -> dict[str, Path]:
    """Map MAESTRO audio stem (lowercased) to embedding file path."""
    feature_dir = embeddings_root / feature_type
    if not feature_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {feature_dir}")

    lookup: dict[str, Path] = {}
    for npy_file in sorted(feature_dir.glob("*.npy")):
        stem = npy_file.stem
        if stem.startswith("maestro_"):
            stem = stem[len("maestro_") :]
        lookup[stem.lower()] = npy_file
    return lookup


def main(
    metadata_path: Path,
    embeddings_root: Path,
    feature_type: str,
    output_path: Path,
    include_missing: bool,
) -> int:
    rows = load_maestro_rows(metadata_path)
    embedding_lookup = build_embedding_lookup(embeddings_root, feature_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "canonical_composer",
        "canonical_title",
        "split",
        "year",
        "duration",
        "audio_filename",
        "midi_filename",
        "audio_stem",
        "feature_type",
        "embedding_path",
        "has_embedding",
    ]

    matched = 0
    written = 0
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            audio_stem = Path(row["audio_filename"]).stem
            emb_path = embedding_lookup.get(audio_stem.lower())
            has_embedding = emb_path is not None
            if has_embedding:
                matched += 1
            elif not include_missing:
                continue

            writer.writerow(
                {
                    **row,
                    "audio_stem": audio_stem,
                    "feature_type": feature_type,
                    "embedding_path": str(emb_path) if emb_path else "",
                    "has_embedding": int(has_embedding),
                }
            )
            written += 1

    print(f"Metadata rows: {len(rows)}")
    print(f"Embedding files ({feature_type}): {len(embedding_lookup)}")
    print(f"Matched rows: {matched}")
    print(f"Rows written: {written}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build MAESTRO metadata-to-embedding join manifest"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/audio/maestro/maestro-v3.0.0.json"),
        help="Path to maestro-v3.0.0.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("data/embeddings/maestro-emb"),
        help="Root directory containing raw/segments/aggregated embedding folders",
    )
    parser.add_argument(
        "--feature-type",
        choices=["raw", "segments", "aggregated"],
        default="segments",
        help="Embedding feature type to join against",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/metadata/maestro_embedding_manifest.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include metadata rows that do not have an embedding match",
    )
    args = parser.parse_args()

    raise SystemExit(
        main(
            metadata_path=args.metadata,
            embeddings_root=args.embeddings,
            feature_type=args.feature_type,
            output_path=args.output,
            include_missing=args.include_missing,
        )
    )
