#!/usr/bin/env python3
"""Build a clip-level index from extracted segment embeddings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mess.datasets.clip_index import ClipIndex, ClipRecord


def _hop_seconds(segment_duration: float, overlap_ratio: float) -> float:
    hop = segment_duration * (1.0 - overlap_ratio)
    if hop <= 0:
        raise ValueError("segment_duration and overlap_ratio must produce hop > 0")
    return hop


def _load_recording_map(
    manifest_path: Path | None,
    track_col: str,
    recording_col: str,
) -> dict[str, str]:
    if manifest_path is None:
        return {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    mapping: dict[str, str] = {}
    for row in rows:
        track_id = row.get(track_col)
        recording_id = row.get(recording_col)
        if not track_id or not recording_id:
            continue
        mapping[str(track_id)] = str(recording_id)
    return mapping


def build_clip_records(
    dataset_id: str,
    segments_dir: Path,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
    default_split: str = "unspecified",
    recording_map: dict[str, str] | None = None,
) -> list[ClipRecord]:
    """Build clip records from per-track segment embedding files."""
    if not segments_dir.exists():
        raise FileNotFoundError(f"Segment directory not found: {segments_dir}")

    recording_lookup = recording_map or {}
    hop = _hop_seconds(segment_duration, overlap_ratio)

    records: list[ClipRecord] = []
    for embedding_file in sorted(segments_dir.glob("*.npy")):
        track_id = embedding_file.stem
        recording_id = recording_lookup.get(track_id, track_id)
        raw = np.load(embedding_file, mmap_mode="r")
        if raw.ndim < 3:
            raise ValueError(
                f"Expected segment embeddings shape [segments, 13, ...], got {raw.shape}"
            )
        n_segments = int(raw.shape[0])

        for segment_idx in range(n_segments):
            start_sec = segment_idx * hop
            end_sec = start_sec + segment_duration
            clip_id = f"{dataset_id}:{track_id}:{segment_idx:05d}"
            records.append(
                ClipRecord(
                    clip_id=clip_id,
                    dataset_id=dataset_id,
                    recording_id=recording_id,
                    track_id=track_id,
                    segment_idx=segment_idx,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    split=default_split,
                    embedding_path=str(embedding_file),
                )
            )
    return records


def main(
    dataset_id: str,
    segments_dir: Path,
    output_path: Path,
    segment_duration: float,
    overlap_ratio: float,
    assign_splits: bool,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    manifest_path: Path | None,
    manifest_track_col: str,
    manifest_recording_col: str,
) -> int:
    recording_map = _load_recording_map(
        manifest_path=manifest_path,
        track_col=manifest_track_col,
        recording_col=manifest_recording_col,
    )

    records = build_clip_records(
        dataset_id=dataset_id,
        segments_dir=segments_dir,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
        recording_map=recording_map,
    )
    index = ClipIndex(records)

    if assign_splits:
        index = index.assign_recording_splits(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

    index.to_csv(output_path)

    print(f"Dataset ID: {dataset_id}")
    print(f"Segment files: {len(sorted(segments_dir.glob('*.npy')))}")
    print(f"Total clips: {len(index)}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build clip-level index CSV from segment embeddings"
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="Dataset ID label (e.g., smd, maestro)",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        required=True,
        help="Directory containing per-track segment embedding .npy files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output clip index CSV path",
    )
    parser.add_argument("--segment-duration", type=float, default=5.0)
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument(
        "--assign-splits",
        action="store_true",
        help="Assign train/val/test splits by recording_id",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--recording-manifest",
        type=Path,
        default=None,
        help="Optional CSV mapping tracks to recording IDs",
    )
    parser.add_argument(
        "--manifest-track-col",
        default="track_id",
        help="Track ID column name in recording manifest",
    )
    parser.add_argument(
        "--manifest-recording-col",
        default="recording_id",
        help="Recording ID column name in recording manifest",
    )
    args = parser.parse_args()

    raise SystemExit(
        main(
            dataset_id=args.dataset_id,
            segments_dir=args.segments_dir,
            output_path=args.output,
            segment_duration=args.segment_duration,
            overlap_ratio=args.overlap_ratio,
            assign_splits=args.assign_splits,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            manifest_path=args.recording_manifest,
            manifest_track_col=args.manifest_track_col,
            manifest_recording_col=args.manifest_recording_col,
        )
    )
