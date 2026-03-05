#!/usr/bin/env python3
"""Build a clip-level index from extracted segment embeddings."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mess.datasets.clip_index import ClipIndex, build_clip_records


def _load_id_maps(
    manifest_path: Path | None,
    track_col: str,
    recording_col: str,
    work_col: str,
) -> tuple[dict[str, str], dict[str, str]]:
    if manifest_path is None:
        return {}, {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    recording_map: dict[str, str] = {}
    work_map: dict[str, str] = {}
    for row in rows:
        track_id = row.get(track_col)
        if not track_id:
            continue
        track_id_str = str(track_id)

        recording_id = row.get(recording_col)
        if recording_id:
            recording_map[track_id_str] = str(recording_id)

        work_id = row.get(work_col)
        if work_id:
            work_map[track_id_str] = str(work_id)

    return recording_map, work_map


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
    manifest_work_col: str,
) -> int:
    recording_map, work_map = _load_id_maps(
        manifest_path=manifest_path,
        track_col=manifest_track_col,
        recording_col=manifest_recording_col,
        work_col=manifest_work_col,
    )

    records = build_clip_records(
        dataset_id=dataset_id,
        segments_dir=segments_dir,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
        recording_map=recording_map,
        work_map=work_map,
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
    parser.add_argument(
        "--manifest-work-col",
        default="work_id",
        help="Optional work ID column name in recording manifest",
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
            manifest_work_col=args.manifest_work_col,
        )
    )
