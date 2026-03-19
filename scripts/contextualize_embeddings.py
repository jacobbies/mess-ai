#!/usr/bin/env python3
"""Run trained contextualizer on track segments and save global+local embeddings."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch

from mess.datasets.clip_index import ClipIndex
from mess.training.context_export import (
    contextualize_tracks,
    export_contextualizer_track_artifact,
    load_contextualizer_from_state,
    save_contextualized_embeddings,
)
from mess.training.context_trainer import group_records_by_track, load_track_segments


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trained contextualizer and save global+local embeddings"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to contextualizer_checkpoint.pt",
    )
    parser.add_argument(
        "--clip-index",
        type=Path,
        required=True,
        help="Path to clip index CSV/parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save contextualized embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size",
    )
    parser.add_argument(
        "--use-target-encoder",
        action="store_true",
        help="Use EMA target encoder instead of online model",
    )
    parser.add_argument(
        "--export-artifact",
        action="store_true",
        help="Export global vectors as a deployable FAISS track artifact",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Artifact output root (default: <output-dir>/artifacts)",
    )
    parser.add_argument(
        "--artifact-name",
        default="contextualizer_track_index",
        help="Artifact logical name",
    )
    parser.add_argument(
        "--artifact-dataset",
        default=None,
        help="Dataset name for artifact manifest",
    )
    parser.add_argument(
        "--artifact-created-stamp",
        default=None,
        help="Optional deterministic artifact stamp (YYYYMMDDTHHMMSSZ)",
    )

    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_kwargs = checkpoint["model_kwargs"]
    config = checkpoint.get("config", {})

    state_key = "target_state_dict" if args.use_target_encoder else "online_state_dict"
    state_dict = checkpoint[state_key]

    model = load_contextualizer_from_state(state_dict, **model_kwargs)

    # Load tracks
    layer = config.get("input_layer")
    index = ClipIndex.from_path(args.clip_index)
    records = list(index)
    records_by_track = group_records_by_track(records)
    tracks = load_track_segments(records_by_track, layer=layer)

    print(f"Loaded {len(tracks)} tracks from {len(records)} clip records")

    # Run inference
    global_vectors, local_matrices = contextualize_tracks(
        model, tracks, batch_size=args.batch_size
    )
    track_ids = [t.track_id for t in tracks]

    # Save embeddings
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_contextualized_embeddings(global_vectors, local_matrices, track_ids, args.output_dir)
    print(f"Saved {len(track_ids)} global + local embeddings to {args.output_dir}")

    # Optional artifact export
    if args.export_artifact:
        artifact_root = args.artifact_root or (args.output_dir / "artifacts")
        dataset = args.artifact_dataset
        if dataset is None:
            dataset_ids = sorted({t.dataset_id for t in tracks})
            dataset = dataset_ids[0] if dataset_ids else "unknown"

        artifact_dir = export_contextualizer_track_artifact(
            global_vectors=global_vectors,
            track_ids=track_ids,
            artifact_root=artifact_root,
            artifact_name=args.artifact_name,
            dataset=dataset,
            layer=layer,
            created_stamp=args.artifact_created_stamp,
        )
        print(f"Saved track artifact: {artifact_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
