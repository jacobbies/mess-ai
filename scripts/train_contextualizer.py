#!/usr/bin/env python3
"""Train a segment transformer contextualizer on precomputed segment embeddings."""

from __future__ import annotations

import argparse
import json
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
from mess.training.context_config import ContextualizerConfig
from mess.training.context_trainer import (
    group_records_by_track,
    load_track_segments,
    train_contextualizer,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train segment transformer contextualizer on track segment embeddings"
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
        help="Directory to save outputs",
    )

    # Input
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="MERT layer to load (0-12); use -1 for flattened input",
    )
    parser.add_argument(
        "--train-split",
        action="append",
        default=["train"],
        help="Split label(s) to include for training pool (repeatable)",
    )

    # Model architecture
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--num-transformer-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--max-segments", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--pool-mode",
        choices=["mean", "cls"],
        default="mean",
    )

    # Training
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.07)

    # Loss weights
    parser.add_argument("--global-loss-weight", type=float, default=1.0)
    parser.add_argument("--local-loss-weight", type=float, default=0.5)

    # Mining
    parser.add_argument("--search-k", type=int, default=64)
    parser.add_argument("--positives-per-query", type=int, default=1)
    parser.add_argument("--negatives-per-query", type=int, default=8)
    parser.add_argument("--min-time-separation", type=float, default=5.0)
    parser.add_argument("--cross-recording-positives", action="store_true")
    parser.add_argument("--allow-same-recording-negatives", action="store_true")

    # Index / EMA
    parser.add_argument("--refresh-every", type=int, default=100)
    parser.add_argument("--ema-decay", type=float, default=0.995)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")

    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    split_set = set(args.train_split)
    if not split_set:
        raise ValueError("At least one --train-split is required")

    # Load clip index
    index = ClipIndex.from_path(args.clip_index)
    records = list(index)
    if not records:
        raise ValueError("Clip index is empty")

    # Determine layer and input dim
    layer = None if args.layer < 0 else args.layer
    # Input dim: 768 for single layer, 13*768 for flattened
    input_dim = 768 if layer is not None else 13 * 768

    # Group records by track and load segments
    print(f"Loading segments from {len(records)} clip records...")
    records_by_track = group_records_by_track(records)
    tracks = load_track_segments(records_by_track, layer=layer)
    print(f"Loaded {len(tracks)} tracks")

    if not tracks:
        raise ValueError("No tracks found after loading segments")

    config = ContextualizerConfig(
        input_dim=input_dim,
        context_dim=args.context_dim,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        max_segments=args.max_segments,
        dropout=args.dropout,
        pool_mode=args.pool_mode,
        input_layer=layer,
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        temperature=args.temperature,
        global_loss_weight=args.global_loss_weight,
        local_loss_weight=args.local_loss_weight,
        search_k=args.search_k,
        positives_per_query=args.positives_per_query,
        negatives_per_query=args.negatives_per_query,
        min_time_separation_sec=args.min_time_separation,
        require_cross_recording_positive=args.cross_recording_positives,
        exclude_same_recording_negative=not args.allow_same_recording_negatives,
        refresh_every=args.refresh_every,
        ema_decay=args.ema_decay,
        seed=args.seed,
        device=args.device,
        train_splits=tuple(sorted(split_set)),
    )

    # Train
    print("Starting contextualizer training...")
    result = train_contextualizer(tracks, config)

    # Save outputs
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "train_config.json"
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "contextualizer_checkpoint.pt"

    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    checkpoint_payload = {
        "input_dim": result.input_dim,
        "context_dim": result.context_dim,
        "used_device": result.used_device,
        "steps_completed": result.steps_completed,
        "skipped_steps": result.skipped_steps,
        "final_index_version": result.final_index_version,
        "online_state_dict": result.online_state_dict,
        "target_state_dict": result.target_state_dict,
        "model_kwargs": result.model_kwargs,
        "config": config.to_dict(),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    print(f"Tracks used: {len(tracks)}")
    print(f"Input dim: {result.input_dim}")
    print(f"Context dim: {result.context_dim}")
    print(f"Used device: {result.used_device}")
    print(f"Completed steps: {result.steps_completed}")
    print(f"Skipped steps: {result.skipped_steps}")
    print(f"Saved config: {config_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved checkpoint: {checkpoint_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
