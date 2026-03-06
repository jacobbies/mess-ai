#!/usr/bin/env python3
"""Train a retrieval-augmented projection head on precomputed clip embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mess.datasets.clip_index import ClipIndex, ClipRecord
from mess.datasets.stores import NpySegmentEmbeddingStore
from mess.search.faiss_index import upload_artifact_to_s3
from mess.training import (
    RetrievalSSLConfig,
    export_projection_clip_artifact,
    train_projection_head,
)


def _load_base_vectors(
    index: ClipIndex,
    *,
    layer: int | None,
    flatten: bool,
) -> tuple[np.ndarray, list[ClipRecord]]:
    store = NpySegmentEmbeddingStore(index=index, layer=layer, flatten=flatten, mmap=True)

    vectors: list[np.ndarray] = []
    records = list(index)
    for record in records:
        vector = np.asarray(store.get(record.clip_id), dtype=np.float32)
        if vector.ndim != 1:
            vector = vector.reshape(-1)
        vectors.append(vector)

    if not vectors:
        raise ValueError("No clip embeddings loaded from index")

    return np.vstack(vectors).astype(np.float32), records


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train retrieval-augmented projection head on clip embeddings"
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

    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="MERT layer to load (0-12); use -1 for flattened input",
    )
    parser.add_argument(
        "--flatten-input",
        action="store_true",
        help="Flatten embedding matrix input (forces full vector mode)",
    )
    parser.add_argument(
        "--train-split",
        action="append",
        default=["train"],
        help="Split label(s) to include for training pool (repeatable)",
    )

    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)

    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)

    parser.add_argument("--search-k", type=int, default=128)
    parser.add_argument("--positives-per-query", type=int, default=4)
    parser.add_argument("--negatives-per-query", type=int, default=16)
    parser.add_argument("--min-time-separation", type=float, default=5.0)
    parser.add_argument("--cross-recording-positives", action="store_true")
    parser.add_argument("--allow-same-recording-negatives", action="store_true")

    parser.add_argument("--refresh-every", type=int, default=50)
    parser.add_argument("--ema-decay", type=float, default=0.995)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument(
        "--export-artifact",
        action="store_true",
        help="Export projected clip vectors as a deployable FAISS artifact",
    )
    parser.add_argument(
        "--export-target-encoder",
        action="store_true",
        help="Export the EMA target encoder instead of the online projection head",
    )
    parser.add_argument(
        "--export-batch-size",
        type=int,
        default=1024,
        help="Batch size for projection export pass",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Artifact output root (default: <output-dir>/artifacts)",
    )
    parser.add_argument(
        "--artifact-name",
        default="projection_clip_index",
        help="Artifact logical name",
    )
    parser.add_argument(
        "--artifact-dataset",
        default=None,
        help="Override dataset name stored in artifact manifest",
    )
    parser.add_argument(
        "--artifact-index-type",
        choices=["flatip", "ivfflat"],
        default="flatip",
        help="FAISS index type for exported artifact",
    )
    parser.add_argument(
        "--artifact-nlist",
        type=int,
        default=1024,
        help="IVF centroid count when --artifact-index-type=ivfflat",
    )
    parser.add_argument(
        "--artifact-model-name",
        default="m-a-p/MERT-v1-95M+projection-head",
        help="Model name metadata saved in artifact manifest",
    )
    parser.add_argument(
        "--artifact-created-stamp",
        default=None,
        help="Optional deterministic artifact stamp (YYYYMMDDTHHMMSSZ)",
    )
    parser.add_argument(
        "--artifact-s3-bucket",
        default=None,
        help="Optional S3 bucket for artifact upload",
    )
    parser.add_argument(
        "--artifact-s3-prefix",
        default="mess/faiss",
        help="S3 prefix root for uploaded artifacts",
    )
    parser.add_argument(
        "--artifact-upload-latest",
        action="store_true",
        help="Upload and set latest pointer in S3",
    )

    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    split_set = set(args.train_split)
    if not split_set:
        raise ValueError("At least one --train-split is required")
    if not args.export_artifact:
        if args.artifact_s3_bucket is not None:
            raise ValueError("--artifact-s3-bucket requires --export-artifact")
        if args.artifact_upload_latest:
            raise ValueError("--artifact-upload-latest requires --export-artifact")

    index = ClipIndex.from_path(args.clip_index)
    index = index.filter(splits=split_set)

    if len(index) == 0:
        raise ValueError(f"No clip rows available after split filter: {sorted(split_set)}")

    layer = None if args.layer < 0 else args.layer
    flatten_input = args.flatten_input or layer is None
    vectors, records = _load_base_vectors(index=index, layer=layer, flatten=flatten_input)

    config = RetrievalSSLConfig(
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
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

    result = train_projection_head(
        base_vectors=vectors,
        records=records,
        config=config,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "train_config.json"
    metrics_path = output_dir / "metrics.json"
    checkpoint_path = output_dir / "projection_head_checkpoint.pt"

    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    checkpoint_payload = {
        "input_dim": result.input_dim,
        "output_dim": result.output_dim,
        "used_device": result.used_device,
        "steps_completed": result.steps_completed,
        "skipped_steps": result.skipped_steps,
        "final_index_version": result.final_index_version,
        "online_state_dict": result.online_state_dict,
        "target_state_dict": result.target_state_dict,
        "config": config.to_dict(),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    print(f"Rows used: {len(records)}")
    print(f"Input dim: {result.input_dim}")
    print(f"Projection dim: {result.output_dim}")
    print(f"Used device: {result.used_device}")
    print(f"Completed steps: {result.steps_completed}")
    print(f"Skipped steps: {result.skipped_steps}")
    print(f"Saved config: {config_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved checkpoint: {checkpoint_path}")

    if args.export_artifact:
        export_state_dict = (
            result.target_state_dict if args.export_target_encoder else result.online_state_dict
        )
        artifact_root = args.artifact_root or (output_dir / "artifacts")
        artifact_dir = export_projection_clip_artifact(
            base_vectors=vectors,
            records=records,
            state_dict=export_state_dict,
            input_dim=result.input_dim,
            projection_dim=result.output_dim,
            hidden_dim=config.hidden_dim,
            artifact_root=artifact_root,
            artifact_name=args.artifact_name,
            feature_source_dir=str(args.clip_index),
            dataset=args.artifact_dataset,
            layer=layer,
            index_type=args.artifact_index_type,
            model_name=args.artifact_model_name,
            nlist=args.artifact_nlist,
            created_stamp=args.artifact_created_stamp,
            batch_size=args.export_batch_size,
        )
        print(f"Saved projection artifact: {artifact_dir}")

        if args.artifact_s3_bucket:
            uploaded = upload_artifact_to_s3(
                artifact_dir,
                bucket=args.artifact_s3_bucket,
                prefix=args.artifact_s3_prefix,
                upload_latest_pointer=args.artifact_upload_latest,
            )
            print(
                "Uploaded "
                f"{len(uploaded)} object(s) to s3://{args.artifact_s3_bucket}/{args.artifact_s3_prefix}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
