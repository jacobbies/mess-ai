#!/usr/bin/env python3
"""Evaluate retrieval quality and emit deterministic JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mess.datasets.clip_index import ClipIndex, ClipRecord
from mess.datasets.stores import NpySegmentEmbeddingStore
from mess.evaluation.metrics import evaluate_rankings
from mess.training.trainer import ProjectionHead


def _parse_k_values(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(k <= 0 for k in values):
        raise ValueError("--k-values must contain positive integers (for example: 1,5,10)")
    return values


def _sample_query_indices(
    total: int,
    max_queries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if total <= 0:
        return np.array([], dtype=np.int64)
    if max_queries <= 0 or max_queries >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=max_queries, replace=False)).astype(np.int64)


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
        raise ValueError("No clip embeddings loaded from clip index")
    return np.vstack(vectors).astype(np.float32), records


def _normalize(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty [n, d] vectors, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def _rank_clip_indices(normalized_vectors: np.ndarray, query_idx: int) -> list[int]:
    similarities = normalized_vectors @ normalized_vectors[query_idx]
    similarities = similarities.astype(np.float32, copy=True)
    similarities[query_idx] = -np.inf
    ranked = np.argsort(-similarities)
    return ranked.astype(np.int64).tolist()


def _dedupe_track_ranking(ranked_clip_indices: list[int], track_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    ranked_tracks: list[str] = []
    for clip_idx in ranked_clip_indices:
        track_id = track_ids[clip_idx]
        if track_id in seen:
            continue
        seen.add(track_id)
        ranked_tracks.append(track_id)
    return ranked_tracks


def _evaluate_vectors(
    vectors: np.ndarray,
    records: list[ClipRecord],
    *,
    query_indices: np.ndarray,
    k_values: list[int],
    include_clip_to_clip: bool,
    include_clip_to_track: bool,
) -> dict[str, dict[str, object]]:
    if query_indices.size == 0:
        raise ValueError("No query indices available for evaluation")

    normalized = _normalize(vectors)
    track_ids = [record.track_id for record in records]
    per_track_indices: dict[str, set[int]] = {}
    for idx, track_id in enumerate(track_ids):
        per_track_indices.setdefault(track_id, set()).add(idx)

    clip_rankings: list[list[int]] = []
    clip_relevants: list[set[int]] = []
    track_rankings: list[list[str]] = []
    track_relevants: list[set[str]] = []

    for query_idx in query_indices:
        ranking = _rank_clip_indices(normalized, int(query_idx))
        query_track_id = track_ids[int(query_idx)]

        if include_clip_to_clip:
            relevant_clips = per_track_indices[query_track_id] - {int(query_idx)}
            if relevant_clips:
                clip_rankings.append(ranking)
                clip_relevants.append(relevant_clips)

        if include_clip_to_track:
            track_rankings.append(_dedupe_track_ranking(ranking, track_ids))
            track_relevants.append({query_track_id})

    results: dict[str, dict[str, object]] = {}
    if include_clip_to_clip:
        results["clip_to_clip"] = evaluate_rankings(
            rankings=clip_rankings,
            relevant_sets=clip_relevants,
            k_values=k_values,
        )
    if include_clip_to_track:
        results["clip_to_track"] = evaluate_rankings(
            rankings=track_rankings,
            relevant_sets=track_relevants,
            k_values=k_values,
        )
    return results


def _cpu_state_dict(payload: dict[str, object], key: str) -> dict[str, torch.Tensor]:
    raw_state = payload.get(key)
    if not isinstance(raw_state, dict):
        raise ValueError(f"Checkpoint missing `{key}` state dict")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if isinstance(value, torch.Tensor):
            state[str(name)] = value.detach().cpu()
        else:
            state[str(name)] = torch.as_tensor(value)
    return state


def _project_vectors(
    vectors: np.ndarray,
    *,
    checkpoint_path: Path,
    use_target_encoder: bool,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    if batch_size <= 0:
        raise ValueError("--projection-batch-size must be > 0")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Projection checkpoint payload must be a dict")

    input_dim = int(payload["input_dim"])
    output_dim = int(payload["output_dim"])
    config_payload = payload.get("config")
    hidden_dim: int | None = None
    if isinstance(config_payload, dict):
        hidden_value = config_payload.get("hidden_dim")
        hidden_dim = int(hidden_value) if hidden_value is not None else None

    state_key = "target_state_dict" if use_target_encoder else "online_state_dict"
    model = ProjectionHead(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    ).cpu()
    model.load_state_dict(_cpu_state_dict(payload, key=state_key))
    model.eval()

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected base vectors [n, d], got shape {arr.shape}")
    if arr.shape[1] != input_dim:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} does not match vectors width={arr.shape[1]}"
        )

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, arr.shape[0], batch_size):
            end = min(start + batch_size, arr.shape[0])
            batch = torch.from_numpy(arr[start:end])
            outputs.append(model(batch).cpu().numpy().astype(np.float32))

    return (
        np.vstack(outputs).astype(np.float32),
        {
            "checkpoint": str(checkpoint_path),
            "encoder": "target" if use_target_encoder else "online",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
        },
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate clip retrieval and emit JSON report with Recall@K/MRR/nDCG"
    )
    parser.add_argument("--clip-index", type=Path, required=True, help="Clip index CSV/parquet")
    parser.add_argument("--report-json", type=Path, required=True, help="Output report JSON path")
    parser.add_argument("--layer", type=int, default=0, help="MERT layer (0-12), use -1 to flatten")
    parser.add_argument(
        "--flatten-input",
        action="store_true",
        help="Flatten full embedding matrix input before evaluation",
    )
    parser.add_argument("--k-values", default="1,5,10", help="Comma-separated K values")
    parser.add_argument("--max-queries", type=int, default=500, help="Max sampled query clips")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--protocol",
        choices=["clip_to_clip", "clip_to_track", "both"],
        default="both",
        help="Evaluation protocol mode",
    )
    parser.add_argument(
        "--projection-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint produced by scripts/train_retrieval_ssl.py",
    )
    parser.add_argument(
        "--use-target-encoder",
        action="store_true",
        help="Use target encoder weights from checkpoint for projection evaluation",
    )
    parser.add_argument(
        "--projection-batch-size",
        type=int,
        default=1024,
        help="Batch size for projection checkpoint forward pass",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not args.clip_index.exists():
        print(f"Error: clip index not found: {args.clip_index}")
        return 1
    if args.projection_checkpoint is not None and not args.projection_checkpoint.exists():
        print(f"Error: projection checkpoint not found: {args.projection_checkpoint}")
        return 1

    k_values = _parse_k_values(args.k_values)
    layer = None if args.layer < 0 else args.layer
    flatten = args.flatten_input or layer is None

    index = ClipIndex.from_path(args.clip_index)
    vectors, records = _load_base_vectors(index=index, layer=layer, flatten=flatten)

    rng = np.random.default_rng(args.seed)
    query_indices = _sample_query_indices(len(records), max_queries=args.max_queries, rng=rng)

    include_clip_to_clip = args.protocol in {"clip_to_clip", "both"}
    include_clip_to_track = args.protocol in {"clip_to_track", "both"}

    systems: dict[str, dict[str, object]] = {}
    baseline_protocols = _evaluate_vectors(
        vectors=vectors,
        records=records,
        query_indices=query_indices,
        k_values=k_values,
        include_clip_to_clip=include_clip_to_clip,
        include_clip_to_track=include_clip_to_track,
    )
    systems["baseline"] = {
        "vector_dim": int(vectors.shape[1]),
        "protocols": baseline_protocols,
    }

    if args.projection_checkpoint is not None:
        projected_vectors, projection_meta = _project_vectors(
            vectors=vectors,
            checkpoint_path=args.projection_checkpoint,
            use_target_encoder=args.use_target_encoder,
            batch_size=args.projection_batch_size,
        )
        projection_protocols = _evaluate_vectors(
            vectors=projected_vectors,
            records=records,
            query_indices=query_indices,
            k_values=k_values,
            include_clip_to_clip=include_clip_to_clip,
            include_clip_to_track=include_clip_to_track,
        )
        systems["projection"] = {
            "vector_dim": int(projected_vectors.shape[1]),
            "metadata": projection_meta,
            "protocols": projection_protocols,
        }

    report = {
        "run_config": {
            "clip_index": str(args.clip_index),
            "k_values": k_values,
            "max_queries": int(args.max_queries),
            "seed": int(args.seed),
            "protocol": args.protocol,
            "layer": None if layer is None else int(layer),
            "flatten_input": bool(flatten),
            "projection_checkpoint": (
                str(args.projection_checkpoint) if args.projection_checkpoint is not None else None
            ),
            "use_target_encoder": bool(args.use_target_encoder),
        },
        "dataset_summary": {
            "num_clips": len(records),
            "num_tracks": len({record.track_id for record in records}),
            "num_query_clips": int(len(query_indices)),
        },
        "systems": systems,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote report: {args.report_json}")
    print(f"Systems evaluated: {', '.join(systems.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
