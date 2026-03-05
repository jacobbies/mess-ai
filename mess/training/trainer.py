"""Training loop for retrieval-augmented projection-head learning."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..datasets.clip_index import ClipRecord
from .config import RetrievalSSLConfig
from .index import FaissRetrievalIndex, should_refresh_index
from .losses import batch_similarity_stats, multi_positive_info_nce
from .mining import MiningParams, mine_batch_indices


class ProjectionHead(nn.Module):
    """Projection head over frozen input embeddings."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.network: nn.Module
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")

        if hidden_dim is None:
            self.network = nn.Linear(input_dim, output_dim)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.network(x)
        return torch.nn.functional.normalize(projected, dim=-1)


@dataclass
class TrainResult:
    """Serializable outputs from a training run."""

    input_dim: int
    output_dim: int
    used_device: str
    steps_completed: int
    skipped_steps: int
    final_index_version: int
    metrics: list[dict[str, float | int]]
    online_state_dict: dict[str, torch.Tensor]
    target_state_dict: dict[str, torch.Tensor]


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _embed_all(
    model: nn.Module,
    base_vectors: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, base_vectors.shape[0], batch_size):
            end = min(start + batch_size, base_vectors.shape[0])
            batch = base_vectors[start:end].to(device)
            outputs.append(model(batch).cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def _ema_update(online_model: nn.Module, target_model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(
            target_model.parameters(), online_model.parameters(), strict=True
        ):
            target_param.mul_(decay).add_(online_param, alpha=(1.0 - decay))


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def train_projection_head(
    base_vectors: np.ndarray,
    records: Sequence[ClipRecord],
    config: RetrievalSSLConfig,
) -> TrainResult:
    """Train projection head with retrieval-mined positives and negatives."""
    config.validate()

    vectors = np.asarray(base_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError(f"base_vectors must be non-empty 2D array, got {vectors.shape}")
    if vectors.shape[0] != len(records):
        raise ValueError(
            "records length must match base_vectors rows: "
            f"{len(records)} vs {vectors.shape[0]}"
        )

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    device = _resolve_device(config.device)
    input_dim = int(vectors.shape[1])

    online_model = ProjectionHead(
        input_dim=input_dim,
        output_dim=config.projection_dim,
        hidden_dim=config.hidden_dim,
    ).to(device)
    target_model = copy.deepcopy(online_model).to(device)
    target_model.eval()

    optimizer = torch.optim.AdamW(
        online_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    base_tensor = torch.from_numpy(vectors)
    train_indices = np.array(
        [idx for idx, row in enumerate(records) if row.split in set(config.train_splits)],
        dtype=np.int64,
    )
    if train_indices.size == 0:
        raise ValueError(f"No rows found for train_splits={config.train_splits}")

    mining_params = MiningParams(
        positives_per_query=config.positives_per_query,
        negatives_per_query=config.negatives_per_query,
        min_time_separation_sec=config.min_time_separation_sec,
        require_cross_recording_positive=config.require_cross_recording_positive,
        exclude_same_recording_negative=config.exclude_same_recording_negative,
    )

    target_vectors = _embed_all(
        model=target_model,
        base_vectors=base_tensor,
        device=device,
        batch_size=max(config.batch_size, 128),
    )
    retrieval_index = FaissRetrievalIndex.build(target_vectors)
    target_tensor = torch.from_numpy(target_vectors).to(device)

    metrics: list[dict[str, float | int]] = []
    skipped_steps = 0

    for step in range(1, config.num_steps + 1):
        if step > 1 and should_refresh_index(step, config.refresh_every):
            target_vectors = _embed_all(
                model=target_model,
                base_vectors=base_tensor,
                device=device,
                batch_size=max(config.batch_size, 128),
            )
            retrieval_index.rebuild(target_vectors)
            target_tensor = torch.from_numpy(target_vectors).to(device)

        replace = train_indices.size < config.batch_size
        batch_indices = rng.choice(
            train_indices,
            size=config.batch_size,
            replace=replace,
        ).astype(np.int64)

        _, neighbor_indices = retrieval_index.search(
            target_vectors[batch_indices],
            k=min(config.search_k, len(records)),
        )

        positive_indices, positive_mask_np, negative_indices = mine_batch_indices(
            records=records,
            query_indices=batch_indices,
            neighbor_indices=neighbor_indices,
            params=mining_params,
            rng=rng,
        )

        if not positive_mask_np.any():
            skipped_steps += 1
            continue

        query_input = base_tensor[batch_indices].to(device)
        positive_idx_t = torch.from_numpy(positive_indices).to(device)
        negative_idx_t = torch.from_numpy(negative_indices).to(device)
        positive_mask_t = torch.from_numpy(positive_mask_np).to(device)

        online_model.train()
        query_embeddings = online_model(query_input)
        positive_embeddings = target_tensor[positive_idx_t]
        negative_embeddings = target_tensor[negative_idx_t]

        loss = multi_positive_info_nce(
            query_embeddings=query_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            positive_mask=positive_mask_t,
            temperature=config.temperature,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        _ema_update(online_model=online_model, target_model=target_model, decay=config.ema_decay)

        stats = batch_similarity_stats(
            query_embeddings=query_embeddings.detach(),
            positive_embeddings=positive_embeddings.detach(),
            negative_embeddings=negative_embeddings.detach(),
            positive_mask=positive_mask_t,
        )
        metrics.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "pos_sim_mean": float(stats["pos_sim_mean"]),
                "neg_sim_mean": float(stats["neg_sim_mean"]),
                "valid_queries": int(positive_mask_np.any(axis=1).sum()),
                "index_version": retrieval_index.version,
            }
        )

    if not metrics:
        raise RuntimeError(
            "Training produced no valid steps; mining may be too strict for this dataset"
        )

    return TrainResult(
        input_dim=input_dim,
        output_dim=config.projection_dim,
        used_device=str(device),
        steps_completed=len(metrics),
        skipped_steps=skipped_steps,
        final_index_version=retrieval_index.version,
        metrics=metrics,
        online_state_dict=_cpu_state_dict(online_model),
        target_state_dict=_cpu_state_dict(target_model),
    )
