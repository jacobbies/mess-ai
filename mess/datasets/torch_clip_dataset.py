"""Clip-first torch dataset for downstream expressive retrieval experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .clip_index import ClipIndex
from .stores import AudioStore, EmbeddingStore, FeatureStore, TargetStore
from .torch_dataset import default_meta_builder


class ClipDataset(Dataset[dict[str, Any]]):
    """Torch dataset that returns clip embeddings with optional targets/features."""

    def __init__(
        self,
        index: ClipIndex,
        embedding_store: EmbeddingStore,
        feature_store: FeatureStore | None = None,
        target_store: TargetStore | None = None,
        audio_store: AudioStore | None = None,
        target_keys: Sequence[str] | None = None,
    ) -> None:
        self._records = index.records
        self._embedding_store = embedding_store
        self._feature_store = feature_store
        self._target_store = target_store
        self._audio_store = audio_store
        self._target_keys = list(target_keys or [])
        self._target_key_to_idx = {name: i for i, name in enumerate(self._target_keys)}

    def __len__(self) -> int:
        return len(self._records)

    def _to_tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.float()
        return torch.from_numpy(np.asarray(value, dtype=np.float32))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._records[idx]
        clip_id = record.clip_id

        embedding = self._to_tensor(self._embedding_store.get(clip_id))
        features = self._feature_store.get(clip_id) if self._feature_store else None
        targets = self._target_store.get(clip_id) if self._target_store else None
        audio = self._audio_store.get(clip_id) if self._audio_store else None

        sample: dict[str, Any] = {
            "clip_id": clip_id,
            "embedding": embedding,
            "features": features,
            "targets": targets,
            "has_targets": targets is not None and len(targets) > 0,
            "meta": default_meta_builder(record),
        }
        if audio is not None:
            sample["audio"] = self._to_tensor(audio)

        if self._target_keys:
            values = torch.full((len(self._target_keys),), float("nan"), dtype=torch.float32)
            mask = torch.zeros(len(self._target_keys), dtype=torch.bool)
            if targets is not None:
                for key, raw_value in targets.items():
                    key_idx = self._target_key_to_idx.get(key)
                    if key_idx is None:
                        continue
                    values[key_idx] = float(raw_value)
                    mask[key_idx] = True
            sample["target_values"] = values
            sample["target_mask"] = mask

        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Batch clips with optional target tensors and heterogeneous metadata."""
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        collated: dict[str, Any] = {
            "clip_id": [item["clip_id"] for item in batch],
            "embedding": torch.stack([item["embedding"] for item in batch]),
            "meta": [item["meta"] for item in batch],
            "has_targets": torch.tensor([bool(item["has_targets"]) for item in batch]),
            "targets": [item["targets"] for item in batch],
            "features": [item["features"] for item in batch],
        }

        if "audio" in batch[0]:
            collated["audio"] = torch.stack([item["audio"] for item in batch])

        if "target_values" in batch[0]:
            collated["target_values"] = torch.stack([item["target_values"] for item in batch])
            collated["target_mask"] = torch.stack([item["target_mask"] for item in batch])

        return collated
