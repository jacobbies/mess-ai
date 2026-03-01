"""General-purpose torch dataset adapter for clip-indexed records."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from torch.utils.data import Dataset

from .clip_index import ClipIndex, ClipRecord

Resolver = Callable[[str], Any]
MetaBuilder = Callable[[ClipRecord], dict[str, Any]]


def default_meta_builder(record: ClipRecord) -> dict[str, Any]:
    """Default metadata payload attached to each sample."""
    return {
        "dataset_id": record.dataset_id,
        "recording_id": record.recording_id,
        "track_id": record.track_id,
        "segment_idx": record.segment_idx,
        "start_sec": record.start_sec,
        "end_sec": record.end_sec,
        "split": record.split,
        "embedding_path": record.embedding_path,
    }


class GeneralTorchDataset(Dataset[dict[str, Any]]):
    """
    Generic torch dataset for clip-indexed workflows.

    Field values are resolved by callables keyed by output field name.
    """

    def __init__(
        self,
        index: ClipIndex,
        field_resolvers: Mapping[str, Resolver] | None = None,
        meta_builder: MetaBuilder = default_meta_builder,
    ) -> None:
        self._records = index.records
        self._field_resolvers = dict(field_resolvers or {})
        self._meta_builder = meta_builder

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self._records[idx]
        sample: dict[str, Any] = {
            "clip_id": record.clip_id,
            "meta": self._meta_builder(record),
        }
        for field_name, resolver in self._field_resolvers.items():
            sample[field_name] = resolver(record.clip_id)
        return sample
