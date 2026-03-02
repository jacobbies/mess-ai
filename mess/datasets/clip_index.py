"""Clip-level index contract for downstream retrieval and training workflows."""

from __future__ import annotations

import csv
import random
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = (
    "clip_id",
    "dataset_id",
    "recording_id",
    "track_id",
    "segment_idx",
    "start_sec",
    "end_sec",
    "split",
    "embedding_path",
)


@dataclass(frozen=True)
class ClipRecord:
    """One clip row in the clip index."""

    clip_id: str
    dataset_id: str
    recording_id: str
    track_id: str
    segment_idx: int
    start_sec: float
    end_sec: float
    split: str
    embedding_path: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> ClipRecord:
        """Build a typed clip record from a CSV/parquet-like row."""
        missing = [name for name in REQUIRED_COLUMNS if name not in row]
        if missing:
            raise ValueError(f"Clip row missing required columns: {missing}")
        return cls(
            clip_id=str(row["clip_id"]),
            dataset_id=str(row["dataset_id"]),
            recording_id=str(row["recording_id"]),
            track_id=str(row["track_id"]),
            segment_idx=int(row["segment_idx"]),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            split=str(row["split"]),
            embedding_path=str(row["embedding_path"]),
        )


class ClipIndex:
    """In-memory clip index with filtering and recording-level split helpers."""

    def __init__(self, records: Iterable[ClipRecord]):
        ordered = sorted(records, key=lambda r: r.clip_id)
        clip_ids = [record.clip_id for record in ordered]
        duplicates = {clip_id for clip_id, count in Counter(clip_ids).items() if count > 1}
        if duplicates:
            dupes = sorted(duplicates)
            raise ValueError(f"Duplicate clip_id values found: {dupes[:5]}")
        self._records = ordered

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, str]]) -> ClipIndex:
        """Load from row dictionaries (e.g., csv.DictReader output)."""
        return cls(ClipRecord.from_row(row) for row in rows)

    @classmethod
    def from_csv(cls, path: str | Path) -> ClipIndex:
        """Load clip index from CSV."""
        index_path = Path(path)
        with index_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        return cls.from_rows(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> ClipIndex:
        """Load clip index from parquet if pandas is available."""
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading parquet clip index requires pandas. "
                "Install optional dependency `pandas`."
            ) from exc

        frame = pd.read_parquet(Path(path))
        rows = frame.to_dict(orient="records")
        normalized_rows = [{str(k): str(v) for k, v in row.items()} for row in rows]
        return cls.from_rows(normalized_rows)

    @classmethod
    def from_path(cls, path: str | Path) -> ClipIndex:
        """Load from CSV or parquet by file extension."""
        index_path = Path(path)
        suffix = index_path.suffix.lower()
        if suffix == ".csv":
            return cls.from_csv(index_path)
        if suffix in {".parquet", ".pq"}:
            return cls.from_parquet(index_path)
        raise ValueError(f"Unsupported clip index format: {index_path.suffix}")

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[ClipRecord]:
        return iter(self._records)

    def __getitem__(self, idx: int) -> ClipRecord:
        return self._records[idx]

    @property
    def records(self) -> list[ClipRecord]:
        """Return records as a list copy."""
        return list(self._records)

    def filter(
        self,
        splits: set[str] | None = None,
        dataset_ids: set[str] | None = None,
        recording_ids: set[str] | None = None,
        track_ids: set[str] | None = None,
        clip_ids: set[str] | None = None,
    ) -> ClipIndex:
        """Filter records by one or more dimensions."""
        filtered = []
        for record in self._records:
            if splits is not None and record.split not in splits:
                continue
            if dataset_ids is not None and record.dataset_id not in dataset_ids:
                continue
            if recording_ids is not None and record.recording_id not in recording_ids:
                continue
            if track_ids is not None and record.track_id not in track_ids:
                continue
            if clip_ids is not None and record.clip_id not in clip_ids:
                continue
            filtered.append(record)
        return ClipIndex(filtered)

    def assign_recording_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> ClipIndex:
        """Assign train/val/test split by recording_id, never by clip."""
        if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("Split ratios must be >= 0 and train_ratio > 0")
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Split ratios must sum to 1.0")

        recordings = sorted({record.recording_id for record in self._records})
        if not recordings:
            return ClipIndex([])

        rng = random.Random(seed)
        shuffled = recordings[:]
        rng.shuffle(shuffled)

        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_set = set(shuffled[:n_train])
        val_set = set(shuffled[n_train:n_train + n_val])
        test_set = set(shuffled[n_train + n_val :])

        updated = []
        for record in self._records:
            if record.recording_id in train_set:
                split = "train"
            elif record.recording_id in val_set:
                split = "val"
            elif record.recording_id in test_set:
                split = "test"
            else:
                raise RuntimeError(f"Unassigned recording_id: {record.recording_id}")
            updated.append(
                ClipRecord(
                    clip_id=record.clip_id,
                    dataset_id=record.dataset_id,
                    recording_id=record.recording_id,
                    track_id=record.track_id,
                    segment_idx=record.segment_idx,
                    start_sec=record.start_sec,
                    end_sec=record.end_sec,
                    split=split,
                    embedding_path=record.embedding_path,
                )
            )
        return ClipIndex(updated)

    def to_csv(self, path: str | Path) -> None:
        """Write clip index to CSV."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
            writer.writeheader()
            for record in self._records:
                writer.writerow(
                    {
                        "clip_id": record.clip_id,
                        "dataset_id": record.dataset_id,
                        "recording_id": record.recording_id,
                        "track_id": record.track_id,
                        "segment_idx": str(record.segment_idx),
                        "start_sec": f"{record.start_sec:.6f}",
                        "end_sec": f"{record.end_sec:.6f}",
                        "split": record.split,
                        "embedding_path": record.embedding_path,
                    }
                )
