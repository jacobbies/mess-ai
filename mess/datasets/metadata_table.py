"""Dataset metadata table contract for canonical recording/work/track identity."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

REQUIRED_METADATA_COLUMNS = (
    "recording_id",
    "work_id",
    "track_id",
)


def _normalize_value(value: object) -> str:
    if value is None:
        return ""
    # NaN check without hard dependency on numpy/pandas scalar types.
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value).strip()


class DatasetMetadataTable:
    """In-memory metadata table keyed by canonical track_id."""

    def __init__(self, rows: Iterable[Mapping[str, object]]) -> None:
        normalized_rows: list[dict[str, str]] = []
        for idx, row in enumerate(rows):
            normalized: dict[str, str] = {
                str(key): _normalize_value(value) for key, value in row.items()
            }

            missing = [name for name in REQUIRED_METADATA_COLUMNS if name not in normalized]
            if missing:
                raise ValueError(
                    f"Metadata row {idx} missing required columns: {missing}"
                )

            empty = [name for name in REQUIRED_METADATA_COLUMNS if not normalized[name]]
            if empty:
                raise ValueError(
                    f"Metadata row {idx} has empty required values: {empty}"
                )

            normalized_rows.append(normalized)

        by_track: dict[str, dict[str, str]] = {}
        for row in normalized_rows:
            track_id = row["track_id"]
            if track_id in by_track:
                raise ValueError(f"Duplicate track_id values found in metadata table: {track_id}")
            by_track[track_id] = row

        self._rows = normalized_rows
        self._by_track = by_track

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, object]]) -> DatasetMetadataTable:
        """Build metadata table from row-like mappings."""
        return cls(rows)

    @classmethod
    def from_csv(cls, path: str | Path) -> DatasetMetadataTable:
        """Load metadata table from CSV."""
        table_path = Path(path)
        with table_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        return cls.from_rows(rows)

    @classmethod
    def from_parquet(cls, path: str | Path) -> DatasetMetadataTable:
        """Load metadata table from parquet if pandas is available."""
        try:
            import pandas as pd  # type: ignore[import-untyped]
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading parquet metadata requires pandas. "
                "Install optional dependency `pandas`."
            ) from exc

        frame = pd.read_parquet(Path(path))
        rows = frame.to_dict(orient="records")
        return cls.from_rows(rows)

    @classmethod
    def from_path(cls, path: str | Path) -> DatasetMetadataTable:
        """Load metadata table from CSV or parquet by file extension."""
        table_path = Path(path)
        suffix = table_path.suffix.lower()
        if suffix == ".csv":
            return cls.from_csv(table_path)
        if suffix in {".parquet", ".pq"}:
            return cls.from_parquet(table_path)
        raise ValueError(f"Unsupported metadata table format: {table_path.suffix}")

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, str]]:
        return iter(self.rows)

    @property
    def rows(self) -> list[dict[str, str]]:
        """Return metadata rows as a list copy."""
        return [dict(row) for row in self._rows]

    def get_row(self, track_id: str) -> dict[str, str] | None:
        """Return metadata row for track_id, if available."""
        row = self._by_track.get(track_id)
        if row is None:
            return None
        return dict(row)

    def recording_id_for_track(self, track_id: str) -> str | None:
        """Return recording_id for track_id, if available."""
        row = self._by_track.get(track_id)
        if row is None:
            return None
        return row["recording_id"]

    def work_id_for_track(self, track_id: str) -> str | None:
        """Return work_id for track_id, if available."""
        row = self._by_track.get(track_id)
        if row is None:
            return None
        return row["work_id"]

    def text_for_track(
        self,
        track_id: str,
        fields: Iterable[str] | None = None,
    ) -> str | None:
        """Return a normalized metadata text blob for lightweight keyword matching."""
        row = self._by_track.get(track_id)
        if row is None:
            return None

        selected_fields = list(fields) if fields is not None else sorted(row.keys())
        values = [
            row.get(field, "").strip()
            for field in selected_fields
            if row.get(field, "").strip()
        ]
        if not values:
            return ""
        return " ".join(values)

    def to_recording_map(self) -> dict[str, str]:
        """Return track_id -> recording_id mapping."""
        return {row["track_id"]: row["recording_id"] for row in self._rows}

    def to_work_map(self) -> dict[str, str]:
        """Return track_id -> work_id mapping."""
        return {row["track_id"]: row["work_id"] for row in self._rows}

    def to_csv(self, path: str | Path) -> None:
        """Write metadata table to CSV."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = list(REQUIRED_METADATA_COLUMNS)
        extra_fields = sorted(
            {
                key
                for row in self._rows
                for key in row
                if key not in REQUIRED_METADATA_COLUMNS
            }
        )
        fieldnames.extend(extra_fields)

        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
