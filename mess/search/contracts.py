"""Shared clip metadata/search contracts for retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass

from ..datasets.clip_index import ClipRecord


@dataclass(frozen=True)
class ClipLocation:
    """Index location and time span for one segment embedding."""

    track_id: str
    segment_idx: int
    start_time: float
    end_time: float


@dataclass(frozen=True)
class ClipMetadata:
    """Clip identity and provenance aligned with ClipIndex rows."""

    clip_id: str
    dataset_id: str
    recording_id: str
    work_id: str
    track_id: str
    segment_idx: int
    start_time: float
    end_time: float
    split: str

    @classmethod
    def from_clip_record(cls, record: ClipRecord) -> ClipMetadata:
        return cls(
            clip_id=record.clip_id,
            dataset_id=record.dataset_id,
            recording_id=record.recording_id,
            work_id=record.work_id,
            track_id=record.track_id,
            segment_idx=record.segment_idx,
            start_time=record.start_sec,
            end_time=record.end_sec,
            split=record.split,
        )

    @classmethod
    def from_clip_location(
        cls,
        location: ClipLocation,
        *,
        dataset_id: str = "",
        clip_id: str = "",
        recording_id: str = "",
        work_id: str = "",
        split: str = "",
    ) -> ClipMetadata:
        return cls(
            clip_id=clip_id,
            dataset_id=dataset_id,
            recording_id=recording_id,
            work_id=work_id,
            track_id=location.track_id,
            segment_idx=location.segment_idx,
            start_time=location.start_time,
            end_time=location.end_time,
            split=split,
        )

    def to_location(self) -> ClipLocation:
        return ClipLocation(
            track_id=self.track_id,
            segment_idx=self.segment_idx,
            start_time=self.start_time,
            end_time=self.end_time,
        )


@dataclass(frozen=True)
class ClipSearchResult(ClipMetadata):
    """Search result for clip-level retrieval with preserved metadata."""

    similarity: float

    @classmethod
    def from_clip_metadata(
        cls,
        clip: ClipMetadata,
        *,
        similarity: float,
    ) -> ClipSearchResult:
        return cls(
            clip_id=clip.clip_id,
            dataset_id=clip.dataset_id,
            recording_id=clip.recording_id,
            work_id=clip.work_id,
            track_id=clip.track_id,
            segment_idx=clip.segment_idx,
            start_time=clip.start_time,
            end_time=clip.end_time,
            split=clip.split,
            similarity=similarity,
        )
