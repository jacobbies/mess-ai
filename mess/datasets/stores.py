"""Storage interfaces and local store implementations for clip-indexed data."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

try:
    from torchcodec.decoders import AudioDecoder  # type: ignore[import-untyped]
except Exception:
    AudioDecoder = None  # type: ignore[assignment,misc]

from .clip_index import ClipIndex, ClipRecord


class EmbeddingStore(Protocol):
    """Protocol for clip embedding retrieval."""

    def get(self, clip_id: str) -> np.ndarray:
        """Return embedding array for one clip."""
        ...


class TargetStore(Protocol):
    """Protocol for clip target retrieval."""

    def get(self, clip_id: str) -> dict[str, float] | None:
        """Return optional target dictionary for one clip."""
        ...


class FeatureStore(Protocol):
    """Protocol for clip engineered feature retrieval."""

    def get(self, clip_id: str) -> dict[str, float] | None:
        """Return optional feature dictionary for one clip."""
        ...


class AudioStore(Protocol):
    """Protocol for clip audio retrieval."""

    def get(self, clip_id: str) -> np.ndarray:
        """Return waveform slice for one clip."""
        ...


class _IndexBackedStore:
    """Base helper for stores backed by clip index records."""

    def __init__(self, index: ClipIndex) -> None:
        self.index = index
        self._by_clip_id = {record.clip_id: record for record in index}

    def _resolve_record(self, clip_id: str) -> ClipRecord:
        try:
            return self._by_clip_id[clip_id]
        except KeyError as exc:
            raise KeyError(f"Unknown clip_id: {clip_id}") from exc


AudioPathResolver = Callable[[ClipRecord], str | Path]


class TorchCodecAudioStore(_IndexBackedStore):
    """Decode clip windows with TorchCodec using clip index time bounds."""

    def __init__(
        self,
        index: ClipIndex,
        audio_path_resolver: AudioPathResolver | None = None,
        audio_root: str | Path | None = None,
        audio_extensions: Sequence[str] = (".wav", ".flac", ".mp3", ".ogg", ".m4a"),
        sample_rate: int = 24000,
        num_channels: int = 1,
    ) -> None:
        super().__init__(index)
        if AudioDecoder is None:
            raise RuntimeError(
                "TorchCodec AudioDecoder is unavailable. Install FFmpeg shared libraries "
                "compatible with torchcodec in this runtime."
            )
        if audio_path_resolver is None and audio_root is None:
            raise ValueError("Provide either audio_path_resolver or audio_root")

        self.audio_path_resolver = audio_path_resolver
        self.audio_root = Path(audio_root) if audio_root is not None else None
        self.audio_extensions = tuple(
            suffix if suffix.startswith(".") else f".{suffix}"
            for suffix in audio_extensions
        )
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        self._decoder_cache: dict[str, AudioDecoder] = {}
        self._audio_path_cache: dict[str, Path] = {}

    def _record_cache_key(self, record: ClipRecord) -> str:
        return f"{record.dataset_id}:{record.track_id}"

    def _resolve_audio_path(self, record: ClipRecord) -> Path:
        if self.audio_path_resolver is not None:
            candidate = Path(self.audio_path_resolver(record))
            if not candidate.exists():
                raise FileNotFoundError(f"Audio file not found: {candidate}")
            return candidate

        assert self.audio_root is not None
        key = self._record_cache_key(record)
        cached = self._audio_path_cache.get(key)
        if cached is not None:
            return cached

        matches = [
            path
            for path in self.audio_root.rglob(f"{record.track_id}.*")
            if path.is_file() and path.suffix.lower() in self.audio_extensions
        ]
        if not matches:
            raise FileNotFoundError(
                f"No audio file found for track_id={record.track_id} under {self.audio_root}"
            )
        if len(matches) > 1:
            sample = ", ".join(str(path) for path in sorted(matches)[:3])
            raise ValueError(
                f"Multiple audio files found for track_id={record.track_id}: {sample}"
            )

        resolved = matches[0]
        self._audio_path_cache[key] = resolved
        return resolved

    def _decoder_for_path(self, audio_path: Path) -> AudioDecoder:
        key = str(audio_path)
        decoder = self._decoder_cache.get(key)
        if decoder is None:
            assert AudioDecoder is not None
            decoder = AudioDecoder(
                str(audio_path),
                sample_rate=self.sample_rate,
                num_channels=self.num_channels,
            )
            self._decoder_cache[key] = decoder
        return decoder

    def get(self, clip_id: str) -> np.ndarray:
        record = self._resolve_record(clip_id)
        if record.end_sec <= record.start_sec:
            raise ValueError(
                f"Invalid clip bounds for {clip_id}: start={record.start_sec}, end={record.end_sec}"
            )

        audio_path = self._resolve_audio_path(record)
        decoder = self._decoder_for_path(audio_path)
        samples = decoder.get_samples_played_in_range(record.start_sec, record.end_sec)
        waveform = samples.data

        if waveform.ndim != 2:
            raise ValueError(
                f"Expected decoded waveform shape [channels, samples], got {waveform.shape}"
            )
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return np.asarray(waveform.squeeze(0).cpu().numpy(), dtype=np.float32)


class NpySegmentEmbeddingStore(_IndexBackedStore):
    """Load clip embeddings from per-track numpy files referenced by clip index rows."""

    def __init__(
        self,
        index: ClipIndex,
        layer: int | None = None,
        flatten: bool = False,
        mmap: bool = True,
    ) -> None:
        super().__init__(index)
        self.layer = layer
        self.flatten = flatten
        self.mmap = mmap
        self._file_cache: dict[str, np.ndarray] = {}

    def _load_file(self, path: str) -> np.ndarray:
        if path not in self._file_cache:
            mmap_mode: Literal["r"] | None = "r" if self.mmap else None
            self._file_cache[path] = np.load(path, mmap_mode=mmap_mode)
        return self._file_cache[path]

    def _vector_from_row(self, row: ClipRecord) -> np.ndarray:
        matrix = self._load_file(row.embedding_path)

        if matrix.ndim == 2 and matrix.shape[0] == 13:
            if row.segment_idx != 0:
                raise IndexError(
                    f"clip_id={row.clip_id} has segment_idx={row.segment_idx} but embedding "
                    "file contains one matrix only."
                )
            segment = matrix
        elif matrix.ndim == 3 and matrix.shape[1] == 13:
            segment = matrix[row.segment_idx]
        elif matrix.ndim == 4 and matrix.shape[1] == 13:
            segment = matrix[row.segment_idx].mean(axis=1)
        else:
            raise ValueError(
                f"Unsupported embedding shape {matrix.shape} in {row.embedding_path}"
            )

        if self.layer is not None:
            return np.asarray(segment[self.layer], dtype=np.float32)

        segment_array = np.asarray(segment, dtype=np.float32)
        if self.flatten:
            return segment_array.reshape(-1)
        return segment_array

    def get(self, clip_id: str) -> np.ndarray:
        row = self._resolve_record(clip_id)
        return self._vector_from_row(row)


class NPZSegmentTargetStore(_IndexBackedStore):
    """Load clip-level scalar targets from per-track segment target npz files."""

    def __init__(self, index: ClipIndex, targets_dir: str | Path) -> None:
        super().__init__(index)
        self.targets_dir = Path(targets_dir)
        self._file_cache: dict[str, Mapping[str, object] | None] = {}

    def _target_path_for_track(self, track_id: str) -> Path:
        return self.targets_dir / f"{track_id}_segment_targets.npz"

    def _load_target_file(self, track_id: str) -> Mapping[str, object] | None:
        if track_id in self._file_cache:
            return self._file_cache[track_id]

        path = self._target_path_for_track(track_id)
        if not path.exists():
            self._file_cache[track_id] = None
            return None

        data = np.load(path, allow_pickle=True)
        unpacked: dict[str, object] = {}
        for key in data.files:
            unpacked[key] = data[key].item()
        self._file_cache[track_id] = unpacked
        return unpacked

    def get(self, clip_id: str) -> dict[str, float] | None:
        row = self._resolve_record(clip_id)
        nested = self._load_target_file(row.track_id)
        if nested is None:
            return None

        values: dict[str, float] = {}
        for category, payload in nested.items():
            if not isinstance(payload, dict):
                continue
            for field_name, arr in payload.items():
                if not isinstance(arr, np.ndarray):
                    continue
                if arr.ndim != 1 or row.segment_idx >= len(arr):
                    continue
                scalar = arr[row.segment_idx]
                if np.isnan(scalar):
                    continue
                values[f"{category}.{field_name}"] = float(scalar)

        return values if values else None
