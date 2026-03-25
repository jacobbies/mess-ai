"""Production-oriented FAISS index artifacts for track and clip retrieval."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import re
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .._runtime import configure_faiss_runtime
from ..datasets.clip_index import ClipIndex, ClipRecord
from ..datasets.stores import NpySegmentEmbeddingStore
from .contracts import ClipLocation, ClipMetadata
from .search import load_features, load_segment_features

IndexKind = Literal["track", "clip"]
IndexType = Literal["flatip", "ivfflat", "factory"]
SCHEMA_VERSION = 2
REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_name",
    "artifact_version_id",
    "kind",
    "index_type",
    "metric",
    "dataset",
    "feature_source_dir",
    "layer",
    "dimension",
    "ntotal",
    "created_at_utc",
    "model_name",
}
CHECKSUM_FILE = "checksums.json"
INDEX_FILE = "index.faiss"
MANIFEST_FILE = "manifest.json"
TRACK_NAMES_FILE = "track_names.json"
CLIP_RECORDS_FILE = "clip_records.json.gz"
CLIP_LOCATIONS_FILE = "clip_locations.json.gz"
VECTORS_FILE = "vectors.npy"
_STAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")


class ArtifactValidationError(ValueError):
    """Raised when a persisted artifact is malformed or fails integrity checks."""


@dataclass(frozen=True)
class ArtifactManifest:
    """Serialized metadata for one persisted FAISS artifact."""

    schema_version: int
    artifact_name: str
    artifact_version_id: str
    kind: IndexKind
    index_type: IndexType
    metric: str
    dataset: str
    feature_source_dir: str
    layer: int | None
    dimension: int
    ntotal: int
    created_at_utc: str
    model_name: str
    nlist: int | None = None
    default_nprobe: int | None = None
    factory_string: str | None = None
    train_size: int | None = None


@dataclass(frozen=True)
class FAISSArtifact:
    """In-memory bundle of a FAISS index and its lookup metadata."""

    index: Any
    manifest: ArtifactManifest
    track_names: list[str] | None = None
    clip_records: list[ClipMetadata] | None = None
    vectors: np.ndarray | None = None

    @property
    def clip_locations(self) -> list[ClipLocation] | None:
        if self.clip_records is None:
            return None
        return [record.to_location() for record in self.clip_records]

    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
        nprobe: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run cosine-similarity search on the artifact's FAISS index."""
        if k <= 0:
            raise ValueError("k must be > 0")

        faiss = _require_faiss()
        _set_index_nprobe(self.index, nprobe)
        query = np.array(query_vectors, dtype=np.float32, copy=True)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.manifest.dimension:
            raise ValueError(
                "Query dimension mismatch: "
                f"expected {self.manifest.dimension}, got {query.shape[1]}"
            )

        faiss.normalize_L2(query)
        distances, indices = self.index.search(query, k)
        return np.asarray(distances), np.asarray(indices)


def _require_faiss() -> Any:
    try:
        return configure_faiss_runtime(importlib.import_module("faiss"))
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "faiss is required for index build/load. Install it with `pip install faiss-cpu`."
        ) from exc


def _require_boto3() -> Any:
    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "boto3 is required for S3 publishing. Install it with `pip install boto3`."
        ) from exc


def _now_utc_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _now_utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _new_artifact_version_id() -> str:
    return f"vid-{uuid.uuid4().hex}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _md5_file(path: Path) -> str:
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _size_file(path: Path) -> int:
    return path.stat().st_size


def _build_checksums(paths: list[Path]) -> dict[str, dict[str, Any]]:
    checksums: dict[str, dict[str, Any]] = {}
    for path in paths:
        checksums[path.name] = {
            "sha256": _sha256_file(path),
            "md5": _md5_file(path),
            "size": _size_file(path),
        }
    return checksums


def _validate_manifest_payload(payload: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_MANIFEST_FIELDS - set(payload))
    if missing:
        raise ArtifactValidationError(
            f"Manifest missing required fields: {', '.join(missing)}"
        )

    schema_version = payload["schema_version"]
    if schema_version != SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"Unsupported schema_version={schema_version}. Expected {SCHEMA_VERSION}."
        )

    artifact_version_id = payload["artifact_version_id"]
    if not isinstance(artifact_version_id, str) or not artifact_version_id.startswith("vid-"):
        raise ArtifactValidationError("Manifest has invalid artifact_version_id format.")

    index_type = payload["index_type"]
    factory_string = payload.get("factory_string")
    if index_type == "factory":
        if not isinstance(factory_string, str) or not factory_string.strip():
            raise ArtifactValidationError(
                "Factory index manifest requires non-empty factory_string."
            )
    default_nprobe = payload.get("default_nprobe")
    if default_nprobe is not None and (not isinstance(default_nprobe, int) or default_nprobe <= 0):
        raise ArtifactValidationError("Manifest default_nprobe must be a positive integer.")


def _validate_checksums(root: Path, checksums: dict[str, dict[str, Any]]) -> None:
    for name, info in checksums.items():
        path = root / name
        if not path.exists():
            raise ArtifactValidationError(f"Missing file referenced by checksums: {name}")

        expected_size = int(info["size"])
        actual_size = _size_file(path)
        if actual_size != expected_size:
            raise ArtifactValidationError(
                f"File size mismatch for {name}: expected {expected_size}, got {actual_size}"
            )

        expected_sha = str(info["sha256"])
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise ArtifactValidationError(
                f"SHA256 mismatch for {name}: expected {expected_sha}, got {actual_sha}"
            )


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    faiss = _require_faiss()
    arr: NDArray[np.float32] = np.asarray(vectors, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D vectors, got shape {arr.shape}")
    faiss.normalize_L2(arr)
    return cast(np.ndarray, arr)


def _build_faiss_index(
    vectors: np.ndarray,
    index_type: IndexType = "flatip",
    nlist: int = 256,
    factory_string: str | None = None,
    nprobe: int | None = None,
) -> Any:
    return _build_faiss_index_impl(
        _normalize_vectors(vectors),
        index_type=index_type,
        nlist=nlist,
        factory_string=factory_string,
        nprobe=nprobe,
    )


def _build_faiss_index_impl(
    normalized_vectors: np.ndarray,
    *,
    index_type: IndexType,
    nlist: int,
    factory_string: str | None,
    nprobe: int | None,
) -> Any:
    faiss = _require_faiss()
    arr = np.asarray(normalized_vectors, dtype=np.float32)
    dim = arr.shape[1]

    if index_type == "flatip":
        if factory_string is not None:
            raise ValueError("factory_string is only valid when index_type='factory'")
        if nprobe is not None:
            raise ValueError("nprobe is only valid for ivfflat indexes")
        index = faiss.IndexFlatIP(dim)
        index.add(arr)
        return index

    if index_type == "ivfflat":
        if factory_string is not None:
            raise ValueError("factory_string is only valid when index_type='factory'")
        if nlist <= 0:
            raise ValueError("nlist must be > 0 for ivfflat")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
        index.train(arr)
        index.add(arr)
        _set_index_nprobe(index, nprobe)
        return index

    if index_type == "factory":
        if nprobe is not None:
            raise ValueError("nprobe is only valid for ivfflat indexes")
        if not factory_string or not factory_string.strip():
            raise ValueError("factory_string is required when index_type='factory'")
        index = faiss.index_factory(dim, factory_string, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            index.train(arr)
        index.add(arr)
        return index

    raise ValueError(f"Unsupported index_type '{index_type}'")


def _set_index_nprobe(index: Any, nprobe: int | None) -> None:
    if nprobe is None:
        return
    if nprobe <= 0:
        raise ValueError("nprobe must be > 0")
    if not hasattr(index, "nprobe"):
        raise ValueError("nprobe is only supported for IVF indexes")
    index.nprobe = int(nprobe)


def _serialize_clip_locations(locations: list[ClipLocation]) -> list[dict[str, Any]]:
    return [asdict(loc) for loc in locations]


def _deserialize_clip_locations(rows: list[dict[str, Any]]) -> list[ClipLocation]:
    return [ClipLocation(**row) for row in rows]


def _serialize_clip_records(records: Sequence[ClipMetadata]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def _deserialize_clip_records(rows: list[dict[str, Any]]) -> list[ClipMetadata]:
    return [ClipMetadata(**row) for row in rows]


def _coerce_clip_records(
    records: Sequence[ClipMetadata | ClipRecord],
) -> list[ClipMetadata]:
    coerced: list[ClipMetadata] = []
    for record in records:
        if isinstance(record, ClipMetadata):
            coerced.append(record)
        else:
            coerced.append(ClipMetadata.from_clip_record(record))
    return coerced


def _resolve_clip_index(index: ClipIndex | str | Path) -> ClipIndex:
    if isinstance(index, ClipIndex):
        return index
    return ClipIndex.from_path(index)


def _resolve_dataset_from_records(records: Sequence[ClipMetadata]) -> str:
    unique_dataset_ids: set[str] = {
        record.dataset_id for record in records if record.dataset_id
    }
    dataset_ids: list[str] = sorted(unique_dataset_ids)
    if not dataset_ids:
        raise ValueError("Cannot infer dataset from clip records without dataset_id")
    if len(dataset_ids) > 1:
        raise ValueError(
            "Multiple dataset IDs found in clip records. Provide dataset explicitly."
        )
    return cast(str, dataset_ids[0])


def _validate_vectors_payload(manifest: ArtifactManifest, vectors: np.ndarray) -> None:
    if vectors.ndim != 2:
        raise ArtifactValidationError(f"Vectors payload must be 2D, got {vectors.shape}")
    if vectors.shape[0] != manifest.ntotal:
        raise ArtifactValidationError(
            f"Vectors ntotal mismatch: manifest {manifest.ntotal}, vectors {vectors.shape[0]}"
        )
    if vectors.shape[1] != manifest.dimension:
        raise ArtifactValidationError(
            f"Vectors dimension mismatch: manifest {manifest.dimension}, vectors {vectors.shape[1]}"
        )


def _artifact_dir(root: str | Path, artifact_name: str, created_stamp: str) -> Path:
    return Path(root) / artifact_name / created_stamp


def _upload_and_validate_file(
    s3: Any,
    *,
    bucket: str,
    key: str,
    local_path: Path,
    expected_md5: str,
    expected_size: int,
) -> dict[str, str]:
    s3.upload_file(str(local_path), bucket, key)
    head = s3.head_object(Bucket=bucket, Key=key)

    content_length = int(head.get("ContentLength", -1))
    if content_length != expected_size:
        raise ArtifactValidationError(
            f"S3 content length mismatch for {key}: expected {expected_size}, got {content_length}"
        )

    etag = str(head.get("ETag", "")).strip('"')
    if etag and "-" not in etag and etag != expected_md5:
        raise ArtifactValidationError(
            f"S3 ETag mismatch for {key}: expected md5 {expected_md5}, got {etag}"
        )

    result: dict[str, str] = {}
    if head.get("VersionId"):
        result["version_id"] = str(head["VersionId"])
    if etag:
        result["etag"] = etag
    return result


def build_track_artifact(
    *,
    dataset: str,
    features_dir: str | Path,
    artifact_name: str = "track_index",
    layer: int | None = None,
    index_type: IndexType = "flatip",
    model_name: str = "m-a-p/MERT-v1-95M",
    nlist: int = 256,
    factory_string: str | None = None,
    nprobe: int | None = None,
) -> FAISSArtifact:
    """Build track-level FAISS artifact from aggregated features."""
    vectors, track_names = load_features(str(features_dir), layer=layer)
    index = _build_faiss_index(
        vectors,
        index_type=index_type,
        nlist=nlist,
        factory_string=factory_string,
        nprobe=nprobe,
    )

    manifest = ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        artifact_name=artifact_name,
        artifact_version_id=_new_artifact_version_id(),
        kind="track",
        index_type=index_type,
        metric="cosine_ip",
        dataset=dataset,
        feature_source_dir=str(features_dir),
        layer=layer,
        dimension=int(vectors.shape[1]),
        ntotal=int(vectors.shape[0]),
        created_at_utc=_now_utc_iso(),
        model_name=model_name,
        nlist=nlist if index_type == "ivfflat" else None,
        default_nprobe=nprobe if index_type == "ivfflat" else None,
        factory_string=factory_string if index_type == "factory" else None,
        train_size=int(vectors.shape[0]) if index_type in {"ivfflat", "factory"} else None,
    )
    return FAISSArtifact(index=index, manifest=manifest, track_names=track_names)


def build_clip_artifact(
    *,
    dataset: str | None = None,
    features_dir: str | Path | None = None,
    clip_index: ClipIndex | str | Path | None = None,
    artifact_name: str = "clip_index",
    layer: int | None = None,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
    index_type: IndexType = "flatip",
    model_name: str = "m-a-p/MERT-v1-95M",
    nlist: int = 1024,
    factory_string: str | None = None,
    nprobe: int | None = None,
) -> FAISSArtifact:
    """Build clip-level FAISS artifact from ClipIndex rows or legacy segment features."""
    if clip_index is not None:
        resolved_index = _resolve_clip_index(clip_index)
        source_dir = str(clip_index) if not isinstance(clip_index, ClipIndex) else "<clip-index>"
        return build_clip_artifact_from_records(
            records=resolved_index.records,
            dataset=dataset,
            artifact_name=artifact_name,
            layer=layer,
            index_type=index_type,
            model_name=model_name,
            nlist=nlist,
            feature_source_dir=source_dir,
            factory_string=factory_string,
            nprobe=nprobe,
        )

    if features_dir is None:
        raise ValueError("Provide either clip_index or features_dir for clip artifact build")
    if dataset is None:
        raise ValueError("dataset is required when building clip artifacts from features_dir")

    vectors, clip_locations = load_segment_features(
        str(features_dir),
        layer=layer,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
    )
    return build_clip_artifact_from_vectors(
        dataset=dataset,
        vectors=vectors,
        clip_locations=clip_locations,
        artifact_name=artifact_name,
        layer=layer,
        index_type=index_type,
        model_name=model_name,
        nlist=nlist,
        feature_source_dir=str(features_dir),
        factory_string=factory_string,
        nprobe=nprobe,
    )


def build_clip_artifact_from_records(
    *,
    records: Sequence[ClipRecord],
    dataset: str | None = None,
    artifact_name: str = "clip_index",
    layer: int | None = None,
    index_type: IndexType = "flatip",
    model_name: str = "m-a-p/MERT-v1-95M",
    nlist: int = 1024,
    feature_source_dir: str = "<clip-index>",
    factory_string: str | None = None,
    nprobe: int | None = None,
) -> FAISSArtifact:
    """Build clip artifact vectors and metadata directly from ClipIndex records."""
    if not records:
        raise ValueError("records must be non-empty for clip artifact build")

    index = ClipIndex(records)
    ordered_records = index.records
    store = NpySegmentEmbeddingStore(index, layer=layer, flatten=layer is None)
    vectors = np.vstack(
        [store.get(record.clip_id) for record in ordered_records]
    ).astype(np.float32)
    clip_records = [ClipMetadata.from_clip_record(record) for record in ordered_records]
    resolved_dataset = dataset or _resolve_dataset_from_records(clip_records)

    return build_clip_artifact_from_vectors(
        dataset=resolved_dataset,
        vectors=vectors,
        clip_records=clip_records,
        artifact_name=artifact_name,
        layer=layer,
        index_type=index_type,
        model_name=model_name,
        nlist=nlist,
        feature_source_dir=feature_source_dir,
        factory_string=factory_string,
        nprobe=nprobe,
    )


def build_clip_artifact_from_vectors(
    *,
    dataset: str,
    vectors: np.ndarray,
    clip_records: Sequence[ClipMetadata | ClipRecord] | None = None,
    clip_locations: list[ClipLocation] | None = None,
    artifact_name: str = "clip_index",
    layer: int | None = None,
    index_type: IndexType = "flatip",
    model_name: str = "m-a-p/MERT-v1-95M",
    nlist: int = 1024,
    feature_source_dir: str = "<in-memory>",
    factory_string: str | None = None,
    nprobe: int | None = None,
) -> FAISSArtifact:
    """Build clip-level FAISS artifact from in-memory vectors + clip metadata."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D vectors, got shape {arr.shape}")

    if clip_records is not None and clip_locations is not None:
        raise ValueError("Provide clip_records or clip_locations, not both")
    if clip_records is None and clip_locations is None:
        raise ValueError("clip_records or clip_locations is required")

    if clip_records is not None:
        normalized_clip_records = _coerce_clip_records(clip_records)
    else:
        assert clip_locations is not None
        normalized_clip_records = [
            ClipMetadata.from_clip_location(location, dataset_id=dataset)
            for location in clip_locations
        ]

    if len(normalized_clip_records) != int(arr.shape[0]):
        raise ValueError(
            "clip metadata length must match vectors rows: "
            f"{len(normalized_clip_records)} vs {arr.shape[0]}"
        )

    normalized_vectors = _normalize_vectors(vectors)
    index = _build_faiss_index_impl(
        normalized_vectors,
        index_type=index_type,
        nlist=nlist,
        factory_string=factory_string,
        nprobe=nprobe,
    )

    manifest = ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        artifact_name=artifact_name,
        artifact_version_id=_new_artifact_version_id(),
        kind="clip",
        index_type=index_type,
        metric="cosine_ip",
        dataset=dataset,
        feature_source_dir=feature_source_dir,
        layer=layer,
        dimension=int(arr.shape[1]),
        ntotal=int(arr.shape[0]),
        created_at_utc=_now_utc_iso(),
        model_name=model_name,
        nlist=nlist if index_type == "ivfflat" else None,
        default_nprobe=nprobe if index_type == "ivfflat" else None,
        factory_string=factory_string if index_type == "factory" else None,
        train_size=int(arr.shape[0]) if index_type in {"ivfflat", "factory"} else None,
    )
    return FAISSArtifact(
        index=index,
        manifest=manifest,
        clip_records=normalized_clip_records,
        vectors=normalized_vectors,
    )


def save_artifact(
    artifact: FAISSArtifact,
    *,
    artifact_root: str | Path,
    created_stamp: str | None = None,
) -> Path:
    """Persist FAISS index + lookup metadata + manifest/checksums and return artifact directory."""
    faiss = _require_faiss()
    stamp = created_stamp or _now_utc_stamp()
    out_dir = _artifact_dir(artifact_root, artifact.manifest.artifact_name, stamp)
    out_dir.mkdir(parents=True, exist_ok=False)

    index_path = out_dir / INDEX_FILE
    manifest_path = out_dir / MANIFEST_FILE
    tracks_path = out_dir / TRACK_NAMES_FILE
    clip_records_path = out_dir / CLIP_RECORDS_FILE
    vectors_path = out_dir / VECTORS_FILE
    checksums_path = out_dir / CHECKSUM_FILE

    faiss.write_index(artifact.index, str(index_path))
    manifest_path.write_text(json.dumps(asdict(artifact.manifest), indent=2), encoding="utf-8")

    payload_files = [index_path, manifest_path]
    if artifact.track_names is not None:
        tracks_path.write_text(json.dumps(artifact.track_names), encoding="utf-8")
        payload_files.append(tracks_path)
    if artifact.clip_records is not None:
        with gzip.open(clip_records_path, "wt", encoding="utf-8") as f:
            json.dump(_serialize_clip_records(artifact.clip_records), f)
        payload_files.append(clip_records_path)
    if artifact.vectors is not None:
        np.save(vectors_path, np.asarray(artifact.vectors, dtype=np.float32))
        payload_files.append(vectors_path)

    checksums = _build_checksums(payload_files)
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return out_dir


def load_artifact(artifact_dir: str | Path) -> FAISSArtifact:
    """Load persisted artifact directory with strict schema and checksum validation."""
    faiss = _require_faiss()
    root = Path(artifact_dir)
    if not root.exists():
        raise FileNotFoundError(f"Artifact directory not found: {root}")

    manifest_path = root / MANIFEST_FILE
    checksums_path = root / CHECKSUM_FILE
    index_path = root / INDEX_FILE
    if not manifest_path.exists():
        raise ArtifactValidationError(f"Missing required file: {manifest_path.name}")
    if not checksums_path.exists():
        raise ArtifactValidationError(f"Missing required file: {checksums_path.name}")
    if not index_path.exists():
        raise ArtifactValidationError(f"Missing required file: {index_path.name}")

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise ArtifactValidationError("Manifest content must be a JSON object.")
    _validate_manifest_payload(manifest_payload)
    manifest = ArtifactManifest(**manifest_payload)

    checksums_payload = json.loads(checksums_path.read_text(encoding="utf-8"))
    if not isinstance(checksums_payload, dict):
        raise ArtifactValidationError("Checksums content must be a JSON object.")
    _validate_checksums(root, checksums_payload)

    index = faiss.read_index(str(index_path))
    if index.d != manifest.dimension:
        raise ArtifactValidationError(
            f"Index dimension mismatch: manifest {manifest.dimension}, index {index.d}"
        )
    if int(index.ntotal) != manifest.ntotal:
        raise ArtifactValidationError(
            f"Index ntotal mismatch: manifest {manifest.ntotal}, index {index.ntotal}"
        )
    try:
        _set_index_nprobe(index, manifest.default_nprobe)
    except ValueError as exc:
        raise ArtifactValidationError(f"Manifest/index nprobe mismatch: {exc}") from exc

    track_names: list[str] | None = None
    clip_records: list[ClipMetadata] | None = None
    vectors: np.ndarray | None = None

    tracks_file = root / TRACK_NAMES_FILE
    if tracks_file.exists():
        track_names = json.loads(tracks_file.read_text(encoding="utf-8"))

    clip_records_file = root / CLIP_RECORDS_FILE
    if clip_records_file.exists():
        with gzip.open(clip_records_file, "rt", encoding="utf-8") as f:
            clip_records = _deserialize_clip_records(json.load(f))
    else:
        clips_file = root / CLIP_LOCATIONS_FILE
        if clips_file.exists():
            with gzip.open(clips_file, "rt", encoding="utf-8") as f:
                clip_locations = _deserialize_clip_locations(json.load(f))
            clip_records = [
                ClipMetadata.from_clip_location(location, dataset_id=manifest.dataset)
                for location in clip_locations
            ]
        else:
            legacy_clips = root / "clip_locations.json"
            if legacy_clips.exists():
                clip_locations = _deserialize_clip_locations(
                    json.loads(legacy_clips.read_text(encoding="utf-8"))
                )
                clip_records = [
                    ClipMetadata.from_clip_location(location, dataset_id=manifest.dataset)
                    for location in clip_locations
                ]

    vectors_file = root / VECTORS_FILE
    if vectors_file.exists():
        vectors = np.load(vectors_file, mmap_mode="r")
        _validate_vectors_payload(manifest, vectors)

    return FAISSArtifact(
        index=index,
        manifest=manifest,
        track_names=track_names,
        clip_records=clip_records,
        vectors=vectors,
    )


def find_latest_artifact_dir(
    artifact_root: str | Path,
    *,
    artifact_name: str,
) -> Path:
    """Find newest timestamped artifact directory under artifact_root/artifact_name."""
    base = Path(artifact_root) / artifact_name
    if not base.exists():
        raise FileNotFoundError(f"Artifact name directory not found: {base}")

    candidates = [
        p
        for p in base.iterdir()
        if p.is_dir() and _STAMP_RE.match(p.name) and (p / MANIFEST_FILE).exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No saved artifacts found under {base}")

    return max(candidates, key=lambda p: p.name)


def upload_artifact_to_s3(
    artifact_dir: str | Path,
    *,
    bucket: str,
    prefix: str,
    upload_latest_pointer: bool = True,
) -> list[str]:
    """
    Upload a local artifact directory to S3.

    Upload validation checks size and (when available) non-multipart ETag/MD5.
    Pointer `latest.json` is only written after all uploads validate.
    """
    boto3 = _require_boto3()

    root = Path(artifact_dir)
    if not root.exists():
        raise FileNotFoundError(f"Artifact directory not found: {root}")

    manifest_payload = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    _validate_manifest_payload(manifest_payload)
    checksums_payload = json.loads((root / CHECKSUM_FILE).read_text(encoding="utf-8"))
    if not isinstance(checksums_payload, dict):
        raise ArtifactValidationError("Checksums content must be a JSON object.")
    _validate_checksums(root, checksums_payload)

    s3 = boto3.client("s3")
    base_prefix = prefix.strip("/")
    artifact_prefix = f"{base_prefix}/{root.parent.name}/{root.name}".strip("/")
    uploaded_keys: list[str] = []
    upload_meta: dict[str, dict[str, str]] = {}

    for filename, info in checksums_payload.items():
        path = root / filename
        key = f"{artifact_prefix}/{filename}"
        meta = _upload_and_validate_file(
            s3,
            bucket=bucket,
            key=key,
            local_path=path,
            expected_md5=str(info["md5"]),
            expected_size=int(info["size"]),
        )
        upload_meta[filename] = meta
        uploaded_keys.append(key)

    checksums_key = f"{artifact_prefix}/{CHECKSUM_FILE}"
    _upload_and_validate_file(
        s3,
        bucket=bucket,
        key=checksums_key,
        local_path=root / CHECKSUM_FILE,
        expected_md5=_md5_file(root / CHECKSUM_FILE),
        expected_size=_size_file(root / CHECKSUM_FILE),
    )
    uploaded_keys.append(checksums_key)

    if upload_latest_pointer:
        latest_payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_name": manifest_payload["artifact_name"],
            "artifact_version_id": manifest_payload["artifact_version_id"],
            "artifact_prefix": artifact_prefix,
            "created_at_utc": manifest_payload["created_at_utc"],
            "checksums": checksums_payload,
            "uploaded_objects": upload_meta,
        }
        latest_key = f"{base_prefix}/{root.parent.name}/latest.json".strip("/")
        s3.put_object(
            Bucket=bucket,
            Key=latest_key,
            Body=json.dumps(latest_payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        uploaded_keys.append(latest_key)

    return uploaded_keys


def download_artifact_from_s3(
    *,
    bucket: str,
    prefix: str,
    artifact_name: str,
    created_stamp: str,
    local_root: str | Path,
) -> Path:
    """Download one artifact directory from S3 and return local path."""
    boto3 = _require_boto3()
    s3 = boto3.client("s3")
    base_prefix = prefix.strip("/")
    artifact_prefix = f"{base_prefix}/{artifact_name}/{created_stamp}".strip("/")
    local_dir = Path(local_root) / artifact_name / created_stamp
    local_dir.mkdir(parents=True, exist_ok=True)

    response = s3.list_objects_v2(Bucket=bucket, Prefix=f"{artifact_prefix}/")
    contents = response.get("Contents", [])
    if not contents:
        raise FileNotFoundError(f"No S3 objects found under s3://{bucket}/{artifact_prefix}/")

    for obj in contents:
        key = str(obj["Key"])
        if key.endswith("/"):
            continue
        rel = key.removeprefix(f"{artifact_prefix}/")
        out_path = local_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(out_path))

    return local_dir


def load_latest_from_s3(
    *,
    bucket: str,
    prefix: str,
    artifact_name: str,
    local_root: str | Path,
) -> FAISSArtifact:
    """Read pointer latest.json, download artifact locally, validate, then load it."""
    boto3 = _require_boto3()
    s3 = boto3.client("s3")
    base_prefix = prefix.strip("/")
    latest_key = f"{base_prefix}/{artifact_name}/latest.json".strip("/")
    response = s3.get_object(Bucket=bucket, Key=latest_key)
    body = response["Body"].read().decode("utf-8")
    pointer = json.loads(body)

    if pointer.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"Unsupported pointer schema_version={pointer.get('schema_version')}."
        )

    if pointer.get("artifact_name") != artifact_name:
        raise ArtifactValidationError(
            f"Pointer artifact_name mismatch: expected {artifact_name}, "
            f"got {pointer.get('artifact_name')}"
        )

    artifact_prefix = pointer.get("artifact_prefix")
    if not isinstance(artifact_prefix, str) or not artifact_prefix:
        raise ArtifactValidationError("Pointer missing artifact_prefix.")

    created_stamp = Path(artifact_prefix).name
    local_dir = download_artifact_from_s3(
        bucket=bucket,
        prefix=prefix,
        artifact_name=artifact_name,
        created_stamp=created_stamp,
        local_root=local_root,
    )

    pointer_checksums = pointer.get("checksums")
    if not isinstance(pointer_checksums, dict):
        raise ArtifactValidationError("Pointer missing checksums map.")
    local_checksums = json.loads((local_dir / CHECKSUM_FILE).read_text(encoding="utf-8"))
    if local_checksums != pointer_checksums:
        raise ArtifactValidationError("Pointer checksums do not match downloaded checksums file.")

    artifact = load_artifact(local_dir)

    expected_vid = pointer.get("artifact_version_id")
    if expected_vid != artifact.manifest.artifact_version_id:
        raise ArtifactValidationError(
            "Pointer artifact_version_id does not match downloaded manifest."
        )

    return artifact


def remove_local_artifact_dir(path: str | Path) -> None:
    """Utility for cleanup in deployment workflows."""
    root = Path(path)
    if root.exists():
        shutil.rmtree(root)
