"""Production-oriented FAISS index artifacts for track and clip retrieval."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .search import ClipLocation, load_features, load_segment_features

IndexKind = Literal["track", "clip"]
IndexType = Literal["flatip", "ivfflat"]
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
CLIP_LOCATIONS_FILE = "clip_locations.json.gz"
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


@dataclass(frozen=True)
class FAISSArtifact:
    """In-memory bundle of a FAISS index and its lookup metadata."""

    index: Any
    manifest: ArtifactManifest
    track_names: list[str] | None = None
    clip_locations: list[ClipLocation] | None = None

    def search(
        self,
        query_vectors: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run cosine-similarity search on the artifact's FAISS index."""
        if k <= 0:
            raise ValueError("k must be > 0")

        faiss = _require_faiss()
        query = np.asarray(query_vectors, dtype=np.float32)
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
        return importlib.import_module("faiss")
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


def _build_faiss_index(
    vectors: np.ndarray,
    index_type: IndexType = "flatip",
    nlist: int = 256,
) -> Any:
    faiss = _require_faiss()
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D vectors, got shape {arr.shape}")

    faiss.normalize_L2(arr)
    dim = arr.shape[1]

    if index_type == "flatip":
        index = faiss.IndexFlatIP(dim)
        index.add(arr)
        return index

    if index_type == "ivfflat":
        if nlist <= 0:
            raise ValueError("nlist must be > 0 for ivfflat")
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, int(nlist), faiss.METRIC_INNER_PRODUCT)
        index.train(arr)
        index.add(arr)
        return index

    raise ValueError(f"Unsupported index_type '{index_type}'")


def _serialize_clip_locations(locations: list[ClipLocation]) -> list[dict[str, Any]]:
    return [asdict(loc) for loc in locations]


def _deserialize_clip_locations(rows: list[dict[str, Any]]) -> list[ClipLocation]:
    return [ClipLocation(**row) for row in rows]


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
) -> FAISSArtifact:
    """Build track-level FAISS artifact from aggregated features."""
    vectors, track_names = load_features(str(features_dir), layer=layer)
    index = _build_faiss_index(vectors, index_type=index_type, nlist=nlist)

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
    )
    return FAISSArtifact(index=index, manifest=manifest, track_names=track_names)


def build_clip_artifact(
    *,
    dataset: str,
    features_dir: str | Path,
    artifact_name: str = "clip_index",
    layer: int | None = None,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
    index_type: IndexType = "flatip",
    model_name: str = "m-a-p/MERT-v1-95M",
    nlist: int = 1024,
) -> FAISSArtifact:
    """Build clip-level FAISS artifact from segment features."""
    vectors, clip_locations = load_segment_features(
        str(features_dir),
        layer=layer,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
    )
    index = _build_faiss_index(vectors, index_type=index_type, nlist=nlist)

    manifest = ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        artifact_name=artifact_name,
        artifact_version_id=_new_artifact_version_id(),
        kind="clip",
        index_type=index_type,
        metric="cosine_ip",
        dataset=dataset,
        feature_source_dir=str(features_dir),
        layer=layer,
        dimension=int(vectors.shape[1]),
        ntotal=int(vectors.shape[0]),
        created_at_utc=_now_utc_iso(),
        model_name=model_name,
    )
    return FAISSArtifact(index=index, manifest=manifest, clip_locations=clip_locations)


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
    clips_path = out_dir / CLIP_LOCATIONS_FILE
    checksums_path = out_dir / CHECKSUM_FILE

    faiss.write_index(artifact.index, str(index_path))
    manifest_path.write_text(json.dumps(asdict(artifact.manifest), indent=2), encoding="utf-8")

    payload_files = [index_path, manifest_path]
    if artifact.track_names is not None:
        tracks_path.write_text(json.dumps(artifact.track_names), encoding="utf-8")
        payload_files.append(tracks_path)
    if artifact.clip_locations is not None:
        with gzip.open(clips_path, "wt", encoding="utf-8") as f:
            json.dump(_serialize_clip_locations(artifact.clip_locations), f)
        payload_files.append(clips_path)

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

    track_names: list[str] | None = None
    clip_locations: list[ClipLocation] | None = None

    tracks_file = root / TRACK_NAMES_FILE
    if tracks_file.exists():
        track_names = json.loads(tracks_file.read_text(encoding="utf-8"))

    clips_file = root / CLIP_LOCATIONS_FILE
    if clips_file.exists():
        with gzip.open(clips_file, "rt", encoding="utf-8") as f:
            clip_locations = _deserialize_clip_locations(json.load(f))
    else:
        legacy_clips = root / "clip_locations.json"
        if legacy_clips.exists():
            clip_locations = _deserialize_clip_locations(
                json.loads(legacy_clips.read_text(encoding="utf-8"))
            )

    return FAISSArtifact(
        index=index,
        manifest=manifest,
        track_names=track_names,
        clip_locations=clip_locations,
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
