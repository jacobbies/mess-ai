"""Tests for production FAISS artifact persistence + S3 publishing helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mess.search.faiss_index import (
    ArtifactValidationError,
    build_clip_artifact,
    build_track_artifact,
    download_artifact_from_s3,
    find_latest_artifact_dir,
    load_artifact,
    load_latest_from_s3,
    save_artifact,
    upload_artifact_to_s3,
)


class TestFAISSArtifactPersistence:
    def test_track_artifact_round_trip(self, tmp_path):
        arr_a = np.random.default_rng(0).standard_normal((13, 768)).astype(np.float32)
        arr_b = np.random.default_rng(1).standard_normal((13, 768)).astype(np.float32)
        feature_dir = tmp_path / "aggregated"
        feature_dir.mkdir()
        np.save(feature_dir / "track_a.npy", arr_a)
        np.save(feature_dir / "track_b.npy", arr_b)

        artifact = build_track_artifact(
            dataset="smd",
            features_dir=feature_dir,
            artifact_name="track_index",
            layer=0,
        )
        out_dir = save_artifact(
            artifact,
            artifact_root=tmp_path / "indices",
            created_stamp="20260101T000000Z",
        )

        loaded = load_artifact(out_dir)
        assert loaded.manifest.kind == "track"
        assert loaded.manifest.schema_version == 2
        assert loaded.manifest.artifact_version_id.startswith("vid-")
        assert loaded.manifest.ntotal == 2
        assert loaded.track_names == ["track_a", "track_b"]
        assert loaded.clip_locations is None

        query = arr_a[0]
        scores, ids = loaded.search(query, k=2)
        assert scores.shape == (1, 2)
        assert ids.shape == (1, 2)
        assert int(ids[0, 0]) == 0

    def test_clip_artifact_round_trip(self, tmp_path):
        segs = np.zeros((3, 13, 768), dtype=np.float32)
        segs[0, 0, 0] = 1.0
        segs[1, 0, 1] = 1.0
        segs[2, 0, :2] = np.array([0.6, 0.8], dtype=np.float32)

        feature_dir = tmp_path / "segments"
        feature_dir.mkdir()
        np.save(feature_dir / "track_x.npy", segs)

        artifact = build_clip_artifact(
            dataset="smd",
            features_dir=feature_dir,
            artifact_name="clip_index",
            layer=0,
            segment_duration=5.0,
            overlap_ratio=0.5,
        )
        out_dir = save_artifact(
            artifact,
            artifact_root=tmp_path / "indices",
            created_stamp="20260101T000000Z",
        )

        assert (out_dir / "clip_locations.json.gz").exists()
        loaded = load_artifact(out_dir)
        assert loaded.manifest.kind == "clip"
        assert loaded.track_names is None
        assert loaded.clip_locations is not None
        assert len(loaded.clip_locations) == 3
        assert loaded.clip_locations[1].start_time == pytest.approx(2.5)

    def test_schema_validation_errors_are_explicit(self, tmp_path):
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "index.faiss").write_bytes(b"not-faiss")
        (bad / "checksums.json").write_text("{}", encoding="utf-8")
        (bad / "manifest.json").write_text(
            json.dumps({"schema_version": 999}),
            encoding="utf-8",
        )

        with pytest.raises(ArtifactValidationError, match="missing required fields"):
            load_artifact(bad)


class TestLatestArtifactSelection:
    def test_find_latest_artifact_dir(self, tmp_path):
        base = tmp_path / "indices" / "clip_index"
        d1 = base / "20260101T000000Z"
        d2 = base / "20260102T000000Z"
        d1.mkdir(parents=True)
        d2.mkdir(parents=True)
        (d1 / "manifest.json").write_text("{}", encoding="utf-8")
        (d2 / "manifest.json").write_text("{}", encoding="utf-8")

        latest = find_latest_artifact_dir(tmp_path / "indices", artifact_name="clip_index")
        assert latest == d2


class TestS3PublishingAndDownload:
    def test_upload_download_and_load_latest(self, tmp_path, monkeypatch):
        artifact_dir = tmp_path / "indices" / "clip_index" / "20260101T000000Z"
        artifact_dir.mkdir(parents=True)

        manifest = {
            "schema_version": 2,
            "artifact_name": "clip_index",
            "artifact_version_id": "vid-test123",
            "kind": "clip",
            "index_type": "flatip",
            "metric": "cosine_ip",
            "dataset": "smd",
            "feature_source_dir": "x",
            "layer": 0,
            "dimension": 768,
            "ntotal": 3,
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "model_name": "m-a-p/MERT-v1-95M",
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (artifact_dir / "index.faiss").write_bytes(b"index")
        with gzip.open(artifact_dir / "clip_locations.json.gz", "wt", encoding="utf-8") as f:
            json.dump([], f)

        # Local checksums used by uploader
        import mess.search.faiss_index as m

        checksums = {
            name: {
                "sha256": m._sha256_file(artifact_dir / name),  # noqa: SLF001
                "md5": m._md5_file(artifact_dir / name),  # noqa: SLF001
                "size": (artifact_dir / name).stat().st_size,
            }
            for name in ["index.faiss", "manifest.json", "clip_locations.json.gz"]
        }
        (artifact_dir / "checksums.json").write_text(json.dumps(checksums), encoding="utf-8")

        class FakeBody:
            def __init__(self, raw: bytes):
                self._raw = raw

            def read(self):
                return self._raw

        class FakeS3Client:
            def __init__(self):
                self.objects: dict[str, bytes] = {}

            def upload_file(self, filename, bucket, key):
                _ = bucket
                self.objects[key] = Path(filename).read_bytes()

            def head_object(self, Bucket, Key):
                _ = Bucket
                data = self.objects[Key]
                etag = hashlib.md5(data, usedforsecurity=False).hexdigest()
                return {"ContentLength": len(data), "ETag": f'"{etag}"', "VersionId": "v1"}

            def put_object(self, Bucket, Key, Body, ContentType):
                _ = Bucket, ContentType
                payload = Body if isinstance(Body, bytes) else Body.encode("utf-8")
                self.objects[Key] = payload

            def list_objects_v2(self, Bucket, Prefix):
                _ = Bucket
                keys = [k for k in self.objects if k.startswith(Prefix)]
                return {"Contents": [{"Key": key} for key in sorted(keys)]}

            def download_file(self, bucket, key, filename):
                _ = bucket
                Path(filename).write_bytes(self.objects[key])

            def get_object(self, Bucket, Key):
                _ = Bucket
                return {"Body": FakeBody(self.objects[Key])}

        fake_s3 = FakeS3Client()

        class FakeBoto3:
            @staticmethod
            def client(name):
                assert name == "s3"
                return fake_s3

        real_import_module = m.importlib.import_module

        def fake_import_module(name):
            if name == "boto3":
                return FakeBoto3
            return real_import_module(name)

        monkeypatch.setattr(m.importlib, "import_module", fake_import_module)

        keys = upload_artifact_to_s3(
            artifact_dir,
            bucket="my-bucket",
            prefix="mess/faiss",
            upload_latest_pointer=True,
        )
        assert any(key.endswith("latest.json") for key in keys)

        dl_dir = download_artifact_from_s3(
            bucket="my-bucket",
            prefix="mess/faiss",
            artifact_name="clip_index",
            created_stamp="20260101T000000Z",
            local_root=tmp_path / "downloaded",
        )
        assert (dl_dir / "manifest.json").exists()
        assert (dl_dir / "checksums.json").exists()

        # Patch load_artifact to avoid FAISS binary parsing from fake bytes.

        def fake_load_artifact(_):
            class DummyManifest:
                artifact_version_id = "vid-test123"

            class DummyArtifact:
                manifest = DummyManifest()

            return DummyArtifact()

        monkeypatch.setattr(m, "load_artifact", fake_load_artifact)

        loaded = load_latest_from_s3(
            bucket="my-bucket",
            prefix="mess/faiss",
            artifact_name="clip_index",
            local_root=tmp_path / "downloaded",
        )
        assert loaded.manifest.artifact_version_id == "vid-test123"
