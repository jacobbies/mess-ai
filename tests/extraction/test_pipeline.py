"""Unit tests for ExtractionPipeline orchestration."""

from pathlib import Path

import numpy as np
import pytest

from mess.extraction.pipeline import ExtractionPipeline

pytestmark = pytest.mark.integration


class _FakeExtractor:
    target_sample_rate = 24000
    segment_duration = 5.0
    overlap_ratio = 0.5

    def __init__(self):
        self.called = []

    def extract_track_features(self, *args, **kwargs):
        self.called.append((args, kwargs))
        if "fail" in str(args[0]):
            raise RuntimeError("boom")
        return {"aggregated": np.zeros((13, 768), dtype=np.float32)}

    def _extract_feature_views_from_segments(self, segments, include_raw, include_segments):
        _ = include_raw, include_segments
        return {"aggregated": np.mean(np.asarray(segments), axis=0)}


class TestDiscoverAudioFiles:
    def test_falls_back_to_recursive_wav_discovery(self, tmp_path):
        extractor = _FakeExtractor()
        pipeline = ExtractionPipeline(extractor)

        (tmp_path / "2018").mkdir()
        (tmp_path / "2019").mkdir()
        (tmp_path / "2018" / "a.wav").touch()
        (tmp_path / "2019" / "b.WAV").touch()

        files = pipeline._discover_audio_files(tmp_path, "*.wav")
        names = [f.name for f in files]
        assert names == ["a.wav", "b.WAV"]


class TestPreprocessWorker:
    def test_returns_cached_when_required_types_exist(self, monkeypatch, tmp_path):
        extractor = _FakeExtractor()
        pipeline = ExtractionPipeline(extractor)
        audio_path = tmp_path / "track.wav"
        audio_path.touch()

        monkeypatch.setattr(
            "mess.extraction.pipeline.features_exist_for_types", lambda *_a, **_k: True
        )

        result = pipeline._preprocess_worker(
            audio_path=audio_path,
            skip_existing=True,
            output_dir=tmp_path,
            dataset="smd",
            requested_feature_types=["aggregated", "segments"],
        )

        assert result["status"] == "cached"
        assert result["track_id"] == "track"

    def test_returns_ready_with_segments(self, monkeypatch, tmp_path):
        extractor = _FakeExtractor()
        pipeline = ExtractionPipeline(extractor)
        audio_path = tmp_path / "track.wav"
        audio_path.touch()

        monkeypatch.setattr(
            "mess.extraction.pipeline.features_exist_for_types", lambda *_a, **_k: False
        )
        monkeypatch.setattr(
            "mess.extraction.pipeline.load_audio", lambda *_a, **_k: np.ones(24, dtype=np.float32)
        )
        monkeypatch.setattr(
            "mess.extraction.pipeline.segment_audio",
            lambda *_a, **_k: [np.ones((13, 768), dtype=np.float32)],
        )

        result = pipeline._preprocess_worker(
            audio_path=audio_path,
            skip_existing=True,
            output_dir=tmp_path,
        )

        assert result["status"] == "ready"
        assert result["track_id"] == "track"
        assert len(result["segments"]) == 1

    def test_returns_error_on_failure(self, monkeypatch, tmp_path):
        extractor = _FakeExtractor()
        pipeline = ExtractionPipeline(extractor)
        audio_path = tmp_path / "track.wav"
        audio_path.touch()

        monkeypatch.setattr(
            "mess.extraction.pipeline.features_exist_for_types", lambda *_a, **_k: False
        )
        monkeypatch.setattr(
            "mess.extraction.pipeline.load_audio",
            lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad audio")),
        )

        result = pipeline._preprocess_worker(
            audio_path=audio_path,
            skip_existing=False,
            output_dir=tmp_path,
        )

        assert result["status"] == "error"
        assert result["track_id"] == "track"
        assert "bad audio" in result["error"]


class TestRun:
    def test_run_delegates_to_parallel_when_workers_gt_one(self, monkeypatch, tmp_path):
        extractor = _FakeExtractor()
        pipeline = ExtractionPipeline(extractor)

        monkeypatch.setattr(
            pipeline,
            "run_parallel",
            lambda *args, **kwargs: {"delegated": True},
        )

        result = pipeline.run(
            audio_dir=tmp_path,
            output_dir=tmp_path / "out",
            num_workers=2,
        )
        assert result == {"delegated": True}

    def test_run_sequential_continues_after_file_error(self, tmp_path):
        extractor = _FakeExtractor()
        pipeline = ExtractionPipeline(extractor)

        (tmp_path / "ok.wav").touch()
        (tmp_path / "fail.wav").touch()

        result = pipeline.run(
            audio_dir=tmp_path,
            output_dir=tmp_path / "out",
            num_workers=1,
            skip_existing=False,
        )

        assert result is None
        # Both files should be attempted; failure should not stop loop.
        assert len(extractor.called) == 2


class TestRunParallel:
    def test_parallel_counts_ready_cached_error(self, monkeypatch, tmp_path):
        extractor = _FakeExtractor()
        pipeline = ExtractionPipeline(extractor)
        output_dir = tmp_path / "out"

        files = [
            tmp_path / "ready.wav",
            tmp_path / "cached.wav",
            tmp_path / "error.wav",
        ]
        for f in files:
            f.touch()

        def fake_worker(audio_path, *_args, **_kwargs):
            name = Path(audio_path).stem
            if name == "ready":
                return {
                    "status": "ready",
                    "path": str(audio_path),
                    "track_id": name,
                    "segments": [np.ones((13, 768), dtype=np.float32)],
                }
            if name == "cached":
                return {"status": "cached", "path": str(audio_path), "track_id": name}
            return {
                "status": "error",
                "path": str(audio_path),
                "track_id": name,
                "error": "boom",
            }

        saved = {"n": 0}

        monkeypatch.setattr(pipeline, "_preprocess_worker", fake_worker)
        monkeypatch.setattr(
            "mess.extraction.pipeline.save_features",
            lambda *_a, **_k: saved.__setitem__("n", saved["n"] + 1),
        )

        result = pipeline.run_parallel(
            audio_dir=tmp_path,
            output_dir=output_dir,
            file_pattern="*.wav",
            skip_existing=True,
            num_workers=1,
            include_raw=False,
            include_segments=False,
        )

        assert result["total_files"] == 3
        assert result["processed"] == 1
        assert result["cached"] == 1
        assert result["failed"] == 1
        assert len(result["errors"]) == 1
        assert saved["n"] == 1
