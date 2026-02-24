"""Unit tests for FeatureExtractor control flow without loading MERT."""

from pathlib import Path

import numpy as np
import pytest

from mess.extraction.extractor import FeatureExtractor


def _make_extractor() -> FeatureExtractor:
    """Build a FeatureExtractor instance without running __init__."""
    extractor = object.__new__(FeatureExtractor)
    extractor.device = "cpu"
    extractor.batch_size = 4
    extractor.segment_duration = 5.0
    extractor.overlap_ratio = 0.5
    extractor.target_sample_rate = 24000
    return extractor


class TestFeatureViews:
    def test_extract_feature_views_requires_segments(self):
        extractor = _make_extractor()

        with pytest.raises(ValueError, match="No audio segments"):
            extractor._extract_feature_views_from_segments([])

    def test_extract_feature_views_returns_requested_views(self):
        extractor = _make_extractor()
        batch = np.zeros((2, 13, 3, 768), dtype=np.float32)
        batch[0, :, :, :] = 1.0
        batch[1, :, :, :] = 3.0
        extractor._extract_mert_features_batched_with_oom_recovery = lambda _: batch

        result = extractor._extract_feature_views_from_segments(
            segments=[np.zeros(8), np.zeros(8)],
            include_raw=True,
            include_segments=True,
        )

        assert set(result) == {"raw", "segments", "aggregated"}
        assert result["raw"].shape == (2, 13, 3, 768)
        assert result["segments"].shape == (2, 13, 768)
        assert result["aggregated"].shape == (13, 768)
        np.testing.assert_allclose(result["aggregated"], 2.0, atol=1e-6)


class TestOOMRecovery:
    def test_oom_recovery_reduces_batch_size_and_restores(self, monkeypatch):
        extractor = _make_extractor()
        extractor.device = "cuda"
        extractor.batch_size = 8

        monkeypatch.setattr(
            "mess.config.mess_config.MERT_CUDA_AUTO_OOM_RECOVERY", True
        )

        calls = {"n": 0, "cleared": 0}

        def fake_batched(_segments):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("CUDA out of memory")
            return np.zeros((1, 13, 2, 768), dtype=np.float32)

        def fake_clear():
            calls["cleared"] += 1

        extractor._extract_mert_features_batched = fake_batched
        extractor.clear_gpu_cache = fake_clear

        result = extractor._extract_mert_features_batched_with_oom_recovery(
            [np.zeros(16)]
        )

        assert result.shape == (1, 13, 2, 768)
        assert calls["n"] == 2
        assert calls["cleared"] == 1
        assert extractor.batch_size == 8


class TestExtractTrackFeatures:
    def test_extract_track_features_returns_cached_selection(self, monkeypatch, tmp_path):
        extractor = _make_extractor()
        cached = {"aggregated": np.ones((13, 768), dtype=np.float32)}

        monkeypatch.setattr("mess.extraction.extractor.features_exist", lambda *_a, **_k: True)
        monkeypatch.setattr(
            "mess.extraction.extractor.load_selected_features",
            lambda *_a, **_k: cached,
        )

        audio_called = {"n": 0}

        def fail_if_called(*_args, **_kwargs):
            audio_called["n"] += 1
            raise AssertionError("load_audio should not be called for cached features")

        monkeypatch.setattr("mess.extraction.extractor.load_audio", fail_if_called)

        result = extractor.extract_track_features(
            audio_path=tmp_path / "track.wav",
            output_dir=tmp_path,
            skip_existing=True,
            include_raw=False,
            include_segments=False,
        )

        assert result is cached
        assert audio_called["n"] == 0

    def test_extract_track_features_safe_returns_validation_error(self, monkeypatch, tmp_path):
        extractor = _make_extractor()

        monkeypatch.setattr(
            "mess.extraction.extractor.validate_audio_file",
            lambda *_a, **_k: {"valid": False, "errors": ["too short"]},
        )

        result, error = extractor.extract_track_features_safe(
            audio_path=tmp_path / "bad.wav",
            validate=True,
        )

        assert result is None
        assert error == "Validation failed: too short"


class TestDatasetDelegation:
    def test_extract_dataset_features_delegates_to_pipeline(self, monkeypatch, tmp_path):
        extractor = _make_extractor()
        captured = {}

        class FakePipeline:
            def __init__(self, passed_extractor):
                assert passed_extractor is extractor

            def run(
                self,
                audio_dir,
                output_dir,
                file_pattern,
                skip_existing,
                num_workers,
                dataset,
                include_raw,
                include_segments,
            ):
                captured["args"] = (
                    Path(audio_dir),
                    Path(output_dir),
                    file_pattern,
                    skip_existing,
                    num_workers,
                    dataset,
                    include_raw,
                    include_segments,
                )
                return {"processed": 1}

        monkeypatch.setattr("mess.extraction.pipeline.ExtractionPipeline", FakePipeline)

        result = extractor.extract_dataset_features(
            audio_dir=tmp_path / "audio",
            output_dir=tmp_path / "embeddings",
            num_workers=2,
            dataset="smd",
            include_raw=False,
            include_segments=True,
        )

        assert result == {"processed": 1}
        assert captured["args"][2:] == ("*.wav", True, 2, "smd", False, True)
