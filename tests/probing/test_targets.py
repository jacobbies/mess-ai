"""Tests for mess.probing.targets dataset-level target creation."""

import numpy as np
import pytest

from mess.probing.targets import create_target_dataset

pytestmark = pytest.mark.integration


class _FakeTargets:
    sample_rate = 24000

    def generate_all_targets(self, audio_path):
        return {
            "rhythm": {"tempo": np.array([120.0], dtype=np.float32)},
            "timbre": {"spectral_centroid": np.array([1000.0], dtype=np.float32)},
        }

    @staticmethod
    def validate_target_structure(targets):
        return targets


def test_create_target_dataset_discovers_nested_wavs(monkeypatch, tmp_path):
    audio_dir = tmp_path / "audio" / "maestro"
    (audio_dir / "2018").mkdir(parents=True)
    (audio_dir / "2019").mkdir(parents=True)
    (audio_dir / "2018" / "track_a.wav").touch()
    (audio_dir / "2019" / "track_b.WAV").touch()
    (audio_dir / "2019" / "ignore.txt").touch()

    output_dir = tmp_path / "proxy_targets"

    monkeypatch.setattr("mess.probing.targets.MusicalAspectTargets", _FakeTargets)

    result = create_target_dataset(
        audio_dir=audio_dir,
        output_dir=output_dir,
        validate=True,
        use_mlflow=False,
    )

    assert result == {"total": 2, "success": 2, "failed": 0}
    assert (output_dir / "track_a_targets.npz").exists()
    assert (output_dir / "track_b_targets.npz").exists()
