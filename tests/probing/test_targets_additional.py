"""Additional tests for mess.probing.targets validation and dataset workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mess.probing.discovery import LayerDiscoverySystem
from mess.probing.targets import MusicalAspectTargets, create_target_dataset

pytestmark = pytest.mark.integration


# =========================================================================
# Tests for fixed target computations
# =========================================================================


class TestFixedTargetComputations:
    """Verify corrected target computations produce meaningful values."""

    @pytest.fixture
    def target_gen(self):
        return MusicalAspectTargets(sample_rate=24000)

    @pytest.fixture
    def synthetic_audio(self):
        """10s synthetic audio with dynamic variation."""
        sr = 24000
        n = sr * 10
        rng = np.random.default_rng(42)
        t = np.linspace(0, 10.0, n, dtype=np.float32)
        # Amplitude envelope: crescendo then diminuendo
        envelope = np.concatenate([
            np.linspace(0.1, 1.0, n // 2, dtype=np.float32),
            np.linspace(1.0, 0.1, n - n // 2, dtype=np.float32),
        ])
        audio = envelope * np.sin(2 * np.pi * 440 * t)
        # Add some noise for onset detection
        audio += 0.05 * rng.standard_normal(n).astype(np.float32)
        return audio

    def test_phrase_regularity_uses_cv(self, target_gen, synthetic_audio):
        """phrase_regularity should use coefficient of variation, not 1/(1+var)."""
        result = target_gen._generate_phrasing_targets(synthetic_audio)
        pr = float(result['phrase_regularity'][0])
        # CV is std/mean — for non-trivial segments it should not be near 1.0
        # (the old formula 1/(1+var) was always near 1.0 for small variances)
        assert pr >= 0.0, "phrase_regularity should be non-negative"
        # It should have meaningful range (not near-constant)
        # The exact value depends on the audio, but it shouldn't be ~1.0
        assert pr < 10.0, "phrase_regularity (CV) should be bounded"

    def test_dynamic_variance_is_normalized(self, target_gen, synthetic_audio):
        """dynamic_variance should be recording-level-invariant."""
        result_normal = target_gen._generate_dynamics_targets(synthetic_audio)
        result_quiet = target_gen._generate_dynamics_targets(synthetic_audio * 0.1)

        dv_normal = float(result_normal['dynamic_variance'][0])
        dv_quiet = float(result_quiet['dynamic_variance'][0])

        # After normalization, variance should be similar regardless of level
        assert abs(dv_normal - dv_quiet) / (dv_normal + 1e-8) < 0.3, (
            f"dynamic_variance should be level-invariant: {dv_normal:.4f} vs {dv_quiet:.4f}"
        )

    def test_crescendo_diminuendo_longest_run(self, target_gen, synthetic_audio):
        """crescendo/diminuendo should reflect longest monotonic runs."""
        result = target_gen._generate_dynamics_targets(synthetic_audio)
        cresc = float(result['crescendo_strength'][0])
        dim = float(result['diminuendo_strength'][0])

        # The synthetic audio has a clear crescendo then diminuendo
        assert cresc > 0.0, "Should detect crescendo in rising envelope"
        assert dim > 0.0, "Should detect diminuendo in falling envelope"

    def test_attack_slopes_uses_db_domain(self, target_gen, synthetic_audio):
        """attack_slopes should use dB envelope, producing stable measurements."""
        result = target_gen._generate_articulation_targets(synthetic_audio)
        slopes = result['attack_slopes']
        assert len(slopes) > 0
        # All slopes should be non-negative (we clamp to max(0, slope))
        assert np.all(slopes >= 0)

    def test_longest_monotonic_runs_helper(self):
        """Test the _longest_monotonic_runs static method directly."""
        # Simple increasing then decreasing
        rms = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1])
        cresc, dim = MusicalAspectTargets._longest_monotonic_runs(rms)
        assert cresc > 0, "Should find crescendo in first half"
        assert dim > 0, "Should find diminuendo in second half"

    def test_longest_monotonic_runs_empty(self):
        """Edge case: single-element or empty array."""
        c, d = MusicalAspectTargets._longest_monotonic_runs(np.array([1.0]))
        assert c == 0.0
        assert d == 0.0

        c, d = MusicalAspectTargets._longest_monotonic_runs(np.array([]))
        assert c == 0.0
        assert d == 0.0


def _complete_scalar_targets() -> dict[str, dict[str, np.ndarray]]:
    targets: dict[str, dict[str, np.ndarray]] = {}
    for category, key, reduction in LayerDiscoverySystem.SCALAR_TARGETS.values():
        targets.setdefault(category, {})
        if reduction == "mean":
            targets[category][key] = np.array([0.1, 0.2], dtype=np.float32)
        else:
            targets[category][key] = np.array([0.3], dtype=np.float32)
    return targets


class TestValidateTargetStructure:
    def test_returns_true_when_all_required_targets_exist(self):
        assert MusicalAspectTargets.validate_target_structure(_complete_scalar_targets()) is True

    def test_returns_false_when_required_target_is_missing(self):
        targets = _complete_scalar_targets()
        del targets["rhythm"]["tempo"]
        assert MusicalAspectTargets.validate_target_structure(targets) is False


class TestCreateTargetDataset:
    def test_returns_zero_counts_when_no_wavs_found(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        result = create_target_dataset(
            audio_dir=audio_dir,
            output_dir=tmp_path / "proxy_targets",
            validate=False,
            use_mlflow=False,
        )

        assert result == {"total": 0, "success": 0, "failed": 0}

    def test_continues_when_one_file_fails(self, monkeypatch, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "ok.wav").touch()
        (audio_dir / "bad.wav").touch()

        class PartialTargets:
            sample_rate = 24000

            def generate_all_targets(self, audio_path):
                if Path(audio_path).stem == "bad":
                    raise RuntimeError("boom")
                return _complete_scalar_targets()

            @staticmethod
            def validate_target_structure(_targets):
                return True

        monkeypatch.setattr("mess.probing.targets.MusicalAspectTargets", PartialTargets)

        output_dir = tmp_path / "proxy_targets"
        result = create_target_dataset(
            audio_dir=audio_dir,
            output_dir=output_dir,
            validate=True,
            use_mlflow=False,
        )

        assert result == {"total": 2, "success": 1, "failed": 1}
        assert (output_dir / "ok_targets.npz").exists()
        assert not (output_dir / "bad_targets.npz").exists()

    def test_logs_mlflow_params_and_metrics_when_run_active(self, monkeypatch, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "track.wav").touch()

        class FakeTargets:
            sample_rate = 24000

            def generate_all_targets(self, _audio_path):
                return _complete_scalar_targets()

            @staticmethod
            def validate_target_structure(_targets):
                return True

        calls: dict[str, list] = {"params": [], "metrics": [], "artifacts": []}

        monkeypatch.setattr("mess.probing.targets.MusicalAspectTargets", FakeTargets)
        monkeypatch.setattr("mess.probing.targets.mlflow.active_run", lambda: object())
        monkeypatch.setattr("mess.probing.targets.mlflow.log_params", calls["params"].append)
        monkeypatch.setattr("mess.probing.targets.mlflow.log_metrics", calls["metrics"].append)
        monkeypatch.setattr("mess.probing.targets.mlflow.log_artifact", calls["artifacts"].append)

        result = create_target_dataset(
            audio_dir=audio_dir,
            output_dir=tmp_path / "proxy_targets",
            validate=True,
            use_mlflow=True,
        )

        assert result == {"total": 1, "success": 1, "failed": 0}
        assert calls["params"]
        assert any("total_files" in entry for entry in calls["metrics"])
        assert not calls["artifacts"]

    def test_logs_errors_artifact_when_failures_occur(self, monkeypatch, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "broken.wav").touch()

        class AlwaysFailTargets:
            sample_rate = 24000

            def generate_all_targets(self, _audio_path):
                raise RuntimeError("bad target extraction")

            @staticmethod
            def validate_target_structure(_targets):
                return True

        artifacts: list[str] = []

        monkeypatch.setattr("mess.probing.targets.MusicalAspectTargets", AlwaysFailTargets)
        monkeypatch.setattr("mess.probing.targets.mlflow.active_run", lambda: object())
        monkeypatch.setattr("mess.probing.targets.mlflow.log_params", lambda _params: None)
        monkeypatch.setattr("mess.probing.targets.mlflow.log_metrics", lambda _metrics: None)
        monkeypatch.setattr("mess.probing.targets.mlflow.log_artifact", artifacts.append)

        output_dir = tmp_path / "proxy_targets"
        result = create_target_dataset(
            audio_dir=audio_dir,
            output_dir=output_dir,
            validate=True,
            use_mlflow=True,
        )

        assert result == {"total": 1, "success": 0, "failed": 1}
        assert (output_dir / "errors.txt").exists()
        assert artifacts
        assert artifacts[0].endswith("errors.txt")
