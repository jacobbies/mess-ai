"""Tests for mess.config.MESSConfig."""

import pytest

from mess.config import MESSConfig

pytestmark = pytest.mark.unit


class TestDefaults:
    def test_default_device_is_cpu(self):
        config = MESSConfig()
        assert config.device == "cpu"

    def test_default_sample_rate(self):
        config = MESSConfig()
        assert config.target_sample_rate == 24000

    def test_default_segment_duration(self):
        config = MESSConfig()
        assert config.segment_duration == 5.0

    def test_default_overlap_ratio(self):
        config = MESSConfig()
        assert config.overlap_ratio == 0.5

    def test_default_max_workers(self):
        config = MESSConfig()
        assert config.max_workers == 4

    def test_default_cache_dir_is_none(self):
        config = MESSConfig()
        assert config.cache_dir is None


class TestBatchSize:
    @pytest.mark.parametrize(
        "device, expected",
        [("cpu", 4), ("cuda", 16), ("mps", 8)],
    )
    def test_batch_size_per_device(self, device, expected):
        config = MESSConfig()
        config.MERT_DEVICE = device
        assert config.batch_size == expected


class TestValidation:
    def test_validate_passes_with_defaults(self, tmp_path):
        config = MESSConfig()
        config.project_root = tmp_path
        config.validate_config()

    def test_validate_rejects_negative_segment_duration(self, tmp_path):
        config = MESSConfig()
        config.project_root = tmp_path
        config.MERT_SEGMENT_DURATION = -1.0
        with pytest.raises(ValueError, match="segment_duration must be positive"):
            config.validate_config()

    def test_validate_rejects_zero_segment_duration(self, tmp_path):
        config = MESSConfig()
        config.project_root = tmp_path
        config.MERT_SEGMENT_DURATION = 0
        with pytest.raises(ValueError, match="segment_duration must be positive"):
            config.validate_config()

    def test_validate_rejects_overlap_ratio_ge_1(self, tmp_path):
        config = MESSConfig()
        config.project_root = tmp_path
        config.MERT_OVERLAP_RATIO = 1.0
        with pytest.raises(ValueError, match="overlap_ratio"):
            config.validate_config()

    def test_validate_rejects_invalid_device(self, tmp_path):
        config = MESSConfig()
        config.project_root = tmp_path
        config.MERT_DEVICE = "tpu"
        with pytest.raises(ValueError, match="Invalid device"):
            config.validate_config()

    def test_validate_rejects_zero_batch_size(self, tmp_path):
        config = MESSConfig()
        config.project_root = tmp_path
        config.MERT_BATCH_SIZE_CPU = 0
        with pytest.raises(ValueError, match="batch_size must be positive"):
            config.validate_config()

    def test_validate_rejects_zero_max_workers(self, tmp_path):
        config = MESSConfig()
        config.project_root = tmp_path
        config.MERT_MAX_WORKERS = 0
        with pytest.raises(ValueError, match="max_workers must be at least 1"):
            config.validate_config()


class TestEnvOverrides:
    def test_device_override(self, monkeypatch):
        monkeypatch.setenv("MESS_DEVICE", "mps")
        config = MESSConfig()
        assert config.device == "mps"

    def test_workers_override(self, monkeypatch):
        monkeypatch.setenv("MESS_WORKERS", "8")
        config = MESSConfig()
        assert config.max_workers == 8

    def test_batch_size_override(self, monkeypatch):
        monkeypatch.setenv("MESS_BATCH_SIZE", "32")
        config = MESSConfig()
        assert config.MERT_BATCH_SIZE_CUDA == 32
        assert config.MERT_BATCH_SIZE_MPS == 32
        assert config.MERT_BATCH_SIZE_CPU == 32


class TestPaths:
    def test_data_root(self, isolated_config, tmp_path):
        assert isolated_config.data_root == tmp_path / "data"

    def test_probing_results_file(self, isolated_config, tmp_path):
        expected = tmp_path / "mess" / "probing" / "layer_discovery_results.json"
        assert isolated_config.probing_results_file == expected

    def test_proxy_targets_dir(self, isolated_config, tmp_path):
        assert isolated_config.proxy_targets_dir == tmp_path / "data" / "proxy_targets"

    def test_cache_dir_returns_path_when_set(self):
        config = MESSConfig()
        config.MERT_CACHE_DIR = "/tmp/hf_cache"
        from pathlib import Path
        assert config.cache_dir == Path("/tmp/hf_cache")
