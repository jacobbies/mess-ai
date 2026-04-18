"""Root test fixtures for mess-ai test suite."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from mess.config import MESSConfig

SYSTEM_MARKER_FILES = {
    "test_setup_demo_data_script.py",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu", action="store_true", default=False, help="Run GPU tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        skip_gpu = None
    else:
        skip_gpu = pytest.mark.skip(reason="Need --run-gpu option to run")

    for item in items:
        if skip_gpu is not None and "gpu" in item.keywords:
            item.add_marker(skip_gpu)

        path = Path(str(item.fspath))
        if "library" not in item.keywords and "system" not in item.keywords:
            is_system = (
                "workflow" in item.keywords
                or "workflows" in path.parts
                or path.name in SYSTEM_MARKER_FILES
            )
            item.add_marker(pytest.mark.system if is_system else pytest.mark.library)


@pytest.fixture
def sample_audio_array():
    """1-second 24kHz sine wave as float32 array."""
    sr = 24000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def sample_embeddings():
    """Dict of 5 tracks with [13, 768] aggregated embeddings."""
    rng = np.random.default_rng(42)
    tracks = [f"track_{i:02d}" for i in range(5)]
    return {
        name: rng.standard_normal((13, 768)).astype(np.float32)
        for name in tracks
    }


@pytest.fixture
def sample_discovery_results():
    """Synthetic discovery results with known best layers."""
    results = {}
    targets = ["spectral_centroid", "spectral_rolloff", "tempo"]
    for layer in range(13):
        results[layer] = {}
        for target in targets:
            results[layer][target] = {
                "r2_score": 0.3 + 0.01 * layer,
                "correlation": 0.5 + 0.01 * layer,
                "rmse": 1.0 - 0.01 * layer,
            }
    # Make layer 0 best for spectral_centroid (R2=0.95)
    results[0]["spectral_centroid"]["r2_score"] = 0.95
    # Make layer 5 best for spectral_rolloff (R2=0.85)
    results[5]["spectral_rolloff"]["r2_score"] = 0.85
    # Make layer 12 best for tempo (R2=0.45) — below medium threshold
    results[12]["tempo"]["r2_score"] = 0.45
    return results


@pytest.fixture
def isolated_config(tmp_path):
    """MESSConfig with project_root pointed at tmp_path."""
    config = MESSConfig()
    config.project_root = tmp_path
    return config
