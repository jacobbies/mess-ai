"""Fixtures for extraction tests."""

import numpy as np
import pytest


@pytest.fixture
def long_audio_array():
    """10-second 24kHz sine wave for segmentation tests."""
    sr = 24000
    t = np.linspace(0, 10.0, sr * 10, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t)
