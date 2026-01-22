"""
Saarland Music Dataset (SMD) implementation
"""
from pathlib import Path
from typing import List
from .base import BaseDataset


class SMDDataset(BaseDataset):
    """Saarland Music Dataset - 50 classical piano pieces at 44kHz."""

    def get_audio_files(self) -> List[Path]:
        """Get list of SMD audio files."""
        wav_dir = self.data_dir / "smd" / "wav-44"

        if not wav_dir.exists():
            return []

        # Return sorted list of .wav files
        return sorted(wav_dir.glob("*.wav"))

    @property
    def name(self) -> str:
        return "Saarland Music Dataset (SMD)"

    @property
    def description(self) -> str:
        return "Classical piano dataset with 50 recordings (44kHz WAV)"