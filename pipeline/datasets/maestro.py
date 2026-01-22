"""
MAESTRO Dataset implementation
"""
from pathlib import Path
from typing import List
from .base import BaseDataset


class MAESTRODataset(BaseDataset):
    """MAESTRO Dataset - Large-scale classical piano performances."""

    def get_audio_files(self) -> List[Path]:
        """Get list of MAESTRO audio files."""
        wav_dir = self.data_dir / "maestro" / "wav"

        if not wav_dir.exists():
            return []

        # Return sorted list of .wav files
        return sorted(wav_dir.glob("*.wav"))

    @property
    def name(self) -> str:
        return "MAESTRO Dataset"

    @property
    def description(self) -> str:
        return "Classical piano performances dataset (MIDI Aligned Edited Synchronized TRack of Orchestral)"