"""
Saarland Music Dataset (SMD) implementation
"""
from pathlib import Path

from .base import BaseDataset


class SMDDataset(BaseDataset):
    """
    Saarland Music Dataset - 50 classical piano pieces at 44kHz.

    Self-contained dataset with its own directory structure:
    - Audio: data_root/audio/smd/wav-44/
    - Embeddings: data_root/embeddings/smd-emb/
    """

    @property
    def dataset_id(self) -> str:
        return "smd"

    @property
    def audio_dir(self) -> Path:
        """Directory containing SMD audio files."""
        return self.data_root / "audio" / "smd" / "wav-44"

    @property
    def embeddings_dir(self) -> Path:
        """Directory for SMD MERT embeddings."""
        return self.data_root / "embeddings" / "smd-emb"

    @property
    def name(self) -> str:
        return "Saarland Music Dataset (SMD)"

    @property
    def description(self) -> str:
        return "Classical piano dataset with 50 recordings (44kHz WAV)"
