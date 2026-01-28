"""
MAESTRO Dataset implementation
"""
from pathlib import Path
from .base import BaseDataset


class MAESTRODataset(BaseDataset):
    """
    MAESTRO Dataset - Large-scale classical piano performances.

    Self-contained dataset with its own directory structure:
    - Audio: data_root/audio/maestro/
    - Embeddings: data_root/embeddings/maestro-emb/
    """

    @property
    def audio_dir(self) -> Path:
        """Directory containing MAESTRO audio files."""
        return self.data_root / "audio" / "maestro"

    @property
    def embeddings_dir(self) -> Path:
        """Directory for MAESTRO MERT embeddings."""
        return self.data_root / "embeddings" / "maestro-emb"

    @property
    def name(self) -> str:
        return "MAESTRO Dataset"

    @property
    def description(self) -> str:
        return "Classical piano performances dataset (MIDI Aligned Edited Synchronized TRack of Orchestral)"