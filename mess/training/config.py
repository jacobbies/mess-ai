"""Configuration objects for retrieval-augmented SSL training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RetrievalSSLConfig:
    """Hyperparameters for projection-head retrieval-augmented training."""

    num_steps: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07

    projection_dim: int = 256
    hidden_dim: int | None = 512

    search_k: int = 128
    positives_per_query: int = 4
    negatives_per_query: int = 16
    min_time_separation_sec: float = 5.0
    require_cross_recording_positive: bool = False
    exclude_same_recording_negative: bool = True

    refresh_every: int = 50
    ema_decay: float = 0.995

    seed: int = 42
    device: str = "cpu"
    train_splits: tuple[str, ...] = ("train",)

    def validate(self) -> None:
        """Validate training hyperparameters."""
        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be > 0")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0 when provided")
        if self.search_k <= 1:
            raise ValueError("search_k must be > 1")
        if self.positives_per_query <= 0:
            raise ValueError("positives_per_query must be > 0")
        if self.negatives_per_query <= 0:
            raise ValueError("negatives_per_query must be > 0")
        if self.min_time_separation_sec < 0:
            raise ValueError("min_time_separation_sec must be >= 0")
        if self.refresh_every <= 0:
            raise ValueError("refresh_every must be > 0")
        if not 0 < self.ema_decay < 1:
            raise ValueError("ema_decay must be in (0, 1)")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if not self.train_splits:
            raise ValueError("train_splits cannot be empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize config to a plain dict."""
        return asdict(self)
