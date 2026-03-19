"""Configuration for segment transformer contextualizer training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ContextualizerConfig:
    """Hyperparameters for segment transformer contextualizer training."""

    # Model architecture
    input_dim: int = 768
    context_dim: int = 256
    num_transformer_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 512
    max_segments: int = 512
    dropout: float = 0.1
    pool_mode: str = "mean"

    # Input layer selection
    input_layer: int | None = None

    # Training
    num_steps: int = 2000
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    temperature: float = 0.07

    # Dual loss weights
    global_loss_weight: float = 1.0
    local_loss_weight: float = 0.5

    # Mining
    search_k: int = 64
    positives_per_query: int = 1
    negatives_per_query: int = 8
    min_time_separation_sec: float = 5.0
    require_cross_recording_positive: bool = False
    exclude_same_recording_negative: bool = True

    # Index / EMA
    refresh_every: int = 100
    ema_decay: float = 0.995

    # Reproducibility
    seed: int = 42
    device: str = "cpu"
    train_splits: tuple[str, ...] = ("train",)

    def validate(self) -> None:
        """Validate contextualizer hyperparameters."""
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be > 0")
        if self.num_transformer_layers <= 0:
            raise ValueError("num_transformer_layers must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if self.context_dim % self.num_heads != 0:
            raise ValueError("context_dim must be divisible by num_heads")
        if self.ff_dim <= 0:
            raise ValueError("ff_dim must be > 0")
        if self.max_segments <= 0:
            raise ValueError("max_segments must be > 0")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.pool_mode not in {"mean", "cls"}:
            raise ValueError("pool_mode must be 'mean' or 'cls'")
        if self.input_layer is not None and not 0 <= self.input_layer <= 12:
            raise ValueError("input_layer must be in [0, 12] or None")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if self.global_loss_weight < 0:
            raise ValueError("global_loss_weight must be >= 0")
        if self.local_loss_weight < 0:
            raise ValueError("local_loss_weight must be >= 0")
        if self.global_loss_weight == 0 and self.local_loss_weight == 0:
            raise ValueError("At least one loss weight must be > 0")
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
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps")
        if not self.train_splits:
            raise ValueError("train_splits cannot be empty")

    def to_dict(self) -> dict[str, object]:
        """Serialize config to a plain dict."""
        return asdict(self)
