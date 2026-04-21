"""
MERT Layer Discovery: find what each layer encodes via linear probing.

Uses Ridge regression with cross-validation to test whether frozen MERT
layer embeddings linearly predict musical descriptors (proxy targets).
High R² means the layer explicitly encodes that musical aspect.

Also provides model inspection utilities (inventory, activation tracing).

Usage:
    from mess.probing.discovery import LayerDiscoverySystem

    discovery = LayerDiscoverySystem()
    results = discovery.discover(n_samples=50)
    # results[layer][target] = {'r2_score': 0.944, 'correlation': 0.97, 'rmse': 0.15}

    best = LayerDiscoverySystem.best_layers(results)
    # best['spectral_centroid'] = {'layer': 0, 'r2_score': 0.944, 'confidence': 'high'}
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..config import mess_config

logger = logging.getLogger(__name__)
mlflow: Any

ProbeMode = Literal["best_layer", "weighted_sum", "both"]

try:
    import mlflow  # type: ignore[import-not-found]
except ModuleNotFoundError:
    class _MlflowStub:
        """No-op MLflow shim when mlflow is not installed."""

        @staticmethod
        def active_run() -> None:
            return None

        @staticmethod
        def log_params(_params: dict[str, Any]) -> None:
            return None

        @staticmethod
        def log_metrics(_metrics: dict[str, float]) -> None:
            return None

        @staticmethod
        def log_metric(_name: str, _value: float) -> None:
            return None

        @staticmethod
        def log_artifact(_path: str) -> None:
            return None
    mlflow = _MlflowStub()

NUM_LAYERS = 13
EMBEDDING_DIM = 768


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D vector."""
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp))


@lru_cache(maxsize=1)
def _require_sklearn() -> tuple[Any, ...]:
    """Import sklearn probe primitives lazily for lightweight serving installs."""
    try:
        from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
        from sklearn.metrics import mean_squared_error, r2_score  # type: ignore[import-untyped]
        from sklearn.model_selection import (  # type: ignore[import-untyped]
            GroupKFold,
            KFold,
            cross_val_predict,
        )
        from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for layer probing. "
            "Install with `mess-ai[ml]`."
        ) from exc

    return (
        GroupKFold,
        Ridge,
        mean_squared_error,
        r2_score,
        KFold,
        cross_val_predict,
        make_pipeline,
        StandardScaler,
    )

# Targets viable at 5s segment duration for segment-level probing.
# Excludes tempo (needs >5s for beat tracking), phrase_regularity,
# num_phrases (need whole piece), onset_density (too noisy at 5s).
SEGMENT_TARGETS: set[str] = {
    'spectral_centroid',
    'spectral_rolloff',
    'spectral_bandwidth',
    'zero_crossing_rate',
    'dynamic_range',
    'dynamic_variance',
    'crescendo_strength',
    'diminuendo_strength',
    'harmonic_complexity',
    'attack_slopes',
    'attack_sharpness',
}


# =============================================================================
# Model inspection utilities
# =============================================================================

def inspect_model(model_name: str | None = None) -> dict[str, Any]:
    """
    Inventory MERT model: layer names, parameter counts, shapes.

    Returns dict with total_params, trainable_params, and per-module breakdown.
    Requires transformers and torch dependencies.
    """
    from transformers import AutoModel  # type: ignore[import-untyped]

    name = model_name or mess_config.model_name
    model = AutoModel.from_pretrained(name, trust_remote_code=True)

    inventory: dict[str, Any] = {
        'model_name': name,
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'modules': [],
    }

    for mod_name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params > 0:
            shapes = {
                pname: list(p.shape)
                for pname, p in module.named_parameters(recurse=False)
            }
            inventory['modules'].append({
                'name': mod_name,
                'type': type(module).__name__,
                'params': params,
                'shapes': shapes,
            })

    return inventory


def trace_activations(
    audio_path: str,
    model_name: str | None = None,
    segment_duration: float = 5.0,
) -> dict[str, Any]:
    """
    Run a forward pass through MERT and capture activation shapes per hidden layer.

    Useful for verifying the model produces expected output dimensions before
    running full feature extraction.

    Requires transformers, torch, and torchaudio dependencies.
    """
    import torch
    import torchaudio  # type: ignore[import-untyped]
    from transformers import (  # type: ignore[import-untyped]
        AutoModel,
        Wav2Vec2FeatureExtractor,
    )

    name = model_name or mess_config.model_name
    processor = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, trust_remote_code=True, output_hidden_states=True)
    model.eval()

    audio, sr = torchaudio.load(audio_path)
    if sr != mess_config.target_sample_rate:
        audio = torchaudio.transforms.Resample(sr, mess_config.target_sample_rate)(audio)
    audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio.squeeze(0)

    # Take first segment only
    n_samples = int(segment_duration * mess_config.target_sample_rate)
    audio_np = audio[:n_samples].numpy()

    inputs = processor(audio_np, sampling_rate=mess_config.target_sample_rate, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states

    return {
        'audio_path': audio_path,
        'input_shape': list(inputs['input_values'].shape),
        'num_hidden_layers': len(hidden_states),
        'hidden_state_shapes': [list(h.shape) for h in hidden_states],
        'output_shape': list(outputs.last_hidden_state.shape),
    }


# =============================================================================
# Core layer discovery via linear probing
# =============================================================================

class LayerDiscoverySystem:
    """
    Discover what musical aspects each MERT layer encodes using linear probing.

    For each (layer, proxy target) pair, trains a Ridge regression on frozen
    embeddings and reports cross-validated R², correlation, and RMSE.

    R² interpretation:
        >0.9  Excellent - use this layer confidently
        >0.8  Good - use with awareness
        >0.7  Promising - experimental use
        <0.5  Weak - avoid
    """

    # Scalar targets to extract from proxy target npz files.
    # Format: target_name -> (category_key, field_key, reduction)
    SCALAR_TARGETS = {
        # Timbre (audio-derived)
        'spectral_centroid':   ('timbre', 'spectral_centroid', 'mean'),
        'spectral_rolloff':    ('timbre', 'spectral_rolloff', 'mean'),
        'spectral_bandwidth':  ('timbre', 'spectral_bandwidth', 'mean'),
        'zero_crossing_rate':  ('timbre', 'zero_crossing_rate', 'mean'),
        # Rhythm (audio-derived)
        'tempo':               ('rhythm', 'tempo', 'first'),
        'onset_density':       ('rhythm', 'onset_density', 'first'),
        # Dynamics (audio-derived)
        'dynamic_range':       ('dynamics', 'dynamic_range', 'first'),
        'dynamic_variance':    ('dynamics', 'dynamic_variance', 'first'),
        'crescendo_strength':  ('dynamics', 'crescendo_strength', 'first'),
        'diminuendo_strength': ('dynamics', 'diminuendo_strength', 'first'),
        # Harmony (audio-derived)
        'harmonic_complexity': ('harmony', 'harmonic_complexity', 'first'),
        # Articulation (audio-derived)
        'attack_slopes':       ('articulation', 'attack_slopes', 'mean'),
        'attack_sharpness':    ('articulation', 'attack_sharpness', 'mean'),
        # Phrasing (audio-derived)
        'phrase_regularity':   ('phrasing', 'phrase_regularity', 'first'),
        'num_phrases':         ('phrasing', 'num_phrases', 'first'),
        # Expression (MIDI-derived, optional)
        'rubato':              ('expression', 'rubato', 'first'),
        'velocity_mean':       ('expression', 'velocity_mean', 'first'),
        'velocity_std':        ('expression', 'velocity_std', 'first'),
        'velocity_range':      ('expression', 'velocity_range', 'first'),
        'articulation_ratio':  ('expression', 'articulation_ratio', 'first'),
        'tempo_variability':   ('expression', 'tempo_variability', 'first'),
        'onset_timing_std':    ('expression', 'onset_timing_std', 'first'),
    }

    # Categories where missing data is expected (tracks without MIDI).
    # Tracks missing optional targets are still included for required targets;
    # optional values are stored as NaN and filtered before probing.
    OPTIONAL_CATEGORIES: set[str] = {'expression'}

    def __init__(
        self,
        dataset_name: str = "smd",
        alpha: float = 1.0,
        n_folds: int = 5,
        probe_mode: ProbeMode = "both",
    ):
        from mess.datasets.factory import DatasetFactory

        self.dataset = DatasetFactory.get_dataset(dataset_name)
        self.features_dir = self.dataset.embeddings_dir / "raw"
        self.targets_dir = mess_config.proxy_targets_dir
        # Optional explicit override for segment-level targets directory.
        # When unset, segment discovery uses config defaults while still
        # honoring custom ``targets_dir`` overrides in tests/workflows.
        self.segment_targets_dir: Path | None = None
        self.alpha = alpha
        self.n_folds = n_folds
        self.probe_mode: ProbeMode = probe_mode

    def load_features(self, audio_files: list[str]) -> tuple[dict[int, np.ndarray], list[str]]:
        """Load MERT layer embeddings, averaged across segments and time steps."""
        per_layer: dict[int, list[np.ndarray]] = {
            layer_idx: [] for layer_idx in range(NUM_LAYERS)
        }
        loaded: list[str] = []

        for path in audio_files:
            feat_file = self.features_dir / f"{Path(path).stem}.npy"
            if not feat_file.exists():
                continue
            raw = np.load(feat_file)  # [segments, 13, time, 768]
            for layer in range(NUM_LAYERS):
                per_layer[layer].append(raw[:, layer, :, :].mean(axis=(0, 1)))
            loaded.append(path)

        features = {layer_idx: np.array(values) for layer_idx, values in per_layer.items()}
        logger.info(f"Loaded features for {len(loaded)}/{len(audio_files)} files")
        return features, loaded

    def load_targets(self, audio_files: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        """Load proxy targets (legacy scalars + registered curve/MIDI) for probing."""
        collectors: dict[str, list[float]] = {name: [] for name in self.SCALAR_TARGETS}
        loaded: list[str] = []

        for path in audio_files:
            target_file = self.targets_dir / f"{Path(path).stem}_targets.npz"
            if not target_file.exists():
                continue

            data = np.load(target_file, allow_pickle=True)
            row: dict[str, float] = {}
            ok = True

            for name, (category, key, reduction) in self.SCALAR_TARGETS.items():
                try:
                    cat_data = data[category].item()
                    val = cat_data[key]
                    if isinstance(val, np.ndarray):
                        val = float(np.mean(val)) if reduction == 'mean' else float(val.flat[0])
                    else:
                        val = float(val)
                    row[name] = val
                except (KeyError, IndexError, TypeError):
                    if category in self.OPTIONAL_CATEGORIES:
                        row[name] = float('nan')
                    else:
                        ok = False
                        break

            if ok:
                for name in self.SCALAR_TARGETS:
                    collectors[name].append(row.get(name, float('nan')))
                loaded.append(path)

        # Only keep targets with variance (constant targets can't be probed).
        # Optional categories may contain NaN for tracks without that metadata.
        targets: dict[str, np.ndarray] = {}
        for name, values in collectors.items():
            arr = np.array(values, dtype=float)
            valid = arr[~np.isnan(arr)]

            if valid.size > 0 and np.std(valid) > 1e-10:
                targets[name] = arr
            elif valid.size == 0:
                logger.warning(f"Skipping target '{name}': no valid values")
            elif len(arr) > 0:
                logger.warning(f"Skipping target '{name}': constant values")

        # Layer on registered curve/MIDI targets (Phase 2). Missing fields
        # become NaN rows and get dropped per-track by ``_row_valid_mask``.
        registered = self._load_registered_targets(loaded)
        targets.update(registered)

        logger.info(f"Loaded {len(targets)} targets for {len(loaded)} files")
        return targets, loaded

    def _load_registered_targets(
        self, audio_files: list[str],
    ) -> dict[str, np.ndarray]:
        """Load every descriptor-registered target for the given tracks.

        Iterates ``mess.probing.targets._registry.all_names()``; each entry
        dispatches through ``_schema.load_target_field`` which handles
        SCALAR / CURVE / MIDI_SCALAR / MIDI_CURVE types uniformly.

        Missing fields (no ``.npz``, missing key, shape mismatch, MIDI
        unavailable) produce NaN rows that ``_row_valid_mask`` drops at
        probe time. Targets with zero valid rows are skipped entirely.
        """
        from ._schema import TargetType, load_target_field
        from .targets._registry import _GENERATORS, all_names

        out: dict[str, np.ndarray] = {}
        names = all_names()
        if not names:
            return out

        n_folds = getattr(self, "n_folds", 5)
        for name in names:
            descriptor = _GENERATORS[name][0]
            rows: list[np.ndarray | None] = []
            for path in audio_files:
                npz_path = self.targets_dir / f"{Path(path).stem}_targets.npz"
                rows.append(load_target_field(npz_path, descriptor))

            n_frames = (
                descriptor.curve_spec.n_frames
                if descriptor.curve_spec is not None
                else 1
            )
            is_curve = descriptor.type in (TargetType.CURVE, TargetType.MIDI_CURVE)

            if is_curve:
                matrix = np.full((len(audio_files), n_frames), np.nan, dtype=float)
                for i, row in enumerate(rows):
                    if row is not None and row.shape == (n_frames,):
                        matrix[i] = row
                n_valid = int(np.sum(~np.all(np.isnan(matrix), axis=1)))
                if n_valid >= n_folds:
                    out[name] = matrix
                else:
                    logger.warning(
                        f"Skipping registered target '{name}': only "
                        f"{n_valid} valid rows (need >= {n_folds})"
                    )
            else:
                vector = np.full(len(audio_files), np.nan, dtype=float)
                for i, row in enumerate(rows):
                    if row is not None and row.size > 0:
                        vector[i] = float(row.flat[0])
                valid = vector[~np.isnan(vector)]
                if valid.size >= n_folds and np.std(valid) > 1e-10:
                    out[name] = vector
                else:
                    logger.warning(
                        f"Skipping registered target '{name}': "
                        f"{valid.size} valid, std={np.std(valid):.3g}"
                    )
        return out

    def _row_valid_mask(self, y: np.ndarray) -> np.ndarray:
        """Return a row-level valid mask for scalar or curve targets."""
        if y.ndim == 1:
            mask = ~np.isnan(y)
        else:
            # Curve: drop rows that are entirely NaN (MIDI unavailable / missing).
            mask = ~np.all(np.isnan(y), axis=1)
        return np.asarray(mask, dtype=bool)

    def _make_cv(
        self,
        n: int,
        groups: np.ndarray | None,
    ) -> tuple[Any, bool]:
        """Build the CV splitter shared by scalar and curve paths."""
        (GroupKFold, _Ridge, _mse, _r2, KFold, *_rest) = _require_sklearn()
        if groups is not None:
            n_groups = int(np.unique(groups).size)
            if n_groups >= 2:
                return GroupKFold(n_splits=min(self.n_folds, n_groups)), True
            logger.warning(
                "Segment probe has only one track group after filtering; "
                "falling back to ungrouped KFold."
            )
        return KFold(n_splits=min(self.n_folds, n), shuffle=True, random_state=42), False

    def _probe_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Cross-validated Ridge regression for one (layer, target) pair.

        Dispatches on ``y`` shape:

        * ``(N,)``: scalar Ridge. Returns ``{r2_score, correlation, rmse}``
          (unchanged from pre-F1 behavior).
        * ``(N, T)``: multi-output Ridge over a curve target. Returns
          ``{r2_mean, r2_pc1, rmse_mean}``.

        NaN rows (whole-row for curves, element-wise for scalars) are
        dropped before probing. When rows are dropped the dict also includes
        ``n_valid`` and ``coverage`` so MIDI-backed targets expose mask
        statistics to the caller.
        """
        n_total = len(y)
        valid = self._row_valid_mask(y)
        dropped_any = not bool(np.all(valid))
        if dropped_any:
            X, y = X[valid], y[valid]
            if groups is not None:
                groups = groups[valid]

        n = len(X)
        if y.ndim == 1:
            metrics = self._probe_scalar(X, y, groups, n)
        else:
            metrics = self._probe_curve(X, y, groups, n)

        if dropped_any:
            metrics["n_valid"] = float(n)
            metrics["coverage"] = float(n / n_total) if n_total else 0.0
        return metrics

    def _probe_scalar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None,
        n: int,
    ) -> dict[str, float]:
        """Scalar Ridge probe — preserves exact pre-F1 output shape."""
        if n < self.n_folds:
            return {"r2_score": -999.0, "correlation": 0.0, "rmse": 999.0}

        (
            _GroupKFold,
            Ridge,
            mean_squared_error,
            r2_score,
            _KFold,
            cross_val_predict,
            make_pipeline,
            StandardScaler,
        ) = _require_sklearn()

        cv, uses_grouped = self._make_cv(n, groups)
        probe = make_pipeline(
            StandardScaler(),
            Ridge(alpha=self.alpha, random_state=42),
        )
        if uses_grouped and groups is not None:
            y_pred = cross_val_predict(probe, X, y, cv=cv, groups=groups)
        else:
            y_pred = cross_val_predict(probe, X, y, cv=cv)

        r2 = float(r2_score(y, y_pred))
        corr = (
            float(np.corrcoef(y, y_pred)[0, 1])
            if np.std(y) > 1e-10 and np.std(y_pred) > 1e-10
            else 0.0
        )
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        return {"r2_score": r2, "correlation": corr, "rmse": rmse}

    def _probe_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None,
        n: int,
    ) -> dict[str, float]:
        """Multi-output Ridge probe for curve-valued targets.

        Reports ``r2_mean`` (mean per-frame R²), ``r2_pc1`` (R² against
        the first principal component of ``y``, with PCA fit inside each
        training fold to avoid leakage), and ``rmse_mean``.
        """
        bad = {"r2_mean": -999.0, "r2_pc1": -999.0, "rmse_mean": 999.0}
        if n < self.n_folds:
            return bad

        (
            _GroupKFold,
            Ridge,
            mean_squared_error,
            r2_score,
            _KFold,
            _cross_val_predict,
            make_pipeline,
            StandardScaler,
        ) = _require_sklearn()

        cv, uses_grouped = self._make_cv(n, groups)
        split_iter = cv.split(X, y, groups) if uses_grouped else cv.split(X, y)
        splits = list(split_iter)
        if not splits:
            return bad

        y_pred_curve = np.full_like(y, fill_value=np.nan)
        pc1_true = np.full(n, fill_value=np.nan)
        pc1_pred = np.full(n, fill_value=np.nan)

        for train_idx, test_idx in splits:
            y_train = y[train_idx]
            curve_probe = make_pipeline(
                StandardScaler(),
                Ridge(alpha=self.alpha, random_state=42),
            )
            curve_probe.fit(X[train_idx], y_train)
            y_pred_curve[test_idx] = curve_probe.predict(X[test_idx])

            # PC1 axis is fit on train-fold y only to avoid leaking test
            # variance into the projection.
            train_mean = y_train.mean(axis=0, keepdims=True)
            _u, _s, vt = np.linalg.svd(y_train - train_mean, full_matrices=False)
            axis = vt[0]
            pc1_probe = make_pipeline(
                StandardScaler(),
                Ridge(alpha=self.alpha, random_state=42),
            )
            pc1_probe.fit(X[train_idx], (y_train - train_mean) @ axis)
            pc1_true[test_idx] = (y[test_idx] - train_mean) @ axis
            pc1_pred[test_idx] = pc1_probe.predict(X[test_idx])

        per_frame_r2 = r2_score(y, y_pred_curve, multioutput="raw_values")
        r2_mean = float(np.mean(per_frame_r2))
        rmse_mean = float(np.sqrt(mean_squared_error(y, y_pred_curve)))
        r2_pc1 = float(r2_score(pc1_true, pc1_pred)) if np.std(pc1_true) > 1e-10 else 0.0
        return {"r2_mean": r2_mean, "r2_pc1": r2_pc1, "rmse_mean": rmse_mean}

    def _single_layer_r2(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None,
        n: int,
    ) -> float:
        """R² for a single layer — dispatches on ``y`` shape."""
        if y.ndim == 1:
            return float(self._probe_scalar(X, y, groups, n).get("r2_score", -np.inf))
        return float(self._probe_curve(X, y, groups, n).get("r2_mean", -np.inf))

    def _probe_weighted_sum(
        self,
        X_all_layers: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """SUPERB-style weighted-sum probe across layers.

        ``X_all_layers`` has shape ``(N, 13, 768)``. We parameterise layer
        weights as ``softmax(w / tau)`` and fuse as
        ``sum_j softmax(w)[j] * X_all_layers[:, j, :]``. A small grid over
        sparsity temperatures (``tau``) and a starting weight vector selects
        the configuration that maximises K-Fold R²; the reported metrics use
        the selected fused features with the same CV pipeline as the per-
        layer probes.

        Supports scalar (``y.ndim==1``), curve (``y.ndim==2``), and MIDI-
        masked (NaN rows) targets via the same dispatch as ``_probe_single``.

        Returns a dict with:
            * ``r2_score`` — K-Fold CV R² of the fused probe (mean across
              frames for curves).
            * ``layer_weights`` — list of 13 floats summing to 1.
            * ``correlation`` — Pearson correlation of the fused probe
              (``0.0`` for curve targets where a single scalar correlation
              is not defined).
            * ``rmse`` — mirrors ``_probe_scalar``/``_probe_curve`` for the
              fused features.
            * ``r2_gain_over_best_single`` — fused R² minus the best
              single-layer R².
        """
        n_total = len(y)
        valid = self._row_valid_mask(y)
        dropped_any = not bool(np.all(valid))
        if dropped_any:
            X_all_layers = X_all_layers[valid]
            y = y[valid]
            if groups is not None:
                groups = groups[valid]

        n = len(X_all_layers)
        bad: dict[str, Any] = {
            "r2_score": -999.0,
            "layer_weights": [1.0 / NUM_LAYERS] * NUM_LAYERS,
            "correlation": 0.0,
            "rmse": 999.0,
            "r2_gain_over_best_single": 0.0,
        }
        if n < self.n_folds:
            if dropped_any:
                bad["n_valid"] = float(n)
                bad["coverage"] = float(n / n_total) if n_total else 0.0
            return bad

        # Per-layer baseline R² — serves both the gain term and a warm-start
        # vector for the temperature grid search below.
        per_layer_r2 = np.array(
            [
                self._single_layer_r2(X_all_layers[:, layer, :], y, groups, n)
                for layer in range(NUM_LAYERS)
            ],
            dtype=float,
        )
        best_single_r2 = float(per_layer_r2.max(initial=-np.inf))

        # Grid search over sparsity temperatures — cheap, interpretable, and
        # avoids the bookkeeping headache of nested-CV L-BFGS. Candidates:
        #   * uniform: ``softmax(0) = 1/13`` — mean across layers.
        #   * warm-start: softmax over per-layer R² — tilts toward the
        #     layers that individually probe well.
        #   * spike on argmax: guarantees the fused probe matches the best
        #     single layer as ``tau → 0``.
        warm_start = np.clip(per_layer_r2, 0.0, None)
        if not np.any(warm_start > 0):
            warm_start = np.zeros(NUM_LAYERS)
        spike = np.zeros(NUM_LAYERS)
        spike[int(np.argmax(per_layer_r2))] = 1.0

        best_metrics: dict[str, Any] = dict(bad)
        best_metrics["r2_score"] = -np.inf
        for tau in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
            for base in (np.zeros(NUM_LAYERS), warm_start, spike):
                weights = _softmax(base / tau)
                fused = np.einsum("j,njk->nk", weights, X_all_layers)
                m = (
                    self._probe_scalar(fused, y, groups, n)
                    if y.ndim == 1
                    else self._probe_curve(fused, y, groups, n)
                )
                r2 = float(m.get("r2_score", m.get("r2_mean", -np.inf)))
                if r2 > best_metrics["r2_score"]:
                    best_metrics = {
                        "r2_score": r2,
                        "layer_weights": [float(w) for w in weights],
                        "correlation": float(m.get("correlation", 0.0)),
                        "rmse": float(m.get("rmse", m.get("rmse_mean", 0.0))),
                    }

        best_metrics["r2_gain_over_best_single"] = float(
            best_metrics["r2_score"] - best_single_r2
        )
        if dropped_any:
            best_metrics["n_valid"] = float(n)
            best_metrics["coverage"] = float(n / n_total) if n_total else 0.0
        return best_metrics

    def discover(
        self,
        n_samples: int = 50,
        probe_mode: ProbeMode | None = None,
    ) -> dict[Any, dict[str, dict[str, float]]]:
        """
        Run full layer discovery: probe all 13 layers against all proxy targets.

        Logs all parameters and metrics to MLflow if a run is active.

        Args:
            n_samples: Cap on audio files to probe.
            probe_mode: Override the instance-level :attr:`probe_mode`. When
                ``"best_layer"`` only the per-layer loop runs; when
                ``"weighted_sum"`` only the SUPERB-style fused probe runs;
                ``"both"`` (default) runs both.

        Returns:
            ``{layer_idx: {target: metrics}, ...}`` with integer layer keys
            for the per-layer section. When ``probe_mode`` includes
            ``weighted_sum``, a top-level ``"weighted_sum"`` key is also
            present with the same target->metrics shape.
        """
        mode: ProbeMode = probe_mode or getattr(self, "probe_mode", "both")  # type: ignore[assignment]
        audio_files = sorted(str(f) for f in self.dataset.get_audio_files())[:n_samples]
        logger.info(f"Running discovery with up to {len(audio_files)} audio files")

        features, feat_files = self.load_features(audio_files)
        targets, tgt_files = self.load_targets(audio_files)

        # Align to files that have both features and targets
        common = sorted(set(feat_files) & set(tgt_files))
        if len(common) < self.n_folds:
            logger.error(f"Only {len(common)} common files, need >= {self.n_folds}")
            return {}

        # Align feature/target rows by the same sorted `common` order.
        # Membership-only filtering can misalign rows if load order differs.
        feat_pos = {path: idx for idx, path in enumerate(feat_files)}
        tgt_pos = {path: idx for idx, path in enumerate(tgt_files)}
        feat_idx = [feat_pos[path] for path in common]
        tgt_idx = [tgt_pos[path] for path in common]

        features = {
            layer_idx: values[feat_idx] for layer_idx, values in features.items()
        }
        targets = {name: v[tgt_idx] for name, v in targets.items()}

        n_tracks = len(common)

        # Require enough valid rows per target after NaN filtering.
        min_valid_samples = self.n_folds
        valid_counts: dict[str, int] = {}
        filtered_targets: dict[str, np.ndarray] = {}
        for name, values in targets.items():
            n_valid = int(np.sum(~np.isnan(values)))
            if n_valid < min_valid_samples:
                logger.warning(
                    f"Skipping target '{name}': only {n_valid} valid samples "
                    f"(need >= {min_valid_samples})"
                )
                continue
            filtered_targets[name] = values
            valid_counts[name] = n_valid

        targets = filtered_targets
        n_targets = len(targets)
        if n_targets == 0:
            logger.error(
                "No probeable targets after validity filtering "
                f"(need >= {min_valid_samples} valid samples per target)"
            )
            return {}

        logger.info(f"Probing {NUM_LAYERS} layers x {n_targets} targets on {n_tracks} tracks")

        # Log experiment parameters to MLflow
        if mlflow.active_run():
            mlflow.log_params({
                'alpha': self.alpha,
                'n_folds': self.n_folds,
                'n_samples': n_samples,
                'n_tracks_used': n_tracks,
                'n_targets': n_targets,
                'dataset': self.dataset.name if hasattr(self.dataset, 'name') else 'unknown',
                'targets': ','.join(sorted(targets.keys())),
                'probe_mode': mode,
            })
            try:  # Some mlflow stubs don't support set_tag.
                mlflow.set_tag("probe_mode", mode)
            except AttributeError:
                pass

        results: dict[Any, dict[str, dict[str, float]]] = {}

        run_best_layer = mode in ("best_layer", "both")
        run_weighted_sum = mode in ("weighted_sum", "both")

        if run_best_layer:
            for layer in range(NUM_LAYERS):
                results[layer] = {}
                for target_name, target_values in targets.items():
                    metrics = self._probe_single(features[layer], target_values)
                    n_valid = valid_counts[target_name]
                    metrics.setdefault('n_valid', float(n_valid))
                    metrics.setdefault('coverage', float(n_valid / n_tracks))
                    results[layer][target_name] = metrics

                    r2 = metrics['r2_score']
                    marker = (
                        ' ***' if r2 > 0.9 else ' **' if r2 > 0.8 else ' *' if r2 > 0.7 else ''
                    )
                    logger.info(
                        f"  Layer {layer:2d} | {target_name:25s} | "
                        f"R²={r2:7.4f}  corr={metrics['correlation']:6.3f}  "
                        f"RMSE={metrics['rmse']:8.4f}  "
                        f"n_valid={n_valid:4d}  cov={metrics['coverage']:.3f}{marker}"
                    )

                    # Log per-(layer, target) metrics to MLflow
                    if mlflow.active_run():
                        prefix = f"L{layer}_{target_name}"
                        mlflow.log_metrics({
                            f"{prefix}_r2": metrics['r2_score'],
                            f"{prefix}_corr": metrics['correlation'],
                            f"{prefix}_rmse": metrics['rmse'],
                            f"{prefix}_n_valid": metrics['n_valid'],
                            f"{prefix}_coverage": metrics['coverage'],
                        })

            # Log best-layer summary metrics
            if mlflow.active_run():
                best = self.best_layers(results)
                for target_name, info in best.items():
                    mlflow.log_metric(f"best_r2_{target_name}", info['r2_score'])
                    mlflow.log_metric(f"best_layer_{target_name}", info['layer'])

        if run_weighted_sum:
            # Stack per-layer features into (N, 13, 768) for the fused probe.
            stacked = np.stack(
                [features[layer] for layer in range(NUM_LAYERS)], axis=1
            )
            ws_results: dict[str, dict[str, Any]] = {}
            for target_name, target_values in targets.items():
                metrics = self._probe_weighted_sum(stacked, target_values)
                n_valid = valid_counts[target_name]
                metrics.setdefault("n_valid", float(n_valid))
                metrics.setdefault("coverage", float(n_valid / n_tracks))
                ws_results[target_name] = metrics

                r2 = metrics["r2_score"]
                marker = ' ***' if r2 > 0.9 else ' **' if r2 > 0.8 else ' *' if r2 > 0.7 else ''
                logger.info(
                    f"  WSUM     | {target_name:25s} | "
                    f"R²={r2:7.4f}  gain={metrics['r2_gain_over_best_single']:+.4f}  "
                    f"n_valid={n_valid:4d}  cov={metrics['coverage']:.3f}{marker}"
                )
                if mlflow.active_run():
                    prefix = f"WSUM_{target_name}"
                    mlflow.log_metrics({
                        f"{prefix}_r2": metrics["r2_score"],
                        f"{prefix}_gain": metrics["r2_gain_over_best_single"],
                    })
            results["weighted_sum"] = ws_results

        return results

    @staticmethod
    def best_layers(
        results: dict[Any, dict[str, dict[str, float]]],
    ) -> dict[str, dict[str, Any]]:
        """
        Find the best layer for each proxy target from discovery results.

        Ignores non-integer top-level keys (e.g. ``"weighted_sum"``) so the
        best-single-layer summary only reflects per-layer probes.

        Returns:
            {target_name: {'layer': int, 'r2_score': float, 'confidence': str}}
        """
        if not results:
            return {}

        per_layer = {k: v for k, v in results.items() if isinstance(k, int)}
        if not per_layer:
            return {}

        all_targets: set[str] = set()
        for layer_results in per_layer.values():
            all_targets.update(layer_results.keys())

        best: dict[str, dict[str, Any]] = {}
        for target in sorted(all_targets):
            best_layer = -1
            best_r2 = -999.0
            for layer, layer_results in per_layer.items():
                if target in layer_results:
                    r2 = layer_results[target]['r2_score']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_layer = layer

            if best_r2 > 0.8:
                confidence = 'high'
            elif best_r2 > 0.5:
                confidence = 'medium'
            else:
                confidence = 'low'

            best[target] = {
                'layer': best_layer,
                'r2_score': round(best_r2, 4),
                'confidence': confidence,
            }
            sample_metrics = per_layer.get(best_layer, {}).get(target, {})
            if 'n_valid' in sample_metrics:
                best[target]['n_valid'] = int(sample_metrics['n_valid'])
            if 'coverage' in sample_metrics:
                best[target]['coverage'] = float(sample_metrics['coverage'])

        return best

    def load_segment_features(
        self, audio_files: list[str],
    ) -> tuple[dict[int, np.ndarray], list[str], np.ndarray]:
        """Load per-segment MERT embeddings from the segments/ directory.

        Returns:
            Tuple of (features_per_layer, loaded_files, segment_counts) where:
            - features_per_layer: ``{layer: np.ndarray[total_segments, 768]}``
            - loaded_files: list of audio file paths that had features
            - segment_counts: array of segment count per track (for alignment)
        """
        segments_dir = self.features_dir.parent / "segments"
        per_layer: dict[int, list[np.ndarray]] = {i: [] for i in range(NUM_LAYERS)}
        loaded: list[str] = []
        seg_counts: list[int] = []

        for path in audio_files:
            feat_file = segments_dir / f"{Path(path).stem}.npy"
            if not feat_file.exists():
                continue
            raw = np.load(feat_file)  # [num_segments, 13, 768]
            n_seg = raw.shape[0]
            for layer in range(NUM_LAYERS):
                per_layer[layer].append(raw[:, layer, :])  # [n_seg, 768]
            loaded.append(path)
            seg_counts.append(n_seg)

        if not loaded:
            features = {i: np.empty((0, EMBEDDING_DIM)) for i in range(NUM_LAYERS)}
            return features, loaded, np.array([], dtype=int)

        features = {
            layer: np.concatenate(values, axis=0) for layer, values in per_layer.items()
        }
        logger.info(
            f"Loaded segment features for {len(loaded)}/{len(audio_files)} files "
            f"({features[0].shape[0]} total segments)"
        )
        return features, loaded, np.array(seg_counts, dtype=int)

    def load_segment_targets(
        self, audio_files: list[str],
    ) -> tuple[dict[str, np.ndarray], list[str], np.ndarray]:
        """Load per-segment proxy targets from segment target npz files.

        Returns:
            Tuple of (targets, loaded_files, segment_counts) where:
            - targets: ``{target_name: np.ndarray[total_segments]}``
            - loaded_files: list of audio file paths that had targets
            - segment_counts: array of segment count per track
        """
        seg_targets_dir = self._resolve_segment_targets_dir()

        collectors: dict[str, list[np.ndarray]] = {
            name: [] for name in SEGMENT_TARGETS
        }
        loaded: list[str] = []
        seg_counts: list[int] = []

        for path in audio_files:
            target_file = seg_targets_dir / f"{Path(path).stem}_segment_targets.npz"
            if not target_file.exists():
                continue

            data = np.load(target_file, allow_pickle=True)

            # Determine segment count from first available target
            first_count = None
            row: dict[str, np.ndarray] = {}
            ok = True

            for name, (category, key, _reduction) in self.SCALAR_TARGETS.items():
                if name not in SEGMENT_TARGETS:
                    continue
                try:
                    cat_data = data[category].item()
                    arr = cat_data[key]
                    if not isinstance(arr, np.ndarray) or arr.ndim == 0:
                        ok = False
                        break
                    if first_count is None:
                        first_count = len(arr)
                    elif len(arr) != first_count:
                        ok = False
                        break
                    row[name] = arr
                except (KeyError, IndexError, TypeError):
                    ok = False
                    break

            if ok and first_count is not None and first_count > 0:
                for name in SEGMENT_TARGETS:
                    if name in row:
                        collectors[name].append(row[name])
                loaded.append(path)
                seg_counts.append(first_count)

        if not loaded:
            empty_targets = {name: np.array([]) for name in SEGMENT_TARGETS}
            return empty_targets, loaded, np.array([], dtype=int)

        targets: dict[str, np.ndarray] = {}
        for name, arrays in collectors.items():
            if arrays:
                combined = np.concatenate(arrays)
                valid = combined[~np.isnan(combined)]
                if valid.size > 0 and np.std(valid) > 1e-10:
                    targets[name] = combined

        logger.info(
            f"Loaded {len(targets)} segment targets for {len(loaded)} files"
        )
        return targets, loaded, np.array(seg_counts, dtype=int)

    def _resolve_segment_targets_dir(self) -> Path:
        """Resolve segment target path with instance-level overrides."""
        segment_targets_dir = getattr(self, "segment_targets_dir", None)
        if segment_targets_dir is not None:
            return Path(segment_targets_dir)

        targets_dir = getattr(self, "targets_dir", None)
        if targets_dir is not None and Path(targets_dir) != mess_config.proxy_targets_dir:
            return Path(targets_dir)

        return mess_config.proxy_targets_segments_dir

    def discover_segments(
        self,
        n_samples: int = 50,
        probe_mode: ProbeMode | None = None,
    ) -> dict[Any, dict[str, dict[str, float]]]:
        """Run segment-level discovery: probe layers against segment targets.

        Same Ridge regression pipeline as ``discover()`` but uses individual
        5s segment embeddings as samples instead of track-level averages.

        When ``probe_mode`` includes ``weighted_sum`` a top-level
        ``"weighted_sum"`` key is added to the returned dict alongside the
        per-layer sections.
        """
        mode: ProbeMode = probe_mode or getattr(self, "probe_mode", "both")  # type: ignore[assignment]
        audio_files = sorted(str(f) for f in self.dataset.get_audio_files())[:n_samples]
        logger.info(f"Running segment discovery with up to {len(audio_files)} audio files")

        features, feat_files, feat_seg_counts = self.load_segment_features(audio_files)
        targets, tgt_files, tgt_seg_counts = self.load_segment_targets(audio_files)

        # Align to files with both features and targets
        common = sorted(set(feat_files) & set(tgt_files))
        if not common:
            logger.error("No files have both segment features and segment targets")
            return {}

        # Align and verify segment counts match per track
        feat_pos = {path: idx for idx, path in enumerate(feat_files)}
        tgt_pos = {path: idx for idx, path in enumerate(tgt_files)}

        aligned_feat_slices: list[tuple[int, int]] = []
        aligned_tgt_slices: list[tuple[int, int]] = []
        total_segments = 0

        feat_cumsum = np.cumsum(np.concatenate([[0], feat_seg_counts]))
        tgt_cumsum = np.cumsum(np.concatenate([[0], tgt_seg_counts]))

        mismatched = []
        for path in common:
            fi = feat_pos[path]
            ti = tgt_pos[path]
            f_count = int(feat_seg_counts[fi])
            t_count = int(tgt_seg_counts[ti])
            if f_count != t_count:
                mismatched.append((path, f_count, t_count))
                continue
            aligned_feat_slices.append((int(feat_cumsum[fi]), int(feat_cumsum[fi + 1])))
            aligned_tgt_slices.append((int(tgt_cumsum[ti]), int(tgt_cumsum[ti + 1])))
            total_segments += f_count

        if mismatched:
            logger.warning(
                f"Skipped {len(mismatched)} tracks with segment count mismatches"
            )

        if total_segments < self.n_folds:
            logger.error(
                f"Only {total_segments} aligned segments, need >= {self.n_folds}"
            )
            return {}

        # Track-group labels for leakage-safe segment CV.
        # GroupKFold keeps all segments from a track in the same fold.
        segment_groups = np.concatenate(
            [
                np.full(end - start, group_idx, dtype=np.int32)
                for group_idx, (start, end) in enumerate(aligned_feat_slices)
            ]
        )

        # Build aligned feature/target arrays
        aligned_features = {}
        for layer in range(NUM_LAYERS):
            parts = [features[layer][s:e] for s, e in aligned_feat_slices]
            aligned_features[layer] = np.concatenate(parts, axis=0)

        aligned_targets: dict[str, np.ndarray] = {}
        for name, arr in targets.items():
            parts = [arr[s:e] for s, e in aligned_tgt_slices]
            aligned_targets[name] = np.concatenate(parts, axis=0)

        n_tracks = len(aligned_feat_slices)
        min_valid_samples = self.n_folds
        valid_counts: dict[str, int] = {}
        valid_group_counts: dict[str, int] = {}
        filtered_targets: dict[str, np.ndarray] = {}
        for name, values in aligned_targets.items():
            valid_mask = ~np.isnan(values)
            n_valid = int(np.sum(valid_mask))
            n_valid_groups = int(np.unique(segment_groups[valid_mask]).size) if n_valid else 0
            if n_valid < min_valid_samples:
                logger.warning(
                    f"Skipping segment target '{name}': only {n_valid} valid segments "
                    f"(need >= {min_valid_samples})"
                )
                continue
            if n_valid_groups < 2:
                logger.warning(
                    f"Segment target '{name}' has {n_valid_groups} valid track group(s); "
                    "using ungrouped KFold fallback for this target."
                )
            filtered_targets[name] = values
            valid_counts[name] = n_valid
            valid_group_counts[name] = n_valid_groups

        aligned_targets = filtered_targets
        n_targets = len(aligned_targets)

        if n_targets == 0:
            logger.error("No viable segment targets after filtering")
            return {}

        logger.info(
            f"Segment probing: {NUM_LAYERS} layers x {n_targets} targets "
            f"on {total_segments} segments from {n_tracks} tracks"
        )

        # Log to MLflow
        if mlflow.active_run():
            mlflow.log_params({
                'mode': 'segment',
                'alpha': self.alpha,
                'n_folds': self.n_folds,
                'n_samples': n_samples,
                'n_tracks_used': n_tracks,
                'n_total_segments': total_segments,
                'n_targets': n_targets,
                'dataset': self.dataset.name if hasattr(self.dataset, 'name') else 'unknown',
                'targets': ','.join(sorted(aligned_targets.keys())),
                'probe_mode': mode,
            })
            try:
                mlflow.set_tag("probe_mode", mode)
            except AttributeError:
                pass

        results: dict[Any, dict[str, dict[str, float]]] = {}

        run_best_layer = mode in ("best_layer", "both")
        run_weighted_sum = mode in ("weighted_sum", "both")

        if run_best_layer:
            for layer in range(NUM_LAYERS):
                results[layer] = {}
                for target_name, target_values in aligned_targets.items():
                    metrics = self._probe_single(
                        aligned_features[layer],
                        target_values,
                        groups=segment_groups,
                    )
                    n_valid = valid_counts[target_name]
                    metrics.setdefault('n_valid', float(n_valid))
                    metrics.setdefault('coverage', float(n_valid / total_segments))
                    metrics['n_groups'] = float(valid_group_counts[target_name])
                    metrics['n_tracks'] = float(n_tracks)
                    results[layer][target_name] = metrics

                    r2 = metrics['r2_score']
                    marker = (
                        ' ***' if r2 > 0.9 else ' **' if r2 > 0.8 else ' *' if r2 > 0.7 else ''
                    )
                    logger.info(
                        f"  Layer {layer:2d} | {target_name:25s} | "
                        f"R²={r2:7.4f}  corr={metrics['correlation']:6.3f}  "
                        f"RMSE={metrics['rmse']:8.4f}  "
                        f"n_valid={n_valid:5d}  n_trk={n_tracks:3d}  "
                        f"cov={metrics['coverage']:.3f}{marker}"
                    )

                    if mlflow.active_run():
                        prefix = f"seg_L{layer}_{target_name}"
                        mlflow.log_metrics({
                            f"{prefix}_r2": metrics['r2_score'],
                            f"{prefix}_corr": metrics['correlation'],
                            f"{prefix}_rmse": metrics['rmse'],
                            f"{prefix}_n_valid": metrics['n_valid'],
                            f"{prefix}_coverage": metrics['coverage'],
                            f"{prefix}_n_groups": metrics['n_groups'],
                        })

            if mlflow.active_run():
                best = self.best_layers(results)
                for target_name, info in best.items():
                    mlflow.log_metric(f"seg_best_r2_{target_name}", info['r2_score'])
                    mlflow.log_metric(f"seg_best_layer_{target_name}", info['layer'])

        if run_weighted_sum:
            stacked = np.stack(
                [aligned_features[layer] for layer in range(NUM_LAYERS)], axis=1
            )
            ws_results: dict[str, dict[str, Any]] = {}
            for target_name, target_values in aligned_targets.items():
                metrics = self._probe_weighted_sum(
                    stacked, target_values, groups=segment_groups
                )
                n_valid = valid_counts[target_name]
                metrics.setdefault('n_valid', float(n_valid))
                metrics.setdefault('coverage', float(n_valid / total_segments))
                ws_results[target_name] = metrics

                if mlflow.active_run():
                    prefix = f"seg_WSUM_{target_name}"
                    mlflow.log_metrics({
                        f"{prefix}_r2": metrics["r2_score"],
                        f"{prefix}_gain": metrics["r2_gain_over_best_single"],
                    })
            results["weighted_sum"] = ws_results

        return results

    def discover_and_save(
        self,
        n_samples: int = 50,
        path: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run discovery, save results, return best layers summary."""
        results = self.discover(n_samples)
        if results:
            self.save(results, path)
        return self.best_layers(results)

    def save(self, results: dict[int, dict[str, dict[str, float]]], path: str | None = None):
        """Save discovery results to JSON and log as MLflow artifact."""
        out_path = Path(path) if path else mess_config.probing_results_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON requires string keys
        serializable = {str(layer): layer_results for layer, layer_results in results.items()}

        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Results saved to {out_path}")

        if mlflow.active_run():
            mlflow.log_artifact(str(out_path))


# =============================================================================
# Aspect registry: maps user-facing musical aspects to probing targets
# =============================================================================

# Maps user-facing aspect name → probing target(s) that validate it.
# When multiple targets are listed, the one with highest R² is used.
ASPECT_REGISTRY: dict[str, dict[str, Any]] = {
    # -- Legacy scalar aspects (kept for back-compat; some have low R² and --
    # -- will naturally fall below the ``min_r2`` filter in resolve_aspects) --
    'brightness': {
        'targets': ['spectral_centroid'],
        'description': 'Timbral brightness or darkness of the sound',
    },
    'texture': {
        'targets': ['spectral_rolloff', 'zero_crossing_rate'],
        'description': 'Surface texture: smooth vs noisy/rough',
    },
    'warmth': {
        'targets': ['spectral_bandwidth', 'spectral_centroid'],
        'description': 'Tonal warmth and fullness',
    },
    'tempo': {
        'targets': ['tempo'],
        'description': 'Speed and BPM similarity (scalar; see rhythmic_flow for curve)',
    },
    'rhythmic_energy': {
        'targets': ['onset_density'],
        'description': 'Note density and rhythmic activity',
    },
    'dynamics': {
        'targets': ['dynamic_range', 'dynamic_variance'],
        'description': 'Loudness variation and dynamic contrast',
    },
    'harmonic_richness': {
        'targets': ['harmonic_complexity'],
        'description': 'Harmonic content and tonal complexity (scalar; see tension for curve)',
    },
    'articulation': {
        'targets': ['attack_slopes', 'attack_sharpness'],
        'description': (
            'Legato vs staccato (audio-derived; see micro_articulation for MIDI)'
        ),
    },
    'phrasing': {
        'targets': ['phrase_regularity', 'num_phrases'],
        'description': 'Musical sentence structure and regularity',
    },
    # -- New curve-valued aspects (Phase 2: T1-T5) --
    'tension': {
        'targets': ['tis_tension'],
        'description': 'Harmonic tension trajectory (Tonal Interval Space, Bernardes et al.)',
    },
    'dynamic_arc': {
        'targets': ['dynamic_arc'],
        'description': 'Shape of loudness over the passage (crescendo/diminuendo contour)',
    },
    'rhythmic_flow': {
        'targets': ['local_tempo'],
        'description': 'Local tempo curve from tempogram (captures rubato/expressive timing)',
    },
    'brightness_trajectory': {
        'targets': ['centroid_trajectory'],
        'description': 'Relative brightness evolution (z-scored spectral centroid)',
    },
    'structure': {
        'targets': ['novelty'],
        'description': 'Structural novelty / boundary strength (Foote 1999)',
    },
    # -- New MIDI-ground-truth aspects (Phase 2: T6-T7) --
    'micro_articulation': {
        'targets': ['midi_articulation_hist'],
        'description': 'Distribution of articulation ratios (staccato↔legato) from MIDI',
    },
    'performance_expression': {
        'targets': ['midi_velocity_std', 'midi_ioi_std', 'midi_pedal_ratio'],
        'description': 'MIDI-ground-truth velocity variation, rubato proxy, and pedaling ratio',
    },
    # Removed in the 2026-04 rework:
    #   - ``crescendo`` — backed by crescendo_strength/diminuendo_strength (R² < 0);
    #     replaced by ``dynamic_arc`` curve.
    #   - ``rubato``, ``expressiveness``, ``legato`` — advertised MIDI-derived
    #     features that were never actually computed; replaced by the real
    #     MIDI-ground-truth aspects above.
}


def _confidence_label(r2: float) -> str:
    if r2 > 0.8:
        return 'high'
    if r2 > 0.5:
        return 'medium'
    return 'low'


def resolve_aspects(
    min_r2: float = 0.5,
    results_path: Path | None = None,
    probe_mode: Literal['best_layer', 'weighted_sum', 'auto'] = 'auto',
    gain_threshold: float = 0.02,
) -> dict[str, dict[str, Any]]:
    """
    Resolve each aspect in the registry to its best validated MERT probe.

    Loads discovery results and for each aspect, picks the probing target
    and probe mode that maximize R². Supports single-layer mappings
    (backwards compatible) and SUPERB-style weighted-sum layer fusion.

    Args:
        min_r2: Minimum R² to consider an aspect validated.
        results_path: Path to discovery results JSON. Defaults to config.
        probe_mode:
            * ``"best_layer"`` — always return single-layer mapping
              (``{"layer": int, ...}``). Matches pre-I1 behavior.
            * ``"weighted_sum"`` — always return dense layer weights
              (``{"layer_weights": list[13 floats], ...}``) when the
              ``"weighted_sum"`` section is present for the chosen target.
            * ``"auto"`` (default) — prefer weighted-sum when its R²
              exceeds the best-single-layer R² by at least ``gain_threshold``;
              otherwise fall back to best-single-layer.
        gain_threshold: Minimum R² gain of weighted-sum over best-single
            required to switch modes under ``probe_mode="auto"``.

    Returns:
        ``{aspect_name: {target, r2_score, description, confidence, ...}}``
        where each entry additionally carries EITHER ``layer: int`` OR
        ``layer_weights: list[float]`` (mutually exclusive) depending on
        the chosen probe mode. Only aspects meeting ``min_r2`` are returned.
    """
    # Load discovery results
    results_path = results_path or mess_config.probing_results_file
    if not results_path.exists():
        logger.warning("No discovery results found. Run layer discovery first.")
        return {}

    with open(results_path) as f:
        raw = json.load(f)

    # Split per-layer (integer keys) and weighted-sum sections.
    per_layer: dict[int, dict[str, dict[str, Any]]] = {}
    ws_raw = raw.get('weighted_sum', {})
    weighted_sum: dict[str, dict[str, Any]] = ws_raw if isinstance(ws_raw, dict) else {}
    for key, targets in raw.items():
        if key == 'weighted_sum':
            continue
        try:
            layer_idx = int(key)
        except ValueError:
            continue
        per_layer[layer_idx] = targets

    resolved: dict[str, dict[str, Any]] = {}

    for aspect_name, aspect_info in ASPECT_REGISTRY.items():
        # For each candidate target under this aspect, collect best-single-layer
        # R² and (if available) weighted-sum R². Pick the (target, mode) pair
        # that maximizes R² subject to ``probe_mode``.
        best: dict[str, Any] = {'r2': -999.0, 'mode': None, 'target': None}

        for target_name in aspect_info['targets']:
            single_layer = -1
            single_r2 = -np.inf
            for layer, layer_results in per_layer.items():
                if target_name in layer_results:
                    r2 = layer_results[target_name].get('r2_score', -np.inf)
                    if r2 > single_r2:
                        single_r2 = r2
                        single_layer = layer

            ws = weighted_sum.get(target_name)
            ws_r2 = float(ws['r2_score']) if ws and 'r2_score' in ws else -np.inf

            if probe_mode == 'best_layer':
                candidate_r2 = single_r2
                candidate_mode = 'best_layer'
            elif probe_mode == 'weighted_sum':
                candidate_r2 = ws_r2
                candidate_mode = 'weighted_sum'
            else:  # auto
                if ws_r2 > single_r2 + gain_threshold:
                    candidate_r2 = ws_r2
                    candidate_mode = 'weighted_sum'
                else:
                    candidate_r2 = single_r2
                    candidate_mode = 'best_layer'

            if candidate_r2 > best['r2']:
                best = {
                    'r2': candidate_r2,
                    'mode': candidate_mode,
                    'target': target_name,
                    'single_layer': single_layer,
                    'weighted_sum_entry': ws,
                    'single_metrics': (
                        per_layer[single_layer].get(target_name, {})
                        if single_layer >= 0 else {}
                    ),
                }

        if best['r2'] < min_r2 or best['mode'] is None:
            logger.debug(
                f"Aspect '{aspect_name}' not validated: best R²={best['r2']:.4f} < {min_r2}"
            )
            continue

        entry: dict[str, Any] = {
            'target': best['target'],
            'r2_score': round(float(best['r2']), 4),
            'description': aspect_info['description'],
            'confidence': _confidence_label(float(best['r2'])),
        }
        if best['mode'] == 'weighted_sum':
            ws_entry = best['weighted_sum_entry']
            entry['layer_weights'] = [float(w) for w in ws_entry['layer_weights']]
            if 'r2_gain_over_best_single' in ws_entry:
                entry['r2_gain'] = float(ws_entry['r2_gain_over_best_single'])
            source_metrics = ws_entry
        else:
            entry['layer'] = int(best['single_layer'])
            source_metrics = best['single_metrics']

        if 'n_valid' in source_metrics:
            entry['n_valid'] = int(source_metrics['n_valid'])
        if 'coverage' in source_metrics:
            entry['coverage'] = float(source_metrics['coverage'])

        resolved[aspect_name] = entry

    logger.info(f"Resolved {len(resolved)}/{len(ASPECT_REGISTRY)} aspects from discovery results")
    return resolved


def main():
    """Run layer discovery and print summary."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    discovery = LayerDiscoverySystem()
    results = discovery.discover(n_samples=50)

    if not results:
        print("Discovery failed - insufficient data")
        return 1

    print("\n" + "=" * 70)
    print("BEST LAYER PER TARGET")
    print("=" * 70)

    best = LayerDiscoverySystem.best_layers(results)
    for target, info in best.items():
        r2 = info['r2_score']
        tag = '  EXCELLENT' if r2 > 0.9 else '  GOOD' if r2 > 0.8 else ''
        print(f"  {target:25s} -> Layer {info['layer']:2d}  (R²={r2:.4f}){tag}")

    discovery.save(results)
    print(f"\nResults saved to {mess_config.probing_results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
