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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from ..config import mess_config

logger = logging.getLogger(__name__)

NUM_LAYERS = 13
EMBEDDING_DIM = 768


# =============================================================================
# Model inspection utilities (require mess-ai[ml])
# =============================================================================

def inspect_model(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Inventory MERT model: layer names, parameter counts, shapes.

    Returns dict with total_params, trainable_params, and per-module breakdown.
    Requires mess-ai[ml] dependencies (transformers, torch).
    """
    from transformers import AutoModel

    name = model_name or mess_config.model_name
    model = AutoModel.from_pretrained(name, trust_remote_code=True)

    inventory: Dict[str, Any] = {
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
    model_name: Optional[str] = None,
    segment_duration: float = 5.0,
) -> Dict[str, Any]:
    """
    Run a forward pass through MERT and capture activation shapes per hidden layer.

    Useful for verifying the model produces expected output dimensions before
    running full feature extraction.

    Requires mess-ai[ml] dependencies (transformers, torch, torchaudio).
    """
    import torch
    import torchaudio
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

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
        # Timbre
        'spectral_centroid':   ('timbre', 'spectral_centroid', 'mean'),
        'spectral_rolloff':    ('timbre', 'spectral_rolloff', 'mean'),
        'spectral_bandwidth':  ('timbre', 'spectral_bandwidth', 'mean'),
        'zero_crossing_rate':  ('timbre', 'zero_crossing_rate', 'mean'),
        # Rhythm
        'tempo':               ('rhythm', 'tempo', 'first'),
        'onset_density':       ('rhythm', 'onset_density', 'first'),
        # Dynamics
        'dynamic_range':       ('dynamics', 'dynamic_range', 'first'),
        'dynamic_variance':    ('dynamics', 'dynamic_variance', 'first'),
        'crescendo_strength':  ('dynamics', 'crescendo_strength', 'first'),
        'diminuendo_strength': ('dynamics', 'diminuendo_strength', 'first'),
        # Harmony
        'harmonic_complexity': ('harmony', 'harmonic_complexity', 'first'),
        # Articulation
        'attack_slopes':       ('articulation', 'attack_slopes', 'mean'),
        'attack_sharpness':    ('articulation', 'attack_sharpness', 'mean'),
        # Phrasing
        'phrase_regularity':   ('phrasing', 'phrase_regularity', 'first'),
        'num_phrases':         ('phrasing', 'num_phrases', 'first'),
    }

    def __init__(self, dataset_name: str = "smd", alpha: float = 1.0, n_folds: int = 5):
        from mess.datasets.factory import DatasetFactory

        self.dataset = DatasetFactory.get_dataset(dataset_name)
        self.features_dir = self.dataset.embeddings_dir / "raw"
        self.targets_dir = mess_config.proxy_targets_dir
        self.alpha = alpha
        self.n_folds = n_folds

    def load_features(self, audio_files: List[str]) -> Tuple[Dict[int, np.ndarray], List[str]]:
        """Load MERT layer embeddings, averaged across segments and time steps."""
        per_layer: Dict[int, List[np.ndarray]] = {l: [] for l in range(NUM_LAYERS)}
        loaded: List[str] = []

        for path in audio_files:
            feat_file = self.features_dir / f"{Path(path).stem}.npy"
            if not feat_file.exists():
                continue
            raw = np.load(feat_file)  # [segments, 13, time, 768]
            for layer in range(NUM_LAYERS):
                per_layer[layer].append(raw[:, layer, :, :].mean(axis=(0, 1)))
            loaded.append(path)

        features = {l: np.array(v) for l, v in per_layer.items()}
        logger.info(f"Loaded features for {len(loaded)}/{len(audio_files)} files")
        return features, loaded

    def load_targets(self, audio_files: List[str]) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Load scalar proxy targets from npz files for probing."""
        collectors: Dict[str, List[float]] = {name: [] for name in self.SCALAR_TARGETS}
        loaded: List[str] = []

        for path in audio_files:
            target_file = self.targets_dir / f"{Path(path).stem}_targets.npz"
            if not target_file.exists():
                continue

            data = np.load(target_file, allow_pickle=True)
            row: Dict[str, float] = {}
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
                    ok = False
                    break

            if ok:
                for name, val in row.items():
                    collectors[name].append(val)
                loaded.append(path)

        # Only keep targets with variance (constant targets can't be probed)
        targets: Dict[str, np.ndarray] = {}
        for name, values in collectors.items():
            arr = np.array(values)
            if len(arr) > 0 and np.std(arr) > 1e-10:
                targets[name] = arr
            elif len(arr) > 0:
                logger.warning(f"Skipping target '{name}': constant values")

        logger.info(f"Loaded {len(targets)} targets for {len(loaded)} files")
        return targets, loaded

    def _probe_single(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Cross-validated Ridge regression for one (layer, target) pair."""
        n = len(X)
        if n < self.n_folds:
            return {'r2_score': -999.0, 'correlation': 0.0, 'rmse': 999.0}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kf = KFold(n_splits=min(self.n_folds, n), shuffle=True, random_state=42)
        probe = Ridge(alpha=self.alpha, random_state=42)

        y_pred = cross_val_predict(probe, X_scaled, y, cv=kf)

        r2 = float(r2_score(y, y_pred))
        corr = float(np.corrcoef(y, y_pred)[0, 1]) if np.std(y_pred) > 1e-10 else 0.0
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        return {'r2_score': r2, 'correlation': corr, 'rmse': rmse}

    def discover(self, n_samples: int = 50) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Run full layer discovery: probe all 13 layers against all proxy targets.

        Logs all parameters and metrics to MLflow if a run is active.

        Returns:
            {layer_idx: {target_name: {'r2_score', 'correlation', 'rmse'}}}
        """
        audio_files = sorted(str(f) for f in self.dataset.audio_dir.glob("*.wav"))[:n_samples]
        logger.info(f"Running discovery with up to {len(audio_files)} audio files")

        features, feat_files = self.load_features(audio_files)
        targets, tgt_files = self.load_targets(audio_files)

        # Align to files that have both features and targets
        common = sorted(set(feat_files) & set(tgt_files))
        if len(common) < self.n_folds:
            logger.error(f"Only {len(common)} common files, need >= {self.n_folds}")
            return {}

        feat_idx = [i for i, f in enumerate(feat_files) if f in common]
        tgt_idx = [i for i, f in enumerate(tgt_files) if f in common]

        features = {l: v[feat_idx] for l, v in features.items()}
        targets = {name: v[tgt_idx] for name, v in targets.items()}

        n_tracks = len(common)
        n_targets = len(targets)
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
            })

        results: Dict[int, Dict[str, Dict[str, float]]] = {}

        for layer in range(NUM_LAYERS):
            results[layer] = {}
            for target_name, target_values in targets.items():
                metrics = self._probe_single(features[layer], target_values)
                results[layer][target_name] = metrics

                r2 = metrics['r2_score']
                marker = ' ***' if r2 > 0.9 else ' **' if r2 > 0.8 else ' *' if r2 > 0.7 else ''
                logger.info(
                    f"  Layer {layer:2d} | {target_name:25s} | "
                    f"R²={r2:7.4f}  corr={metrics['correlation']:6.3f}  "
                    f"RMSE={metrics['rmse']:8.4f}{marker}"
                )

                # Log per-(layer, target) metrics to MLflow
                if mlflow.active_run():
                    prefix = f"L{layer}_{target_name}"
                    mlflow.log_metrics({
                        f"{prefix}_r2": metrics['r2_score'],
                        f"{prefix}_corr": metrics['correlation'],
                        f"{prefix}_rmse": metrics['rmse'],
                    })

        # Log best-layer summary metrics
        if mlflow.active_run():
            best = self.best_layers(results)
            for target_name, info in best.items():
                mlflow.log_metric(f"best_r2_{target_name}", info['r2_score'])
                mlflow.log_metric(f"best_layer_{target_name}", info['layer'])

        return results

    @staticmethod
    def best_layers(
        results: Dict[int, Dict[str, Dict[str, float]]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Find the best layer for each proxy target from discovery results.

        Returns:
            {target_name: {'layer': int, 'r2_score': float, 'confidence': str}}
        """
        if not results:
            return {}

        all_targets: set[str] = set()
        for layer_results in results.values():
            all_targets.update(layer_results.keys())

        best: Dict[str, Dict[str, Any]] = {}
        for target in sorted(all_targets):
            best_layer = -1
            best_r2 = -999.0
            for layer, layer_results in results.items():
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

        return best

    def discover_and_save(self, n_samples: int = 50, path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run discovery, save results, return best layers summary."""
        results = self.discover(n_samples)
        if results:
            self.save(results, path)
        return self.best_layers(results)

    def save(self, results: Dict[int, Dict[str, Dict[str, float]]], path: Optional[str] = None):
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
