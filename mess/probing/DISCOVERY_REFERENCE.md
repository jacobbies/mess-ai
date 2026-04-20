# Discovery Reference

## Goal
Find which MERT layer(s) best linearly predict each musical proxy target.

## Inputs
- Audio files: `data/audio/<dataset>/*.wav`
- Precomputed embeddings: `data/embeddings/<dataset>-emb/raw/<track>.npy`
  - Shape per track: `[segments, 13, time, 768]`
- Proxy targets: `data/proxy_targets/<track>_targets.npz`

## Target Types

Targets are declared via `mess.probing._schema.TargetDescriptor`:

| Type          | `.npz` location                          | Probe                              |
|---------------|------------------------------------------|------------------------------------|
| `SCALAR`      | `<category>/<name>` (nested dict)        | Ridge with 1-D y                   |
| `CURVE`       | `curves/<name>` — shape `(T,)`           | Multi-output Ridge                 |
| `MIDI_SCALAR` | `midi/<name>` (scalar)                   | Ridge with NaN-masked rows         |
| `MIDI_CURVE`  | `midi/<name>` — shape `(T,)`             | Multi-output Ridge, NaN-masked     |

Curve targets use a fixed frame rate (default 2 Hz) and fixed duration
(default 30 s), i.e. `T = 60` frames. The spec is declared by
`mess.probing._schema.CurveSpec` and shared via `DEFAULT_CURVE_SPEC`.

MIDI-backed targets also carry a top-level boolean `midi_available` flag
in the `.npz` so tracks without MIDI are dropped during probing instead
of breaking the pipeline. Existing scalar `.npz` files (no `curves` or
`midi` sections) remain fully compatible.

## Probe Modes

`LayerDiscoverySystem` accepts `probe_mode: Literal["best_layer", "weighted_sum", "both"]`
(default `"both"`). Choose via the constructor or the `discover()` /
`discover_segments()` argument.

| Mode            | What it does                                                                                  | When to use                                                       |
|-----------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `best_layer`    | Legacy loop: probes each of the 13 layers independently and reports the winner per target.    | Historic comparisons; cheapest mode.                              |
| `weighted_sum`  | SUPERB-style fused probe: stacks all layers into `(N, 13, 768)` and learns softmax weights.   | Low-spread aspects where several layers each encode part of the signal. |
| `both`          | Runs both; results JSON carries both sections.                                                | Default. Lets `resolve_aspects` pick the cheaper best-layer path while exposing weighted-sum gains for phase-3 fusion. |

## Math (Minimal)

For a scalar target `y` and layer features `X`:
- Model: Ridge regression in a `Pipeline(StandardScaler, Ridge)`
- Validation: K-Fold CV (shuffle, fixed seed); GroupKFold for segment probing
- Metrics: `R² = 1 - SS_res/SS_tot`, `corr(y, y_pred)`, `RMSE`

For curve targets `y ∈ R^{N × T}`:
- Same pipeline, multi-output Ridge (fits one linear map `X -> y`)
- `r2_mean = mean(per_frame_R²)` via `sklearn.metrics.r2_score(..., multioutput="raw_values")`
- `r2_pc1`: fit PCA on the train fold, project to the first principal
  component, run a scalar Ridge on that projection — probes the dominant
  shape axis without leaking test-fold variance
- `rmse_mean`: sqrt of MSE across all N·T entries

For weighted-sum probing:
- Parameterise layer weights as `softmax(w / tau)`.
- Grid search over `tau ∈ {0.1, 0.25, 0.5, 1, 2, 4}` and three starting
  vectors (uniform, per-layer-R² warm-start, spike on the best single
  layer) — cheap, interpretable, and guaranteed never to underperform
  the best single layer. (An L-BFGS variant is acceptable in follow-up
  units; the grid is the simpler baseline.)
- Report: `r2_score`, `layer_weights` (13 floats summing to 1),
  `correlation`, `rmse`, `r2_gain_over_best_single`.

## Outputs

Results JSON (`mess/probing/layer_discovery_results.json`):

```jsonc
{
  "0":  {"spectral_centroid": {"r2_score": 0.93, "correlation": 0.96, "rmse": 33.6}, "...": {}},
  "1":  { /* ... */ },
  "12": { /* ... */ },

  // Present when probe_mode includes "weighted_sum":
  "weighted_sum": {
    "spectral_centroid": {
      "r2_score": 0.95,
      "layer_weights": [0.05, 0.4, /* ... 13 floats ... */],
      "correlation": 0.97,
      "rmse": 31.1,
      "r2_gain_over_best_single": 0.016
    }
  }
}
```

Curve targets swap the scalar keys for `r2_mean`, `r2_pc1`, `rmse_mean`.
MIDI-masked targets add `n_valid` and `coverage` fields at the
per-target level.

`resolve_aspects()` currently reads only the top-level integer keys and
silently ignores `"weighted_sum"`. Phase 3 / Unit I1 extends the resolver
to consume dense layer weights.

## Best-Layer Summary

`LayerDiscoverySystem.best_layers(results)` returns
`{target: {layer, r2_score, confidence}}`, ignoring the `"weighted_sum"`
section. Confidence labels:
- `high`: R² > 0.8
- `medium`: R² > 0.5
- `low`: otherwise

## Quick Interpretation

- High R² for `(layer, target)` means that layer linearly encodes the
  musical property.
- When best-single-layer R² is close for many layers (low layer-spread),
  `weighted_sum` typically wins by ~5-15% and is the right probe to use
  for aspect resolution.
- `r2_pc1` tells you whether the dominant shape of a curve target is
  learnable even when individual frames are noisy.
