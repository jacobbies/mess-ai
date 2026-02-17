# Discovery Reference (Simple)

## Goal
Find which MERT layer best linearly predicts each musical proxy target.

## Inputs
- Audio files: `data/audio/<dataset>/*.wav`
- Precomputed embeddings: `data/embeddings/<dataset>-emb/raw/<track>.npy`
  - Shape per track: `[segments, 13, time, 768]`
- Proxy targets: `data/proxy_targets/<track>_targets.npz`

## Data Flow
1. Select up to `n_samples` tracks.
2. Load embeddings for tracks that exist.
3. For each layer `l` (0..12), average over segments and time:
   - `x_l = mean(raw[:, l, :, :], axes=(0,1))` → 768-d vector per track.
4. Load scalar targets from `.npz` (15 targets total).
5. Keep only tracks that have both embeddings and targets.
6. For each `(layer, target)` pair, run CV Ridge regression.
7. Store metrics and choose best layer per target by highest `R²`.

## Math (Minimal)
For each target `y` and layer features `X`:
- Model: Ridge regression
  - `w* = argmin_w ||y - Xw||^2 + alpha * ||w||^2`
- Validation: K-Fold CV (`k=n_folds`, shuffled, fixed seed).
- Important: scaling is inside CV via sklearn pipeline:
  - `Pipeline(StandardScaler, Ridge)`
  - Prevents CV leakage.

Metrics:
- `R² = 1 - SS_res/SS_tot` (higher is better)
- `corr = corr(y, y_pred)`
- `RMSE = sqrt(mean((y - y_pred)^2))` (lower is better)

## Outputs
- Full results: `mess/probing/layer_discovery_results.json`
  - `{layer: {target: {r2_score, correlation, rmse}}}`
- Best layer summary (computed in code):
  - `{target: {layer, r2_score, confidence}}`

## Confidence Labels
- `high`: `R² > 0.8`
- `medium`: `R² > 0.5`
- `low`: otherwise

## Quick Interpretation
- A high `R²` for `(layer, target)` means that layer linearly encodes that musical property.
- Best layer per target is used later to resolve user-facing search aspects.
