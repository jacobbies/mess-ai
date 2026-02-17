You are working in the `mess-ai` repo. Continue from current state and execute the core next step toward clip-level + score-aware retrieval.

Context:
- We finished repo path normalization and pushed commits:
  - `9cfadea` (script path fixes + extraction updates)
  - `40c84e1` (remaining path normalization + cross-OS-safe file discovery)
  - `7800fcc` (added remaining local files incl. mlruns artifacts)
- Current architecture:
  - MERT extraction outputs:
    - `raw`: `[segments, 13, time, 768]`
    - `segments`: `[segments, 13, 768]`
    - `aggregated`: `[13, 768]`
  - Discovery/probing is in `mess/probing/` with aspect resolution logic.
  - Search is in `mess/search/search.py` (track-level cosine + FAISS IndexFlatIP).
- MAESTRO dataset status:
  - Full local pair availability confirmed: 1276 WAV + 1276 MIDI + metadata JSON.
  - `data/audio/maestro/maestro-v3.0.0.json` has composer/title/split/year/midi/audio/duration.
  - Previously only 10 MAESTRO embeddings were present (all test), so coverage needs regeneration.
- Important fixes already made:
  - `mess/extraction/pipeline.py` now supports recursive fallback and case-insensitive extension matching (Linux/macOS consistency).
  - `scripts/extract_features.py` writes to `dataset.embeddings_dir` (`data/embeddings/<dataset>-emb`).
  - Added `scripts/build_maestro_manifest.py` to join metadata ↔ embeddings.

Goal now:
Execute the core next step: regenerate MAESTRO embeddings at scale and establish a reliable manifest for clip-level/score-aware work.

Do this end-to-end:
1. Verify environment dependencies and run extraction for MAESTRO:
   - `python scripts/extract_features.py --dataset maestro --workers 4 --force`
   - If dependency issues occur, resolve pragmatically and continue.
2. Validate output coverage:
   - Count files in `data/embeddings/maestro-emb/{raw,segments,aggregated}`.
   - Confirm counts are aligned and significantly above previous 10.
3. Build metadata join manifest:
   - `python scripts/build_maestro_manifest.py --feature-type segments --include-missing --output data/metadata/maestro_embedding_manifest.csv`
4. Report:
   - Total MAESTRO metadata rows
   - Number matched to embeddings
   - Split-wise match counts (train/validation/test)
   - Any mismatches/path anomalies
5. If needed, patch code for robust matching (case/stem/path) and rerun.
6. Commit and push only relevant changes.

Constraints:
- Be blunt and practical.
- Don’t redesign architecture yet; focus on this execution step.
- Don’t touch unrelated files unless required.
- Provide clear final summary with exact commands run and resulting counts.
