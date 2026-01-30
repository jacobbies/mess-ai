#!/usr/bin/env python3
"""
Inspect MERT Embeddings: Hands-on exploration of what embeddings contain.

Shows:
- Embedding shapes and data types
- Per-layer statistics
- Validated layer specializations
- Cosine similarity between two tracks
"""

import numpy as np
from pathlib import Path
from mess.extraction.config import mess_config

def main():
    # Load first two tracks
    aggregated_features_dir = mess_config.aggregated_features_dir
    feature_files = list(aggregated_features_dir.glob("*.npy"))

    if len(feature_files) < 2:
        print("Need at least 2 feature files. Run extract_features.py first.")
        return 1

    track1_path, track2_path = feature_files[0], feature_files[1]
    emb1 = np.load(track1_path)
    emb2 = np.load(track2_path)

    print(f"Track 1: {track1_path.stem}")
    print(f"Shape: {emb1.shape} → [13 layers, 768 dimensions]")
    print(f"Data type: {emb1.dtype}")
    print(f"Memory: {emb1.nbytes / 1024:.2f} KB per track\n")

    # Show layer statistics
    print("Per-Layer Statistics:")
    print("Layer | Mean    | Std     | Specialization (from probing)")
    print("-" * 70)

    specializations = {
        0: "Spectral brightness (R²=0.944)",
        1: "Timbral texture (R²=0.922)",
        2: "Acoustic structure (R²=0.933)",
    }

    for layer in range(13):
        mean = emb1[layer].mean()
        std = emb1[layer].std()
        spec = specializations.get(layer, "Unknown")
        print(f"{layer:5d} | {mean:7.4f} | {std:7.4f} | {spec}")

    # Compute cosine similarity per layer
    print(f"\nCosine Similarity: {track1_path.stem} vs {track2_path.stem}")
    print("Layer | Similarity | Interpretation")
    print("-" * 60)

    for layer in range(3):  # Show first 3 validated layers
        cos_sim = np.dot(emb1[layer], emb2[layer]) / (
            np.linalg.norm(emb1[layer]) * np.linalg.norm(emb2[layer])
        )
        spec = specializations[layer].split("(")[0].strip()
        print(f"{layer:5d} | {cos_sim:10.4f} | Similar in {spec}")

    print("\n✅ Embeddings loaded successfully!")
    print("Next: Try demo_recommendations.py to find similar tracks")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
