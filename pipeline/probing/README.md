# MERT Layer Discovery System

This directory contains the empirical validation system that discovered which MERT layers encode specific musical aspects.

## Key Files

### Core System
- **`layer_discovery.py`** - Main discovery system that probes all MERT layers for musical aspects
- **`proxy_targets.py`** - Generates musical aspect targets (rhythm, harmony, timbre, etc.)
- **`layer_discovery_results.json`** - Validated findings from our experiments

### Key Findings (Validated with R² > 0.9)
- **Layer 0**: Spectral brightness/centroid (R² = 0.944) - Best for timbral similarity
- **Layer 1**: Timbral texture (R² = 0.922) - Instrumental characteristics  
- **Layer 2**: Acoustic structure (R² = 0.933) - Resonance patterns

## Usage

```bash
# Run full discovery (takes ~10 minutes)
python layer_discovery.py

# Test proxy target generation
python proxy_targets.py
```

## Methodology

1. **Proxy Targets**: Generate measurable musical descriptors from audio (spectral centroid, tempo, etc.)
2. **Linear Probing**: Test each MERT layer's ability to predict these descriptors using Ridge regression
3. **Cross-Validation**: Use 5-fold CV to ensure robust R² scores
4. **Validation**: Only layers with R² > 0.5 are considered validated for recommendations

This empirical approach replaced arbitrary feature slicing with scientifically validated layer specializations.