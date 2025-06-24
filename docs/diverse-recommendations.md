# Diverse Recommendations in MESS-AI

## Current Challenge

Your insight is correct - the high similarity scores (99%+) indicate that the 50 classical tracks cluster tightly in MERT's embedding space. This is because:

1. **Small dataset size** - Only 50 tracks
2. **Genre homogeneity** - All classical music
3. **MERT's vast embedding space** - Trained on 160k hours of diverse music

## Implemented Solutions

### 1. **Multiple Recommendation Strategies**

The system now supports 5 different strategies via the `strategy` parameter:

```bash
# Traditional similarity (highest scores)
GET /recommend/{track}?strategy=similar

# Balanced diversity using MMR (λ=0.7) - RECOMMENDED DEFAULT
GET /recommend/{track}?strategy=balanced  

# More diverse using MMR (λ=0.5)
GET /recommend/{track}?strategy=diverse

# Complementary tracks (0.5-0.75 similarity) - empty on this dataset
GET /recommend/{track}?strategy=complementary

# Exploration mode (~0.6 similarity) - empty on this dataset
GET /recommend/{track}?strategy=exploration
```

### 2. **Maximal Marginal Relevance (MMR)**

The "balanced" and "diverse" strategies use MMR to balance relevance and diversity:
- Considers both similarity to query AND dissimilarity to already selected items
- Prevents repetitive recommendations
- Tunable λ parameter (0.7 for balanced, 0.5 for diverse)

### 3. **Strategy Comparison Endpoint**

Compare all strategies at once:
```bash
GET /recommend/{track}/compare?top_k=5
```

## Results with Current Dataset

Due to the small, homogeneous dataset:
- **Similar**: Bach → Bach (99%), Scriabin (98%), Liszt (97%)
- **Balanced**: Same tracks but potentially reordered by MMR
- **Diverse**: Includes Mozart (96%), Haydn (96%) for more variety
- **Complementary/Exploration**: Empty (no tracks in 0.5-0.75 range)

## Recommendations for Better Diversity

### 1. **Expand the Dataset**
- Add non-classical genres (jazz, rock, electronic, world music)
- Include modern recordings
- Target 500+ diverse tracks

### 2. **Feature Engineering**
```python
# Implemented in src/mess_ai/analysis/embedding_diversity.py
- Alternative similarity metrics (Manhattan, Angular, Correlation)
- Weighted similarity based on musical attributes
- Feature selection/dimensionality reduction
```

### 3. **Metadata-Enhanced Recommendations**
- Filter by era: "Find similar but from different era"
- Composer diversity: "Similar style, different composer"
- Temporal diversity: Avoid recommending same recording session

### 4. **Advanced Techniques**
- **Clustering**: Pre-cluster tracks, ensure recommendations span clusters
- **Graph-based**: Build similarity graph, use graph algorithms for diversity
- **Learning to Rank**: Train model on user feedback for better diversity

### 5. **UI Enhancements**
Add controls for users to tune recommendations:
- Similarity threshold slider
- Diversity preference
- Genre/era filters
- "Surprise me" mode

## Quick Test Commands

```bash
# See the difference between strategies
curl -s "http://localhost:8000/recommend/Bach_BWV849-01_001_20090916-SMD?strategy=similar&top_k=5" | jq '.recommendations[] | {title, composer, similarity_score}'

curl -s "http://localhost:8000/recommend/Bach_BWV849-01_001_20090916-SMD?strategy=diverse&top_k=5" | jq '.recommendations[] | {title, composer, similarity_score}'

# Compare all strategies
curl -s "http://localhost:8000/recommend/Bach_BWV849-01_001_20090916-SMD/compare?top_k=3" | jq
```

## Next Steps

1. **Immediate**: Use "diverse" strategy as default, add UI controls
2. **Short-term**: Add genre-diverse tracks to the dataset
3. **Long-term**: Implement clustering and personalization

The core issue isn't the algorithm - it's the dataset. With more diverse music, the existing strategies will produce much more interesting recommendations.