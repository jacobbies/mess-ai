# PyTorch Training Integration - TODO

## Current Architecture Status
**All ML is non-neural:**
- MERT: Frozen (inference only)
- Layer Discovery: Linear probing (sklearn Ridge, not neural)
- Search: Unsupervised FAISS (no learned parameters)

## 4 PyTorch Training Integration Options

### OPTION 1: Fine-tune MERT
**Location**: `mess/extraction/mert_finetuner.py` (new file)

**What it does:**
- Takes frozen MERT model
- Adds classification/regression head
- Backpropagates through MERT layers
- Adapts to specific dataset (SMD/MAESTRO)

**Training data needed:**
- Audio pairs with similarity labels
- Or: Audio + metadata (composer, style, tempo labels)
- Or: Contrastive pairs (similar/dissimilar)

**Pros:**
- Could improve embeddings for classical music
- End-to-end learned representations
- Published research path

**Cons:**
- Expensive (requires GPU, lots of data)
- Might overfit on 50-track dataset
- Breaks frozen feature assumption (need to re-extract all features)
- Training time: hours/days

**Code structure:**
```python
# mess/extraction/mert_finetuner.py
class MERTFinetuner:
    def __init__(self, base_model='m-a-p/MERT-v1-95M'):
        self.model = AutoModel.from_pretrained(base_model)
        self.classifier = nn.Linear(768, num_classes)

    def train(self, train_loader, epochs=10):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        for epoch in epochs:
            for batch in train_loader:
                loss = self.forward(batch)
                loss.backward()
                optimizer.step()
```

---

### OPTION 2: Layer Fusion Network ⭐ **RECOMMENDED**
**Location**: `mess/search/layer_fusion.py` (new file)

**What it does:**
- Input: All 13 MERT layers `[13, 768]`
- Learn attention weights or MLP fusion
- Output: Single optimal embedding `[fusion_dim]`
- **Replaces manual layer selection and averaging**

**Training data needed:**
- Triplets: (anchor, positive, negative) tracks
- Or: Pairwise similarity labels
- Or: User click data / playlist co-occurrence

**Why BEST:**
- Solves actual problem: "Which layers matter?"
- Training is offline (doesn't slow search)
- Works with frozen MERT (no re-extraction)
- Small model (fast training, easy to debug)
- Can use existing 50-track dataset

**Pros:**
- Learns optimal layer combination (better than averaging)
- Fast training (<1 hour on CPU)
- Integrates cleanly with existing search
- Can validate against linear probing results

**Cons:**
- Needs training labels (similarity judgments)
- Adds preprocessing step before search

**Code structure:**
```python
# mess/search/layer_fusion.py
class LayerFusionNetwork(nn.Module):
    """Learn optimal combination of 13 MERT layers."""

    def __init__(self, fusion_dim=256):
        super().__init__()
        # Attention-based fusion
        self.layer_attention = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softmax(dim=0)
        )
        self.projection = nn.Linear(768, fusion_dim)

    def forward(self, layer_features):
        # layer_features: [13, 768]
        weights = self.layer_attention(layer_features)  # [13, 1]
        fused = (layer_features * weights).sum(dim=0)   # [768]
        return self.projection(fused)  # [fusion_dim]

class ContrastiveFusionTrainer:
    """Train fusion network with triplet loss."""

    def train(self, triplets):
        for anchor, pos, neg in triplets:
            anchor_emb = self.model(anchor)
            pos_emb = self.model(pos)
            neg_emb = self.model(neg)

            loss = triplet_loss(anchor_emb, pos_emb, neg_emb, margin=0.2)
            loss.backward()
            optimizer.step()
```

**Implementation plan:**
1. Create `mess/search/layer_fusion.py`
   - LayerFusionNetwork (attention-based or MLP)
   - ContrastiveFusionTrainer (triplet loss)

2. Create `research/scripts/train_fusion.py`
   - Load MERT features [13, 768]
   - Generate triplets from metadata
   - Train fusion network
   - Save checkpoint
   - Log to MLflow

3. Update `mess/search.py`
   - Add build_fused_features() function
   - Pre-compute fused embeddings
   - Use in search

4. Create `research/notebooks/fusion_analysis.ipynb`
   - Visualize learned layer weights
   - Compare to linear probing results
   - Evaluate search quality

---

### OPTION 3: Neural Layer Discovery
**Location**: `mess/probing/neural_probe.py` (new file)

**What it does:**
- Replace Ridge regression with neural network probe
- Multi-task learning: predict all 15 targets simultaneously
- Learn which layers encode which aspects

**Training data needed:**
- Same proxy targets from `targets.py`
- MERT layer embeddings
- (Already have this!)

**Pros:**
- Can capture nonlinear relationships
- Multi-task learning shares knowledge across targets
- More expressive than linear probing

**Cons:**
- Linear probing already works well (R² > 0.9)
- Adds complexity without clear benefit
- Harder to interpret than linear models

**Code structure:**
```python
# mess/probing/neural_probe.py
class NeuralProbe(nn.Module):
    """Multi-task neural probe for layer discovery."""

    def __init__(self, input_dim=768, num_targets=15):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.target_heads = nn.ModuleList([
            nn.Linear(128, 1) for _ in range(num_targets)
        ])

    def forward(self, layer_embedding):
        shared = self.shared_encoder(layer_embedding)
        predictions = [head(shared) for head in self.target_heads]
        return torch.stack(predictions, dim=1)
```

---

### OPTION 4: Learned Similarity Metric
**Location**: `mess/search/metric_learning.py` (new file)

**What it does:**
- Siamese or Triplet network
- Learn what "similar" means for music
- Replace fixed cosine similarity

**Training data needed:**
- Triplets or pairs with similarity labels
- Could use metadata (same composer = similar)
- Could crowdsource judgments

**Pros:**
- Captures musical similarity better than dot product
- Published research area (metric learning for audio)

**Cons:**
- Adds inference cost (neural forward pass per comparison)
- Harder to scale than FAISS
- May not integrate cleanly with FAISS

**Code structure:**
```python
# mess/search/metric_learning.py
class SiameseNetwork(nn.Module):
    """Learn musical similarity metric."""

    def __init__(self, input_dim=768, embedding_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)

    def similarity(self, x1, x2):
        e1 = self.forward(x1)
        e2 = self.forward(x2)
        return F.cosine_similarity(e1, e2)
```

---

## Training Data Sources

### A. Self-Supervised (No labels needed)
```python
# Use existing metadata as proxy for similarity
# Same composer → similar
# Same tempo range → similar
# Same key → similar

triplets = []
for anchor in dataset:
    positive = random_track_same_composer(anchor)
    negative = random_track_different_composer(anchor)
    triplets.append((anchor, positive, negative))
```

### B. Proxy Labels from Features
```python
# Use extracted targets as supervision
# Tracks with similar spectral_centroid → similar brightness
# Can generate infinite triplets from proxy targets
```

### C. User Annotations (Expensive but accurate)
```python
# Crowdsource pairwise judgments
# "Which track is more similar to Beethoven Op. 27?"
# → Collect human preferences
```

---

## Recommendation

**Build OPTION 2: Layer Fusion Network**

**Why:**
1. ✅ Solves "what layers are what": Learns layer importance via attention
2. ✅ Improves search: Better embeddings → better recommendations
3. ✅ Works with existing data: No new labels needed (use metadata)
4. ✅ Fast to prototype: Small model, trains in <1 hour
5. ✅ Integrates cleanly: Preprocessing step, doesn't break search
6. ✅ Validates linear probing: Can compare learned weights vs. R² scores

---

## Data Flow Diagram

```
CURRENT PIPELINE (No PyTorch Training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Audio Files
    ↓
[MERT Extraction] ← Frozen inference only
    ↓
MERT Features [13 layers, 768 dims]
    ↓
[Layer Discovery] ← sklearn Ridge regression (non-neural)
    ↓
Layer Mappings (R² scores saved to JSON)
    ↓
[Search] ← FAISS cosine similarity (unsupervised)
    ↓
Results


PROPOSED: Add Layer Fusion Network
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Audio Files
    ↓
[MERT Extraction] ← Frozen
    ↓
MERT Features [13, 768]
    ↓
[Layer Fusion Network] ← NEW! PyTorch training here
  • Attention-based layer weighting
  • Triplet loss on metadata
  • Learns optimal combination
    ↓
Fused Embeddings [fusion_dim]
    ↓
[Search] ← FAISS on learned embeddings
    ↓
Better Results
```

---

## Next Steps

1. **Decide on approach** (recommend Option 2)
2. **Choose training labels** (recommend self-supervised from metadata)
3. **Create implementation plan**
4. **Set up training infrastructure** (PyTorch, data loaders, MLflow tracking)
5. **Implement and validate** against linear probing baselines
