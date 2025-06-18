# mess-ai 🎵

A deep learning-powered music similarity search system that finds musically similar classical pieces using state-of-the-art MERT (Music Understanding Model) embeddings.

## 🚀 Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start the web server:**
```bash
cd src/api && python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

3. **Open your browser** to `http://localhost:8000`

## ✨ Features

### 🎼 Music Player
- Interactive web-based music player
- Real-time waveform visualization (on-demand)
- Support for 50 classical music tracks from SMD dataset

### 🔍 AI-Powered Similarity Search
- **MERT-based recommendations** using transformer embeddings (13 layers, 768 dimensions)
- **Cosine similarity search** across 94GB of precomputed features
- **Real-time recommendations** with similarity percentage scores
- **User-controlled discovery** - click "Get Recommendations" when ready

### 🎯 Smart UI
- Background waveform loading with caching
- Clickable recommendations with seamless playback
- Responsive Bootstrap 5 interface
- Loading states and error handling

## 🏗️ Architecture

### Backend
- **FastAPI** web server with async endpoints
- **MusicRecommender** class for similarity search
- **FeatureExtractor** for MERT embedding generation
- **Apple Silicon (MPS)** acceleration support

### ML Pipeline
- **MERT-v1-95M** transformer model for music understanding
- **Multi-scale features**: raw, segments, and aggregated representations
- **Hierarchical storage**: 150 .npy files organized by feature type
- **Background processing**: ~2.6 minutes to process full dataset

### Data Organization
```
data/
├── smd/wav-44/          # 50 classical recordings (44kHz WAV)
├── processed/features/   # MERT embeddings (94GB total)
│   ├── raw/             # Full temporal features
│   ├── segments/        # Time-averaged segments  
│   └── aggregated/      # Track-level vectors
└── models/              # Training checkpoints (future)
```

## 🎹 Dataset

**Saarland Music Dataset (SMD)**: 50 classical recordings featuring:
- Bach, Beethoven, Chopin, Mozart, Brahms, and more
- High-quality WAV files at 44kHz sample rate
- Performance annotations and metadata
- MIDI representations for symbolic analysis

## 🛠️ Tech Stack

### Core Technologies
- **Backend**: Python 3.11+, FastAPI, PyTorch 2.6+
- **ML**: Transformers (Hugging Face), MERT, scikit-learn
- **Audio**: librosa, torchaudio, soundfile
- **Frontend**: HTML5, Bootstrap 5, Vanilla JavaScript

### Performance
- **Apple Silicon**: MPS acceleration on M3 Pro
- **Processing Speed**: 3.1s average per track
- **Memory Efficient**: Smart caching and background loading
- **Real-time Search**: Sub-second similarity calculations

## 📊 Project Status

### ✅ Completed
- [x] **MERT Feature Extraction Pipeline** - Complete with MPS acceleration
- [x] **Similarity Search System** - Cosine similarity with MERT embeddings  
- [x] **Web Interface** - Interactive player with recommendations
- [x] **API Endpoints** - `/recommend/{track}`, `/tracks`, `/waveform/{track}`
- [x] **Data Processing** - 94GB of precomputed features ready
- [x] **Background Loading** - Optimized UX with caching

### 🚧 In Progress
- [ ] Model fine-tuning on SMD dataset
- [ ] Alternative similarity metrics (Euclidean, Manhattan)
- [ ] Advanced recommendation algorithms

### 📋 Planned
- [ ] Docker containerization
- [ ] AWS S3 integration for cloud storage
- [ ] Expanded dataset support
- [ ] User preference learning
- [ ] Comprehensive testing suite

## 🚀 Performance Metrics

- **Feature Extraction**: 2.6 minutes for 50 tracks
- **Similarity Search**: <100ms per query
- **Storage**: 94GB features, 2MB aggregated vectors
- **Accuracy**: Based on MERT state-of-the-art music understanding

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.