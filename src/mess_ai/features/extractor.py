import librosa
import torch
import numpy as np
from pathlib import Path

class FeatureExtractor:
    def __init__(self, sr=22050, n_mfcc=20, embedding_dim=128):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.embedding_dim = embedding_dim
        
    def extract_features(self, audio_path):
        """Extract audio features and convert to embedding"""
        # Load audio file
        y, sr = librosa.load(str(audio_path), sr=self.sr)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Compute statistics
        features = []
        for feat in [mfccs, chroma, spectral_contrast]:
            features.extend([
                np.mean(feat, axis=1),
                np.std(feat, axis=1)
            ])
        
        # Flatten and concatenate
        features = np.concatenate([f.flatten() for f in features])
        
        # Normalize
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # Convert to embedding with correct dimensionality
        # Simple dimensionality reduction for now
        if len(features) > self.embedding_dim:
            # Use PCA-like approach for dimension reduction
            features = features[:self.embedding_dim]
        else:
            # Pad if needed
            padding = np.zeros(self.embedding_dim - len(features))
            features = np.concatenate([features, padding])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def build_embeddings(self, audio_dir):
        """Build embeddings for all audio files in directory"""
        audio_dir = Path(audio_dir)
        embeddings = {}
        
        for audio_file in audio_dir.glob('*.wav'):
            print(f"Processing {audio_file.name}...")
            embedding = self.extract_features(audio_file)
            embeddings[audio_file.name] = embedding
            
        return embeddings
