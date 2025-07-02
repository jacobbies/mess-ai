import librosa
import torch
import torchaudio
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import warnings
import os


class FeatureExtractor:
    """MERT-based feature extractor for music similarity analysis."""
    
    def __init__(self, model_name="m-a-p/MERT-v1-95M", device=None, cache_dir=None):
        """
        Initialize MERT feature extractor.
        
        Args:
            model_name: Hugging Face model name for MERT
            device: torch device ('mps', 'cuda', or 'cpu'). Auto-detects if None.
            cache_dir: Directory for model cache
        """
        self.model_name = model_name
        # Prefer MPS on Apple Silicon, then CUDA, then CPU
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.cache_dir = cache_dir
        
        # Default output directory
        self.output_dir = Path(__file__).parent.parent.parent.parent / "data" / "processed" / "features"
        
        # MERT requires 24kHz audio
        self.target_sample_rate = 24000
        self.segment_duration = 5.0  # 5-second segments
        self.overlap_ratio = 0.5  # 50% overlap between segments
        
        self._load_model()
        
    def _load_model(self):
        """Load MERT model and processor."""
        try:
            logging.info(f"Loading MERT model: {self.model_name}")
            
            # Load processor and model
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=self.cache_dir
            )
            
            # Try to move to target device with fallback
            try:
                self.model = self.model.to(self.device)
                # Test a small tensor to verify MPS compatibility
                if self.device == 'mps':
                    test_tensor = torch.randn(1, 1).to(self.device)
                    _ = test_tensor + 1  # Simple operation test
                logging.info(f"MERT model loaded successfully on {self.device}")
            except Exception as device_error:
                if self.device == 'mps':
                    logging.warning(f"MPS failed ({device_error}), falling back to CPU")
                    self.device = 'cpu'
                    self.model = self.model.to(self.device)
                    logging.info(f"MERT model loaded successfully on {self.device}")
                else:
                    raise device_error
            
            self.model.eval()
            
        except Exception as e:
            logging.error(f"Failed to load MERT model: {e}")
            raise
    
    def _preprocess_audio(self, audio_path):
        """
        Load and preprocess audio for MERT.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        try:
            # Load audio
            audio, orig_sr = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Resample to 24kHz if needed
            if orig_sr != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, 
                    new_freq=self.target_sample_rate
                )
                audio = resampler(audio)
            
            # Convert to numpy and squeeze
            audio = audio.squeeze().numpy()
            
            return audio
            
        except Exception as e:
            logging.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def _segment_audio(self, audio):
        """
        Segment audio into overlapping windows.
        
        Args:
            audio: Audio array
            
        Returns:
            List of audio segments
        """
        segment_samples = int(self.segment_duration * self.target_sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap_ratio))
        
        segments = []
        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            end = start + segment_samples
            segments.append(audio[start:end])
        
        # Add final segment if remaining audio
        if len(audio) % hop_samples != 0:
            segments.append(audio[-segment_samples:])
            
        return segments
    
    def _extract_mert_features(self, audio_segment):
        """
        Extract MERT features from audio segment.
        
        Args:
            audio_segment: Audio array for single segment
            
        Returns:
            MERT features [num_layers, time_steps, feature_dim]
        """
        try:
            # Process audio with MERT processor
            inputs = self.processor(
                audio_segment,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Stack all hidden states [num_layers, batch, time, features]
                hidden_states = torch.stack(outputs.hidden_states)
                
                # Remove batch dimension [num_layers, time, features]
                hidden_states = hidden_states.squeeze(1)
                
            return hidden_states.cpu().numpy()
            
        except Exception as e:
            logging.error(f"Error extracting MERT features: {e}")
            raise
    
    def extract_track_features(self, audio_path, output_dir=None, track_id=None, dataset=None):
        """
        Extract features for a single track.
        
        Args:
            audio_path: Path to audio file
            output_dir: Directory to save features (optional)
            track_id: Custom track ID for file naming (optional)
            dataset: Dataset name for organizing features (optional)
            
        Returns:
            dict with 'raw', 'segments', 'aggregated' features
        """
        try:
            logging.info(f"Extracting features for: {audio_path}")
            
            # Preprocess audio
            audio = self._preprocess_audio(audio_path)
            
            # Segment audio
            segments = self._segment_audio(audio)
            
            # Extract features for each segment
            segment_features = []
            for segment in segments:
                features = self._extract_mert_features(segment)
                segment_features.append(features)
            
            # Convert to numpy array [num_segments, num_layers, time_steps, feature_dim]
            segment_features = np.array(segment_features)
            
            # Create different feature representations
            results = {
                'raw': segment_features,  # Full features
                'segments': np.mean(segment_features, axis=2),  # Time-averaged per segment
                'aggregated': np.mean(segment_features, axis=(0, 2))  # Single track vector
            }
            
            # Save features if output directory provided
            if output_dir:
                self._save_features(results, audio_path, output_dir, track_id, dataset)
            
            return results
            
        except Exception as e:
            logging.error(f"Error extracting track features for {audio_path}: {e}")
            raise
    
    def _save_features(self, features, audio_path, output_dir, track_id=None, dataset=None):
        """Save extracted features to disk."""
        output_dir = Path(output_dir)
        
        # Use custom track_id or fallback to filename
        filename = track_id if track_id else Path(audio_path).stem
        
        # Add dataset subdirectory if specified
        if dataset:
            output_dir = output_dir / dataset
        
        # Save to respective directories
        for feature_type, data in features.items():
            type_dir = output_dir / feature_type
            type_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = type_dir / f"{filename}.npy"
            np.save(save_path, data)
            
        logging.info(f"Features saved for {filename}")
    
    def extract_dataset_features(self, audio_dir, output_dir, file_pattern="*.wav"):
        """
        Extract features for entire dataset.
        
        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save extracted features
            file_pattern: File pattern to match
        """
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)
        
        # Find all audio files
        audio_files = list(audio_dir.glob(file_pattern))
        
        if not audio_files:
            logging.warning(f"No audio files found in {audio_dir} with pattern {file_pattern}")
            return
        
        logging.info(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        for audio_file in tqdm(audio_files, desc="Extracting features"):
            try:
                self.extract_track_features(audio_file, output_dir)
            except Exception as e:
                logging.error(f"Failed to process {audio_file}: {e}")
                continue
        
        logging.info(f"Feature extraction complete. Results saved to {output_dir}")


# Convenience function for easy usage
def extract_features(audio_dir="data/smd/wav-44", output_dir="data/processed/features"):
    """
    Extract MERT features for SMD dataset.
    
    Args:
        audio_dir: Directory containing wav files
        output_dir: Directory to save features
    """
    extractor = FeatureExtractor()
    extractor.extract_dataset_features(audio_dir, output_dir)
