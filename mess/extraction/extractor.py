"""
MERT Feature Extraction Pipeline

Extracts hierarchical embeddings from audio using MERT transformer (13 layers, 768 dims).

Output Formats:
  - raw: [segments, 13, time, 768] - full temporal resolution
  - segments: [segments, 13, 768] - time-averaged per segment
  - aggregated: [13, 768] - track-level (for similarity search)

Pipeline: audio → mono → 24kHz → 5s segments → MERT → aggregate

Usage:
    extractor = FeatureExtractor()
    features = extractor.extract_track_features("audio.wav")
"""

import torch
import torchaudio
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers.models.wav2vec2 import Wav2Vec2FeatureExtractor
from transformers.models.auto.modeling_auto import AutoModel

# Import unified configuration
from .config import mess_config


class FeatureExtractor:
    """
    MERT-based feature extractor for music similarity.

    Handles: audio loading → preprocessing → segmentation → MERT → aggregation
    """
    
    def __init__(self, model_name=None, device=None, cache_dir=None, output_dir=None, batch_size=None):
        """
        Initialize MERT feature extractor.

        Args:
            model_name: Hugging Face model name (default: m-a-p/MERT-v1-330M)
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
            cache_dir: Model cache directory
            output_dir: Feature output directory
            batch_size: Segments per batch (default: 16 for CUDA, 8 for MPS, 4 for CPU)
        """
        # Use configuration with parameter overrides
        self.model_name = model_name or mess_config.model_name
        self.device = device or mess_config.device
        self.cache_dir = cache_dir or mess_config.cache_dir
        self.output_dir = Path(output_dir) if output_dir else mess_config.output_dir

        # Audio processing settings from config
        self.target_sample_rate = mess_config.target_sample_rate
        self.segment_duration = mess_config.segment_duration
        self.overlap_ratio = mess_config.overlap_ratio

        self._load_model()

        # Set batch size based on device if not specified
        if batch_size is None:
            if self.device == 'cuda':
                self.batch_size = 16  # CUDA GPUs handle larger batches well
            elif self.device == 'mps':
                self.batch_size = 8   # M3 Pro optimal
            else:
                self.batch_size = 4   # CPU: smaller batches
        else:
            self.batch_size = batch_size
        
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
        Load audio and preprocess for MERT (mono, 24kHz).

        Args:
            audio_path: Path to audio file

        Returns:
            Audio array (1D, 24kHz)
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
        Segment audio into 5-second overlapping windows (50% overlap).

        Args:
            audio: Audio array (1D, 24kHz)

        Returns:
            List of segments [120k samples each]
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
        Extract MERT embeddings from a single segment.

        Args:
            audio_segment: Audio array [120k samples, 24kHz]

        Returns:
            Features [13, time_steps, 768]
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

    def _extract_mert_features_batched(self, audio_segments):
        """
        Extract MERT features for multiple segments in parallel.

        Processes segments in batches for GPU efficiency (2-5x faster than sequential).
        - CUDA: batch_size=16 (default)
        - MPS: batch_size=8
        - CPU: batch_size=4

        Args:
            audio_segments: List of audio arrays [120k samples each]

        Returns:
            Features [num_segments, 13, time_steps, 768]
        """
        try:
            all_features = []
            num_segments = len(audio_segments)

            # Process in batches
            for batch_start in range(0, num_segments, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_segments)
                batch_segments = audio_segments[batch_start:batch_end]

                # Process batch through MERT processor
                # Note: processor handles batching internally when given a list
                inputs = self.processor(
                    batch_segments,
                    sampling_rate=self.target_sample_rate,
                    return_tensors="pt",
                    padding=True  # Pad to same length if needed
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Extract features for entire batch
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)

                    # Stack all hidden states [num_layers, batch_size, time, features]
                    hidden_states = torch.stack(outputs.hidden_states)

                    # Transpose to [batch_size, num_layers, time, features]
                    hidden_states = hidden_states.permute(1, 0, 2, 3)

                # Move to CPU and convert to numpy
                batch_features = hidden_states.cpu().numpy()
                all_features.append(batch_features)

            # Concatenate all batches [num_segments, num_layers, time_steps, feature_dim]
            return np.concatenate(all_features, axis=0)

        except Exception as e:
            logging.error(f"Error extracting MERT features (batched): {e}")
            raise

    def _features_exist(self, audio_path, output_dir, track_id=None, dataset=None):
        """
        Check if features already exist on disk.

        Args:
            audio_path: Path to audio file
            output_dir: Feature directory
            track_id: Custom track ID (optional)
            dataset: Dataset name (optional)

        Returns:
            True if features exist, False otherwise
        """
        if not output_dir:
            return False

        output_dir = Path(output_dir)
        filename = track_id if track_id else Path(audio_path).stem

        # Add dataset subdirectory if specified
        if dataset:
            output_dir = output_dir / dataset

        # Check if aggregated features exist
        aggregated_path = output_dir / "aggregated" / f"{filename}.npy"
        return aggregated_path.exists()

    def _load_existing_features(self, audio_path, output_dir, track_id=None, dataset=None):
        """
        Load pre-extracted features from disk (~1000x faster than re-extracting).

        Args:
            audio_path: Path to audio file
            output_dir: Feature directory
            track_id: Custom track ID (optional)
            dataset: Dataset name (optional)

        Returns:
            Dict with 'raw', 'segments', 'aggregated' keys, or None if missing
        """
        try:
            output_dir = Path(output_dir)
            filename = track_id if track_id else Path(audio_path).stem

            # Add dataset subdirectory if specified
            if dataset:
                output_dir = output_dir / dataset

            # Load all three feature types
            features = {}
            for feature_type in ['raw', 'segments', 'aggregated']:
                feature_path = output_dir / feature_type / f"{filename}.npy"
                if not feature_path.exists():
                    return None  # Missing features, need to re-extract
                features[feature_type] = np.load(feature_path)

            return features

        except Exception as e:
            logging.warning(f"Error loading existing features for {audio_path}: {e}")
            return None

    def extract_track_features(self, audio_path, output_dir=None, track_id=None, dataset=None, skip_existing=True):
        """
        Extract MERT features for an audio track.

        Pipeline: audio → preprocess → segment → MERT (batched) → aggregate

        Output formats:
          - 'raw': [segments, 13, time, 768] - full temporal resolution
          - 'segments': [segments, 13, 768] - time-averaged per segment
          - 'aggregated': [13, 768] - track-level (used for similarity search)

        Args:
            audio_path: Path to audio file
            output_dir: Feature save directory (optional)
            track_id: Custom track ID (optional)
            dataset: Dataset name for subdirectory (optional)
            skip_existing: Skip if features already exist (default: True)

        Returns:
            Dict with 'raw', 'segments', 'aggregated' keys
        """
        try:
            # Check cache if skip_existing enabled
            if skip_existing and output_dir:
                existing = self._load_existing_features(audio_path, output_dir, track_id, dataset)
                if existing is not None:
                    logging.info(f"Loading cached features for: {audio_path}")
                    return existing

            logging.info(f"Extracting features for: {audio_path}")

            # Preprocess audio
            audio = self._preprocess_audio(audio_path)

            # Segment audio
            segments = self._segment_audio(audio)

            # Extract features in batches (GPU-optimized)
            segment_features = self._extract_mert_features_batched(segments)

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
    
    def extract_dataset_features(self, audio_dir, output_dir, file_pattern="*.wav", skip_existing=True):
        """
        Extract features for entire dataset.

        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save extracted features
            file_pattern: File pattern to match (default: "*.wav")
            skip_existing: Skip already-extracted files (default: True)
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
                self.extract_track_features(audio_file, output_dir, skip_existing=skip_existing)
            except Exception as e:
                logging.error(f"Failed to process {audio_file}: {e}")
                continue

        logging.info(f"Feature extraction complete. Results saved to {output_dir}")


def extract_features(audio_dir="data/smd/wav-44", output_dir="data/processed/features"):
    """
    Extract MERT features for SMD dataset.
    
    Args:
        audio_dir: Directory containing wav files
        output_dir: Directory to save features
    """
    extractor = FeatureExtractor()
    extractor.extract_dataset_features(audio_dir, output_dir)
