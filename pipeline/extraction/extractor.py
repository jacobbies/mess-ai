"""
MERT Feature Extraction Pipeline

This module extracts embeddings from audio files using the MERT (Music Understanding Model)
transformer architecture. MERT is pre-trained on 160K hours of music and produces hierarchical
representations across 13 layers, where each layer captures different musical aspects.

Key Concepts:
-----------
**Embeddings**: Dense vector representations (768 dimensions) that capture musical semantics.
  Similar music → similar embeddings (measured by cosine similarity).

**Layer Specialization**: Each of MERT's 13 layers encodes different musical aspects:
  - Layer 0: Spectral brightness (R²=0.944) - timbral quality
  - Layer 1: Timbral texture (R²=0.922) - instrumental characteristics
  - Layer 2: Acoustic structure (R²=0.933) - resonance patterns
  - Layers 3-12: Various other aspects (some validated, some exploratory)

**Why 24kHz Sample Rate?**: MERT was pre-trained on 24kHz audio. Using the same rate ensures:
  - No frequency information is lost or added
  - Model sees the same frequency range it was trained on
  - Nyquist limit: max frequency = 12kHz (sufficient for music analysis)

**Why 5-Second Segments?**: Balances context window vs memory/computation:
  - Too short: Misses musical structure (phrases, motifs)
  - Too long: Excessive memory, diminishing returns
  - 5 seconds: Captures local patterns, manageable memory

**Output Formats**:
  - raw: [segments, 13 layers, time_steps, 768 dims] - Full temporal resolution
  - segments: [segments, 13 layers, 768 dims] - Time-averaged per segment
  - aggregated: [13 layers, 768 dims] - Track-level (used for similarity search)

Pipeline Flow:
-------------
1. Load audio file (any format, any sample rate)
2. Convert to mono (if stereo)
3. Resample to 24kHz (MERT's expected rate)
4. Segment into 5-second overlapping windows (50% overlap)
5. Extract MERT features for each segment (13 layers × time × 768 dims)
6. Aggregate features (time-average within segments, then across segments)
7. Save three representations (raw, segments, aggregated)

Usage:
------
    from pipeline.extraction.extractor import FeatureExtractor

    extractor = FeatureExtractor()
    features = extractor.extract_track_features("path/to/audio.wav")

    # features['aggregated'] is [13, 768] - ready for similarity search
    # features['raw'] is full temporal data - for detailed analysis
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
from .config import pipeline_config


class FeatureExtractor:
    """
    MERT-based feature extractor for music similarity analysis.

    Extracts hierarchical embeddings from audio using the MERT transformer model.
    Each of MERT's 13 layers captures different musical aspects, validated through
    systematic layer discovery experiments.

    The extractor handles the full pipeline: audio loading → preprocessing →
    segmentation → MERT feature extraction → aggregation → saving.

    See module docstring for detailed domain concepts and pipeline flow.
    """
    
    def __init__(self, model_name=None, device=None, cache_dir=None, output_dir=None):
        """
        Initialize MERT feature extractor.
        
        Args:
            model_name: Hugging Face model name for MERT (uses config default if None)
            device: torch device ('mps', 'cuda', or 'cpu'). Uses config auto-detection if None.
            cache_dir: Directory for model cache (uses config if None)
            output_dir: Output directory for features (uses config if None)
        """
        # Use configuration with parameter overrides
        self.model_name = model_name or pipeline_config.model_name
        self.device = device or pipeline_config.device
        self.cache_dir = cache_dir or pipeline_config.cache_dir
        self.output_dir = Path(output_dir) if output_dir else pipeline_config.output_dir
        
        # Audio processing settings from config
        self.target_sample_rate = pipeline_config.target_sample_rate
        self.segment_duration = pipeline_config.segment_duration
        self.overlap_ratio = pipeline_config.overlap_ratio
        
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
        Load and preprocess audio for MERT feature extraction.

        Preprocessing Steps:
        1. Load audio file (supports WAV, MP3, FLAC, etc.)
        2. Convert stereo → mono (average channels) if needed
        3. Resample to 24kHz (MERT's pre-training rate)

        Why These Steps?
        ---------------
        **Mono conversion**: MERT expects single-channel audio. Averaging channels
          preserves all frequency information while reducing to one channel.

        **Resampling to 24kHz**: MERT was pre-trained on 24kHz audio. Using the same
          rate ensures:
          - Model sees the same frequency range it was trained on
          - No information loss (original rate is usually 44.1kHz, which is higher)
          - Nyquist limit: 24kHz → max frequency 12kHz (sufficient for music)
          - Lower rate → faster processing, smaller memory footprint

        Args:
            audio_path: Path to audio file

        Returns:
            Preprocessed audio as numpy array (1D, 24kHz sample rate)
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
        Segment audio into overlapping windows for MERT processing.

        Why Segment Audio?
        -----------------
        - **Transformer context window**: MERT's transformer has a fixed maximum context length
        - **Memory constraints**: Processing 3-minute tracks in one pass = too much memory
        - **Local pattern capture**: 5-second segments capture local musical patterns (phrases, motifs)

        Segmentation Parameters (from config):
        - **segment_duration**: 5.0 seconds (120,000 samples at 24kHz)
        - **overlap_ratio**: 0.5 (50% overlap between consecutive segments)

        Why 50% Overlap?
        ----------------
        - **Smooth transitions**: Avoids boundary artifacts (musical phrases split mid-segment)
        - **Redundancy**: Each moment in audio appears in 2 segments → more robust features
        - **Standard practice**: Common in audio ML (used in speech recognition, music analysis)

        Example:
        --------
        30-second audio (720,000 samples at 24kHz):
        - Segment 1: samples 0-120,000 (0-5 sec)
        - Segment 2: samples 60,000-180,000 (2.5-7.5 sec) ← 50% overlap
        - Segment 3: samples 120,000-240,000 (5-10 sec)
        - ... (11 segments total)

        Args:
            audio: Audio array (numpy, 1D, 24kHz)

        Returns:
            List of audio segments (each segment is 5 seconds = 120,000 samples)
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
        Extract MERT embeddings from a single audio segment.

        What MERT Does:
        --------------
        MERT (Music Understanding Model) is a transformer-based model trained on 160K hours
        of music using self-supervised learning. It outputs hierarchical representations:

        - **13 layers**: Each layer captures different musical aspects (low-level → high-level)
        - **768 dimensions**: Dense vector representation (like word embeddings, but for audio)
        - **Temporal resolution**: Features for each time step (usually ~100-300 time steps per 5s segment)

        Layer Specialization (empirically validated):
        - Layer 0: Spectral brightness (R²=0.944) - "how bright/dark does this sound?"
        - Layer 1: Timbral texture (R²=0.922) - "what instrument characteristics?"
        - Layer 2: Acoustic structure (R²=0.933) - "what resonance patterns?"

        Output Shape Explained:
        ----------------------
        [num_layers, time_steps, feature_dim] = [13, ~200, 768]
        - 13 layers: MERT's hierarchical representations
        - ~200 time_steps: Temporal resolution (varies by segment length)
        - 768 feature_dim: Embedding dimensionality (fixed by model architecture)

        Why Hidden States?
        -----------------
        We extract `output_hidden_states=True` to get ALL 13 layers, not just the final layer.
        This is crucial because:
        - Early layers encode low-level features (brightness, texture)
        - Middle layers encode mid-level patterns (structure, rhythm)
        - Late layers encode high-level concepts (style, genre)
        - We want ALL layers for layer-specific similarity search!

        Args:
            audio_segment: Audio array for single 5-second segment (numpy, 1D, 24kHz)

        Returns:
            MERT features as numpy array with shape [13, time_steps, 768]
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
        Extract MERT features for a single audio track with multiple output formats.

        Pipeline Flow:
        -------------
        1. Preprocess audio (load → mono → resample to 24kHz)
        2. Segment into 5-second overlapping windows (50% overlap)
        3. Extract MERT features for each segment (13 layers × time × 768 dims)
        4. Generate three output formats (raw, segments, aggregated)

        Output Formats Explained:
        ------------------------
        This method returns THREE representations of the same audio, each useful for different purposes:

        **'raw'**: [num_segments, 13, time_steps, 768]
          - Full temporal resolution for detailed analysis
          - Example: [11 segments, 13 layers, ~200 time_steps, 768 dims]
          - Use case: Temporal analysis, beat tracking, onset detection
          - Size: ~10 MB per 30-second track

        **'segments'**: [num_segments, 13, 768]
          - Time-averaged within each segment (time dimension removed)
          - Example: [11 segments, 13 layers, 768 dims]
          - Use case: Segment-level similarity, chorus detection
          - Size: ~500 KB per 30-second track

        **'aggregated'**: [13, 768]
          - Single track-level representation (segments + time averaged)
          - Example: [13 layers, 768 dims]
          - Use case: **Track similarity search** (this is what we use!)
          - Size: ~40 KB per track

        Why Multiple Formats?
        --------------------
        - **Flexibility**: Different research questions need different temporal resolutions
        - **Speed**: Aggregated features are tiny → fast similarity search
        - **Completeness**: Raw features preserve all temporal information if needed later

        For Music Similarity:
        --------------------
        We primarily use **'aggregated'** because:
        - Track-level similarity (not segment-level)
        - Fast cosine similarity computation (13×768 = ~10K floats)
        - Layer selection (use Layer 0 for brightness, Layer 1 for texture, etc.)

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            output_dir: Directory to save features (optional, uses config default if None)
            track_id: Custom track ID for file naming (optional, uses filename if None)
            dataset: Dataset name for organizing features into subdirectories (optional)

        Returns:
            dict with three keys:
              - 'raw': Full features [segments, 13, time, 768]
              - 'segments': Time-averaged [segments, 13, 768]
              - 'aggregated': Track-level [13, 768]
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


def extract_features(audio_dir="data/smd/wav-44", output_dir="data/processed/features"):
    """
    Extract MERT features for SMD dataset.
    
    Args:
        audio_dir: Directory containing wav files
        output_dir: Directory to save features
    """
    extractor = FeatureExtractor()
    extractor.extract_dataset_features(audio_dir, output_dir)
