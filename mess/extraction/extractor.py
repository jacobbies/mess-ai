"""
MERT Feature Extraction — Core Extractor

Model lifecycle, inference, and track-level feature extraction.
Dataset-level batch processing lives in pipeline.py.

Output Formats:
  - raw: [segments, 13, time, 768] - full temporal resolution
  - segments: [segments, 13, 768] - time-averaged per segment
  - aggregated: [13, 768] - track-level (for similarity search)

Usage:
    extractor = FeatureExtractor()
    features = extractor.extract_track_features("audio.wav")
"""

import torch
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any

from transformers.models.wav2vec2 import Wav2Vec2FeatureExtractor
from transformers.models.auto.modeling_auto import AutoModel

from ..config import mess_config
from .audio import load_audio, segment_audio, validate_audio_file
from .cache import FeatureCache


class FeatureExtractor:
    """
    MERT-based feature extractor for music similarity.

    Handles: model loading → inference → track-level extraction.
    Audio preprocessing delegated to audio.py, persistence to cache.py.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Initialize MERT feature extractor.

        Args:
            model_name: Hugging Face model name (default: m-a-p/MERT-v1-330M)
            device: 'cuda', 'mps', or 'cpu' (auto-detected if None)
            cache_dir: Model cache directory
            output_dir: Feature output directory
            batch_size: Segments per batch (default: 16 for CUDA, 8 for MPS, 4 for CPU)
        """
        self.model_name = model_name or mess_config.model_name
        self.device = device or mess_config.device
        self.cache_dir = cache_dir or mess_config.cache_dir
        self.output_dir = Path(output_dir) if output_dir else None

        # Audio processing settings from config
        self.target_sample_rate = mess_config.target_sample_rate
        self.segment_duration = mess_config.segment_duration
        self.overlap_ratio = mess_config.overlap_ratio

        # Feature cache for persistence
        self._cache = FeatureCache()

        self._load_model()

        # Set batch size based on device if not specified
        if batch_size is None:
            if self.device == 'cuda':
                self.batch_size = 16
            elif self.device == 'mps':
                self.batch_size = 8
            else:
                self.batch_size = 4
        else:
            self.batch_size = batch_size

    def _load_model(self) -> None:
        """Load MERT model and processor."""
        try:
            logging.info(f"Loading MERT model: {self.model_name}")

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

            self._move_to_device()
            self.model.eval()

        except Exception as e:
            logging.error(f"Failed to load MERT model: {e}")
            raise

    def _move_to_device(self) -> None:
        """
        Move model to target device with automatic fallback chain.

        Fallback priority: CUDA → MPS → CPU
        """
        DEVICE_FALLBACK = {
            'cuda': ['cuda', 'mps', 'cpu'],
            'mps': ['mps', 'cpu'],
            'cpu': ['cpu']
        }

        device_chain = DEVICE_FALLBACK.get(self.device, ['cpu'])
        last_error = None

        for device in device_chain:
            try:
                self.model = self.model.to(device)

                # Test device with a simple operation
                test_tensor = torch.randn(1, 1).to(device)
                _ = test_tensor + 1

                if device != self.device:
                    logging.warning(f"{self.device.upper()} unavailable, using {device.upper()}")
                    self.device = device

                logging.info(f"MERT model loaded successfully on {device.upper()}")
                return

            except Exception as e:
                last_error = e
                if device != device_chain[-1]:
                    logging.warning(f"{device.upper()} failed ({e}), trying next device...")
                continue

        raise RuntimeError(f"Failed to load model on any device. Last error: {last_error}")

    def _extract_mert_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extract MERT embeddings from a single segment.

        Args:
            audio_segment: Audio array [120k samples, 24kHz]

        Returns:
            Features [13, time_steps, 768]
        """
        try:
            inputs = self.processor(
                audio_segment,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = torch.stack(outputs.hidden_states)
                hidden_states = hidden_states.squeeze(1)

            return hidden_states.cpu().numpy()

        except Exception as e:
            logging.error(f"Error extracting MERT features: {e}")
            raise

    def _extract_mert_features_batched(self, audio_segments: List[np.ndarray]) -> np.ndarray:
        """
        Extract MERT features for multiple segments in parallel batches.

        Processes segments in batches for GPU efficiency (2-5x faster than sequential).

        Args:
            audio_segments: List of audio arrays [120k samples each]

        Returns:
            Features [num_segments, 13, time_steps, 768]
        """
        try:
            all_features = []
            num_segments = len(audio_segments)

            for batch_start in range(0, num_segments, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_segments)
                batch_segments = audio_segments[batch_start:batch_end]

                inputs = self.processor(
                    batch_segments,
                    sampling_rate=self.target_sample_rate,
                    return_tensors="pt",
                    padding=True
                )

                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = torch.stack(outputs.hidden_states)
                    hidden_states = hidden_states.permute(1, 0, 2, 3)

                batch_features = hidden_states.cpu().numpy()
                all_features.append(batch_features)

            return np.concatenate(all_features, axis=0)

        except Exception as e:
            logging.error(f"Error extracting MERT features (batched): {e}")
            raise

    def extract_track_features(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        track_id: Optional[str] = None,
        dataset: Optional[str] = None,
        skip_existing: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract MERT features for an audio track.

        Pipeline: audio → preprocess → segment → MERT (batched) → aggregate

        Output formats:
          - 'raw': [segments, 13, time, 768]
          - 'segments': [segments, 13, 768]
          - 'aggregated': [13, 768]

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
                existing = self._cache.load(audio_path, output_dir, track_id, dataset)
                if existing is not None:
                    logging.info(f"Loading cached features for: {audio_path}")
                    return existing

            logging.info(f"Extracting features for: {audio_path}")

            # Preprocess audio (delegated to audio.py)
            audio = load_audio(audio_path, target_sr=self.target_sample_rate)

            # Segment audio (delegated to audio.py)
            segments = segment_audio(
                audio,
                segment_duration=self.segment_duration,
                overlap_ratio=self.overlap_ratio,
                sample_rate=self.target_sample_rate
            )

            # Extract features in batches (GPU-optimized)
            segment_features = self._extract_mert_features_batched(segments)

            # Create different feature representations
            results = {
                'raw': segment_features,
                'segments': np.mean(segment_features, axis=2),
                'aggregated': np.mean(segment_features, axis=(0, 2))
            }

            # Save features if output directory provided
            if output_dir:
                self._cache.save(results, audio_path, output_dir, track_id, dataset)

            return results

        except Exception as e:
            logging.error(f"Error extracting track features for {audio_path}: {e}")
            raise

    def extract_track_features_safe(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        track_id: Optional[str] = None,
        dataset: Optional[str] = None,
        skip_existing: bool = True,
        validate: bool = True
    ) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[str]]:
        """
        Safe version of extract_track_features with comprehensive error handling.

        Returns (features, error_message) tuple instead of raising exceptions.

        Args:
            audio_path: Path to audio file
            output_dir: Feature save directory (optional)
            track_id: Custom track ID (optional)
            dataset: Dataset name (optional)
            skip_existing: Skip if features exist (default: True)
            validate: Validate audio file before extraction (default: True)

        Returns:
            Tuple of (features_dict or None, error_message or None)
        """
        try:
            if validate:
                validation = validate_audio_file(audio_path, check_corruption=False)
                if not validation['valid']:
                    error_msg = '; '.join(validation['errors'])
                    return None, f"Validation failed: {error_msg}"

            features = self.extract_track_features(
                audio_path, output_dir, track_id, dataset, skip_existing
            )

            return features, None

        except Exception as e:
            return None, str(e)

    def clear_gpu_cache(self) -> None:
        """
        Clear GPU cache to free memory.

        Useful between large batch extractions or when memory errors occur.
        """
        try:
            if self.device == 'mps':
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                    logging.info("MPS cache cleared")
            elif self.device == 'cuda':
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                    logging.info("CUDA cache cleared")
            else:
                logging.info("CPU device - no cache to clear")
        except Exception as e:
            logging.warning(f"Failed to clear GPU cache: {e}")

    def extract_dataset_features(
        self,
        audio_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.wav",
        skip_existing: bool = True,
        num_workers: int = 1,
        dataset: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features for entire dataset (convenience delegator).

        Delegates to ExtractionPipeline. Kept here for backward compatibility
        so existing callers (e.g. scripts/extract_features.py) don't need changes.

        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save extracted features
            file_pattern: File pattern to match (default: "*.wav")
            skip_existing: Skip already-extracted files (default: True)
            num_workers: Number of worker threads (default: 1 for sequential)
            dataset: Dataset name for subdirectory (optional)

        Returns:
            Dict with statistics if num_workers>1, None if sequential
        """
        from .pipeline import ExtractionPipeline

        pipeline = ExtractionPipeline(self)
        return pipeline.run(
            audio_dir, output_dir, file_pattern,
            skip_existing, num_workers, dataset
        )

    def estimate_extraction_time(
        self,
        audio_dir: Union[str, Path],
        file_pattern: str = "*.wav",
        sample_size: int = 5,
        num_workers: int = 1
    ) -> Dict[str, float]:
        """
        Estimate extraction time for dataset (convenience delegator).

        Delegates to ExtractionPipeline.estimate_time().
        """
        from .pipeline import ExtractionPipeline

        pipeline = ExtractionPipeline(self)
        return pipeline.estimate_time(audio_dir, file_pattern, sample_size, num_workers)
