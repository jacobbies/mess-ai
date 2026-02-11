"""
Dataset-level batch extraction pipeline.

Orchestrates feature extraction across directories of audio files,
with support for sequential and parallel (threaded) processing.

Usage:
    from mess.extraction.extractor import FeatureExtractor
    from mess.extraction.pipeline import ExtractionPipeline

    extractor = FeatureExtractor()
    pipeline = ExtractionPipeline(extractor)
    results = pipeline.run(audio_dir, output_dir, num_workers=4)
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .audio import load_audio, segment_audio
from .storage import FeatureStorage


class ExtractionPipeline:
    """
    Dataset-level batch processing for MERT feature extraction.

    Wraps a FeatureExtractor to orchestrate extraction across many files.
    Supports sequential (single-threaded) and parallel (threaded CPU
    preprocessing + sequential GPU inference) modes.
    """

    def __init__(self, extractor) -> None:
        """
        Initialize pipeline with a FeatureExtractor.

        Args:
            extractor: A FeatureExtractor instance (used for MERT inference)
        """
        self.extractor = extractor
        self.storage = FeatureStorage()

    def run(
        self,
        audio_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.wav",
        skip_existing: bool = True,
        num_workers: int = 1,
        dataset: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract features for entire dataset.

        Supports both sequential (num_workers=1) and parallel (num_workers>1) processing.
        Parallel processing provides ~40-50% speedup by overlapping CPU preprocessing
        with GPU inference.

        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save extracted features
            file_pattern: File pattern to match (default: "*.wav")
            skip_existing: Skip already-extracted files (default: True)
            num_workers: Number of worker threads (default: 1 for sequential)
            dataset: Dataset name for subdirectory (optional)

        Returns:
            Dict with statistics if num_workers>1, None if sequential (backward compatible)
        """
        if num_workers > 1:
            return self.run_parallel(
                audio_dir, output_dir, file_pattern,
                skip_existing, num_workers, dataset
            )

        # Sequential implementation (original behavior, backward compatible)
        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)

        audio_files = list(audio_dir.glob(file_pattern))

        if not audio_files:
            logging.warning(f"No audio files found in {audio_dir} with pattern {file_pattern}")
            return None

        logging.info(f"Found {len(audio_files)} audio files to process")

        for audio_file in tqdm(audio_files, desc="Extracting features"):
            try:
                self.extractor.extract_track_features(
                    audio_file,
                    output_dir,
                    skip_existing=skip_existing,
                    dataset=dataset
                )
            except Exception as e:
                logging.error(f"Failed to process {audio_file}: {e}")
                continue

        logging.info(f"Feature extraction complete. Results saved to {output_dir}")
        return None

    def run_parallel(
        self,
        audio_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.wav",
        skip_existing: bool = True,
        num_workers: Optional[int] = None,
        dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract features with parallel audio loading.

        Uses ThreadPoolExecutor to parallelize CPU-bound preprocessing while
        keeping GPU inference sequential for thread safety.

        Pipeline:
          Workers (CPU): Load → Preprocess → Segment → Submit
          Main (GPU):    Collect → MERT Batch → Save → Repeat

        Args:
            audio_dir: Directory containing audio files
            output_dir: Directory to save extracted features
            file_pattern: File pattern to match (default: "*.wav")
            skip_existing: Skip already-extracted files (default: True)
            num_workers: Number of worker threads
            dataset: Dataset name for subdirectory (optional)

        Returns:
            Dict with extraction statistics
        """
        from ..config import mess_config

        audio_dir = Path(audio_dir)
        output_dir = Path(output_dir)

        if num_workers is None:
            num_workers = mess_config.max_workers

        audio_files = list(audio_dir.glob(file_pattern))

        if not audio_files:
            logging.warning(f"No audio files found in {audio_dir} with pattern {file_pattern}")
            return {
                'total_files': 0, 'processed': 0, 'cached': 0,
                'failed': 0, 'errors': [],
                'elapsed_time': 0.0, 'avg_time_per_file': 0.0
            }

        logging.info(f"Found {len(audio_files)} audio files to process with {num_workers} workers")

        start_time = time.time()
        processed_count = 0
        cached_count = 0
        failed_count = 0
        errors = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(
                    self._preprocess_worker, audio_file,
                    skip_existing, output_dir, dataset
                ): audio_file
                for audio_file in audio_files
            }

            with tqdm(total=len(audio_files), desc="Extracting features") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()

                    if result['status'] == 'ready':
                        try:
                            segment_features = self.extractor._extract_mert_features_batched(
                                result['segments']
                            )

                            features = {
                                'raw': segment_features,
                                'segments': np.mean(segment_features, axis=2),
                                'aggregated': np.mean(segment_features, axis=(0, 2))
                            }

                            self.storage.save(
                                features, result['path'],
                                output_dir, result['track_id'], dataset
                            )

                            processed_count += 1

                        except Exception as e:
                            logging.error(f"MERT inference failed for {result['path']}: {e}")
                            failed_count += 1
                            errors.append({
                                'path': result['path'],
                                'error': f"MERT inference error: {str(e)}"
                            })

                    elif result['status'] == 'cached':
                        cached_count += 1

                    elif result['status'] == 'error':
                        failed_count += 1
                        errors.append({
                            'path': result['path'],
                            'error': result['error']
                        })
                        logging.error(f"Failed to process {result['path']}: {result['error']}")

                    pbar.update(1)
                    pbar.set_postfix(
                        processed=processed_count,
                        cached=cached_count,
                        failed=failed_count
                    )

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / len(audio_files) if audio_files else 0.0

        logging.info(
            f"Feature extraction complete. "
            f"Processed: {processed_count}, Cached: {cached_count}, Failed: {failed_count}. "
            f"Time: {elapsed_time:.2f}s (avg: {avg_time:.2f}s/file)"
        )

        return {
            'total_files': len(audio_files),
            'processed': processed_count,
            'cached': cached_count,
            'failed': failed_count,
            'errors': errors,
            'elapsed_time': elapsed_time,
            'avg_time_per_file': avg_time
        }

    def _preprocess_worker(
        self,
        audio_path: Path,
        skip_existing: bool,
        output_dir: Path,
        dataset: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Worker function for parallel audio preprocessing (CPU-bound, thread-safe).

        Executes: cache check → audio loading → resampling → segmentation.
        Does NOT perform MERT inference (GPU-bound, main thread only).

        Returns:
            Dict with status ('ready', 'cached', or 'error') and associated data
        """
        try:
            track_id = audio_path.stem

            if skip_existing and self.storage.exists(audio_path, output_dir, track_id, dataset):
                return {
                    'status': 'cached',
                    'path': str(audio_path),
                    'track_id': track_id
                }

            # CPU-bound preprocessing (thread-safe)
            audio = load_audio(audio_path, target_sr=self.extractor.target_sample_rate)
            segments = segment_audio(
                audio,
                segment_duration=self.extractor.segment_duration,
                overlap_ratio=self.extractor.overlap_ratio,
                sample_rate=self.extractor.target_sample_rate
            )

            return {
                'status': 'ready',
                'path': str(audio_path),
                'segments': segments,
                'track_id': track_id
            }

        except Exception as e:
            return {
                'status': 'error',
                'path': str(audio_path),
                'track_id': audio_path.stem,
                'error': str(e)
            }

    def estimate_time(
        self,
        audio_dir: Union[str, Path],
        file_pattern: str = "*.wav",
        sample_size: int = 5,
        num_workers: int = 1
    ) -> Dict[str, float]:
        """
        Estimate extraction time for dataset by sampling files.

        Processes a small sample of files to estimate total extraction time.

        Args:
            audio_dir: Directory containing audio files
            file_pattern: File pattern to match (default: "*.wav")
            sample_size: Number of files to sample (default: 5)
            num_workers: Number of workers for parallel estimation (default: 1)

        Returns:
            Dict with time estimates (in seconds)
        """
        import random

        audio_dir = Path(audio_dir)
        audio_files = list(audio_dir.glob(file_pattern))

        if not audio_files:
            logging.warning(f"No audio files found in {audio_dir}")
            return {
                'total_files': 0,
                'sampled_files': 0,
                'avg_time_per_file': 0.0,
                'estimated_sequential_time': 0.0,
                'estimated_parallel_time': 0.0,
                'estimated_speedup': 1.0
            }

        sample_files = random.sample(audio_files, min(sample_size, len(audio_files)))

        logging.info(f"Sampling {len(sample_files)} files to estimate extraction time...")

        start_time = time.time()

        for audio_file in sample_files:
            try:
                self.extractor.extract_track_features(
                    audio_file, output_dir=None, skip_existing=False
                )
            except Exception as e:
                logging.warning(f"Sample extraction failed for {audio_file}: {e}")
                continue

        elapsed_time = time.time() - start_time
        avg_time_per_file = elapsed_time / len(sample_files)

        estimated_sequential = avg_time_per_file * len(audio_files)

        # Estimate parallel speedup (empirical: ~1.75x for 4 workers)
        speedup_factor = min(1.0 + (num_workers - 1) * 0.25, num_workers * 0.6)
        estimated_parallel = estimated_sequential / speedup_factor

        result = {
            'total_files': len(audio_files),
            'sampled_files': len(sample_files),
            'avg_time_per_file': avg_time_per_file,
            'estimated_sequential_time': estimated_sequential,
            'estimated_parallel_time': estimated_parallel,
            'estimated_speedup': speedup_factor
        }

        logging.info(f"\nEstimation Results:")
        logging.info(f"  Total files: {result['total_files']}")
        logging.info(f"  Avg time/file: {result['avg_time_per_file']:.2f}s")
        logging.info(f"  Sequential estimate: {result['estimated_sequential_time']:.1f}s ({result['estimated_sequential_time']/60:.1f} min)")
        if num_workers > 1:
            logging.info(f"  Parallel estimate ({num_workers} workers): {result['estimated_parallel_time']:.1f}s ({result['estimated_parallel_time']/60:.1f} min)")
            logging.info(f"  Expected speedup: {result['estimated_speedup']:.2f}x")

        return result
