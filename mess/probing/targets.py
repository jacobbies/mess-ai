"""
Proxy target generators for musical aspects in classical music.
These create ground-truth labels for probing MERT's learned representations.

Data Structure Contract:
    Saves targets as .npz files with nested dict structure:
    {category: {field: np.ndarray}}

    Example:
        {'timbre': {'spectral_centroid': array([...]), 'spectral_rolloff': array([...])},
         'rhythm': {'tempo': array([120.0]), 'onset_density': array([2.5])}}

    This structure is consumed by discovery.py via:
        data = np.load(file, allow_pickle=True)
        value = data[category].item()[field]  # .item() unpacks pickled dict

Targets Generated:
    - Timbre: spectral features, MFCCs, zero-crossing rate
    - Rhythm: tempo, onset density, rhythmic regularity
    - Harmony: chroma, key profiles, harmonic complexity
    - Articulation: attack slopes, attack sharpness
    - Dynamics: RMS energy, dynamic range, crescendo/diminuendo
    - Phrasing: phrase boundaries, regularity, novelty
"""

import logging
import time
from typing import Dict, Any, Union, Optional

import mlflow
import numpy as np
import librosa
import scipy.signal
import torch
import torchaudio
from pathlib import Path

from ..config import mess_config

logger = logging.getLogger(__name__)


class MusicalAspectTargets:
    """
    Generate proxy targets for different musical aspects.

    Extracts ground-truth labels from audio for probing MERT layer representations.
    All targets are saved as numpy arrays in a nested dict structure that
    discovery.py can consume for linear probing experiments.
    """

    def __init__(self, sample_rate: Optional[int] = None):
        """
        Initialize target generator.

        Args:
            sample_rate: Audio sample rate. If None, uses mess_config.target_sample_rate (24kHz).
        """
        self.sample_rate = sample_rate or mess_config.target_sample_rate
    
    def generate_all_targets(self, audio_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate all proxy targets for a given audio file."""
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0)
        
        audio = audio.numpy()
        
        targets = {
            'rhythm': self._generate_rhythm_targets(audio),
            'harmony': self._generate_harmony_targets(audio),
            'timbre': self._generate_timbre_targets(audio),
            'articulation': self._generate_articulation_targets(audio),
            'dynamics': self._generate_dynamics_targets(audio),
            'phrasing': self._generate_phrasing_targets(audio)
        }
        
        return targets

    @staticmethod
    def validate_target_structure(targets: Dict[str, Dict[str, np.ndarray]]) -> bool:
        """
        Verify targets match discovery.py's expected structure.

        Checks that all required (category, field) pairs exist in the targets dict.
        Logs warnings for any missing targets but does not raise errors.

        Args:
            targets: Nested dict of {category: {field: array}}

        Returns:
            True if all required targets are present, False otherwise.
        """
        from .discovery import LayerDiscoverySystem

        missing = []
        for target_name, (category, key, _) in LayerDiscoverySystem.SCALAR_TARGETS.items():
            if category not in targets:
                missing.append(f"{category}/{key} (for '{target_name}')")
            elif key not in targets[category]:
                missing.append(f"{category}/{key} (for '{target_name}')")

        if missing:
            logger.warning(
                f"Missing {len(missing)} target(s) expected by discovery.py:\n  "
                + "\n  ".join(missing)
            )
            return False

        logger.info(f"Target structure validated: all {len(LayerDiscoverySystem.SCALAR_TARGETS)} targets present")
        return True

    def _generate_rhythm_targets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate rhythm-related targets."""
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(
            y=audio, 
            sr=self.sample_rate,
            hop_length=512
        )
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=512
        )
        
        # Rhythmic regularity (autocorrelation of onset strength)
        onset_autocorr = np.correlate(onset_env, onset_env, mode='full')
        onset_autocorr = onset_autocorr[onset_autocorr.size // 2:]
        
        # Peak prominence in rhythm
        peaks, properties = scipy.signal.find_peaks(
            onset_env, 
            height=np.mean(onset_env) * 0.5,
            distance=int(self.sample_rate * 0.1 / 512)  # Min 100ms apart
        )
        
        return {
            'tempo': np.array([tempo]),
            'onset_strength': onset_env,
            'rhythmic_regularity': onset_autocorr[:100],  # First 100 lags
            'onset_density': np.array([len(peaks) / (len(audio) / self.sample_rate)])
        }
    
    def _generate_harmony_targets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate harmony-related targets."""
        # Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=512,
            n_chroma=12
        )
        
        # Key estimation (template-based matching with major/minor profiles)
        # Major and minor key profiles (Krumhansl-Schmuckler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Average chroma over time
        chroma_mean = np.mean(chroma, axis=1)

        # Correlate with all major/minor keys
        correlations = []
        for shift in range(12):
            major_corr = np.corrcoef(chroma_mean, np.roll(major_profile, shift))[0, 1]
            minor_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, shift))[0, 1]
            correlations.extend([major_corr, minor_corr])

        # Best matching key profile
        key = np.array(correlations)
        
        # Harmonic change rate (how fast chroma changes)
        chroma_diff = np.diff(chroma, axis=1)
        harmonic_change = np.mean(np.abs(chroma_diff), axis=0)
        
        # Tonal stability (variance in chroma over time)
        tonal_stability = 1.0 / (1.0 + np.var(chroma, axis=1))
        
        # Harmonic complexity (entropy of chroma distribution)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-8)
        harmonic_complexity = -np.sum(chroma_mean * np.log(chroma_mean + 1e-8))
        
        return {
            'chroma': chroma,
            'key_profile': key,
            'harmonic_change_rate': harmonic_change,
            'tonal_stability': tonal_stability,
            'harmonic_complexity': np.array([harmonic_complexity])
        }
    
    def _generate_timbre_targets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate timbre-related targets."""
        # MFCCs (classic timbre descriptors)
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13,
            hop_length=512
        )
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=512
        )[0]
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio,
            sr=self.sample_rate,
            hop_length=512
        )[0]
        
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            hop_length=512
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate,
            hop_length=512
        )[0]
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            hop_length=512
        )[0]
        
        return {
            'mfccs': mfccs,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_contrast': spectral_contrast,
            'spectral_rolloff': spectral_rolloff,
            'zero_crossing_rate': zcr
        }
    
    def _generate_articulation_targets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate articulation-related targets (attack characteristics)."""
        # Attack slope (how quickly energy rises)
        stft = librosa.stft(audio, hop_length=512)
        magnitude = np.abs(stft)
        
        # Energy envelope
        energy = np.sum(magnitude**2, axis=0)
        
        # Find note onsets
        onsets = librosa.onset.onset_detect(
            y=audio,
            sr=self.sample_rate,
            hop_length=512,
            units='frames'
        )
        
        # Calculate attack slopes at onsets
        attack_slopes = []
        window_size = 5  # frames
        
        for onset in onsets:
            if onset + window_size < len(energy):
                # Slope of energy increase after onset
                energy_window = energy[onset:onset + window_size]
                if len(energy_window) > 1:
                    slope = np.polyfit(range(len(energy_window)), energy_window, 1)[0]
                    attack_slopes.append(max(0, slope))  # Only positive slopes
        
        # Attack sharpness (derivative of energy at onsets)
        energy_diff = np.diff(energy)
        attack_sharpness = []
        for onset in onsets:
            if onset < len(energy_diff):
                attack_sharpness.append(abs(energy_diff[onset]))
        
        return {
            'attack_slopes': np.array(attack_slopes) if attack_slopes else np.array([0.0]),
            'attack_sharpness': np.array(attack_sharpness) if attack_sharpness else np.array([0.0]),
            'onset_density': np.array([len(onsets) / (len(audio) / self.sample_rate)]),
            'energy_envelope': energy
        }
    
    def _generate_dynamics_targets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate dynamics-related targets."""
        # RMS energy (dynamic level)
        rms = librosa.feature.rms(
            y=audio,
            hop_length=512
        )[0]
        
        # Dynamic range
        dynamic_range = np.max(rms) - np.min(rms)
        
        # Dynamic variance (how much dynamics change)
        dynamic_variance = np.var(rms)
        
        # Dynamic trajectory (overall shape)
        # Smooth RMS to see overall dynamic arc
        smoothed_rms = scipy.signal.savgol_filter(rms, 51, 3)
        
        # Crescendo/diminuendo detection
        rms_diff = np.diff(smoothed_rms)
        crescendo_strength = np.mean(rms_diff[rms_diff > 0]) if np.any(rms_diff > 0) else 0.0
        diminuendo_strength = np.mean(np.abs(rms_diff[rms_diff < 0])) if np.any(rms_diff < 0) else 0.0
        
        return {
            'rms_energy': rms,
            'dynamic_range': np.array([dynamic_range]),
            'dynamic_variance': np.array([dynamic_variance]),
            'dynamic_trajectory': smoothed_rms,
            'crescendo_strength': np.array([crescendo_strength]),
            'diminuendo_strength': np.array([diminuendo_strength])
        }
    
    def _generate_phrasing_targets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate phrasing-related targets (musical sentence structure)."""
        # Novelty function for phrase boundary detection
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
        
        # Self-similarity matrix
        similarity_matrix = np.dot(chroma.T, chroma)
        
        # Novelty curve (diagonal tracking) 
        try:
            # Try newer API first
            novelty = librosa.segment.recurrence_matrix(chroma.T).diagonal()
        except:
            # Fallback to older API or manual calculation
            novelty = np.diag(similarity_matrix)
        
        # Phrase boundaries (peaks in novelty)
        boundaries = librosa.segment.agglomerative(
            chroma,
            k=8  # Assume ~8 phrases per piece
        )
        
        # Phrase lengths
        phrase_lengths = np.diff(boundaries)
        
        # Phrase regularity (variance in phrase lengths)
        phrase_regularity = 1.0 / (1.0 + np.var(phrase_lengths))
        
        return {
            'novelty_curve': novelty,
            'phrase_boundaries': boundaries,
            'phrase_lengths': phrase_lengths,
            'phrase_regularity': np.array([phrase_regularity]),
            'num_phrases': np.array([len(phrase_lengths)])
        }


###

def create_target_dataset(
    audio_dir: Union[str, Path],
    output_dir: Union[str, Path],
    validate: bool = True,
    use_mlflow: bool = True,
) -> Dict[str, int]:
    """
    Create proxy target dataset for all audio files.

    Processes all .wav files in audio_dir, extracts proxy targets, and saves
    them as .npz files. Optionally validates against discovery.py's expected
    structure and logs metrics to MLflow.

    Args:
        audio_dir: Directory containing .wav audio files.
        output_dir: Directory to save _targets.npz files.
        validate: Whether to validate target structure against discovery.py.
        use_mlflow: Whether to log processing metrics to MLflow.

    Returns:
        Dict with 'total', 'success', 'failed' counts.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_generator = MusicalAspectTargets()

    # Process all audio files (recursive, case-insensitive extension match)
    audio_files = sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    n_total = len(audio_files)

    if n_total == 0:
        logger.warning(f"No .wav files found in {audio_dir}")
        return {'total': 0, 'success': 0, 'failed': 0}

    logger.info(f"Processing {n_total} audio files from {audio_dir}")

    # Track statistics
    start_time = time.time()
    success_count = 0
    failed_count = 0
    errors = []
    target_stats: Dict[str, list] = {}

    # Log to MLflow if requested and a run is active
    if use_mlflow and mlflow.active_run():
        mlflow.log_params({
            'audio_dir': str(audio_dir),
            'output_dir': str(output_dir),
            'n_files': n_total,
            'sample_rate': target_generator.sample_rate,
        })

    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"[{i}/{n_total}] Processing {audio_file.name}...")

        try:
            targets = target_generator.generate_all_targets(str(audio_file))

            # Optional validation
            if validate:
                MusicalAspectTargets.validate_target_structure(targets)

            # Collect statistics for first file (sample)
            if success_count == 0:
                for category, fields in targets.items():
                    for field_name, field_value in fields.items():
                        key = f"{category}/{field_name}"
                        if isinstance(field_value, np.ndarray):
                            target_stats[key] = {
                                'shape': field_value.shape,
                                'mean': float(np.mean(field_value)),
                                'std': float(np.std(field_value)),
                            }

            # Save targets
            # Note: numpy will pickle the nested dicts automatically (allow_pickle=True implicit)
            target_file = output_dir / f"{audio_file.stem}_targets.npz"
            np.savez_compressed(target_file, **targets)

            logger.info(f"  ✓ Saved to {target_file.name}")
            success_count += 1

        except Exception as e:
            error_msg = f"{audio_file.name}: {str(e)}"
            logger.error(f"  ✗ Error: {error_msg}")
            errors.append(error_msg)
            failed_count += 1
            continue

    elapsed = time.time() - start_time

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Target Dataset Creation Complete")
    logger.info(f"{'='*70}")
    logger.info(f"  Total files:    {n_total}")
    logger.info(f"  Success:        {success_count} ({success_count/n_total*100:.1f}%)")
    logger.info(f"  Failed:         {failed_count}")
    logger.info(f"  Elapsed time:   {elapsed:.2f}s ({elapsed/n_total:.2f}s per file)")
    logger.info(f"  Output dir:     {output_dir}")

    if errors:
        logger.warning(f"\nErrors encountered ({len(errors)}):")
        for error in errors[:5]:  # Show first 5
            logger.warning(f"  - {error}")
        if len(errors) > 5:
            logger.warning(f"  ... and {len(errors) - 5} more")

    # Log to MLflow
    if use_mlflow and mlflow.active_run():
        mlflow.log_metrics({
            'total_files': n_total,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_count / n_total if n_total > 0 else 0.0,
            'elapsed_seconds': elapsed,
            'seconds_per_file': elapsed / n_total if n_total > 0 else 0.0,
        })

        # Log sample statistics for first file
        for key, stats in target_stats.items():
            safe_key = key.replace('/', '_')
            mlflow.log_metrics({
                f"sample_{safe_key}_mean": stats['mean'],
                f"sample_{safe_key}_std": stats['std'],
            })

        # Log errors as text artifact if any
        if errors:
            errors_file = output_dir / "errors.txt"
            errors_file.write_text("\n".join(errors))
            mlflow.log_artifact(str(errors_file))

    return {'total': n_total, 'success': success_count, 'failed': failed_count}


def main():
    """CLI entry point for target dataset creation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate proxy targets for layer discovery")
    parser.add_argument(
        '--dataset',
        type=str,
        default='smd',
        help='Dataset name (default: smd)',
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation against discovery.py structure',
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow tracking',
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    # Load dataset
    from mess.datasets.factory import DatasetFactory
    dataset = DatasetFactory.get_dataset(args.dataset)
    audio_dir = dataset.audio_dir
    output_dir = mess_config.proxy_targets_dir

    logger.info(f"Generating proxy targets for dataset: {args.dataset}")
    logger.info(f"  Audio dir:  {audio_dir}")
    logger.info(f"  Output dir: {output_dir}")

    # Start MLflow run if enabled
    if not args.no_mlflow:
        mlflow.set_experiment("proxy_target_generation")
        with mlflow.start_run():
            mlflow.set_tag("dataset", args.dataset)
            result = create_target_dataset(
                audio_dir=audio_dir,
                output_dir=output_dir,
                validate=not args.no_validate,
                use_mlflow=True,
            )
    else:
        result = create_target_dataset(
            audio_dir=audio_dir,
            output_dir=output_dir,
            validate=not args.no_validate,
            use_mlflow=False,
        )

    # Exit code based on success
    if result['failed'] > 0:
        logger.warning(f"Completed with {result['failed']} failures")
        return 1
    else:
        logger.info("All files processed successfully")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
