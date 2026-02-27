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
from pathlib import Path
from typing import Any

import numpy as np

from ..config import mess_config

logger = logging.getLogger(__name__)

try:
    import mlflow  # type: ignore[import-not-found]
except ModuleNotFoundError:
    class _MlflowStub:
        """No-op MLflow shim when mlflow is not installed."""

        @staticmethod
        def active_run() -> None:
            return None

        @staticmethod
        def log_params(_params: dict[str, Any]) -> None:
            return None

        @staticmethod
        def log_metrics(_metrics: dict[str, float]) -> None:
            return None

        @staticmethod
        def log_artifact(_path: str) -> None:
            return None

        @staticmethod
        def set_experiment(_name: str) -> None:
            return None

        class _RunContext:
            def __enter__(self) -> "_MlflowStub._RunContext":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        @staticmethod
        def start_run() -> "_MlflowStub._RunContext":
            return _MlflowStub._RunContext()

        @staticmethod
        def set_tag(_key: str, _value: str) -> None:
            return None

    mlflow = _MlflowStub()


class MusicalAspectTargets:
    """
    Generate proxy targets for different musical aspects.

    Extracts ground-truth labels from audio for probing MERT layer representations.
    All targets are saved as numpy arrays in a nested dict structure that
    discovery.py can consume for linear probing experiments.
    """

    def __init__(self, sample_rate: int | None = None):
        """
        Initialize target generator.

        Args:
            sample_rate: Audio sample rate. If None, uses mess_config.target_sample_rate (24kHz).
        """
        self.sample_rate = sample_rate or mess_config.target_sample_rate
    
    def generate_all_targets(self, audio_path: str) -> dict[str, dict[str, np.ndarray]]:
        """Generate all proxy targets for a given audio file."""
        import torch
        import torchaudio

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
    def validate_target_structure(targets: dict[str, dict[str, np.ndarray]]) -> bool:
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
            # Skip optional categories (e.g. expression — requires MIDI)
            if category in LayerDiscoverySystem.OPTIONAL_CATEGORIES:
                continue
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

        logger.info(
            "Target structure validated: all %d targets present",
            len(LayerDiscoverySystem.SCALAR_TARGETS),
        )
        return True

    def _generate_rhythm_targets(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Generate rhythm-related targets."""
        import librosa
        import scipy.signal

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
    
    def _generate_harmony_targets(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Generate harmony-related targets."""
        import librosa

        # Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=self.sample_rate,
            hop_length=512,
            n_chroma=12
        )
        
        # Key estimation (template-based matching with major/minor profiles)
        # Major and minor key profiles (Krumhansl-Schmuckler)
        major_profile = np.array(
            [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        )
        minor_profile = np.array(
            [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        )

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
    
    def _generate_timbre_targets(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Generate timbre-related targets."""
        import librosa

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
    
    def _generate_articulation_targets(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Generate articulation-related targets (attack characteristics)."""
        import librosa

        # Compute energy envelope in dB for stable slope measurement
        hop_length = 512
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        energy_db = librosa.power_to_db(rms**2, ref=np.max)

        # Find note onsets
        onsets = librosa.onset.onset_detect(
            y=audio,
            sr=self.sample_rate,
            hop_length=hop_length,
            units='frames'
        )

        # Calculate attack slopes over first 200ms after onset (in dB domain)
        # 200ms ≈ ceil(0.2 * sr / hop_length) frames
        window_frames = max(2, int(np.ceil(0.2 * self.sample_rate / hop_length)))

        attack_slopes = []
        for onset in onsets:
            end = onset + window_frames
            if end <= len(energy_db):
                window = energy_db[onset:end]
                if len(window) > 1:
                    slope = np.polyfit(range(len(window)), window, 1)[0]
                    attack_slopes.append(max(0, slope))  # Only positive slopes

        # Attack sharpness (derivative of dB energy at onsets)
        energy_diff = np.diff(energy_db)
        attack_sharpness = []
        for onset in onsets:
            if onset < len(energy_diff):
                attack_sharpness.append(abs(energy_diff[onset]))

        return {
            'attack_slopes': np.array(attack_slopes) if attack_slopes else np.array([0.0]),
            'attack_sharpness': np.array(attack_sharpness) if attack_sharpness else np.array([0.0]),
            'onset_density': np.array([len(onsets) / (len(audio) / self.sample_rate)]),
            'energy_envelope': rms
        }
    
    def _generate_dynamics_targets(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Generate dynamics-related targets."""
        import librosa
        import scipy.signal

        # RMS energy (dynamic level)
        rms = librosa.feature.rms(
            y=audio,
            hop_length=512
        )[0]
        
        # Dynamic range
        dynamic_range = np.max(rms) - np.min(rms)
        
        # Dynamic variance — normalize by mean so it's recording-level-invariant
        rms_mean = np.mean(rms)
        rms_normalized = rms / (rms_mean + 1e-8)
        dynamic_variance = np.var(rms_normalized)

        # Dynamic trajectory (overall shape)
        # Smooth RMS to see overall dynamic arc
        window_len = min(51, len(rms) if len(rms) % 2 == 1 else len(rms) - 1)
        if window_len >= 5:
            smoothed_rms = scipy.signal.savgol_filter(rms, window_len, 3)
        else:
            smoothed_rms = rms

        # Crescendo/diminuendo detection — longest monotonic run
        crescendo_strength, diminuendo_strength = self._longest_monotonic_runs(smoothed_rms)
        
        return {
            'rms_energy': rms,
            'dynamic_range': np.array([dynamic_range]),
            'dynamic_variance': np.array([dynamic_variance]),
            'dynamic_trajectory': smoothed_rms,
            'crescendo_strength': np.array([crescendo_strength]),
            'diminuendo_strength': np.array([diminuendo_strength])
        }
    
    @staticmethod
    def _longest_monotonic_runs(smoothed_rms: np.ndarray) -> tuple[float, float]:
        """Find longest monotonic increasing/decreasing runs in smoothed RMS.

        Returns (crescendo_strength, diminuendo_strength) where each is
        ``amplitude_change * run_length`` for the longest run found.
        """
        if len(smoothed_rms) < 2:
            return 0.0, 0.0

        diffs = np.diff(smoothed_rms)

        def _best_run(positive: bool) -> float:
            mask = diffs > 0 if positive else diffs < 0
            best_len = 0
            best_start = 0
            cur_len = 0
            cur_start = 0
            for i, m in enumerate(mask):
                if m:
                    if cur_len == 0:
                        cur_start = i
                    cur_len += 1
                    if cur_len > best_len:
                        best_len = cur_len
                        best_start = cur_start
                else:
                    cur_len = 0
            if best_len == 0:
                return 0.0
            amp_change = abs(
                float(smoothed_rms[best_start + best_len] - smoothed_rms[best_start])
            )
            return amp_change * best_len

        return _best_run(True), _best_run(False)

    def _generate_phrasing_targets(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """Generate phrasing-related targets (musical sentence structure)."""
        import librosa

        # Novelty function for phrase boundary detection
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
        
        # Self-similarity matrix
        similarity_matrix = np.dot(chroma.T, chroma)
        
        # Novelty curve (diagonal tracking) 
        try:
            # Try newer API first
            novelty = librosa.segment.recurrence_matrix(chroma.T).diagonal()
        except Exception:
            # Fallback to older API or manual calculation
            novelty = np.diag(similarity_matrix)
        
        # Dynamic k based on track duration
        duration_seconds = len(audio) / self.sample_rate
        k = max(4, min(12, int(duration_seconds / 15)))

        # Phrase boundaries (peaks in novelty)
        boundaries = librosa.segment.agglomerative(
            chroma,
            k=k
        )

        # Phrase lengths
        phrase_lengths = np.diff(boundaries)

        # Phrase regularity — coefficient of variation (higher = more irregular)
        if len(phrase_lengths) > 1:
            phrase_regularity = float(
                np.std(phrase_lengths) / (np.mean(phrase_lengths) + 1e-8)
            )
        else:
            phrase_regularity = 0.0
        
        return {
            'novelty_curve': novelty,
            'phrase_boundaries': boundaries,
            'phrase_lengths': phrase_lengths,
            'phrase_regularity': np.array([phrase_regularity]),
            'num_phrases': np.array([len(phrase_lengths)])
        }


###

def create_target_dataset(
    audio_dir: str | Path,
    output_dir: str | Path,
    validate: bool = True,
    use_mlflow: bool = True,
    dataset_id: str | None = None,
) -> dict[str, int]:
    """
    Create proxy target dataset for all audio files.

    Processes all .wav files in audio_dir, extracts proxy targets, and saves
    them as .npz files. Optionally validates against discovery.py's expected
    structure and logs metrics to MLflow.

    When *dataset_id* is provided, also attempts to generate MIDI expression
    targets for tracks that have corresponding MIDI files.

    Args:
        audio_dir: Directory containing .wav audio files.
        output_dir: Directory to save _targets.npz files.
        validate: Whether to validate target structure against discovery.py.
        use_mlflow: Whether to log processing metrics to MLflow.
        dataset_id: Dataset identifier ('smd', 'maestro') for MIDI lookup.

    Returns:
        Dict with 'total', 'success', 'failed' counts.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_generator = MusicalAspectTargets()

    # Optionally set up MIDI expression target generator
    midi_generator = None
    if dataset_id is not None:
        try:
            from .midi_targets import MidiExpressionTargets, resolve_midi_path
            midi_generator = MidiExpressionTargets()
            logger.info(f"MIDI expression targets enabled for dataset '{dataset_id}'")
        except ImportError:
            logger.info("pretty_midi not installed; skipping MIDI expression targets")

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
    target_stats: dict[str, list] = {}

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

            # Merge MIDI expression targets if available
            if midi_generator is not None and dataset_id is not None:
                midi_path = resolve_midi_path(audio_file, dataset_id)
                if midi_path is not None:
                    try:
                        midi_targets = midi_generator.generate_expression_targets(
                            str(midi_path)
                        )
                        targets.update(midi_targets)
                    except Exception as e:
                        logger.warning(f"  MIDI targets skipped for {audio_file.name}: {e}")

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
    logger.info("Target Dataset Creation Complete")
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
                dataset_id=args.dataset,
            )
    else:
        result = create_target_dataset(
            audio_dir=audio_dir,
            output_dir=output_dir,
            validate=not args.no_validate,
            use_mlflow=False,
            dataset_id=args.dataset,
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
