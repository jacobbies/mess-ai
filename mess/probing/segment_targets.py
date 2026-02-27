"""
Segment-level proxy target generation for segment probing.

Generates per-segment proxy targets aligned with the 5s extraction windows
used by MERT feature extraction. Each target is an array of length
``num_segments`` (one scalar per segment) rather than a single scalar per track.

This enables segment-level probing which dramatically increases sample count
(~2000 segments from 50 tracks) and preserves local temporal information.

Data Structure Contract:
    Returns ``{category: {field: np.ndarray[num_segments]}}`` — same nested
    structure as ``targets.py`` but with arrays instead of scalars.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Targets that are viable at 5s segment duration.
# Excluded: tempo (needs >5s for beat tracking), phrase_regularity,
# num_phrases (need whole piece), onset_density (too noisy at 5s).
SEGMENT_TARGET_NAMES: set[str] = {
    'spectral_centroid',
    'spectral_rolloff',
    'spectral_bandwidth',
    'zero_crossing_rate',
    'dynamic_range',
    'dynamic_variance',
    'crescendo_strength',
    'diminuendo_strength',
    'harmonic_complexity',
    'attack_slopes',
    'attack_sharpness',
}

# Expression targets viable at segment level (from MIDI).
SEGMENT_EXPRESSION_TARGET_NAMES: set[str] = {
    'rubato',
    'velocity_mean',
    'velocity_std',
    'velocity_range',
    'articulation_ratio',
    'tempo_variability',
    'onset_timing_std',
}


def generate_segment_targets(
    audio_path: str | Path,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
    sample_rate: int = 24000,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate per-segment proxy targets for a single audio file.

    Loads audio, segments it using the same windowing as MERT extraction,
    and computes each viable target independently on each 5s segment.

    Args:
        audio_path: Path to a .wav audio file.
        segment_duration: Duration of each segment in seconds.
        overlap_ratio: Overlap between consecutive segments.
        sample_rate: Target sample rate for audio loading.

    Returns:
        ``{category: {field: np.ndarray[num_segments]}}`` where each array
        has one value per segment.
    """
    import torch
    import torchaudio

    from ..extraction.audio import segment_audio

    audio_tensor, sr = torchaudio.load(str(audio_path))
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio_tensor = resampler(audio_tensor)
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0)
    audio = audio_tensor.numpy().squeeze()

    segments = segment_audio(
        audio,
        segment_duration=segment_duration,
        overlap_ratio=overlap_ratio,
        sample_rate=sample_rate,
    )
    # Compute targets per segment
    timbre = _compute_timbre_segments(segments, sample_rate)
    dynamics = _compute_dynamics_segments(segments, sample_rate)
    harmony = _compute_harmony_segments(segments, sample_rate)
    articulation = _compute_articulation_segments(segments, sample_rate)

    return {
        'timbre': timbre,
        'dynamics': dynamics,
        'harmony': harmony,
        'articulation': articulation,
    }


def generate_segment_expression_targets(
    midi_path: str | Path,
    segment_boundaries: list[tuple[float, float]],
    min_notes: int = 5,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate per-segment expression targets from MIDI data.

    Slices MIDI notes into time-aligned segments matching audio extraction
    windows, then computes expression features per segment.

    Args:
        midi_path: Path to a MIDI file.
        segment_boundaries: List of ``(start_time, end_time)`` tuples
            matching the audio segment windows.
        min_notes: Minimum notes required per segment. Segments with
            fewer notes get NaN for all targets.

    Returns:
        ``{'expression': {field: np.ndarray[num_segments]}}``
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))

    # Collect all non-drum notes sorted by onset
    all_notes = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)
    all_notes.sort(key=lambda n: n.start)

    n_segments = len(segment_boundaries)
    fields = list(SEGMENT_EXPRESSION_TARGET_NAMES)
    result: dict[str, np.ndarray] = {f: np.full(n_segments, np.nan) for f in fields}

    for seg_idx, (t_start, t_end) in enumerate(segment_boundaries):
        # Filter notes that start within this segment
        seg_notes = [n for n in all_notes if n.start >= t_start and n.start < t_end]

        if len(seg_notes) < min_notes:
            continue

        onsets = np.array([n.start for n in seg_notes])
        offsets = np.array([n.end for n in seg_notes])
        velocities = np.array([n.velocity for n in seg_notes], dtype=np.float64)
        durations = offsets - onsets

        ioi = np.diff(onsets)
        ioi_positive = ioi[ioi > 0]

        # Rubato
        if len(ioi_positive) >= 3:
            ratios = ioi_positive[:-1] / ioi_positive[1:]
            result['rubato'][seg_idx] = float(np.std(ratios))
        else:
            result['rubato'][seg_idx] = 0.0

        # Velocity stats
        result['velocity_mean'][seg_idx] = float(np.mean(velocities))
        result['velocity_std'][seg_idx] = float(np.std(velocities))
        result['velocity_range'][seg_idx] = float(np.max(velocities) - np.min(velocities))

        # Articulation ratio
        paired_durations = durations[:-1]
        mask = ioi > 0
        if np.any(mask):
            result['articulation_ratio'][seg_idx] = float(
                np.mean(paired_durations[mask] / ioi[mask])
            )
        else:
            result['articulation_ratio'][seg_idx] = 1.0

        # Tempo variability
        if len(ioi_positive) >= 2:
            local_tempos = 60.0 / ioi_positive
            result['tempo_variability'][seg_idx] = float(np.std(local_tempos))
        else:
            result['tempo_variability'][seg_idx] = 0.0

        # Onset timing std
        if len(onsets) >= 4 and len(ioi_positive) >= 2:
            grid_spacing = float(np.median(ioi_positive))
            if grid_spacing > 0:
                grid_positions = np.round(onsets / grid_spacing) * grid_spacing
                deviations = onsets - grid_positions
                result['onset_timing_std'][seg_idx] = float(np.std(deviations))
            else:
                result['onset_timing_std'][seg_idx] = 0.0
        else:
            result['onset_timing_std'][seg_idx] = 0.0

    return {'expression': result}


def get_segment_boundaries(
    audio_length_samples: int,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
    sample_rate: int = 24000,
) -> list[tuple[float, float]]:
    """Compute segment boundary times matching ``segment_audio()`` windowing.

    Returns:
        List of ``(start_seconds, end_seconds)`` tuples.
    """
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap_ratio))

    boundaries = []
    for start in range(0, audio_length_samples - segment_samples + 1, hop_samples):
        boundaries.append((start / sample_rate, (start + segment_samples) / sample_rate))

    # Final segment (if remainder)
    if audio_length_samples % hop_samples != 0:
        end = audio_length_samples / sample_rate
        boundaries.append((end - segment_duration, end))

    return boundaries


# =========================================================================
# Internal per-segment target computations
# =========================================================================

def _compute_timbre_segments(
    segments: list[np.ndarray], sample_rate: int,
) -> dict[str, np.ndarray]:
    import librosa

    n = len(segments)
    centroid = np.zeros(n)
    rolloff = np.zeros(n)
    bandwidth = np.zeros(n)
    zcr = np.zeros(n)

    for i, seg in enumerate(segments):
        centroid[i] = float(np.mean(
            librosa.feature.spectral_centroid(y=seg, sr=sample_rate, hop_length=512)[0]
        ))
        rolloff[i] = float(np.mean(
            librosa.feature.spectral_rolloff(y=seg, sr=sample_rate, hop_length=512)[0]
        ))
        bandwidth[i] = float(np.mean(
            librosa.feature.spectral_bandwidth(y=seg, sr=sample_rate, hop_length=512)[0]
        ))
        zcr[i] = float(np.mean(
            librosa.feature.zero_crossing_rate(y=seg, hop_length=512)[0]
        ))

    return {
        'spectral_centroid': centroid,
        'spectral_rolloff': rolloff,
        'spectral_bandwidth': bandwidth,
        'zero_crossing_rate': zcr,
    }


def _compute_dynamics_segments(
    segments: list[np.ndarray], sample_rate: int,
) -> dict[str, np.ndarray]:
    import librosa
    import scipy.signal

    n = len(segments)
    dyn_range = np.zeros(n)
    dyn_variance = np.zeros(n)
    crescendo = np.zeros(n)
    diminuendo = np.zeros(n)

    for i, seg in enumerate(segments):
        rms = librosa.feature.rms(y=seg, hop_length=512)[0]
        dyn_range[i] = float(np.max(rms) - np.min(rms))

        rms_mean = np.mean(rms)
        rms_norm = rms / (rms_mean + 1e-8)
        dyn_variance[i] = float(np.var(rms_norm))

        # Crescendo/diminuendo via longest monotonic run
        if len(rms) >= 5:
            window_len = min(11, len(rms) if len(rms) % 2 == 1 else len(rms) - 1)
            if window_len >= 5:
                smoothed = scipy.signal.savgol_filter(rms, window_len, 3)
            else:
                smoothed = rms
        else:
            smoothed = rms

        from .targets import MusicalAspectTargets
        c, d = MusicalAspectTargets._longest_monotonic_runs(smoothed)
        crescendo[i] = c
        diminuendo[i] = d

    return {
        'dynamic_range': dyn_range,
        'dynamic_variance': dyn_variance,
        'crescendo_strength': crescendo,
        'diminuendo_strength': diminuendo,
    }


def _compute_harmony_segments(
    segments: list[np.ndarray], sample_rate: int,
) -> dict[str, np.ndarray]:
    import librosa

    n = len(segments)
    complexity = np.zeros(n)

    for i, seg in enumerate(segments):
        chroma = librosa.feature.chroma_cqt(y=seg, sr=sample_rate, hop_length=512)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-8)
        complexity[i] = float(-np.sum(chroma_mean * np.log(chroma_mean + 1e-8)))

    return {'harmonic_complexity': complexity}


def _compute_articulation_segments(
    segments: list[np.ndarray], sample_rate: int,
) -> dict[str, np.ndarray]:
    import librosa

    n = len(segments)
    slopes = np.zeros(n)
    sharpness = np.zeros(n)

    hop_length = 512
    window_frames = max(2, int(np.ceil(0.2 * sample_rate / hop_length)))

    for i, seg in enumerate(segments):
        rms = librosa.feature.rms(y=seg, hop_length=hop_length)[0]
        energy_db = librosa.power_to_db(rms**2, ref=np.max)

        onsets = librosa.onset.onset_detect(
            y=seg, sr=sample_rate, hop_length=hop_length, units='frames'
        )

        seg_slopes = []
        for onset in onsets:
            end = onset + window_frames
            if end <= len(energy_db):
                window = energy_db[onset:end]
                if len(window) > 1:
                    slope = np.polyfit(range(len(window)), window, 1)[0]
                    seg_slopes.append(max(0, slope))

        slopes[i] = float(np.mean(seg_slopes)) if seg_slopes else 0.0

        energy_diff = np.diff(energy_db)
        seg_sharpness = [abs(energy_diff[o]) for o in onsets if o < len(energy_diff)]
        sharpness[i] = float(np.mean(seg_sharpness)) if seg_sharpness else 0.0

    return {
        'attack_slopes': slopes,
        'attack_sharpness': sharpness,
    }


# =========================================================================
# Batch dataset creation
# =========================================================================

def create_segment_target_dataset(
    audio_dir: str | Path,
    output_dir: str | Path,
    segment_duration: float = 5.0,
    overlap_ratio: float = 0.5,
    sample_rate: int = 24000,
    dataset_id: str | None = None,
) -> dict[str, int]:
    """Create segment-level proxy target dataset for all audio files.

    Mirrors ``create_target_dataset()`` from ``targets.py`` but saves
    per-segment arrays to ``data/proxy_targets_segments/``.

    Args:
        audio_dir: Directory containing .wav audio files.
        output_dir: Directory to save ``{stem}_segment_targets.npz`` files.
        segment_duration: Segment duration in seconds.
        overlap_ratio: Overlap between segments.
        sample_rate: Target sample rate.
        dataset_id: Dataset identifier for MIDI lookup ('smd', 'maestro').

    Returns:
        Dict with 'total', 'success', 'failed' counts.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up MIDI generator if available
    midi_resolver = None
    if dataset_id is not None:
        try:
            from .midi_targets import resolve_midi_path
            midi_resolver = resolve_midi_path
            logger.info(f"MIDI segment targets enabled for dataset '{dataset_id}'")
        except ImportError:
            logger.info("pretty_midi not installed; skipping MIDI segment targets")

    audio_files = sorted(
        p for p in audio_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    )
    n_total = len(audio_files)

    if n_total == 0:
        logger.warning(f"No .wav files found in {audio_dir}")
        return {'total': 0, 'success': 0, 'failed': 0}

    logger.info(f"Processing {n_total} audio files for segment targets")

    start_time = time.time()
    success = 0
    failed = 0

    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"[{i}/{n_total}] Processing {audio_file.name}...")

        try:
            targets = generate_segment_targets(
                audio_file,
                segment_duration=segment_duration,
                overlap_ratio=overlap_ratio,
                sample_rate=sample_rate,
            )

            # Merge MIDI expression targets if available
            if midi_resolver is not None and dataset_id is not None:
                midi_path = midi_resolver(audio_file, dataset_id)
                if midi_path is not None:
                    try:
                        import torchaudio

                        info = torchaudio.info(str(audio_file))
                        audio_length = info.num_frames
                        if info.sample_rate != sample_rate:
                            audio_length = int(
                                audio_length * sample_rate / info.sample_rate
                            )
                        boundaries = get_segment_boundaries(
                            audio_length, segment_duration, overlap_ratio, sample_rate
                        )
                        midi_targets = generate_segment_expression_targets(
                            midi_path, boundaries
                        )
                        targets.update(midi_targets)
                    except Exception as e:
                        logger.warning(
                            f"  MIDI segment targets skipped for {audio_file.name}: {e}"
                        )

            out_file = output_dir / f"{audio_file.stem}_segment_targets.npz"
            np.savez_compressed(out_file, **targets)
            logger.info(f"  Saved to {out_file.name}")
            success += 1

        except Exception as e:
            logger.error(f"  Error: {audio_file.name}: {e}")
            failed += 1

    elapsed = time.time() - start_time
    logger.info(
        f"Segment target creation complete: "
        f"{success}/{n_total} success, {failed} failed, {elapsed:.1f}s"
    )

    return {'total': n_total, 'success': success, 'failed': failed}
