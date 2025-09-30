"""
Proxy target generators for musical aspects in classical music.
These create ground-truth labels for probing MERT's learned representations.
"""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, Any, Tuple
import torch
import torchaudio
from pathlib import Path


class MusicalAspectTargets:
    """Generate proxy targets for different musical aspects."""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
    
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
        
        # Key estimation
        try:
            key = librosa.key.key_to_degrees(
                librosa.key.estimate_key(chroma)
            )
        except:
            key = np.zeros(12)  # Fallback
        
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


def create_target_dataset(audio_dir: str, output_dir: str):
    """Create proxy target dataset for all audio files."""
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_generator = MusicalAspectTargets()
    
    # Process all audio files
    audio_files = list(audio_dir.glob("*.wav"))
    
    for audio_file in audio_files:
        print(f"Processing {audio_file.name}...")
        
        try:
            targets = target_generator.generate_all_targets(str(audio_file))
            
            # Save targets (flatten nested dict structure)
            target_file = output_dir / f"{audio_file.stem}_targets.npz"
            flattened = {}
            for category, category_data in targets.items():
                flattened[category] = category_data
            np.savez_compressed(target_file, **flattened)
            
            print(f"Saved targets to {target_file}")
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue


if __name__ == "__main__":
    # Example usage
    audio_dir = "/Users/jacobbieschke/mess-ai/data/smd/wav-44"
    output_dir = "/Users/jacobbieschke/mess-ai/data/processed/proxy_targets"
    
    create_target_dataset(audio_dir, output_dir)