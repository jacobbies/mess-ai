import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile
import librosa
import pretty_midi
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt

# Base class for audio processing
class AudioProcessor:
    def __init__(self):
        self.processing_history = []
    
    def normalize(self, audio, method='minmax'):
        """Normalize audio data using specified method"""
        if method == 'minmax':
            min_val = np.min(audio)
            max_val = np.max(audio)
            return (audio - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean = np.mean(audio)
            std = np.std(audio)
            return (audio - mean) / std
        elif method == 'peak':
            return audio / np.max(np.abs(audio))
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def filter_signal(self, audio, sr, filter_type='lowpass', cutoff=0.1, order=5):
        """Apply Butterworth filter to audio data"""
        nyquist = 0.5 * sr
        
        if filter_type == 'lowpass':
            b, a = signal.butter(order, cutoff / nyquist, btype='low')
        elif filter_type == 'highpass':
            b, a = signal.butter(order, cutoff / nyquist, btype='high')
        elif filter_type in ['bandpass', 'bandstop']:
            if not isinstance(cutoff, tuple) or len(cutoff) != 2:
                raise ValueError("Cutoff must be a tuple of (low, high) for bandpass/bandstop")
            b, a = signal.butter(order, [c / nyquist for c in cutoff], btype=filter_type)
        
        return signal.filtfilt(b, a, audio)
    
    def extract_features(self, audio, sr, feature_type='mfcc', n_mfcc=13, hop_length=512):
        """Extract audio features"""
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        # Extract features
        if feature_type == 'mfcc':
            return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        elif feature_type == 'chroma':
            return librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=hop_length)
        elif feature_type == 'spectral_contrast':
            return librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)
        elif feature_type == 'tonnetz':
            return librosa.feature.tonnetz(y=audio, sr=sr)
        elif feature_type == 'melspectrogram':
            return librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=hop_length)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

# Class for handling MIDI data
class MidiProcessor:
    def __init__(self):
        self.processing_history = []
    
    def get_note_events(self, midi_data):
        """Extract note events from MIDI data"""
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'onset': note.start,
                    'offset': note.end,
                    'duration': note.end - note.start,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'instrument': instrument.program
                })
        
        return pd.DataFrame(notes)
    
    def get_piano_roll(self, midi_data, fs=100):
        """Get piano roll representation of MIDI data"""
        return midi_data.get_piano_roll(fs=fs)

# Class for visualization
class MusicVisualizer:
    @staticmethod
    def plot_waveform(audio, sr, title='Waveform'):
        """Plot audio waveform"""
        fig, ax = plt.subplots(figsize=(12, 4))
        time = np.arange(0, len(audio)) / sr
        ax.plot(time, audio)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        return fig
    
    @staticmethod
    def plot_spectrogram(audio, sr, title='Spectrogram'):
        """Plot audio spectrogram"""
        # Convert to mono if stereo
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)
        
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        
        fig, ax = plt.subplots(figsize=(12, 6))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        return fig
    
    @staticmethod
    def plot_piano_roll(piano_roll, title='Piano Roll'):
        """Plot MIDI piano roll"""
        fig, ax = plt.subplots(figsize=(12, 6))
        librosa.display.specshow(piano_roll, x_axis='time', y_axis='cqt_note', 
                                fmin=librosa.midi_to_hz(21), ax=ax)
        ax.set_title(title)
        return fig

# Main class for Saarland Music Dataset recordings
class Recording:
    def __init__(self, recording_id, base_path=None):
        """
        Initialize a recording from the Saarland Music Dataset v2
        
        Parameters:
        -----------
        recording_id : str
            ID of the recording in the dataset
        base_path : str, optional
            Base path to the dataset directory
        """
        self.recording_id = recording_id
        self.base_path = base_path or os.getcwd()
        
        # Data containers
        self.audio_22 = None
        self.audio_44 = None
        self.midi_data = None
        self.csv_data = None
        self.wav_midi = None
        
        # Metadata
        self.sr_22 = 22050
        self.sr_44 = 44100
        self.duration = None
        
        # Processors
        self.audio_processor = AudioProcessor()
        self.midi_processor = MidiProcessor()
        self.visualizer = MusicVisualizer()
        
        # Processing history
        self.processing_history = []
    
    def load_audio(self, sample_rate='both'):
        """Load audio data from WAV files"""
        if sample_rate in ['both', '22']:
            try:
                wav_22_path = os.path.join(self.base_path, 'wav-22', f"{self.recording_id}.wav")
                sr, audio = wavfile.read(wav_22_path)
                self.audio_22 = audio
                self.sr_22 = sr
                self.processing_history.append(f"Loaded 22kHz audio from {wav_22_path}")
            except Exception as e:
                print(f"Error loading 22kHz audio: {e}")
        
        if sample_rate in ['both', '44']:
            try:
                wav_44_path = os.path.join(self.base_path, 'wav-44', f"{self.recording_id}.wav")
                sr, audio = wavfile.read(wav_44_path)
                self.audio_44 = audio
                self.sr_44 = sr
                self.processing_history.append(f"Loaded 44kHz audio from {wav_44_path}")
            except Exception as e:
                print(f"Error loading 44kHz audio: {e}")
        
        # Calculate duration if audio was loaded
        if self.audio_44 is not None:
            self.duration = len(self.audio_44) / self.sr_44
        elif self.audio_22 is not None:
            self.duration = len(self.audio_22) / self.sr_22
            
        return self
    
    def load_midi(self):
        """Load MIDI data"""
        try:
            midi_path = os.path.join(self.base_path, 'midi', f"{self.recording_id}.mid")
            self.midi_data = pretty_midi.PrettyMIDI(midi_path)
            self.processing_history.append(f"Loaded MIDI from {midi_path}")
        except Exception as e:
            print(f"Error loading MIDI: {e}")
        
        return self
    
    def load_csv(self):
        """Load CSV annotation data"""
        try:
            csv_path = os.path.join(self.base_path, 'csv', f"{self.recording_id}.csv")
            self.csv_data = pd.read_csv(csv_path)
            self.processing_history.append(f"Loaded CSV from {csv_path}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
        
        return self
    
    def load_wav_midi(self):
        """Load WAV-MIDI aligned audio"""
        try:
            wav_midi_path = os.path.join(self.base_path, 'wav-midi', f"{self.recording_id}.wav")
            sr, audio = wavfile.read(wav_midi_path)
            self.wav_midi = audio
            self.processing_history.append(f"Loaded WAV-MIDI from {wav_midi_path}")
        except Exception as e:
            print(f"Error loading WAV-MIDI: {e}")
        
        return self
    
    def load_all(self):
        """Load all available data for this recording"""
        self.load_audio()
        self.load_midi()
        self.load_csv()
        self.load_wav_midi()
        return self
    
    # Audio processing methods
    def normalize_audio(self, sample_rate='44', method='minmax'):
        """Normalize audio data"""
        if sample_rate == '44' and self.audio_44 is not None:
            self.audio_44 = self.audio_processor.normalize(self.audio_44, method)
            self.processing_history.append(f"Normalized 44kHz audio using {method}")
        elif sample_rate == '22' and self.audio_22 is not None:
            self.audio_22 = self.audio_processor.normalize(self.audio_22, method)
            self.processing_history.append(f"Normalized 22kHz audio using {method}")
        else:
            raise ValueError(f"No audio data loaded for {sample_rate}kHz")
        
        return self
    
    def filter_audio(self, sample_rate='44', filter_type='lowpass', cutoff=0.1, order=5):
        """Apply filter to audio data"""
        if sample_rate == '44' and self.audio_44 is not None:
            self.audio_44 = self.audio_processor.filter_signal(
                self.audio_44, self.sr_44, filter_type, cutoff, order
            )
            self.processing_history.append(f"Applied {filter_type} filter to 44kHz audio")
        elif sample_rate == '22' and self.audio_22 is not None:
            self.audio_22 = self.audio_processor.filter_signal(
                self.audio_22, self.sr_22, filter_type, cutoff, order
            )
            self.processing_history.append(f"Applied {filter_type} filter to 22kHz audio")
        else:
            raise ValueError(f"No audio data loaded for {sample_rate}kHz")
        
        return self
    
    def extract_features(self, sample_rate='44', feature_type='mfcc', n_mfcc=13, hop_length=512):
        """Extract audio features"""
        audio = self.audio_44 if sample_rate == '44' and self.audio_44 is not None else self.audio_22
        sr = self.sr_44 if sample_rate == '44' and self.audio_44 is not None else self.sr_22
        
        if audio is None:
            raise ValueError(f"No audio data loaded for {sample_rate}kHz")
        
        features = self.audio_processor.extract_features(
            audio, sr, feature_type, n_mfcc, hop_length
        )
        self.processing_history.append(f"Extracted {feature_type} features from {sample_rate}kHz audio")
        
        return features
    
    # MIDI processing methods
    def get_note_events(self):
        """Get note events from MIDI data"""
        if self.midi_data is None:
            raise ValueError("No MIDI data loaded")
        
        return self.midi_processor.get_note_events(self.midi_data)
    
    def get_piano_roll(self, fs=100):
        """Get piano roll representation of MIDI data"""
        if self.midi_data is None:
            raise ValueError("No MIDI data loaded")
        
        return self.midi_processor.get_piano_roll(self.midi_data, fs)
    
    # Visualization methods
    def visualize_waveform(self, sample_rate='44'):
        """Visualize audio waveform"""
        audio = self.audio_44 if sample_rate == '44' and self.audio_44 is not None else self.audio_22
        sr = self.sr_44 if sample_rate == '44' and self.audio_44 is not None else self.sr_22
        
        if audio is None:
            raise ValueError(f"No audio data loaded for {sample_rate}kHz")
        
        title = f'Waveform for recording {self.recording_id} ({sample_rate}kHz)'
        return self.visualizer.plot_waveform(audio, sr, title)
    
    def visualize_spectrogram(self, sample_rate='44'):
        """Visualize audio spectrogram"""
        audio = self.audio_44 if sample_rate == '44' and self.audio_44 is not None else self.audio_22
        sr = self.sr_44 if sample_rate == '44' and self.audio_44 is not None else self.sr_22
        
        if audio is None:
            raise ValueError(f"No audio data loaded for {sample_rate}kHz")
        
        title = f'Spectrogram for recording {self.recording_id} ({sample_rate}kHz)'
        return self.visualizer.plot_spectrogram(audio, sr, title)
    
    def visualize_piano_roll(self):
        """Visualize MIDI piano roll"""
        if self.midi_data is None:
            raise ValueError("No MIDI data loaded")
        
        piano_roll = self.get_piano_roll()
        title = f'Piano roll for recording {self.recording_id}'
        return self.visualizer.plot_piano_roll(piano_roll, title)
    
    # Dataset-specific methods
    def align_midi_to_audio(self):
        """Align MIDI events with audio using the CSV annotations"""
        if self.midi_data is None or self.csv_data is None:
            raise ValueError("Both MIDI and CSV data must be loaded first")
        
        # This is a placeholder for actual alignment logic
        # The actual implementation would depend on the specific format of the CSV files
        print("Warning: This is a placeholder for MIDI-audio alignment.")
        print("The actual implementation would depend on the specific format of the dataset.")
        
        return self.get_note_events()

# Dataset manager class
class SaarlandMusicDataset:
    def __init__(self, base_path):
        """
        Initialize the dataset manager
        
        Parameters:
        -----------
        base_path : str
            Base path to the dataset directory
        """
        self.base_path = base_path
        self.recordings = {}
        self._scan_dataset()
    
    def _scan_dataset(self):
        """Scan the dataset directory for available recordings"""
        try:
            wav_22_dir = os.path.join(self.base_path, 'wav-22')
            if os.path.exists(wav_22_dir):
                for filename in os.listdir(wav_22_dir):
                    if filename.endswith('.wav'):
                        recording_id = os.path.splitext(filename)[0]
                        self.recordings[recording_id] = {
                            'id': recording_id,
                            'wav-22': True,
                            'wav-44': False,
                            'midi': False,
                            'csv': False,
                            'wav-midi': False
                        }
            
            # Check for other file types
            for data_type in ['wav-44', 'midi', 'csv', 'wav-midi']:
                data_dir = os.path.join(self.base_path, data_type)
                if os.path.exists(data_dir):
                    ext = '.mid' if data_type == 'midi' else '.wav' if 'wav' in data_type else '.csv'
                    for filename in os.listdir(data_dir):
                        if filename.endswith(ext):
                            recording_id = os.path.splitext(filename)[0]
                            if recording_id in self.recordings:
                                self.recordings[recording_id][data_type] = True
                            else:
                                self.recordings[recording_id] = {
                                    'id': recording_id,
                                    'wav-22': False,
                                    'wav-44': False,
                                    'midi': False,
                                    'csv': False,
                                    'wav-midi': False
                                }
                                self.recordings[recording_id][data_type] = True
            
            print(f"Found {len(self.recordings)} recordings in the dataset")
        except Exception as e:
            print(f"Error scanning dataset: {e}")
    
    def get_recording(self, recording_id):
        """
        Get a recording by ID
        
        Parameters:
        -----------
        recording_id : str
            ID of the recording
            
        Returns:
        --------
        recording : Recording
            Recording object
        """
        if recording_id not in self.recordings:
            raise ValueError(f"Recording {recording_id} not found in the dataset")
        
        return Recording(recording_id, self.base_path)
    
    def get_all_recordings(self):
        """
        Get all recordings in the dataset
        
        Returns:
        --------
        recordings : list
            List of Recording objects
        """
        return [Recording(recording_id, self.base_path) for recording_id in self.recordings]
    
    def get_recordings_with_all_data(self):
        """
        Get recordings that have all data types available
        
        Returns:
        --------
        recordings : list
            List of Recording objects
        """
        complete_recordings = [
            recording_id for recording_id, data in self.recordings.items()
            if data['wav-22'] and data['wav-44'] and data['midi'] and data['csv'] and data['wav-midi']
        ]
        
        return [Recording(recording_id, self.base_path) for recording_id in complete_recordings]
    
    def summary(self):
        """
        Print a summary of the dataset
        """
        total = len(self.recordings)
        with_wav_22 = sum(1 for data in self.recordings.values() if data['wav-22'])
        with_wav_44 = sum(1 for data in self.recordings.values() if data['wav-44'])
        with_midi = sum(1 for data in self.recordings.values() if data['midi'])
        with_csv = sum(1 for data in self.recordings.values() if data['csv'])
        with_wav_midi = sum(1 for data in self.recordings.values() if data['wav-midi'])
        with_all = sum(1 for data in self.recordings.values() 
                      if data['wav-22'] and data['wav-44'] and data['midi'] and data['csv'] and data['wav-midi'])
        
        print(f"Saarland Music Dataset Summary:")
        print(f"Total recordings: {total}")
        print(f"Recordings with 22kHz audio: {with_wav_22} ({with_wav_22/total*100:.1f}%)")
        print(f"Recordings with 44kHz audio: {with_wav_44} ({with_wav_44/total*100:.1f}%)")
        print(f"Recordings with MIDI: {with_midi} ({with_midi/total*100:.1f}%)")
        print(f"Recordings with CSV annotations: {with_csv} ({with_csv/total*100:.1f}%)")
        print(f"Recordings with WAV-MIDI: {with_wav_midi} ({with_wav_midi/total*100:.1f}%)")
        print(f"Recordings with all data types: {with_all} ({with_all/total*100:.1f}%)")
    