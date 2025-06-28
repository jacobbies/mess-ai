import io
import os
import json
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

class MusicLibrary:
    def __init__(self, wav_dir='data/smd/wav-44', datasets=None):
        """
        Initialize MusicLibrary with support for multiple datasets.
        
        Args:
            wav_dir: Default WAV directory (for backward compatibility)
            datasets: Dict mapping dataset names to their metadata files
        """
        self.wav_dir = Path(wav_dir)
        self.current_file = None #path
        self.data = None
        self.sample_rate = 44100
        self.is_playing = False
        
        # Dataset support
        self.datasets = datasets or {}
        self.metadata_cache = {}
        
        # Add default SMD dataset
        self.datasets.setdefault('smd', {
            'wav_dir': self.wav_dir,
            'metadata_file': None
        })
        
        # Add Maestro dataset if available
        maestro_metadata = Path('data/processed/metadata/maestro_metadata.json')
        if maestro_metadata.exists():
            self.datasets['maestro'] = {
                'wav_dir': Path('data/maestro'),
                'metadata_file': maestro_metadata
            }
        
    def load_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Load metadata for a specific dataset."""
        if dataset_name not in self.datasets:
            return None
            
        if dataset_name in self.metadata_cache:
            return self.metadata_cache[dataset_name]
            
        metadata_file = self.datasets[dataset_name].get('metadata_file')
        if metadata_file and Path(metadata_file).exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.metadata_cache[dataset_name] = metadata
                return metadata
        
        return None
    
    def get_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self.datasets.keys())
    
    def list_files(self, dataset_name: str = 'smd'):
        """List all WAV files in the specified dataset directory."""
        if dataset_name not in self.datasets:
            return []
            
        wav_dir = self.datasets[dataset_name]['wav_dir']
        return list(Path(wav_dir).glob('*.wav'))
    
    def list_all_files(self) -> Dict[str, List[Path]]:
        """List all WAV files from all datasets."""
        all_files = {}
        for dataset_name in self.datasets:
            all_files[dataset_name] = self.list_files(dataset_name)
        return all_files
        
    def get_track_info(self, track_id: str, dataset_name: str = None) -> Optional[Dict]:
        """Get track information by ID, optionally from specific dataset."""
        if dataset_name:
            metadata = self.load_dataset_metadata(dataset_name)
            if metadata:
                for track in metadata.get('tracks', []):
                    if track['track_id'] == track_id:
                        return track
        else:
            # Search all datasets
            for ds_name in self.datasets:
                metadata = self.load_dataset_metadata(ds_name)
                if metadata:
                    for track in metadata.get('tracks', []):
                        if track['track_id'] == track_id:
                            return track
        return None
        
    def get_audio_path(self, track_id: str, dataset_name: str = None) -> Optional[Path]:
        """Get audio file path for a track."""
        track_info = self.get_track_info(track_id, dataset_name)
        if track_info and 'audio_path' in track_info:
            return Path(track_info['audio_path'])
        return None
    
    def load_file(self, file_path):
        """Load a WAV file"""
        self.current_file = Path(file_path)
        self.data, self.sample_rate = sf.read(file_path)
        return self.data, self.sample_rate
        
    def load_track(self, track_id: str, dataset_name: str = None):
        """Load a track by ID."""
        audio_path = self.get_audio_path(track_id, dataset_name)
        if audio_path and audio_path.exists():
            return self.load_file(audio_path)
        else:
            raise FileNotFoundError(f"Track {track_id} not found in dataset {dataset_name}")
    
    def plot_waveform(self, save_to_buffer=False):
        """Plot the waveform of the loaded audio"""
        if self.data is not None:

            duration = len(self.data) / self.sample_rate
            time = np.linspace(0, duration, len(self.data))
            plt.figure(figsize=(10, 4))
            plt.plot(time,self.data)
            plt.title(f'Waveform: {self.current_file.name if self.current_file else ""}')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            
            if save_to_buffer:
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close()
                buf.seek(0)
                return buf
            else:
                plt.show()
                return None
        else:
            print("No audio file loaded")
            return None

if __name__ == "__main__":
    library = MusicLibrary()
    wav_files = library.list_files()
    print("Available WAV files:")
    for i, file in enumerate(wav_files):
        print(f"{i+1}. {file.name}")