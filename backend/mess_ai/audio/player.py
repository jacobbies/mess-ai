import io
import os
import json
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MusicLibrary:
    def __init__(self, wav_dir=None, datasets=None, settings=None):
        """
        Initialize MusicLibrary with support for multiple datasets.
        
        Args:
            wav_dir: Default WAV directory (for backward compatibility)
            datasets: Dict mapping dataset names to their metadata files
            settings: Configuration settings object
        """
        # Import settings here to avoid circular imports
        if settings is None:
            from ...core.config import settings as default_settings
            settings = default_settings
        
        self.settings = settings
        
        # Use settings-based paths if available, otherwise use defaults
        if wav_dir is None and settings:
            self.wav_dir = settings.wav_dir
        else:
            self.wav_dir = Path(wav_dir or 'data/smd/wav-44')
            
        self.current_file: Optional[Path] = None
        self.data: Optional[np.ndarray] = None
        self.sample_rate: int = 44100
        self.is_playing: bool = False
        
        # Dataset support
        self.datasets: Dict = datasets or {}
        self.metadata_cache: Dict[str, Dict] = {}
        
        # Add default SMD dataset
        self.datasets.setdefault('smd', {
            'wav_dir': self.wav_dir,
            'metadata_file': None
        })
        
        # Add Maestro dataset if available
        if settings:
            maestro_metadata = settings.metadata_dir / 'maestro_metadata.json'
            maestro_wav_dir = settings.maestro_dir
        else:
            maestro_metadata = Path('data/processed/metadata/maestro_metadata.json')
            maestro_wav_dir = Path('data/maestro')
            
        if maestro_metadata.exists():
            self.datasets['maestro'] = {
                'wav_dir': maestro_wav_dir,
                'metadata_file': maestro_metadata
            }
            logger.info("Maestro dataset detected")
        else:
            logger.info("Maestro dataset not found")
        
    def load_dataset_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Load metadata for a specific dataset."""
        if dataset_name not in self.datasets:
            return None
            
        if dataset_name in self.metadata_cache:
            return self.metadata_cache[dataset_name]
            
        metadata_file = self.datasets[dataset_name].get('metadata_file')
        if metadata_file and Path(metadata_file).exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.metadata_cache[dataset_name] = metadata
                    return metadata
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load metadata for dataset '{dataset_name}': {e}")
                return None
        
        return None
    
    def get_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self.datasets.keys())
    
    def list_files(self, dataset_name: str = 'smd'):
        """List all WAV files in the specified dataset directory."""
        if dataset_name not in self.datasets:
            logger.warning(f"Dataset '{dataset_name}' not found")
            return []
            
        wav_dir = self.datasets[dataset_name]['wav_dir']
        if not Path(wav_dir).exists():
            logger.warning(f"Audio directory '{wav_dir}' does not exist for dataset '{dataset_name}'")
            return []
            
        return list(Path(wav_dir).glob('*.wav'))
    
    def list_all_files(self) -> Dict[str, List[Path]]:
        """List all WAV files from all datasets."""
        all_files = {}
        for dataset_name in self.datasets:
            all_files[dataset_name] = self.list_files(dataset_name)
        return all_files
        
    def get_track_info(self, track_id: str, dataset_name: str = None) -> Optional[Dict]: # type: ignore
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
        
    def get_audio_path(self, track_id: str, dataset_name: str = None) -> Optional[Path]: # type: ignore
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
        
    def load_track(self, track_id: str, dataset_name: str = None): # type: ignore
        """Load a track by ID."""
        audio_path = self.get_audio_path(track_id, dataset_name)
        if audio_path and audio_path.exists():
            return self.load_file(audio_path)
        else:
            dataset_info = f" in dataset '{dataset_name}'" if dataset_name else ""
            raise FileNotFoundError(f"Track '{track_id}' not found{dataset_info}. Check that the track ID is correct and the audio file exists.")
    
    def plot_waveform(self, save_to_buffer=False):
        """Plot the waveform of the loaded audio"""
        if self.data is not None:
            duration = len(self.data) / self.sample_rate
            time = np.linspace(0, duration, len(self.data))
            plt.figure(figsize=(10, 4))
            plt.plot(time, self.data)
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
            logger.warning("No audio file loaded")
            return None
    
    def get_dataset_stats(self) -> Dict[str, Dict]:
        """Get statistics for all available datasets."""
        stats = {}
        for dataset_name in self.datasets:
            try:
                wav_files = self.list_files(dataset_name)
                metadata = self.load_dataset_metadata(dataset_name)
                
                stats[dataset_name] = {
                    'audio_files': len(wav_files),
                    'metadata_tracks': len(metadata.get('tracks', [])) if metadata else 0,
                    'has_metadata': metadata is not None,
                    'wav_dir': str(self.datasets[dataset_name]['wav_dir']),
                    'metadata_file': str(self.datasets[dataset_name].get('metadata_file', 'None'))
                }
            except Exception as e:
                logger.error(f"Error getting stats for dataset '{dataset_name}': {e}")
                stats[dataset_name] = {'error': str(e)}
        
        return stats
    
    def validate_datasets(self) -> bool:
        """Validate that all configured datasets are accessible."""
        all_valid = True
        for dataset_name in self.datasets:
            wav_dir = self.datasets[dataset_name]['wav_dir']
            if not Path(wav_dir).exists():
                logger.error(f"Dataset '{dataset_name}' WAV directory not found: {wav_dir}")
                all_valid = False
            
            metadata_file = self.datasets[dataset_name].get('metadata_file')
            if metadata_file and not Path(metadata_file).exists():
                logger.warning(f"Dataset '{dataset_name}' metadata file not found: {metadata_file}")
        
        return all_valid

if __name__ == "__main__":
    library = MusicLibrary()
    wav_files = library.list_files()
    print("Available WAV files:")
    for i, file in enumerate(wav_files):
        print(f"{i+1}. {file.name}")