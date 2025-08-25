"""Parser for MAESTRO dataset metadata."""
import json
import pandas as pd
from pathlib import Path
from typing import List

from ..models.metadata import TrackMetadata


class MaestroParser:
    """Parser for MAESTRO dataset CSV and JSON metadata."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / "maestro-v3.0.0.csv"
        self.json_path = self.data_dir / "maestro-v3.0.0.json"
        
    def parse_metadata(self) -> List[TrackMetadata]:
        """Parse MAESTRO metadata and return list of TrackMetadata objects."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"MAESTRO CSV not found at {self.csv_path}")
            
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        tracks = []
        for _, row in df.iterrows():
            # Extract filename from audio path
            audio_path = Path(row['audio_filename'])
            filename = audio_path.name
            track_id = f"maestro_{audio_path.stem}"
            
            # Create track metadata
            track = TrackMetadata(
                track_id=track_id,
                filename=filename,
                title=row['canonical_title'],
                composer=row['canonical_composer'],
                composer_full=row['canonical_composer'],  # MAESTRO uses full names
                duration_seconds=float(row['duration']),
                recording_date=None,  # Will need to parse year into date format
                year_composed=None,  # Not provided by MAESTRO
                dataset_source="MAESTRO",
                dataset_version="v3.0.0",
                tags=[row['split'], f"year_{row['year']}"],  # Add split and year as tags
                performer_name=None,  # MAESTRO doesn't provide performer names
                instrument="Piano",  # MAESTRO is piano-only dataset
            )
            
            # Add audio and MIDI paths to tags for reference
            if pd.notna(row['midi_filename']):
                track.tags.append("has_midi")
            
            tracks.append(track)
            
        return tracks
    
    def get_tracks_by_split(self, split: str) -> List[TrackMetadata]:
        """Get tracks filtered by split (train/validation/test)."""
        all_tracks = self.parse_metadata()
        # Check tags since we store split there
        return [t for t in all_tracks if split in t.tags]
    
    def get_unique_composers(self) -> List[str]:
        """Get list of unique composers in the dataset."""
        df = pd.read_csv(self.csv_path)
        return sorted(df['canonical_composer'].unique().tolist())
    
    def get_audio_paths(self) -> dict:
        """Get mapping of track_id to audio file paths."""
        df = pd.read_csv(self.csv_path)
        paths = {}
        for _, row in df.iterrows():
            audio_path = Path(row['audio_filename'])
            track_id = f"maestro_{audio_path.stem}"
            paths[track_id] = str(self.data_dir / row['audio_filename'])
        return paths
    
    def get_midi_paths(self) -> dict:
        """Get mapping of track_id to MIDI file paths."""
        df = pd.read_csv(self.csv_path)
        paths = {}
        for _, row in df.iterrows():
            if pd.notna(row['midi_filename']):
                audio_path = Path(row['audio_filename'])
                track_id = f"maestro_{audio_path.stem}"
                paths[track_id] = str(self.data_dir / row['midi_filename'])
        return paths
    
    def save_processed_metadata(self, output_path: Path):
        """Save processed metadata as JSON for faster loading."""
        tracks = self.parse_metadata()
        audio_paths = self.get_audio_paths()
        midi_paths = self.get_midi_paths()
        
        # Convert to dict format with paths
        tracks_with_paths = []
        for track in tracks:
            track_dict = track.model_dump()
            track_dict['audio_path'] = audio_paths.get(track.track_id)
            track_dict['midi_path'] = midi_paths.get(track.track_id)
            tracks_with_paths.append(track_dict)
        
        metadata_dict = {
            "dataset": "MAESTRO",
            "version": "v3.0.0",
            "total_tracks": len(tracks),
            "tracks": tracks_with_paths
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
            
        print(f"Saved {len(tracks)} MAESTRO tracks to {output_path}")