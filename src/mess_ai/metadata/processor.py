"""
Metadata processing pipeline for music files.
"""
import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
from ..models.metadata import TrackMetadata, COMPOSER_INFO, OPUS_TITLES


class MetadataProcessor:
    """Process and manage music metadata."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default to the project's data directory
            self.data_dir = Path(__file__).parent.parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.csv_path = self.metadata_dir / "music_catalog.csv"
        self.json_path = self.metadata_dir / "music_catalog.json"
    
    def parse_smd_filename(self, filename: str) -> Dict[str, str]:
        """
        Parse SMD filename to extract metadata.
        
        Pattern: Composer_OpusInfo-Movement_PerformerID_Date-SMD.ext
        Example: Bach_BWV849-01_001_20090916-SMD.wav
        """
        # Remove extension
        base_name = filename.replace('.wav', '').replace('.mid', '').replace('.csv', '')
        
        # Remove -SMD suffix
        if base_name.endswith('-SMD'):
            base_name = base_name[:-4]
        
        # Parse components
        parts = base_name.split('_')
        
        if len(parts) < 3:
            return {"track_id": base_name, "filename": filename}
        
        result = {
            "track_id": base_name,
            "filename": filename,
            "composer": parts[0]
        }
        
        # Parse opus and movement
        if len(parts) > 1:
            opus_movement = parts[1]
            if '-' in opus_movement:
                opus, movement = opus_movement.rsplit('-', 1)
                result["opus"] = opus
                result["movement"] = movement
            else:
                result["opus"] = opus_movement
        
        # Parse performer ID
        if len(parts) > 2:
            result["performer_id"] = parts[2]
        
        # Parse date
        if len(parts) > 3:
            date_str = parts[3]
            try:
                # Convert YYYYMMDD to date
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                result["recording_date"] = datetime(year, month, day).date()
            except:
                pass
        
        return result
    
    def enhance_metadata(self, parsed: Dict[str, str]) -> TrackMetadata:
        """
        Enhance parsed filename data with additional metadata.
        """
        composer = parsed.get("composer", "Unknown")
        opus = parsed.get("opus", "")
        
        # Get composer info
        composer_info = COMPOSER_INFO.get(composer, {})
        
        # Get piece title
        title = OPUS_TITLES.get(opus, f"{composer} {opus}")
        if parsed.get("movement"):
            movement_num = parsed.get("movement", "")
            if movement_num.isdigit():
                title += f" - Movement {movement_num}"
        
        # Build tags
        tags = []
        if composer_info.get("era"):
            tags.append(composer_info["era"].lower())
        tags.append("piano")  # All SMD pieces are piano
        tags.append("classical")
        tags.append(composer.lower())
        
        # Determine form from opus info
        form = None
        title_lower = title.lower()
        if "prelude" in title_lower:
            form = "Prelude"
            tags.append("prelude")
        elif "fugue" in title_lower:
            form = "Fugue"
            tags.append("fugue")
        elif "sonata" in title_lower:
            form = "Sonata"
            tags.append("sonata")
        elif "étude" in title_lower or "etude" in title_lower:
            form = "Étude"
            tags.append("etude")
        elif "waltz" in title_lower or "valse" in title_lower:
            form = "Waltz"
            tags.append("waltz")
        elif "polonaise" in title_lower:
            form = "Polonaise"
            tags.append("polonaise")
        elif "nocturne" in title_lower:
            form = "Nocturne"
            tags.append("nocturne")
        elif "ballade" in title_lower:
            form = "Ballade"
            tags.append("ballade")
        elif "variation" in title_lower:
            form = "Variations"
            tags.append("variations")
        
        # Extract key signature from title if present
        key_signature = None
        key_match = re.search(r'in ([A-G]#?\s*(major|minor|flat major|sharp minor))', title)
        if key_match:
            key_signature = key_match.group(1)
        
        return TrackMetadata(
            track_id=parsed["track_id"],
            filename=parsed["filename"],
            title=title,
            composer=composer,
            composer_full=composer_info.get("full_name", composer),
            opus=parsed.get("opus"),
            movement=parsed.get("movement"),
            era=composer_info.get("era"),
            form=form,
            key_signature=key_signature,
            performer_id=parsed.get("performer_id"),
            recording_date=parsed.get("recording_date"),
            year_composed=None,  # Would need additional data source
            dataset_source="SMD",
            tags=tags
        )
    
    def process_smd_directory(self, wav_dir: Optional[Path] = None) -> List[TrackMetadata]:
        """
        Process all files in the SMD wav directory.
        """
        if wav_dir is None:
            wav_dir = self.data_dir / "smd" / "wav-44"
        
        metadata_list = []
        
        for filename in sorted(wav_dir.glob("*.wav")):
            parsed = self.parse_smd_filename(filename.name)
            metadata = self.enhance_metadata(parsed)
            metadata_list.append(metadata)
        
        return metadata_list
    
    def save_to_csv(self, metadata_list: List[TrackMetadata]):
        """
        Save metadata to CSV file.
        """
        # Convert to pandas DataFrame for easy CSV export
        data = [m.dict() for m in metadata_list]
        df = pd.DataFrame(data)
        
        # Convert lists to comma-separated strings for CSV
        df['tags'] = df['tags'].apply(lambda x: ','.join(x) if x else '')
        
        # Convert dates to strings
        df['recording_date'] = df['recording_date'].apply(
            lambda x: x.isoformat() if x else ''
        )
        
        # Save to CSV
        df.to_csv(self.csv_path, index=False)
        print(f"Saved {len(df)} tracks to {self.csv_path}")
        
        return df
    
    def save_to_json(self, metadata_list: List[TrackMetadata]):
        """
        Save metadata to JSON file.
        """
        data = [m.dict() for m in metadata_list]
        
        # Convert dates to strings for JSON
        for item in data:
            if item.get('recording_date'):
                item['recording_date'] = item['recording_date'].isoformat()
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} tracks to {self.json_path}")
    
    def load_from_csv(self) -> pd.DataFrame:
        """
        Load metadata from CSV file.
        """
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found at {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Convert tags back to lists
        df['tags'] = df['tags'].apply(
            lambda x: x.split(',') if pd.notna(x) and x else []
        )
        
        return df
    
    def load_metadata_dict(self) -> Dict[str, TrackMetadata]:
        """
        Load metadata as a dictionary keyed by track_id.
        """
        df = self.load_from_csv()
        metadata_dict = {}
        
        for _, row in df.iterrows():
            track_data = row.to_dict()
            # Handle NaN values
            track_data = {k: v if pd.notna(v) else None for k, v in track_data.items()}
            
            # Convert recording_date string back to date
            if track_data.get('recording_date'):
                try:
                    track_data['recording_date'] = datetime.fromisoformat(
                        track_data['recording_date']
                    ).date()
                except:
                    track_data['recording_date'] = None
            
            metadata = TrackMetadata(**track_data)
            metadata_dict[metadata.track_id] = metadata
        
        return metadata_dict
    
    def generate_composer_summary(self) -> pd.DataFrame:
        """
        Generate a summary of composers in the catalog.
        """
        df = self.load_from_csv()
        
        composer_counts = df.groupby(['composer', 'composer_full', 'era']).size().reset_index(name='track_count')
        composer_counts = composer_counts.sort_values('track_count', ascending=False)
        
        return composer_counts
    
    def search_tracks(self, 
                     composer: Optional[str] = None,
                     era: Optional[str] = None,
                     form: Optional[str] = None,
                     tags: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Search tracks based on filters.
        """
        df = self.load_from_csv()
        
        if composer:
            df = df[df['composer'].str.contains(composer, case=False, na=False)]
        
        if era:
            df = df[df['era'] == era]
        
        if form:
            df = df[df['form'] == form]
        
        if tags:
            # Check if any of the search tags are in the track tags
            def has_tags(track_tags):
                if not track_tags:
                    return False
                track_tag_list = track_tags if isinstance(track_tags, list) else track_tags.split(',')
                return any(tag in track_tag_list for tag in tags)
            
            df = df[df['tags'].apply(has_tags)]
        
        return df


def create_initial_catalog():
    """
    Create the initial music catalog from SMD files.
    """
    processor = MetadataProcessor()
    
    print("Processing SMD directory...")
    metadata_list = processor.process_smd_directory()
    
    print(f"Found {len(metadata_list)} tracks")
    
    # Save to both CSV and JSON
    processor.save_to_csv(metadata_list)
    processor.save_to_json(metadata_list)
    
    # Generate composer summary
    composer_summary = processor.generate_composer_summary()
    print("\nComposer Summary:")
    print(composer_summary)
    
    return metadata_list


if __name__ == "__main__":
    create_initial_catalog()