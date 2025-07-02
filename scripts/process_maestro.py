#!/usr/bin/env python3
"""Process MAESTRO dataset metadata and prepare for feature extraction."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mess_ai.metadata.maestro_csv_parser import MaestroParser


def main():
    # Initialize parser
    maestro_dir = project_root / "data" / "maestro"
    parser = MaestroParser(maestro_dir)
    
    # Parse metadata
    print("Parsing MAESTRO metadata...")
    tracks = parser.parse_metadata()
    print(f"Found {len(tracks)} tracks in MAESTRO dataset")
    
    # Show breakdown by split
    for split in ['train', 'validation', 'test']:
        split_tracks = parser.get_tracks_by_split(split)
        print(f"  {split}: {len(split_tracks)} tracks")
    
    # Show composers
    composers = parser.get_unique_composers()
    print(f"\nFound {len(composers)} unique composers")
    print("Top 10 composers:")
    for composer in composers[:10]:
        print(f"  - {composer}")
    
    # Save processed metadata
    output_path = project_root / "data" / "processed" / "metadata" / "maestro_metadata.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parser.save_processed_metadata(output_path)
    
    # Show sample tracks
    print("\nSample tracks:")
    audio_paths = parser.get_audio_paths()
    for track in tracks[:5]:
        audio_path = audio_paths.get(track.track_id, "N/A")
        print(f"  - {track.track_id}: {track.composer} - {track.title} ({track.duration_seconds:.1f}s)")
        print(f"    Audio: {Path(audio_path).name if audio_path != 'N/A' else 'N/A'}")


if __name__ == "__main__":
    main()