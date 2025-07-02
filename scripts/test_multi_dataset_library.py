#!/usr/bin/env python3
"""Test MusicLibrary with multiple datasets."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mess_ai.audio.player import MusicLibrary


def main():
    # Initialize library
    library = MusicLibrary()
    
    print("Available datasets:")
    datasets = library.get_datasets()
    for dataset in datasets:
        print(f"  - {dataset}")
    
    # Test each dataset
    for dataset_name in datasets:
        print(f"\n--- Testing {dataset_name.upper()} dataset ---")
        
        # Load metadata
        metadata = library.load_dataset_metadata(dataset_name)
        if metadata:
            print(f"Loaded metadata: {metadata['total_tracks']} tracks")
            
            # Show sample tracks
            sample_tracks = metadata['tracks'][:3]
            for track in sample_tracks:
                print(f"  - {track['track_id']}: {track['composer']} - {track['title']}")
                
                # Test track loading
                try:
                    audio_path = library.get_audio_path(track['track_id'], dataset_name)
                    if audio_path and audio_path.exists():
                        print(f"    ✓ Audio file found: {audio_path.name}")
                        
                        # Test loading the track
                        data, sr = library.load_track(track['track_id'], dataset_name)
                        duration = len(data) / sr
                        print(f"    ✓ Loaded: {duration:.1f}s at {sr}Hz")
                    else:
                        print(f"    ✗ Audio file not found")
                except Exception as e:
                    print(f"    ✗ Error loading track: {e}")
        else:
            print(f"No metadata found for {dataset_name}")
            
            # Fallback to file listing
            files = library.list_files(dataset_name)
            print(f"Found {len(files)} audio files")
            for file in files[:3]:
                print(f"  - {file.name}")
    
    print("\nMulti-dataset library test complete!")


if __name__ == "__main__":
    main()