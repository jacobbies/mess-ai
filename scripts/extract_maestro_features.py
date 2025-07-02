#!/usr/bin/env python3
"""Extract MERT features from MAESTRO dataset."""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mess_ai.features.extractor import FeatureExtractor
from src.mess_ai.metadata.maestro_csv_parser import MaestroParser


def main():
    # Load MAESTRO metadata
    maestro_dir = project_root / "data" / "maestro"
    metadata_path = project_root / "data" / "processed" / "metadata" / "maestro_metadata.json"
    
    if not metadata_path.exists():
        print(f"Metadata file not found at {metadata_path}")
        print("Run scripts/process_maestro.py first to generate metadata")
        return
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {metadata['total_tracks']} MAESTRO tracks")
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Process only a subset for testing (e.g., first 10 tracks from test set)
    test_tracks = [t for t in metadata['tracks'] if 'test' in t['tags']][:10]
    
    print(f"\nProcessing {len(test_tracks)} test tracks...")
    
    for i, track in enumerate(test_tracks):
        print(f"\n[{i+1}/{len(test_tracks)}] Processing {track['track_id']}")
        print(f"  Title: {track['title']}")
        print(f"  Composer: {track['composer']}")
        print(f"  Duration: {track['duration_seconds']:.1f}s")
        
        # Extract features
        audio_path = track['audio_path']
        if Path(audio_path).exists():
            try:
                # Process with dataset label
                extractor.extract_track_features(
                    audio_path, 
                    output_dir=extractor.output_dir,
                    track_id=track['track_id'],
                    dataset="maestro"  # Save in maestro subdirectory
                )
                print(f"  ✓ Features extracted successfully")
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            print(f"  ✗ Audio file not found: {audio_path}")
    
    print("\nFeature extraction complete!")
    print(f"Features saved to: {extractor.output_dir}/maestro/")


if __name__ == "__main__":
    main()