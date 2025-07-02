#!/usr/bin/env python3
"""Test multi-dataset FAISS search functionality."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mess_ai.search.similarity import SimilaritySearchEngine


def main():
    # Initialize search engine
    print("Initializing multi-dataset search engine...")
    search_engine = SimilaritySearchEngine(
        features_dir="data/processed/features",
        feature_type="aggregated",
        cache_dir="data/processed/cache/faiss"
    )
    
    # Show available tracks
    track_names = search_engine.get_track_names()
    print(f"\nTotal tracks in index: {len(track_names)}")
    
    # Group by dataset
    datasets = {}
    for track_name in track_names:
        if ":" in track_name:
            dataset, track = track_name.split(":", 1)
        else:
            dataset = "smd"
            track = track_name
        
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(track_name)
    
    # Show dataset breakdown
    for dataset, tracks in datasets.items():
        print(f"  {dataset}: {len(tracks)} tracks")
        if tracks:
            print(f"    Sample: {tracks[0]}")
    
    # Test similarity search with a Maestro track
    maestro_tracks = [t for t in track_names if t.startswith("maestro:")]
    if maestro_tracks:
        test_track = maestro_tracks[0]
        print(f"\nTesting similarity search with: {test_track}")
        
        try:
            similar_tracks = search_engine.search(test_track, top_k=5, exclude_self=True)
            print("Similar tracks:")
            for track, score in similar_tracks:
                dataset = "smd" if ":" not in track else track.split(":")[0]
                print(f"  {score:.3f}: {track} [{dataset}]")
        except Exception as e:
            print(f"Error in similarity search: {e}")
    
    # Test with SMD track if available
    smd_tracks = [t for t in track_names if not t.startswith("maestro:")]
    if smd_tracks:
        test_track = smd_tracks[0]
        print(f"\nTesting similarity search with SMD track: {test_track}")
        
        try:
            similar_tracks = search_engine.search(test_track, top_k=5, exclude_self=True)
            print("Similar tracks:")
            for track, score in similar_tracks:
                dataset = "smd" if ":" not in track else track.split(":")[0]
                print(f"  {score:.3f}: {track} [{dataset}]")
        except Exception as e:
            print(f"Error in similarity search: {e}")
    
    print("\nMulti-dataset search test complete!")


if __name__ == "__main__":
    main()