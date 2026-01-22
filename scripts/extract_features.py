#!/usr/bin/env python3
"""
Feature Extraction Script
Extract MERT embeddings from audio files for similarity search.

Usage:
    python scripts/extract_features.py --dataset smd
    python scripts/extract_features.py --dataset maestro --force
"""

import argparse
import logging
from pathlib import Path

from pipeline.extraction.extractor import FeatureExtractor
from pipeline.datasets.factory import DatasetFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(dataset_name="smd", force_recompute=False):
    """
    Extract features from specified dataset.

    Args:
        dataset_name: Dataset to process (smd, maestro)
        force_recompute: Re-extract even if features exist
    """
    logger.info(f"Starting feature extraction for {dataset_name}")

    # Initialize dataset
    dataset = DatasetFactory.get_dataset(dataset_name)
    audio_files = dataset.get_audio_files()

    logger.info(f"Found {len(audio_files)} audio files")

    # Initialize extractor
    extractor = FeatureExtractor()

    # Process all files
    for audio_path in audio_files:
        logger.info(f"Processing: {audio_path.name}")

        features = extractor.extract(
            audio_path,
            force_recompute=force_recompute
        )

        logger.info(f"  Extracted shape: {features['aggregated'].shape}")

    logger.info("Feature extraction complete!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MERT features from audio")
    parser.add_argument(
        "--dataset",
        default="smd",
        choices=["smd", "maestro"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if features exist"
    )

    args = parser.parse_args()
    raise SystemExit(main(args.dataset, args.force))
