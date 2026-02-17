#!/usr/bin/env python3
"""
Feature Extraction Script
Extract MERT embeddings from audio files for similarity search.

Results are tracked in MLflow — run `mlflow ui` to browse experiments.

Usage:
    # Sequential (original, backward compatible)
    python scripts/extract_features.py --dataset smd

    # Parallel (4 workers, ~40-50% faster)
    python scripts/extract_features.py --dataset smd --workers 4

    # Force re-extraction
    python scripts/extract_features.py --dataset maestro --force --workers 4
"""

import argparse
import logging
from pathlib import Path

import mlflow

from mess.extraction.extractor import FeatureExtractor
from mess.datasets.factory import DatasetFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "feature_extraction"


def main(dataset_name="smd", force_recompute=False, num_workers=1):
    """
    Extract features from specified dataset.

    Args:
        dataset_name: Dataset to process (smd, maestro)
        force_recompute: Re-extract even if features exist
        num_workers: Number of worker threads (1=sequential, >1=parallel)
    """
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        logger.info(f"Starting feature extraction for {dataset_name}")

        mlflow.log_params({
            'dataset': dataset_name,
            'force_recompute': force_recompute,
            'num_workers': num_workers,
        })

        if num_workers > 1:
            logger.info(f"Using parallel extraction with {num_workers} workers")

        # Initialize dataset
        dataset = DatasetFactory.get_dataset(dataset_name)

        # Get dataset paths
        audio_dir = dataset.audio_dir
        output_dir = dataset.embeddings_dir

        logger.info(f"Audio directory: {audio_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Initialize extractor
        extractor = FeatureExtractor()

        # Extract features for entire dataset
        results = extractor.extract_dataset_features(
            audio_dir=audio_dir,
            output_dir=output_dir,
            file_pattern="*.wav",
            skip_existing=not force_recompute,
            num_workers=num_workers,
            dataset=dataset_name
        )

        # Log results to MLflow and print
        if results:
            mlflow.log_metrics({
                'total_files': results['total_files'],
                'processed': results['processed'],
                'cached': results['cached'],
                'failed': results['failed'],
                'elapsed_time': results['elapsed_time'],
                'avg_time_per_file': results['avg_time_per_file'],
            })

            logger.info("\n" + "="*60)
            logger.info("EXTRACTION STATISTICS")
            logger.info("="*60)
            logger.info(f"Total files:     {results['total_files']}")
            logger.info(f"Processed:       {results['processed']}")
            logger.info(f"Cached:          {results['cached']}")
            logger.info(f"Failed:          {results['failed']}")
            logger.info(f"Elapsed time:    {results['elapsed_time']:.2f}s")
            logger.info(f"Avg time/file:   {results['avg_time_per_file']:.2f}s")

            if results['errors']:
                mlflow.log_param('errors', len(results['errors']))
                logger.error("\nErrors:")
                for error in results['errors']:
                    logger.error(f"  {error['path']}: {error['error']}")

            logger.info("="*60)

        logger.info("Feature extraction complete!")
        logger.info(f"MLflow run: {mlflow.active_run().info.run_id}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MERT features from audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sequential extraction (original)
  python scripts/extract_features.py --dataset smd

  # Parallel extraction (4 workers, ~40-50% faster)
  python scripts/extract_features.py --dataset smd --workers 4

  # Force re-extraction with parallel processing
  python scripts/extract_features.py --dataset maestro --force --workers 4
        """
    )
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads (default: 1 for sequential, recommend 4 for parallel)"
    )

    args = parser.parse_args()
    raise SystemExit(main(args.dataset, args.force, args.workers))
