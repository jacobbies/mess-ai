#!/usr/bin/env python3
"""
Feature Extraction Script
Extract MERT embeddings from audio files for similarity search.

Results are tracked in MLflow — run `mlflow ui` to browse experiments.

Usage:
    # CPU mode (safe default)
    python scripts/extract_features.py --dataset smd

    # GPU mode with RTX 3070Ti optimizations (2-2.5x faster)
    python scripts/extract_features.py --dataset smd --device cuda

    # Parallel + GPU for large datasets (~15-20 min for 100GB)
    python scripts/extract_features.py --dataset maestro --device cuda --workers 4

    # Force re-extraction
    python scripts/extract_features.py --dataset maestro --force --workers 4 --device cuda

    # Disable mixed precision (for debugging)
    python scripts/extract_features.py --dataset smd --device cuda --no-mixed-precision
"""

import argparse
import logging

import mlflow

from mess.datasets.factory import DatasetFactory
from mess.extraction.extractor import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "feature_extraction"


def main(
    dataset_name="smd",
    force_recompute=False,
    num_workers=1,
    feature_level="all",
    batch_size=None,
    device="cpu",
    no_mixed_precision=False,
    disable_oom_recovery=False,
):
    """
    Extract features from specified dataset.

    Args:
        dataset_name: Dataset to process (smd, maestro)
        force_recompute: Re-extract even if features exist
        num_workers: Number of worker threads (1=sequential, >1=parallel)
        feature_level: all | segments | aggregated
        batch_size: Optional MERT inference batch size override
        device: Device to use ('cpu', 'cuda', 'mps')
        no_mixed_precision: Disable CUDA mixed precision (FP16)
        disable_oom_recovery: Disable automatic OOM recovery
    """
    from mess.config import mess_config

    # Configure device (CPU default, explicit GPU opt-in)
    mess_config.MERT_DEVICE = device

    # Configure CUDA optimizations (enabled by default when using CUDA)
    if device == 'cuda':
        if no_mixed_precision:
            mess_config.MERT_CUDA_MIXED_PRECISION = False
        if disable_oom_recovery:
            mess_config.MERT_CUDA_AUTO_OOM_RECOVERY = False

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        logger.info(f"Starting feature extraction for {dataset_name}")
        logger.info(f"Device: {mess_config.device.upper()}")
        logger.info(f"Batch size: {mess_config.batch_size}")

        if device == 'cuda':
            logger.info(f"CUDA mixed precision: {mess_config.MERT_CUDA_MIXED_PRECISION}")
            logger.info(f"CUDA OOM recovery: {mess_config.MERT_CUDA_AUTO_OOM_RECOVERY}")

        mlflow.log_params({
            'dataset': dataset_name,
            'device': mess_config.device,
            'force_recompute': force_recompute,
            'num_workers': num_workers,
            'feature_level': feature_level,
            'batch_size': batch_size if batch_size is not None else mess_config.batch_size,
            'mixed_precision': mess_config.MERT_CUDA_MIXED_PRECISION if device == 'cuda' else False,
            'oom_recovery': mess_config.MERT_CUDA_AUTO_OOM_RECOVERY if device == 'cuda' else False,
        })

        if num_workers > 1:
            logger.info(f"Using parallel extraction with {num_workers} workers")

        # Initialize dataset
        dataset = DatasetFactory.get_dataset(dataset_name)

        # Get dataset paths
        audio_dir = dataset.audio_dir
        output_dir = dataset.embeddings_dir

        logger.info(f"Dataset: {dataset.dataset_id}")
        logger.info(f"Audio directory: {audio_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Initialize extractor
        extractor = FeatureExtractor(batch_size=batch_size)

        include_raw = feature_level == "all"
        include_segments = feature_level in {"all", "segments"}

        # Extract features for entire dataset
        results = extractor.extract_dataset_features(
            audio_dir=audio_dir,
            output_dir=output_dir,
            file_pattern="*.wav",
            skip_existing=not force_recompute,
            num_workers=num_workers,
            dataset=None,
            include_raw=include_raw,
            include_segments=include_segments,
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
  # CPU mode (safe default)
  python scripts/extract_features.py --dataset smd

  # GPU mode with RTX 3070Ti optimizations (2-2.5x faster)
  python scripts/extract_features.py --dataset smd --device cuda

  # Parallel + GPU for large datasets (optimal speed)
  python scripts/extract_features.py --dataset maestro --device cuda --workers 4

  # Force re-extraction
  python scripts/extract_features.py --dataset maestro --force --device cuda --workers 4

  # Debug mode (disable optimizations)
  python scripts/extract_features.py --dataset smd --device cuda \\
    --no-mixed-precision --disable-oom-recovery
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
    parser.add_argument(
        "--feature-level",
        choices=["all", "segments", "aggregated"],
        default="all",
        help=(
            "Feature detail to save: all (raw+segments+aggregated), "
            "segments (segments+aggregated), or aggregated (track-level only)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override MERT inference batch size (lower values reduce memory pressure)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to use (default: cpu). Use 'cuda' for GPU acceleration on RTX 3070Ti.",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable CUDA mixed precision (FP16). Default: enabled for CUDA.",
    )
    parser.add_argument(
        "--disable-oom-recovery",
        action="store_true",
        help="Disable automatic OOM recovery. Default: enabled.",
    )

    args = parser.parse_args()
    raise SystemExit(
        main(
            args.dataset,
            args.force,
            args.workers,
            args.feature_level,
            args.batch_size,
            args.device,
            args.no_mixed_precision,
            args.disable_oom_recovery,
        )
    )
