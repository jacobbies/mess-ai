#!/usr/bin/env python3
"""
Layer Discovery & Probing
Run systematic discovery of MERT layer specializations.

Results are tracked in MLflow — run `mlflow ui` to browse experiments.

Usage:
    python research/scripts/run_probing.py
    python research/scripts/run_probing.py --no-save
    python research/scripts/run_probing.py --samples 30 --alpha 0.5
    python research/scripts/run_probing.py --experiment "ridge_tuning"
"""

import argparse
import logging

import mlflow

from mess.config import mess_config
from mess.probing.layer_discovery import LayerDiscoverySystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "layer_discovery"


def main(save_results=True, n_samples=50, alpha=1.0, n_folds=5, experiment_name=None):
    """Run layer discovery experiments to find layer specializations."""
    mlflow.set_experiment(experiment_name or MLFLOW_EXPERIMENT)

    with mlflow.start_run():
        logger.info("Starting layer discovery experiments...")

        discovery = LayerDiscoverySystem(alpha=alpha, n_folds=n_folds)
        results = discovery.discover(n_samples=n_samples)

        if not results:
            logger.error("Discovery failed - no results produced")
            mlflow.log_param("status", "failed")
            return 1

        # Display per-layer results
        print("\n" + "=" * 70)
        print("LAYER DISCOVERY RESULTS")
        print("=" * 70)

        for layer_idx, layer_results in sorted(results.items()):
            print(f"\nLayer {layer_idx}:")
            for proxy_name, metrics in sorted(layer_results.items()):
                print(f"  {proxy_name}:")
                print(f"    R² Score:    {metrics['r2_score']:.4f}")
                print(f"    Correlation: {metrics['correlation']:.4f}")
                print(f"    RMSE:        {metrics['rmse']:.4f}")

        # Display best layers summary
        print("\n" + "=" * 70)
        print("BEST LAYER PER TARGET")
        print("=" * 70)

        best = LayerDiscoverySystem.best_layers(results)
        for target, info in best.items():
            r2 = info['r2_score']
            tag = '  EXCELLENT' if r2 > 0.9 else '  GOOD' if r2 > 0.8 else ''
            print(f"  {target:25s} -> Layer {info['layer']:2d}  (R²={r2:.4f}){tag}")

        # Save if requested (also logs artifact to MLflow)
        if save_results:
            discovery.save(results)
            logger.info(f"Results saved to: {mess_config.probing_results_file}")

        print("\n" + "=" * 70)
        print(f"MLflow run: {mlflow.active_run().info.run_id}")
        print(f"View results: mlflow ui")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MERT layer discovery")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of audio samples to use (default: 50)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regression alpha (default: 1.0)"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="MLflow experiment name (default: layer_discovery)"
    )

    args = parser.parse_args()
    raise SystemExit(main(
        save_results=not args.no_save,
        n_samples=args.samples,
        alpha=args.alpha,
        n_folds=args.folds,
        experiment_name=args.experiment,
    ))
