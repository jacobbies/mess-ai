#!/usr/bin/env python3
"""
Layer Discovery & Probing
Run systematic discovery of MERT layer specializations.

Usage:
    python research/scripts/run_probing.py
    python research/scripts/run_probing.py --no-save
"""

import argparse
import json
import logging
from pathlib import Path

from mess.config import mess_config
from mess.probing.layer_discovery import LayerDiscoverySystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(save_results=True):
    """
    Run layer discovery experiments to find layer specializations.

    Args:
        save_results: Save results to JSON file
    """
    logger.info("Starting layer discovery experiments...")

    # Initialize discovery
    discovery = LayerDiscoverySystem()

    # Run full discovery
    results = discovery.discover_layer_functions()

    # Display results
    print("\n" + "=" * 70)
    print("LAYER DISCOVERY RESULTS")
    print("=" * 70)

    for layer_idx, layer_results in results.items():
        print(f"\nLayer {layer_idx}:")

        for proxy_name, metrics in layer_results.items():
            print(f"  {proxy_name}:")
            print(f"    R² Score: {metrics['r2_score']:.4f}")
            print(f"    Correlation: {metrics['correlation']:.4f}")
            print(f"    RMSE: {metrics['rmse']:.4f}")

    # Save if requested
    if save_results:
        output_path = mess_config.probing_results_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MERT layer discovery")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )

    args = parser.parse_args()
    raise SystemExit(main(save_results=not args.no_save))
