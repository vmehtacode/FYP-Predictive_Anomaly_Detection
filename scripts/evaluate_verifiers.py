#!/usr/bin/env python
"""Run the full verifier benchmark and output results JSON.

This script evaluates multiple verifier configurations on identical
synthetic anomaly data, producing a formatted comparison table and
saving detailed results as JSON for downstream analysis.

Usage:
    PYTHONPATH=src python scripts/evaluate_verifiers.py

    PYTHONPATH=src python scripts/evaluate_verifiers.py \
        --samples 500 --seed 42 \
        --output data/derived/evaluation/benchmark_results.json

    PYTHONPATH=src python scripts/evaluate_verifiers.py \
        --configs baseline hybrid_full decomposition
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fyp.evaluation.benchmark import VerifierBenchmark


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the benchmark script.

    Args:
        verbose: If True, set DEBUG level; else INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def print_summary_table(results: dict) -> None:
    """Print a formatted comparison table to console.

    Args:
        results: Benchmark results dict from VerifierBenchmark.run_benchmark().
    """
    configs = results.get("configurations", {})
    if not configs:
        print("No results to display.")
        return

    # Header
    header = (
        f"{'Configuration':<20} | {'ROC-AUC':>7} | {'PR-AUC':>6} | "
        f"{'Opt-F1':>6} | {'Opt-Thr':>7} | {'Opt-Prec':>8} | {'Opt-Rec':>7} | "
        f"{'Latency':>7} | {'Early-Exit':>10}"
    )
    separator = "-" * len(header)

    print()
    print(separator)
    print(header)
    print(separator)

    for name, cfg_result in configs.items():
        if "error" in cfg_result:
            print(f"{name:<20} | {'ERROR':>7} | {cfg_result['error']}")
            continue

        roc_auc = cfg_result.get("roc_auc")
        pr_auc = cfg_result.get("pr_auc")
        opt_f1 = cfg_result.get("optimal_f1")
        opt_thr = cfg_result.get("optimal_threshold")
        opt_prec = cfg_result.get("optimal_precision")
        opt_rec = cfg_result.get("optimal_recall")
        latency = cfg_result.get("mean_latency_ms", 0.0)
        early_exit = cfg_result.get("early_exit_rate")

        roc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
        pr_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
        opt_f1_str = f"{opt_f1:.4f}" if opt_f1 is not None else "N/A"
        opt_thr_str = f"{opt_thr:.4f}" if opt_thr is not None else "N/A"
        opt_prec_str = f"{opt_prec:.4f}" if opt_prec is not None else "N/A"
        opt_rec_str = f"{opt_rec:.4f}" if opt_rec is not None else "N/A"
        early_exit_str = f"{early_exit * 100:.1f}%" if early_exit is not None else "N/A"

        print(
            f"{name:<20} | {roc_str:>7} | {pr_str:>6} | "
            f"{opt_f1_str:>6} | {opt_thr_str:>7} | {opt_prec_str:>8} | {opt_rec_str:>7} | "
            f"{latency:>7.2f} | {early_exit_str:>10}"
        )

    print(separator)
    print()

    # Test data stats
    stats = results.get("test_data_stats", {})
    if stats:
        print(f"Test Data: {stats.get('total_samples', 0)} samples "
              f"({stats.get('anomalous_samples', 0)} anomalous, "
              f"{stats.get('normal_samples', 0)} normal)")
        dist = stats.get("anomaly_type_distribution", {})
        if dist:
            print(f"Anomaly types: {dist}")
        print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the full verifier benchmark and output results JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of test samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/derived/evaluation/benchmark_results.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Specific configurations to run (default: all)",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=44,
        help="Nodes per graph (must match GNN training config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main benchmark entry point.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Verifier Benchmark")
    logger.info("=" * 60)
    logger.info("  Samples: %d", args.samples)
    logger.info("  Seed: %d", args.seed)
    logger.info("  Nodes: %d", args.num_nodes)
    logger.info("  Output: %s", args.output)
    if args.configs:
        logger.info("  Configs: %s", args.configs)
    logger.info("=" * 60)

    # Create and run benchmark
    benchmark = VerifierBenchmark(
        seed=args.seed,
        num_samples=args.samples,
        num_nodes=args.num_nodes,
    )

    results = benchmark.run_benchmark()

    # Filter to requested configs if specified
    if args.configs:
        filtered = {
            k: v for k, v in results["configurations"].items()
            if k in args.configs
        }
        if not filtered:
            logger.error(
                "None of the requested configs found: %s. Available: %s",
                args.configs,
                list(results["configurations"].keys()),
            )
            return 1
        results["configurations"] = filtered

    # Print summary table
    print_summary_table(results)

    # Save results
    benchmark.save_results(results, args.output)
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
