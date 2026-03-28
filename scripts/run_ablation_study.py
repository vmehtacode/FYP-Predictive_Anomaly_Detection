#!/usr/bin/env python
"""Run ablation study for hybrid verifier component analysis.

This script runs a comprehensive ablation study quantifying each
component's contribution to the hybrid verifier ensemble, including
weight sweep, early-exit trade-off, per-anomaly-type breakdown,
physics compliance measurement, and statistical significance testing.

Usage:
    PYTHONPATH=src python scripts/run_ablation_study.py

    PYTHONPATH=src python scripts/run_ablation_study.py \
        --samples 500 --seed 42 \
        --output data/derived/evaluation/ablation_results.json

    PYTHONPATH=src python scripts/run_ablation_study.py --quick
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fyp.evaluation.ablation import AblationStudy
from fyp.evaluation.benchmark import VerifierBenchmark


def _print_component_table(results: dict) -> None:
    """Print component contribution comparison table."""
    print("\n=== Component Contribution ===")
    print(f"{'Config':<20s} | {'F1':>6s} | {'Prec':>6s} | {'Recall':>6s} | {'Lift vs Baseline':>16s}")
    print("-" * 20 + "-+-" + "-" * 6 + "-+-" + "-" * 6 + "-+-" + "-" * 6 + "-+-" + "-" * 16)

    # Sort by F1 descending, but put baseline first
    configs = sorted(
        results.items(),
        key=lambda x: (-1 if x[0] == "baseline" else 0, -x[1].get("f1", 0)),
    )

    for name, metrics in configs:
        f1 = metrics.get("f1", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        lift = metrics.get("lift_vs_baseline_pct", 0)

        if name == "baseline":
            lift_str = "---"
        else:
            lift_str = f"{lift:+.1f}%"

        print(f"{name:<20s} | {f1:6.4f} | {precision:6.4f} | {recall:6.4f} | {lift_str:>16s}")


def _print_optimal_weights(weight_sweep: dict) -> None:
    """Print optimal weight configuration."""
    optimal = weight_sweep.get("optimal", {})
    if not optimal:
        print("\n=== Optimal Weights ===")
        print("  No optimal found (empty grid)")
        return

    print(f"\n=== Optimal Weights ({weight_sweep['num_combinations']} combinations tested) ===")
    print(f"  physics={optimal['physics_weight']:.1f}, "
          f"gnn={optimal['gnn_weight']:.1f}, "
          f"cascade={optimal['cascade_weight']:.2f} "
          f"-> F1={optimal['f1']:.4f}")


def _print_early_exit_table(early_exit: dict) -> None:
    """Print early-exit threshold sweep table."""
    sweep = early_exit.get("sweep_results", [])
    if not sweep:
        return

    print("\n=== Early-Exit Trade-off ===")
    print(f"{'Threshold':>9s} | {'F1':>6s} | {'Latency(ms)':>11s} | {'Exit Rate':>9s}")
    print("-" * 9 + "-+-" + "-" * 6 + "-+-" + "-" * 11 + "-+-" + "-" * 9)

    for point in sweep:
        threshold = point["threshold"]
        f1 = point["f1"]
        latency = point["mean_latency_ms"]
        exit_rate = point.get("early_exit_rate")
        exit_str = f"{exit_rate * 100:.1f}%" if exit_rate is not None else "N/A"
        print(f"{threshold:9.2f} | {f1:6.3f} | {latency:11.1f} | {exit_str:>9s}")


def _print_anomaly_type_table(per_type: dict) -> None:
    """Print per-anomaly-type breakdown."""
    if not per_type:
        return

    print("\n=== Per-Anomaly-Type Breakdown ===")
    print(f"{'Type':<18s} | {'Count':>5s} | {'Hybrid F1':>9s} | {'Baseline F1':>11s} | {'Winner':>8s}")
    print("-" * 18 + "-+-" + "-" * 5 + "-+-" + "-" * 9 + "-+-" + "-" * 11 + "-+-" + "-" * 8)

    for atype, data in sorted(per_type.items()):
        count = data.get("count", 0)
        h_f1 = data.get("hybrid", {}).get("f1", 0)
        b_f1 = data.get("baseline", {}).get("f1", 0)
        winner = "hybrid" if h_f1 >= b_f1 else "baseline"
        print(f"{atype:<18s} | {count:5d} | {h_f1:9.4f} | {b_f1:11.4f} | {winner:>8s}")


def _print_physics_compliance(compliance: dict) -> None:
    """Print physics compliance rates."""
    if not compliance:
        return

    print("\n=== Physics Compliance ===")
    for name, data in compliance.items():
        rate = data.get("compliance_rate", 0) * 100
        total = data.get("total_detected", 0)
        compliant = data.get("physics_compliant", 0)
        print(f"  {name:<20s}: {compliant}/{total} ({rate:.1f}%)")


def _print_significance(significance: dict) -> None:
    """Print statistical significance test results."""
    if not significance:
        return

    print("\n=== Statistical Significance ===")
    for test_name, data in significance.items():
        method = data.get("method", "unknown")
        p_value = data.get("p_value")
        significant = data.get("significant", False)
        sig_str = "significant" if significant else "not significant"

        if method == "wilcoxon" and p_value is not None:
            print(f"  {data.get('test_name', test_name)}: p={p_value:.4f} ({sig_str})")
        elif method == "bootstrap_ci":
            ci_low = data.get("ci_lower", 0)
            ci_high = data.get("ci_upper", 0)
            print(f"  {data.get('test_name', test_name)}: "
                  f"CI=[{ci_low:.4f}, {ci_high:.4f}] ({sig_str})")
        else:
            note = data.get("note", "")
            print(f"  {data.get('test_name', test_name)}: {sig_str} ({note})")


def _print_summary(results: dict) -> None:
    """Print a summary of which anomaly types benefit from each component."""
    per_type = results.get("per_anomaly_type", {})
    if not per_type:
        return

    print("\n=== Summary: Component Strengths ===")

    hybrid_better = []
    baseline_better = []
    for atype, data in per_type.items():
        h_f1 = data.get("hybrid", {}).get("f1", 0)
        b_f1 = data.get("baseline", {}).get("f1", 0)
        if h_f1 > b_f1:
            hybrid_better.append((atype, h_f1 - b_f1))
        elif b_f1 > h_f1:
            baseline_better.append((atype, b_f1 - h_f1))

    if hybrid_better:
        hybrid_better.sort(key=lambda x: -x[1])
        print("  Hybrid excels at:", ", ".join(
            f"{t} (+{d:.2f} F1)" for t, d in hybrid_better
        ))
    if baseline_better:
        baseline_better.sort(key=lambda x: -x[1])
        print("  Baseline excels at:", ", ".join(
            f"{t} (+{d:.2f} F1)" for t, d in baseline_better
        ))
    if not hybrid_better and not baseline_better:
        print("  Both configurations perform identically across all types")


def main() -> None:
    """Run ablation study with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run ablation study for hybrid verifier components",
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of test samples (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/derived/evaluation/ablation_results.json",
        help="Output JSON path (default: data/derived/evaluation/ablation_results.json)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer weight sweep points, fewer threshold points",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"Ablation Study Configuration:")
    print(f"  Samples: {args.samples}")
    print(f"  Seed: {args.seed}")
    print(f"  Quick mode: {args.quick}")
    print(f"  Output: {args.output}")
    print()

    # Create benchmark and study
    benchmark = VerifierBenchmark(
        num_samples=args.samples,
        seed=args.seed,
    )
    study = AblationStudy(benchmark)

    # Run full ablation
    results = study.run_full_ablation(quick=args.quick)

    # Print formatted tables
    _print_component_table(results.get("component_isolation", {}))
    _print_optimal_weights(results.get("weight_sweep", {}))
    _print_early_exit_table(results.get("early_exit_sweep", {}))
    _print_anomaly_type_table(results.get("per_anomaly_type", {}))
    _print_physics_compliance(results.get("physics_compliance", {}))
    _print_significance(results.get("significance_tests", {}))
    _print_summary(results)

    # Save results
    study.save_results(results, args.output)
    print(f"\nResults saved to {args.output}")

    # Print metadata
    meta = results.get("metadata", {})
    duration = meta.get("duration_seconds", 0)
    print(f"Total duration: {duration:.1f}s")


if __name__ == "__main__":
    main()
