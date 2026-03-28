#!/usr/bin/env python
"""Generate publication-quality figures and LaTeX tables from evaluation results.

This script reads benchmark and ablation results JSON files (produced by
``evaluate_verifiers.py`` and ``run_ablation_study.py``) and generates:

  1. Configuration comparison grouped bar chart
  2. Early-exit trade-off dual-axis line plot
  3. Weight sweep heatmap
  4. Per-anomaly-type breakdown grouped bar chart
  5. Component contribution horizontal bar chart
  6. LaTeX table for thesis results chapter

Usage::

    PYTHONPATH=src python scripts/generate_evaluation_figures.py

    PYTHONPATH=src python scripts/generate_evaluation_figures.py \\
        --benchmark data/derived/evaluation/benchmark_results.json \\
        --ablation data/derived/evaluation/ablation_results.json \\
        --output-dir data/derived/evaluation/figures/ \\
        --format png --dpi 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend for script usage
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication-quality style configuration
# ---------------------------------------------------------------------------

_STYLE = "seaborn-v0_8-whitegrid"

# Color palette (colourblind-friendly)
_COLORS = {
    "baseline": "#1f77b4",
    "hybrid_full": "#2ca02c",
    "physics_only": "#d62728",
    "gnn_only": "#9467bd",
    "cascade_only": "#8c564b",
    "decomposition": "#e377c2",
    "physics_gnn": "#7f7f7f",
    "physics_cascade": "#bcbd22",
    "gnn_cascade": "#17becf",
    "isolation_forest": "#ff7f0e",  # Orange
    "autoencoder": "#aec7e8",      # Light blue
}


def _setup_style() -> None:
    """Configure matplotlib for publication-quality output."""
    plt.style.use(_STYLE)
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.1,
    })


def _color_for(name: str) -> str:
    """Return a colour for a configuration name, with a fallback."""
    return _COLORS.get(name, "#333333")


# ---------------------------------------------------------------------------
# Figure 1: Configuration Comparison
# ---------------------------------------------------------------------------


def plot_configuration_comparison(
    benchmark: dict,
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> Path | None:
    """Create a grouped bar chart comparing metrics across configurations.

    Args:
        benchmark: Benchmark results dict (from evaluate_verifiers.py JSON).
        output_dir: Directory to save the figure.
        fmt: Image format (png, pdf, svg).
        dpi: Resolution for raster formats.

    Returns:
        Path to the saved figure, or None if data is insufficient.
    """
    configs = benchmark.get("configurations", {})
    if not configs:
        logger.warning("No configurations found in benchmark results; skipping comparison chart.")
        return None

    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    config_names = [n for n in configs if "error" not in configs[n]]

    if not config_names:
        logger.warning("All configurations have errors; skipping comparison chart.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics_keys))
    width = 0.8 / len(config_names)

    for i, name in enumerate(config_names):
        vals = [configs[name].get(m, 0.0) for m in metrics_keys]
        offset = (i - len(config_names) / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, vals, width,
            label=name.replace("_", " ").title(),
            color=_color_for(name),
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7, rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_keys])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Anomaly Detection Performance: Verifier Configuration Comparison")
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)

    out_path = output_dir / f"fig_configuration_comparison.{fmt}"
    fig.savefig(out_path, dpi=dpi, format=fmt)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 2: Early-Exit Trade-off
# ---------------------------------------------------------------------------


def plot_early_exit_tradeoff(
    ablation: dict,
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> Path | None:
    """Create a dual-axis line plot of F1 vs latency for early-exit thresholds.

    Args:
        ablation: Ablation results dict (from run_ablation_study.py JSON).
        output_dir: Directory to save the figure.
        fmt: Image format.
        dpi: Resolution for raster formats.

    Returns:
        Path to saved figure, or None if data is insufficient.
    """
    sweep = ablation.get("early_exit_sweep", {}).get("sweep_results", [])
    if not sweep:
        logger.warning("No early-exit sweep data; skipping trade-off plot.")
        return None

    thresholds = [p["threshold"] for p in sweep]
    f1_scores = [p["f1"] for p in sweep]
    latencies = [p["mean_latency_ms"] for p in sweep]
    exit_rates = [p.get("early_exit_rate") for p in sweep]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # F1 on left axis
    color_f1 = "#1f77b4"
    ax1.set_xlabel("Early-Exit Threshold")
    ax1.set_ylabel("F1 Score", color=color_f1)
    line1 = ax1.plot(
        thresholds, f1_scores, "o-",
        color=color_f1, linewidth=2, markersize=8,
        label="F1 Score",
    )
    ax1.tick_params(axis="y", labelcolor=color_f1)
    ax1.set_ylim(0, max(f1_scores) * 1.15 if f1_scores else 1.0)

    # Mark optimal F1 point
    best_idx = int(np.argmax(f1_scores))
    ax1.plot(
        thresholds[best_idx], f1_scores[best_idx],
        marker="*", color="gold", markersize=20,
        zorder=5, markeredgecolor="black", markeredgewidth=1.0,
    )
    ax1.annotate(
        f"Best F1: {f1_scores[best_idx]:.3f}",
        xy=(thresholds[best_idx], f1_scores[best_idx]),
        xytext=(15, 15), textcoords="offset points",
        fontsize=10, fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "black"},
    )

    # Latency on right axis
    ax2 = ax1.twinx()
    color_lat = "#d62728"
    ax2.set_ylabel("Mean Latency (ms)", color=color_lat)
    line2 = ax2.plot(
        thresholds, latencies, "s--",
        color=color_lat, linewidth=2, markersize=8,
        label="Mean Latency",
    )
    ax2.tick_params(axis="y", labelcolor=color_lat)

    # Early-exit rate as annotations
    for i, rate in enumerate(exit_rates):
        if rate is not None:
            pct = rate * 100 if isinstance(rate, float) else 0
            ax1.annotate(
                f"{pct:.0f}%",
                xy=(thresholds[i], f1_scores[i]),
                xytext=(0, -20), textcoords="offset points",
                fontsize=8, ha="center", color="gray",
            )

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.9)

    ax1.set_title("Early-Exit Threshold: Accuracy vs Latency Trade-off")
    fig.tight_layout()

    out_path = output_dir / f"fig_early_exit_tradeoff.{fmt}"
    fig.savefig(out_path, dpi=dpi, format=fmt)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Weight Sweep Heatmap
# ---------------------------------------------------------------------------


def plot_weight_sweep_heatmap(
    ablation: dict,
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> Path | None:
    """Create a heatmap of F1 scores across ensemble weight combinations.

    Args:
        ablation: Ablation results dict.
        output_dir: Output directory.
        fmt: Image format.
        dpi: Resolution.

    Returns:
        Path to saved figure, or None if data is insufficient.
    """
    grid = ablation.get("weight_sweep", {}).get("grid_results", [])
    if not grid:
        logger.warning("No weight sweep data; skipping heatmap.")
        return None

    # Extract unique axes values
    physics_vals = sorted(set(p["physics_weight"] for p in grid))
    gnn_vals = sorted(set(p["gnn_weight"] for p in grid))

    # Build F1 matrix (physics on Y, gnn on X)
    f1_matrix = np.full((len(physics_vals), len(gnn_vals)), np.nan)
    for point in grid:
        pi = physics_vals.index(point["physics_weight"])
        gi = gnn_vals.index(point["gnn_weight"])
        f1_matrix[pi, gi] = point["f1"]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Mask NaN cells
    masked = np.ma.masked_invalid(f1_matrix)
    im = ax.imshow(
        masked, cmap="viridis", aspect="auto",
        origin="lower", interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, label="F1 Score")

    # Annotate cells
    for i in range(len(physics_vals)):
        for j in range(len(gnn_vals)):
            val = f1_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < np.nanmedian(f1_matrix) else "black"
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color=text_color,
                )

    # Mark optimal
    optimal = ablation.get("weight_sweep", {}).get("optimal", {})
    if optimal:
        opt_pi = physics_vals.index(optimal["physics_weight"])
        opt_gi = gnn_vals.index(optimal["gnn_weight"])
        ax.plot(
            opt_gi, opt_pi, marker="*",
            color="red", markersize=20,
            markeredgecolor="white", markeredgewidth=1.5,
        )

    ax.set_xticks(range(len(gnn_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in gnn_vals])
    ax.set_yticks(range(len(physics_vals)))
    ax.set_yticklabels([f"{v:.1f}" for v in physics_vals])
    ax.set_xlabel("GNN Weight")
    ax.set_ylabel("Physics Weight")
    ax.set_title("Ensemble Weight Optimization: F1 Score")

    out_path = output_dir / f"fig_weight_sweep_heatmap.{fmt}"
    fig.savefig(out_path, dpi=dpi, format=fmt)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: Per-Anomaly-Type Breakdown
# ---------------------------------------------------------------------------


def plot_anomaly_type_breakdown(
    ablation: dict,
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> Path | None:
    """Create a grouped bar chart comparing hybrid vs baseline per anomaly type.

    Args:
        ablation: Ablation results dict.
        output_dir: Output directory.
        fmt: Image format.
        dpi: Resolution.

    Returns:
        Path to saved figure, or None if data is insufficient.
    """
    per_type = ablation.get("per_anomaly_type", {})
    if not per_type:
        logger.warning("No per-anomaly-type data; skipping breakdown chart.")
        return None

    # Filter to anomaly types (exclude NORMAL if present)
    anomaly_types = sorted(
        t for t in per_type if t != "NORMAL" and t != "NONE"
    )
    if not anomaly_types:
        anomaly_types = sorted(per_type.keys())

    hybrid_f1s = [per_type[t].get("hybrid", {}).get("f1", 0) for t in anomaly_types]
    baseline_f1s = [per_type[t].get("baseline", {}).get("f1", 0) for t in anomaly_types]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(anomaly_types))
    width = 0.35

    bars_hybrid = ax.bar(
        x - width / 2, hybrid_f1s, width,
        label="Hybrid Full", color=_COLORS["hybrid_full"],
        edgecolor="white", linewidth=0.5,
    )
    bars_baseline = ax.bar(
        x + width / 2, baseline_f1s, width,
        label="Baseline", color=_COLORS["baseline"],
        edgecolor="white", linewidth=0.5,
    )

    # Add lift annotations
    for i, (h, b) in enumerate(zip(hybrid_f1s, baseline_f1s)):
        if b > 0:
            lift = (h - b) / b * 100
            sign = "+" if lift >= 0 else ""
            ax.annotate(
                f"{sign}{lift:.0f}%",
                xy=(x[i], max(h, b) + 0.02),
                ha="center", fontsize=9,
                color="green" if lift >= 0 else "red",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("_", " ").title() for t in anomaly_types],
        rotation=15, ha="right",
    )
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, max(max(hybrid_f1s, default=0), max(baseline_f1s, default=0)) * 1.2 + 0.05)
    ax.set_title("Detection Performance by Anomaly Type")
    ax.legend(loc="upper right", framealpha=0.9)

    out_path = output_dir / f"fig_anomaly_type_breakdown.{fmt}"
    fig.savefig(out_path, dpi=dpi, format=fmt)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Figure 5: Component Contribution (horizontal bar)
# ---------------------------------------------------------------------------


def plot_component_contribution(
    ablation: dict,
    output_dir: Path,
    fmt: str = "png",
    dpi: int = 300,
) -> Path | None:
    """Create a horizontal bar chart of F1 by ablation configuration.

    Args:
        ablation: Ablation results dict.
        output_dir: Output directory.
        fmt: Image format.
        dpi: Resolution.

    Returns:
        Path to saved figure, or None if data is insufficient.
    """
    isolation = ablation.get("component_isolation", {})
    if not isolation:
        logger.warning("No component isolation data; skipping contribution chart.")
        return None

    # Sort by F1 descending
    items = [
        (name, metrics.get("f1", 0.0))
        for name, metrics in isolation.items()
        if "error" not in metrics
    ]
    items.sort(key=lambda x: x[1])

    names = [it[0] for it in items]
    f1_scores = [it[1] for it in items]

    # Categorise bars by type
    def _bar_type(name: str) -> str:
        if name in ("baseline", "decomposition", "isolation_forest", "autoencoder"):
            return "baseline"
        singles = {"physics_only", "gnn_only", "cascade_only"}
        if name in singles:
            return "single"
        pairs = {"physics_gnn", "physics_cascade", "gnn_cascade"}
        if name in pairs:
            return "pair"
        return "full"

    type_colors = {
        "baseline": "#aec7e8",
        "single": "#ff9896",
        "pair": "#ffbb78",
        "full": "#98df8a",
    }
    bar_colors = [type_colors.get(_bar_type(n), "#cccccc") for n in names]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.6)))
    bars = ax.barh(
        range(len(names)), f1_scores,
        color=bar_colors, edgecolor="white", linewidth=0.5,
    )

    # Value labels
    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10,
        )

    # Baseline reference line
    baseline_f1 = isolation.get("baseline", {}).get("f1")
    if baseline_f1 is not None:
        ax.axvline(
            baseline_f1, color="#1f77b4", linestyle="--",
            linewidth=1.5, alpha=0.7, label=f"Baseline F1 ({baseline_f1:.3f})",
        )
        ax.legend(loc="lower right", framealpha=0.9)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ").title() for n in names])
    ax.set_xlabel("F1 Score")
    ax.set_title("Component Ablation: F1 Score by Configuration")

    # Custom legend for bar types
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=type_colors["baseline"], label="Baseline"),
        Patch(facecolor=type_colors["single"], label="Single Component"),
        Patch(facecolor=type_colors["pair"], label="Pair"),
        Patch(facecolor=type_colors["full"], label="Full Ensemble"),
    ]
    ax2_legend = ax.legend(
        handles=legend_patches, loc="lower right",
        framealpha=0.9, title="Configuration Type",
    )
    ax.add_artist(ax2_legend)
    if baseline_f1 is not None:
        ax.axvline(
            baseline_f1, color="#1f77b4", linestyle="--",
            linewidth=1.5, alpha=0.7,
        )

    out_path = output_dir / f"fig_component_contribution.{fmt}"
    fig.savefig(out_path, dpi=dpi, format=fmt)
    plt.close(fig)
    logger.info("Saved: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# LaTeX Table
# ---------------------------------------------------------------------------


def generate_latex_table(
    benchmark: dict,
    output_dir: Path,
) -> Path | None:
    """Generate a LaTeX-formatted results table.

    The table shows accuracy, precision, recall, F1, mean latency, and
    early-exit rate for each configuration, with best values bolded.

    Args:
        benchmark: Benchmark results dict.
        output_dir: Output directory.

    Returns:
        Path to the saved .tex file, or None if data is insufficient.
    """
    configs = benchmark.get("configurations", {})
    if not configs:
        logger.warning("No configurations for LaTeX table.")
        return None

    valid = {n: c for n, c in configs.items() if "error" not in c}
    if not valid:
        logger.warning("All configurations have errors; skipping LaTeX table.")
        return None

    metrics_keys = ["accuracy", "precision", "recall", "f1", "mean_latency_ms"]
    headers = ["Configuration", "Accuracy", "Precision", "Recall", "F1", "Latency (ms)", "Early-Exit"]

    # Find best value per metric (higher is better for all except latency)
    best: dict[str, float] = {}
    for m in metrics_keys:
        values = [c.get(m, 0.0) for c in valid.values()]
        if m == "mean_latency_ms":
            best[m] = min(values) if values else 0.0
        else:
            best[m] = max(values) if values else 0.0

    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Verifier Configuration Comparison}")
    lines.append("\\label{tab:main_results}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\hline")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\hline")

    for name, cfg_r in valid.items():
        display_name = name.replace("_", "\\_")
        cells = [display_name]

        for m in metrics_keys:
            val = cfg_r.get(m, 0.0)
            if m == "mean_latency_ms":
                cell = f"{val:.2f}"
                if abs(val - best[m]) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
            else:
                cell = f"{val:.4f}"
                if abs(val - best[m]) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
            cells.append(cell)

        # Early exit
        ee = cfg_r.get("early_exit_rate")
        if ee is not None:
            cells.append(f"{ee * 100:.1f}\\%")
        else:
            cells.append("N/A")

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex_content = "\n".join(lines) + "\n"

    out_path = output_dir / "table_main_results.tex"
    out_path.write_text(tex_content, encoding="utf-8")
    logger.info("Saved LaTeX table: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="data/derived/evaluation/benchmark_results.json",
        help="Benchmark results JSON path",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default="data/derived/evaluation/ablation_results.json",
        help="Ablation results JSON path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/derived/evaluation/figures/",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure output format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster formats",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    """Generate all figures and tables from evaluation results.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Suppress matplotlib font warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _setup_style()

    # Load benchmark results
    benchmark: dict | None = None
    bench_path = Path(args.benchmark)
    if bench_path.exists():
        with open(bench_path) as f:
            benchmark = json.load(f)
        logger.info("Loaded benchmark results from %s", bench_path)
    else:
        logger.warning("Benchmark file not found: %s — skipping benchmark figures.", bench_path)

    # Load ablation results
    ablation: dict | None = None
    ablation_path = Path(args.ablation)
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation = json.load(f)
        logger.info("Loaded ablation results from %s", ablation_path)
    else:
        logger.warning("Ablation file not found: %s — skipping ablation figures.", ablation_path)

    if benchmark is None and ablation is None:
        logger.error("No input files found. Generate benchmark/ablation results first.")
        return 1

    generated: list[str] = []

    # Figure 1: Configuration comparison (needs benchmark)
    if benchmark:
        path = plot_configuration_comparison(benchmark, output_dir, args.format, args.dpi)
        if path:
            generated.append(str(path))

    # Figure 2: Early-exit trade-off (needs ablation)
    if ablation:
        path = plot_early_exit_tradeoff(ablation, output_dir, args.format, args.dpi)
        if path:
            generated.append(str(path))

    # Figure 3: Weight sweep heatmap (needs ablation)
    if ablation:
        path = plot_weight_sweep_heatmap(ablation, output_dir, args.format, args.dpi)
        if path:
            generated.append(str(path))

    # Figure 4: Per-anomaly-type breakdown (needs ablation)
    if ablation:
        path = plot_anomaly_type_breakdown(ablation, output_dir, args.format, args.dpi)
        if path:
            generated.append(str(path))

    # Figure 5: Component contribution (needs ablation)
    if ablation:
        path = plot_component_contribution(ablation, output_dir, args.format, args.dpi)
        if path:
            generated.append(str(path))

    # LaTeX table (needs benchmark)
    if benchmark:
        path = generate_latex_table(benchmark, output_dir)
        if path:
            generated.append(str(path))

    # Summary
    print()
    print("=" * 60)
    print("Figure Generation Complete")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print(f"  Figures generated: {len(generated)}")
    for p in generated:
        print(f"    - {p}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
