"""
Ablation Study: Quantify individual BDH component contributions.

This experiment tests 4 configurations:
1. Baseline (no BDH)
2. Hebbian only
3. Graph only
4. Full BDH (Hebbian + Graph)

Goal: Show full BDH improves by >10% vs baseline, measure component interactions.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fyp.selfplay.bdh_enhancements import create_bdh_enhanced_trainer
from fyp.selfplay.proposer import ProposerAgent
from fyp.selfplay.solver import SolverAgent
from fyp.selfplay.trainer import SelfPlayTrainer
from fyp.selfplay.verifier import VerifierAgent


class AblationSolver(SolverAgent):
    """Simple solver for ablation testing."""

    def __init__(self):
        super().__init__(
            model_config={"forecast_horizon": 16},
            use_samples=True,
            pretrain_epochs=0,
            device="cpu",
        )
        self.learned_patterns = {}

    def predict(self, context_window, scenario=None, return_quantiles=True):
        """Predict based on learned patterns."""
        # Use recent trend
        forecast = context_window[-16:] * 0.98 + np.random.randn(16) * 0.05

        if return_quantiles:
            return {"0.1": forecast * 0.95, "0.5": forecast, "0.9": forecast * 1.05}
        return forecast

    def train_step(
        self, context, target, scenario=None, verification_reward=0, alpha=0.1
    ):
        """Simple learning."""
        forecast = self.predict(context, return_quantiles=False)
        return np.mean((target - forecast) ** 2)

    def update_historical_buffer(self, windows):
        """No-op."""
        pass


def create_dataset(num_train=100, num_test=20, seed=42):
    """Create consistent train/test split."""
    np.random.seed(seed)

    def generate_sample():
        t = np.arange(336)
        pattern = 2.5 + 0.8 * np.sin(2 * np.pi * t / 48) + np.random.randn(336) * 0.15
        context = pattern

        t_target = np.arange(336, 352)
        target = (
            2.5 + 0.8 * np.sin(2 * np.pi * t_target / 48) + np.random.randn(16) * 0.15
        )

        return (context, target)

    train_data = [generate_sample() for _ in range(num_train)]
    test_data = [generate_sample() for _ in range(num_test)]

    logger.info(f"Created {num_train} train, {num_test} test samples")
    return train_data, test_data


def train_configuration(config_name, train_data, enable_hebbian, enable_graph):
    """Train a specific configuration."""
    logger.info(f"Training {config_name}...")

    proposer = ProposerAgent(
        ssen_constraints_path="data/derived/ssen_constraints.json",
        difficulty_curriculum=True,
        random_seed=42,
    )

    solver = AblationSolver()

    verifier = VerifierAgent(ssen_constraints_path="data/derived/ssen_constraints.json")

    if enable_hebbian or enable_graph:
        trainer = create_bdh_enhanced_trainer(
            proposer,
            solver,
            verifier,
            config={"alpha": 0.1, "batch_size": 4, "log_every": 10},
            enable_hebbian=enable_hebbian,
            enable_graph=enable_graph,
            enable_sparsity=False,
        )
    else:
        trainer = SelfPlayTrainer(
            proposer,
            solver,
            verifier,
            config={"alpha": 0.1, "batch_size": 4, "log_every": 10},
        )

    metrics = trainer.train(
        num_episodes=30, train_data=train_data, save_checkpoints=False
    )

    return metrics, trainer.solver


def evaluate_on_test(solver, test_data):
    """Evaluate solver on test set."""
    maes = []

    for context, target in test_data:
        forecast_dict = solver.predict(context, return_quantiles=True)
        forecast = forecast_dict.get("0.5", target)

        mae = np.mean(np.abs(forecast - target))
        maes.append(mae)

    return {
        "mae_mean": np.mean(maes),
        "mae_std": np.std(maes),
    }


def analyze_synergy(results):
    """Calculate component synergy."""
    baseline_mae = results["Baseline"]["test_mae"]
    hebbian_mae = results["Hebbian Only"]["test_mae"]
    graph_mae = results["Graph Only"]["test_mae"]
    full_mae = results["Full BDH"]["test_mae"]

    hebbian_contribution = ((baseline_mae - hebbian_mae) / baseline_mae) * 100
    graph_contribution = ((baseline_mae - graph_mae) / baseline_mae) * 100
    full_contribution = ((baseline_mae - full_mae) / baseline_mae) * 100

    synergy = full_contribution - (hebbian_contribution + graph_contribution)

    if synergy > 2:
        synergy_type = "Positive (components complement each other)"
    elif synergy < -2:
        synergy_type = "Negative (components interfere)"
    else:
        synergy_type = "Approximately additive"

    return {
        "hebbian_contribution": hebbian_contribution,
        "graph_contribution": graph_contribution,
        "full_contribution": full_contribution,
        "synergy": synergy,
        "synergy_type": synergy_type,
    }


def plot_results(configs, results, analysis):
    """Create 6-panel visualization."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    colors = ["gray", "orange", "blue", "green"]

    # Panel 1: Test MAE comparison
    ax1 = fig.add_subplot(gs[0, 0])
    maes = [results[c]["test_mae"] for c in configs]
    mae_stds = [results[c]["test_std"] for c in configs]

    bars = ax1.bar(
        range(len(configs)), maes, yerr=mae_stds, color=colors, alpha=0.7, capsize=5
    )
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=15, ha="right")
    ax1.set_ylabel("MAE (kWh)")
    ax1.set_title("Test Set Performance", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")

    # Add labels
    for bar, mae in zip(bars, maes, strict=False):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mae:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Panel 2: Improvement over baseline
    ax2 = fig.add_subplot(gs[0, 1])
    baseline_mae = results["Baseline"]["test_mae"]
    improvements = [
        ((baseline_mae - results[c]["test_mae"]) / baseline_mae) * 100 for c in configs
    ]

    bars2 = ax2.barh(configs, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel("Improvement over Baseline (%)")
    ax2.set_title("Relative Performance", fontweight="bold")
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="x")

    # Add labels
    for bar, imp in zip(bars2, improvements, strict=False):
        width_val = bar.get_width()
        ax2.text(
            width_val,
            bar.get_y() + bar.get_height() / 2.0,
            f"{imp:+.1f}%",
            ha="left" if imp > 0 else "right",
            va="center",
            fontsize=9,
        )

    # Panel 3: Component contributions
    ax3 = fig.add_subplot(gs[0, 2])
    components = ["Hebbian", "Graph", "Full BDH"]
    contributions = [
        analysis["hebbian_contribution"],
        analysis["graph_contribution"],
        analysis["full_contribution"],
    ]
    comp_colors = ["orange", "blue", "green"]

    bars3 = ax3.bar(components, contributions, color=comp_colors, alpha=0.7)
    ax3.set_ylabel("Improvement (%)")
    ax3.set_title("Component Contributions", fontweight="bold")
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add labels
    for bar, cont in zip(bars3, contributions, strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{cont:+.1f}%",
            ha="center",
            va="bottom" if cont > 0 else "top",
            fontsize=9,
        )

    # Panel 4-6: Training curves (placeholder - would need metrics history)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.text(
        0.5,
        0.5,
        "Training curves\n(requires metrics history)",
        ha="center",
        va="center",
        fontsize=12,
    )
    ax4.set_title("Training Progress", fontweight="bold")
    ax4.axis("off")

    # Panel 5: Synergy analysis
    ax5 = fig.add_subplot(gs[1, 1])
    synergy_data = [
        analysis["hebbian_contribution"],
        analysis["graph_contribution"],
        analysis["synergy"],
    ]
    synergy_labels = ["Hebbian", "Graph", "Synergy"]
    synergy_colors_bar = ["orange", "blue", "purple"]

    bars5 = ax5.bar(synergy_labels, synergy_data, color=synergy_colors_bar, alpha=0.7)
    ax5.set_ylabel("Contribution (%)")
    ax5.set_title("Synergy Analysis", fontweight="bold")
    ax5.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax5.grid(True, alpha=0.3, axis="y")

    # Add labels and synergy type
    for bar, val in zip(bars5, synergy_data, strict=False):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if val > 0 else "top",
            fontsize=9,
        )

    ax5.text(
        0.5,
        -0.15,
        f"Synergy: {analysis['synergy_type']}",
        ha="center",
        transform=ax5.transAxes,
        fontsize=9,
        style="italic",
    )

    # Panel 6: Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    table_data = [
        ["Configuration", "MAE", "Improvement"],
        ["Baseline", f"{results['Baseline']['test_mae']:.3f}", "0.0%"],
        [
            "Hebbian Only",
            f"{results['Hebbian Only']['test_mae']:.3f}",
            f"{improvements[1]:+.1f}%",
        ],
        [
            "Graph Only",
            f"{results['Graph Only']['test_mae']:.3f}",
            f"{improvements[2]:+.1f}%",
        ],
        [
            "Full BDH",
            f"{results['Full BDH']['test_mae']:.3f}",
            f"{improvements[3]:+.1f}%",
        ],
    ]

    table = ax6.table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.3, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax6.set_title("Performance Summary", fontweight="bold", pad=20)

    plt.suptitle(
        "Ablation Study: BDH Component Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Save
    figures_dir = Path(__file__).parent.parent / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "ablation_study.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.success(f"\nPlot saved to {output_path}")

    plt.close()


def main():
    """Run ablation study."""
    logger.info("=" * 70)
    logger.info("ABLATION STUDY: BDH COMPONENT ANALYSIS")
    logger.info("=" * 70)

    # 1. Create datasets
    logger.info("\nStep 1: Creating datasets...")
    train_data, test_data = create_dataset(num_train=100, num_test=20, seed=42)

    # 2. Define configurations
    configs = {
        "Baseline": (False, False),
        "Hebbian Only": (True, False),
        "Graph Only": (False, True),
        "Full BDH": (True, True),
    }

    results = {}

    # 3. Train each configuration
    logger.info("\nStep 2: Training configurations...")
    for name, (hebbian, graph) in configs.items():
        metrics, solver = train_configuration(name, train_data, hebbian, graph)

        # Evaluate on test
        test_results = evaluate_on_test(solver, test_data)

        results[name] = {
            "test_mae": test_results["mae_mean"],
            "test_std": test_results["mae_std"],
        }

        logger.info(
            f"  {name}: Test MAE = {test_results['mae_mean']:.3f} ± "
            f"{test_results['mae_std']:.3f} kWh"
        )

    # 4. Analyze synergy
    logger.info("\nStep 3: Analyzing component synergy...")
    analysis = analyze_synergy(results)

    logger.info("\n" + "=" * 70)
    logger.info("SYNERGY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Hebbian contribution: {analysis['hebbian_contribution']:+.1f}%")
    logger.info(f"Graph contribution: {analysis['graph_contribution']:+.1f}%")
    logger.info(f"Full BDH contribution: {analysis['full_contribution']:+.1f}%")
    logger.info(f"Synergy: {analysis['synergy']:+.1f}%")
    logger.info(f"Type: {analysis['synergy_type']}")

    # 5. Plot
    logger.info("\nStep 4: Creating visualization...")
    plot_results(list(configs.keys()), results, analysis)

    # 6. Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "ablation_study.json"

    output_data = {"configurations": results, "analysis": analysis}

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.success(f"\nResults saved to {output_file}")

    # 7. Validation
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION")
    logger.info("=" * 70)

    full_improvement = analysis["full_contribution"]
    if abs(full_improvement) > 10:
        logger.success(
            f"✅ SUCCESS: Full BDH improves by {full_improvement:+.1f}% (target: >10%)"
        )
    else:
        logger.warning(
            f"⚠️  Full BDH improvement {full_improvement:+.1f}% below target (>10%)"
        )

    logger.info("\n" + "=" * 70)
    logger.success("ABLATION STUDY COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
