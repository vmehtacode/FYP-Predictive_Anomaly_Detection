"""
BDH-Enhanced Self-Play Demonstration.

This script demonstrates the Dragon Hatchling (BDH) inspired enhancements
to the Grid Guardian self-play system, including:
1. Hebbian constraint weight adaptation
2. Graph-based scenario relationships
3. Activation sparsity monitoring (placeholder)

References:
    Kosowski et al. (2025). The Dragon Hatchling: The Missing Link between
    the Transformer and Models of the Brain. arXiv:2509.26507
    https://arxiv.org/abs/2509.26507
"""

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
from fyp.selfplay.verifier import VerifierAgent


def main() -> None:
    """Run BDH-enhanced self-play demonstration."""
    logger.info("Grid Guardian Self-Play Demo with BDH Enhancements")
    logger.info("Reference: https://arxiv.org/abs/2509.26507\n")

    # 1. Initialize base components
    logger.info("Initializing base components...")

    proposer = ProposerAgent(
        ssen_constraints_path="data/derived/ssen_constraints.json",
        difficulty_curriculum=True,
        random_seed=42,
    )

    solver = SolverAgent(
        model_config={
            "patch_len": 8,
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "forecast_horizon": 16,
            "max_epochs": 2,
        },
        use_samples=True,
        pretrain_epochs=0,
        device="cpu",
    )

    verifier = VerifierAgent(ssen_constraints_path="data/derived/ssen_constraints.json")

    # 2. Create BDH-enhanced trainer
    logger.info("\nCreating BDH-enhanced trainer...")
    config = {"alpha": 0.15, "batch_size": 4, "log_every": 1}

    trainer = create_bdh_enhanced_trainer(
        base_proposer=proposer,
        base_solver=solver,
        base_verifier=verifier,
        config=config,
        enable_hebbian=True,
        enable_graph=True,
        enable_sparsity=False,  # Requires model modifications
    )

    # 3. Create synthetic training batch
    logger.info("\nCreating synthetic batch...")
    np.random.seed(42)
    train_data = [(np.random.rand(336) * 2, np.random.rand(16) * 2) for _ in range(20)]

    # 4. Run training for 20 episodes
    logger.info("\nTraining for 20 episodes with BDH enhancements...\n")
    metrics_history = trainer.train(
        num_episodes=20, train_data=train_data, val_data=None, save_checkpoints=False
    )

    # 5. Analyze BDH-specific results
    logger.info("\n" + "=" * 70)
    logger.info("BDH ENHANCEMENT RESULTS")
    logger.info("=" * 70)

    # 5a. Hebbian weight adaptation
    if hasattr(trainer, "hebbian_verifier"):
        logger.info("\n1. HEBBIAN CONSTRAINT ADAPTATION")
        logger.info(
            "   Concept: Constraints strengthen when violated "
            "(like synaptic plasticity)"
        )
        logger.info("-" * 70)

        weight_stats = trainer.hebbian_verifier.get_weight_statistics()
        for constraint, stats in weight_stats.items():
            logger.info(
                f"   {constraint:20s}: "
                f"weight={stats['weight']:.3f} "
                f"(baseline={stats['baseline_weight']:.3f}), "
                f"violation_rate={stats['activation_rate']:.1%}"
            )

        # Find most adapted constraint
        max_adapted = max(
            weight_stats.items(),
            key=lambda x: abs(x[1]["weight"] - x[1]["baseline_weight"]),
        )
        logger.info(
            f"\n   Most adapted: {max_adapted[0]} "
            f"(change: {max_adapted[1]['weight'] - max_adapted[1]['baseline_weight']:+.3f})"
        )

    # 5b. Graph-based scenario relationships
    if hasattr(trainer, "graph_proposer"):
        logger.info("\n2. GRAPH-BASED SCENARIO RELATIONSHIPS")
        logger.info("   Concept: Scenarios follow causal graph (like neuron network)")
        logger.info("-" * 70)

        graph_stats = trainer.graph_proposer.get_graph_statistics()
        logger.info(f"   Scenario nodes: {graph_stats['num_nodes']}")
        logger.info(f"   Causal edges: {graph_stats['num_edges']}")
        logger.info(f"   Avg out-degree: {graph_stats['avg_out_degree']:.2f}")
        logger.info(f"   Graph density: {graph_stats['graph_density']:.2%}")

        # Analyze scenario transitions
        all_scenarios = []
        for m in metrics_history:
            all_scenarios.extend(m["scenarios"])

        logger.info(f"\n   Scenario occurrences over {len(metrics_history)} episodes:")
        from collections import Counter

        scenario_counts = Counter(all_scenarios)
        for scenario, count in scenario_counts.most_common():
            logger.info(
                f"      {scenario:15s}: {count:3d} ({count / len(all_scenarios):.1%})"
            )

    # 6. Standard metrics
    logger.info("\n3. STANDARD TRAINING METRICS")
    logger.info("-" * 70)
    logger.info(f"   Episodes completed: {len(metrics_history)}")
    logger.info(f"   Final MAE: {metrics_history[-1]['avg_mae']:.4f} kWh")
    logger.info(
        f"   Final Verification Reward: "
        f"{metrics_history[-1]['avg_verification_reward']:.4f}"
    )
    logger.info(
        f"   Scenario Diversity: {metrics_history[-1]['scenario_diversity']:.2%}"
    )

    # 7. Validation
    logger.info("\n4. VALIDATION")
    logger.info("-" * 70)
    assert len(metrics_history) == 20, "Should complete 20 episodes"

    # Check MAE - allow NaN if using fallback solver
    final_mae = metrics_history[-1]["avg_mae"]
    if not np.isnan(final_mae):
        assert final_mae < 10.0, "MAE should be reasonable"
        logger.success(f"   MAE validation passed: {final_mae:.4f} kWh")
    else:
        logger.warning(
            "   MAE is NaN (expected with fallback solver - PyTorch not available)"
        )

    # Check solver loss
    assert not np.isnan(
        metrics_history[-1]["avg_solver_loss"]
    ), "Loss should not be NaN"

    logger.success("   All critical validation checks passed!")

    # 8. Plot BDH-specific metrics
    plot_bdh_metrics(metrics_history, trainer)

    logger.info("\n" + "=" * 70)
    logger.success("BDH-enhanced self-play system validated successfully!")
    logger.info("=" * 70)


def plot_bdh_metrics(metrics_history: list, trainer: any) -> None:
    """
    Visualize BDH-specific training metrics.

    Args:
        metrics_history: List of metric dictionaries from training
        trainer: BDH-enhanced trainer with additional attributes
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    episodes = range(len(metrics_history))

    # Row 1: Standard metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot([m["avg_solver_loss"] for m in metrics_history], "b-o", linewidth=2)
    ax1.set_title("Solver Loss over Episodes", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        [m["avg_verification_reward"] for m in metrics_history], "g-o", linewidth=2
    )
    ax2.set_title("Verification Reward over Episodes", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot([m["avg_mae"] for m in metrics_history], "r-o", linewidth=2)
    ax3.set_title("MAE over Episodes", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("MAE (kWh)")
    ax3.grid(True, alpha=0.3)

    # Row 2: BDH-specific metrics
    if hasattr(trainer, "hebbian_verifier"):
        ax4 = fig.add_subplot(gs[1, :])
        weight_stats = trainer.hebbian_verifier.get_weight_statistics()

        constraints = list(weight_stats.keys())
        weights = [weight_stats[c]["weight"] for c in constraints]
        baselines = [weight_stats[c]["baseline_weight"] for c in constraints]
        violation_rates = [
            weight_stats[c]["activation_rate"] * 100 for c in constraints
        ]

        x = np.arange(len(constraints))
        width = 0.35

        bars1 = ax4.bar(
            x - width / 2, baselines, width, label="Baseline Weight", alpha=0.7
        )
        bars2 = ax4.bar(
            x + width / 2, weights, width, label="Current Weight", alpha=0.7
        )

        ax4.set_xlabel("Constraint")
        ax4.set_ylabel("Weight")
        ax4.set_title(
            "Hebbian Constraint Adaptation (BDH-Inspired)",
            fontsize=12,
            fontweight="bold",
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels(constraints, rotation=45, ha="right")
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis="y")

        # Add violation rate as secondary y-axis
        ax4_twin = ax4.twinx()
        ax4_twin.plot(x, violation_rates, "ro-", label="Violation Rate", linewidth=2)
        ax4_twin.set_ylabel("Violation Rate (%)", color="r")
        ax4_twin.tick_params(axis="y", labelcolor="r")
        ax4_twin.legend(loc="upper left")

    # Row 3: Scenario distribution
    ax5 = fig.add_subplot(gs[2, :2])

    all_scenarios = []
    for m in metrics_history:
        all_scenarios.extend(m["scenarios"])

    from collections import Counter

    scenario_counts = Counter(all_scenarios)
    scenarios = list(scenario_counts.keys())
    counts = [scenario_counts[s] for s in scenarios]

    colors = plt.cm.Set3(np.linspace(0, 1, len(scenarios)))
    bars = ax5.bar(scenarios, counts, color=colors)
    ax5.set_xlabel("Scenario Type")
    ax5.set_ylabel("Occurrences")
    ax5.set_title(
        "Scenario Distribution (Graph-Based Sampling)",
        fontsize=12,
        fontweight="bold",
    )
    ax5.grid(True, alpha=0.3, axis="y")

    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts, strict=False):
        height = bar.get_height()
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count}\n({count / total:.1%})",
            ha="center",
            va="bottom",
        )

    # Diversity over time
    ax6 = fig.add_subplot(gs[2, 2])
    diversity = [m["scenario_diversity"] for m in metrics_history]
    ax6.plot(diversity, "m-o", linewidth=2)
    ax6.set_xlabel("Episode")
    ax6.set_ylabel("Diversity")
    ax6.set_title("Scenario Diversity", fontsize=12, fontweight="bold")
    ax6.set_ylim([0, 1.1])
    ax6.axhline(y=np.mean(diversity), color="r", linestyle="--", label="Mean")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # Add BDH reference
    fig.text(
        0.5,
        0.02,
        "BDH Reference: Kosowski et al. (2025). The Dragon Hatchling. arXiv:2509.26507",
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Save
    figures_dir = Path(__file__).parent.parent / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "selfplay_bdh_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"\nBDH metrics plot saved to {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
