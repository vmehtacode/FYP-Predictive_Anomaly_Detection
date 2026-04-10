"""
Hebbian Stress Test: Validate adaptive constraint weight strengthening.

This experiment creates challenging scenarios designed to violate various
physics constraints, proving that Hebbian adaptation strengthens weights
for frequently violated constraints.

Goal: Show >5% weight change for at least one constraint type.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fyp.selfplay.bdh_enhancements import HebbianVerifier
from fyp.selfplay.proposer import ProposerAgent
from fyp.selfplay.solver import SolverAgent
from fyp.selfplay.trainer import SelfPlayTrainer
from fyp.selfplay.verifier import VerifierAgent


class StressTestSolver(SolverAgent):
    """
    Solver that returns challenging forecasts to stress-test constraints.

    This solver returns predictions that match the high-target scenarios,
    which will violate constraints and trigger Hebbian adaptation.
    """

    def __init__(self):
        """Initialize stress test solver (no real model needed)."""
        super().__init__(
            model_config={"forecast_horizon": 16},
            use_samples=True,
            pretrain_epochs=0,
            device="cpu",
        )
        self.stress_targets = {}  # Cache targets by context hash

    def predict(self, context_window, scenario=None, return_quantiles=True):
        """Return challenging forecast that may violate constraints."""
        # Use a simple hash of context to lookup cached target
        context_hash = hash(context_window.tobytes())

        if context_hash in self.stress_targets:
            forecast = self.stress_targets[context_hash]
        else:
            # Generate a challenging forecast based on context
            # Use high values that might violate household_max
            forecast = np.abs(context_window[-16:]) + np.random.rand(16) * 6.0

        if return_quantiles:
            return {
                "0.1": forecast * 0.8,
                "0.5": forecast,
                "0.9": forecast * 1.2,
            }
        return forecast

    def train_step(
        self, context, target, scenario=None, verification_reward=0, alpha=0.1
    ):
        """Cache target for this context (simulate 'learning')."""
        context_hash = hash(context.tobytes())
        self.stress_targets[context_hash] = target
        return 1.0  # Fixed loss

    def update_historical_buffer(self, windows):
        """No-op for stress test solver."""
        pass


def create_stress_scenarios(num_scenarios: int = 30) -> list:
    """
    Create challenging scenarios designed to violate specific constraints.

    Args:
        num_scenarios: Number of stress test scenarios to generate

    Returns:
        List of (context, target) tuples with deliberate constraint violations
    """
    np.random.seed(42)
    scenarios = []

    # Scenario type 1: Household max violations (high consumption)
    for _ in range(6):
        context = np.random.rand(336) * 2.0  # Normal context
        target = (
            np.random.rand(16) * 12.0 + 8.0
        )  # Force 8-20 kWh (violates 7.5 typical)
        scenarios.append((context, target))

    # Scenario type 2: Ramp rate violations (sudden spikes)
    for _ in range(6):
        context = np.random.rand(336) * 2.0
        target = np.ones(16) * 2.0
        # Sudden spike in middle
        spike_start = np.random.randint(4, 10)
        target[spike_start : spike_start + 2] = 10.0
        target[spike_start + 2 : spike_start + 4] = 2.0  # Sudden drop
        scenarios.append((context, target))

    # Scenario type 3: Temporal pattern violations (abnormal day/night)
    for _ in range(6):
        context = np.random.rand(336) * 2.0
        # Inverted pattern: high at night, low during day
        target = np.array([5.0 if i % 48 < 24 else 1.0 for i in range(16)])
        target += np.random.rand(16) * 0.5  # Add noise
        scenarios.append((context, target))

    # Scenario type 4: Power factor violations (extreme variations)
    for _ in range(6):
        context = np.random.rand(336) * 2.0
        # Highly variable consumption (poor power factor proxy)
        target = np.random.rand(16) * 8.0
        target[::2] += 4.0  # High alternation
        scenarios.append((context, target))

    # Scenario type 5: Voltage violations (very high loads)
    for _ in range(6):
        context = np.random.rand(336) * 2.0
        # Sustained high load (causes voltage drop)
        target = np.ones(16) * 6.0 + np.random.rand(16) * 2.0
        scenarios.append((context, target))

    logger.info(f"Created {len(scenarios)} stress test scenarios")
    return scenarios


def train_with_hebbian(
    proposer: ProposerAgent,
    solver: SolverAgent,
    verifier: VerifierAgent,
    stress_data: list,
    num_episodes: int = 30,
) -> tuple:
    """Train with Hebbian adaptation enabled."""
    logger.info("Training with Hebbian adaptation...")

    # Wrap verifier with Hebbian adaptation
    hebbian_verifier = HebbianVerifier(
        verifier,
        hebbian_rate=0.02,
        decay_rate=0.005,  # Higher rate for stress test
    )

    # Create custom trainer that uses Hebbian verifier
    trainer = SelfPlayTrainer(
        proposer=proposer,
        solver=solver,
        verifier=verifier,  # Use base verifier
        config={"alpha": 0.15, "batch_size": 4, "log_every": 5},
    )

    # Replace verifier with Hebbian version after initialization
    trainer.verifier = hebbian_verifier
    trainer.hebbian_verifier = hebbian_verifier

    metrics = trainer.train(
        num_episodes=num_episodes, train_data=stress_data, save_checkpoints=False
    )

    return metrics, hebbian_verifier


def train_baseline(
    proposer: ProposerAgent,
    solver: SolverAgent,
    verifier: VerifierAgent,
    stress_data: list,
    num_episodes: int = 30,
) -> list:
    """Train without Hebbian adaptation (baseline)."""
    logger.info("Training baseline (no Hebbian)...")

    trainer = SelfPlayTrainer(
        proposer=proposer,
        solver=solver,
        verifier=verifier,
        config={"alpha": 0.15, "batch_size": 4, "log_every": 5},
    )

    metrics = trainer.train(
        num_episodes=num_episodes, train_data=stress_data, save_checkpoints=False
    )

    return metrics


def analyze_results(
    hebbian_metrics: list, baseline_metrics: list, hebbian_verifier: HebbianVerifier
) -> dict:
    """
    Analyze Hebbian adaptation results vs baseline.

    Args:
        hebbian_metrics: Metrics from Hebbian training
        baseline_metrics: Metrics from baseline training
        hebbian_verifier: Verifier instance with weight tracking

    Returns:
        Dictionary with analysis results
    """
    weight_stats = hebbian_verifier.get_weight_statistics()

    results = {
        "hebbian": {
            "final_mae": hebbian_metrics[-1].get("avg_mae", np.nan),
            "final_reward": hebbian_metrics[-1]["avg_verification_reward"],
            "final_loss": hebbian_metrics[-1]["avg_solver_loss"],
        },
        "baseline": {
            "final_mae": baseline_metrics[-1].get("avg_mae", np.nan),
            "final_reward": baseline_metrics[-1]["avg_verification_reward"],
            "final_loss": baseline_metrics[-1]["avg_solver_loss"],
        },
        "weight_changes": {},
        "violation_rates": {},
    }

    logger.info("\n" + "=" * 70)
    logger.info("HEBBIAN ADAPTATION ANALYSIS")
    logger.info("=" * 70)

    for constraint, stats in weight_stats.items():
        change = stats["weight"] - stats["baseline_weight"]
        pct_change = (change / stats["baseline_weight"]) * 100

        results["weight_changes"][constraint] = {
            "baseline": stats["baseline_weight"],
            "final": stats["weight"],
            "absolute_change": change,
            "percent_change": pct_change,
        }

        results["violation_rates"][constraint] = stats["activation_rate"]

        logger.info(
            f"{constraint:20s}: "
            f"{stats['baseline_weight']:.3f} → {stats['weight']:.3f} "
            f"({pct_change:+.1f}%), "
            f"violations: {stats['activation_rate']:.1%}"
        )

    # Find most adapted constraint
    max_change_constraint = max(
        results["weight_changes"].items(), key=lambda x: abs(x[1]["percent_change"])
    )

    logger.info(
        f"\nMost adapted: {max_change_constraint[0]} "
        f"({max_change_constraint[1]['percent_change']:+.1f}%)"
    )

    # Performance comparison
    logger.info("\nPerformance Comparison:")
    logger.info(f"  Hebbian final reward: {results['hebbian']['final_reward']:.4f}")
    logger.info(f"  Baseline final reward: {results['baseline']['final_reward']:.4f}")

    reward_improvement = (
        results["hebbian"]["final_reward"] - results["baseline"]["final_reward"]
    )
    logger.info(f"  Improvement: {reward_improvement:+.4f}")

    results["performance"] = {
        "reward_improvement": reward_improvement,
        "max_weight_change_pct": max_change_constraint[1]["percent_change"],
    }

    return results


def plot_results(
    hebbian_metrics: list,
    baseline_metrics: list,
    hebbian_verifier: HebbianVerifier,
    results: dict,
) -> None:
    """Create 6-panel visualization."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    episodes = range(len(hebbian_metrics))

    # Panel 1: Weight evolution over time
    ax1 = fig.add_subplot(gs[0, 0])
    for constraint, history in hebbian_verifier.weight_history.items():
        ax1.plot(history, label=constraint, linewidth=2, alpha=0.8)
    ax1.set_xlabel("Evaluation Step")
    ax1.set_ylabel("Constraint Weight")
    ax1.set_title("Hebbian Weight Evolution", fontweight="bold")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Weight comparison (baseline vs final)
    ax2 = fig.add_subplot(gs[0, 1])
    constraints = list(results["weight_changes"].keys())
    baseline_weights = [results["weight_changes"][c]["baseline"] for c in constraints]
    final_weights = [results["weight_changes"][c]["final"] for c in constraints]

    x = np.arange(len(constraints))
    width = 0.35

    ax2.bar(x - width / 2, baseline_weights, width, label="Baseline", alpha=0.7)
    ax2.bar(x + width / 2, final_weights, width, label="Adapted", alpha=0.7)
    ax2.set_xlabel("Constraint")
    ax2.set_ylabel("Weight")
    ax2.set_title("Weight Comparison: Baseline vs Adapted", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(constraints, rotation=45, ha="right", fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # Panel 3: Violation rates
    ax3 = fig.add_subplot(gs[0, 2])
    violation_rates = [results["violation_rates"][c] * 100 for c in constraints]
    colors = plt.cm.RdYlGn_r(np.array(violation_rates) / 100)
    bars = ax3.bar(constraints, violation_rates, color=colors, alpha=0.7)
    ax3.set_xlabel("Constraint")
    ax3.set_ylabel("Violation Rate (%)")
    ax3.set_title("Constraint Violation Frequency", fontweight="bold")
    ax3.set_xticklabels(constraints, rotation=45, ha="right", fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    # Add percentage labels
    for bar, rate in zip(bars, violation_rates, strict=False):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Panel 4: Verification rewards over episodes
    ax4 = fig.add_subplot(gs[1, 0])
    hebbian_rewards = [m["avg_verification_reward"] for m in hebbian_metrics]
    baseline_rewards = [m["avg_verification_reward"] for m in baseline_metrics]
    ax4.plot(episodes, hebbian_rewards, "b-o", label="Hebbian", linewidth=2)
    ax4.plot(
        episodes,
        baseline_rewards,
        color="gray",
        linestyle="--",
        marker="s",
        label="Baseline",
        linewidth=2,
    )
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Verification Reward")
    ax4.set_title("Verification Rewards: Hebbian vs Baseline", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Panel 5: Solver loss over episodes
    ax5 = fig.add_subplot(gs[1, 1])
    hebbian_loss = [m["avg_solver_loss"] for m in hebbian_metrics]
    baseline_loss = [m["avg_solver_loss"] for m in baseline_metrics]
    ax5.plot(episodes, hebbian_loss, "b-o", label="Hebbian", linewidth=2)
    ax5.plot(
        episodes,
        baseline_loss,
        color="gray",
        linestyle="--",
        marker="s",
        label="Baseline",
        linewidth=2,
    )
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Solver Loss")
    ax5.set_title("Solver Loss: Hebbian vs Baseline", fontweight="bold")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel 6: Weight changes summary
    ax6 = fig.add_subplot(gs[1, 2])
    pct_changes = [results["weight_changes"][c]["percent_change"] for c in constraints]
    colors_bar = ["green" if pc > 0 else "blue" for pc in pct_changes]
    bars = ax6.barh(constraints, pct_changes, color=colors_bar, alpha=0.7)
    ax6.set_xlabel("Weight Change (%)")
    ax6.set_ylabel("Constraint")
    ax6.set_title("Hebbian Weight Changes from Baseline", fontweight="bold")
    ax6.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax6.grid(True, alpha=0.3, axis="x")

    # Add percentage labels
    for bar, pc in zip(bars, pct_changes, strict=False):
        width_val = bar.get_width()
        ax6.text(
            width_val,
            bar.get_y() + bar.get_height() / 2.0,
            f"{pc:+.1f}%",
            ha="left" if pc > 0 else "right",
            va="center",
            fontsize=9,
        )

    plt.suptitle(
        "Hebbian Stress Test: Adaptive Constraint Weight Strengthening",
        fontsize=16,
        fontweight="bold",
    )

    # Save
    figures_dir = Path(__file__).parent.parent / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "hebbian_adaptation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.success(f"\nPlot saved to {output_path}")

    plt.close()


def main() -> None:
    """Run Hebbian stress test experiment."""
    logger.info("=" * 70)
    logger.info("HEBBIAN STRESS TEST EXPERIMENT")
    logger.info("=" * 70)

    # 1. Create stress scenarios
    logger.info("\nStep 1: Creating stress test scenarios...")
    stress_data = create_stress_scenarios(num_scenarios=30)

    # 2. Initialize components
    logger.info("\nStep 2: Initializing self-play components...")

    proposer = ProposerAgent(
        ssen_constraints_path="data/derived/ssen_constraints.json",
        difficulty_curriculum=True,
        random_seed=42,
    )

    # Use stress test solver that returns challenging forecasts
    solver = StressTestSolver()
    logger.info("Using StressTestSolver for constraint violation testing")

    verifier = VerifierAgent(ssen_constraints_path="data/derived/ssen_constraints.json")

    # 3. Train with Hebbian adaptation
    logger.info("\nStep 3: Training with Hebbian adaptation (30 episodes)...")
    hebbian_metrics, hebbian_verifier = train_with_hebbian(
        proposer, solver, verifier, stress_data, num_episodes=30
    )

    # 4. Train baseline
    logger.info("\nStep 4: Training baseline without Hebbian (30 episodes)...")
    baseline_metrics = train_baseline(
        proposer, solver, verifier, stress_data, num_episodes=30
    )

    # 5. Analyze results
    logger.info("\nStep 5: Analyzing Hebbian adaptation results...")
    results = analyze_results(hebbian_metrics, baseline_metrics, hebbian_verifier)

    # 6. Create visualization
    logger.info("\nStep 6: Creating visualization...")
    plot_results(hebbian_metrics, baseline_metrics, hebbian_verifier, results)

    # 7. Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / "hebbian_stress_test.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"\nResults saved to {output_file}")

    # 8. Validation
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION")
    logger.info("=" * 70)

    max_change = results["performance"]["max_weight_change_pct"]
    logger.info(f"Maximum weight change: {max_change:+.1f}%")

    if abs(max_change) > 5.0:
        logger.success(
            f"✅ SUCCESS: Weight changed by {max_change:+.1f}% (target: >5%)"
        )
    else:
        logger.warning(f"⚠️  Weight change {max_change:+.1f}% below target (5%)")

    logger.info("\n" + "=" * 70)
    logger.success("HEBBIAN STRESS TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
