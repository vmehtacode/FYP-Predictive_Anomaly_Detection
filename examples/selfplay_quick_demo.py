"""
Quick demonstration of self-play training with visualization.

This script tests the full propose-solve-verify loop on sample data to validate
the Grid Guardian self-play system works correctly.

References:
    See docs/selfplay_implementation.md for architecture details.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fyp.selfplay.proposer import ProposerAgent
from fyp.selfplay.solver import SolverAgent
from fyp.selfplay.trainer import SelfPlayTrainer
from fyp.selfplay.verifier import VerifierAgent


def main() -> None:
    """Run quick self-play demonstration."""
    logger.info("Grid Guardian Self-Play Demo")

    # 1. Initialize components
    logger.info("Initializing components...")

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

    trainer = SelfPlayTrainer(
        proposer=proposer,
        solver=solver,
        verifier=verifier,
        config={"alpha": 0.1, "batch_size": 4, "log_every": 1},
    )

    # 2. Create synthetic training batch
    logger.info("Creating synthetic batch...")
    np.random.seed(42)
    train_data = [(np.random.rand(336) * 2, np.random.rand(16) * 2) for _ in range(20)]

    # 3. Run training for 5 episodes
    logger.info("Training for 5 episodes...")
    metrics_history = trainer.train(
        num_episodes=5, train_data=train_data, val_data=None, save_checkpoints=False
    )

    # 4. Analyze results
    logger.info("Training Results:")
    logger.info(f"  Episodes completed: {len(metrics_history)}")
    logger.info(f"  Final MAE: {metrics_history[-1]['avg_mae']:.4f} kWh")
    logger.info(
        f"  Final Verification Reward: "
        f"{metrics_history[-1]['avg_verification_reward']:.4f}"
    )
    logger.info(
        f"  Scenario Diversity: {metrics_history[-1]['scenario_diversity']:.2%}"
    )
    logger.info(f"  Scenario Types: {set(metrics_history[-1]['scenarios'])}")

    # 5. Validate success criteria
    assert len(metrics_history) == 5, "Should complete 5 episodes"
    assert metrics_history[-1]["avg_mae"] < 10.0, "MAE should be reasonable"
    assert not np.isnan(
        metrics_history[-1]["avg_solver_loss"]
    ), "Loss should not be NaN"

    logger.success("Self-play system validated successfully!")

    # 6. Plot metrics
    plot_metrics(metrics_history)


def plot_metrics(metrics_history: list) -> None:
    """
    Visualize training progression.

    Args:
        metrics_history: List of metric dictionaries from training episodes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    episodes = range(len(metrics_history))

    # Solver loss
    axes[0, 0].plot([m["avg_solver_loss"] for m in metrics_history], "b-o")
    axes[0, 0].set_title("Solver Loss over Episodes")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)

    # Verification reward
    axes[0, 1].plot([m["avg_verification_reward"] for m in metrics_history], "g-o")
    axes[0, 1].set_title("Verification Reward over Episodes")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].grid(True, alpha=0.3)

    # MAE
    axes[1, 0].plot([m["avg_mae"] for m in metrics_history], "r-o")
    axes[1, 0].set_title("MAE over Episodes")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("MAE (kWh)")
    axes[1, 0].grid(True, alpha=0.3)

    # Proposer reward
    axes[1, 1].plot([m["avg_proposer_reward"] for m in metrics_history], "m-o")
    axes[1, 1].set_title("Proposer Learnability Reward")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Create figures directory if needed
    figures_dir = Path(__file__).parent.parent / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / "selfplay_demo_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Metrics plot saved to {output_path}")

    plt.close()


if __name__ == "__main__":
    main()
