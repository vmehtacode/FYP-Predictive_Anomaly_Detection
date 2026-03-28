"""Self-play training orchestrator for energy forecasting.

This module implements the main training loop that coordinates the propose→solve→verify
cycle, following the Absolute Zero Reasoning (AZR) approach adapted for time-series
forecasting.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import numpy as np
from tqdm import tqdm

from fyp.data_loader import EnergyDataLoader
from fyp.metrics import mean_absolute_error, mean_absolute_percentage_error
from fyp.selfplay.proposer import ProposerAgent
from fyp.selfplay.solver import SolverAgent
from fyp.selfplay.utils import create_sliding_windows
from fyp.selfplay.verifier import VerifierAgent

logger = logging.getLogger(__name__)


def _safe_mean(values: list[float] | np.ndarray, default: float = 0.0) -> float:
    """Compute mean safely, handling empty lists and NaN values.

    Args:
        values: List or array of values
        default: Default value if list is empty or all NaN

    Returns:
        Mean value or default
    """
    if not values or len(values) == 0:
        return float(default)

    # Convert to numpy array for easier filtering
    vals = np.array(values, dtype=float)

    # Filter out NaN and infinite values
    valid_vals = vals[np.isfinite(vals)]

    if len(valid_vals) == 0:
        return float(default)

    return float(np.mean(valid_vals))


class SelfPlayTrainer:
    """Orchestrates propose→solve→verify self-play training loop."""

    def __init__(
        self,
        proposer: ProposerAgent,
        solver: SolverAgent,
        verifier: VerifierAgent,
        config: dict[str, Any] | None = None,
        graph_data=None,
    ):
        """Initialize self-play trainer.

        Args:
            proposer: ProposerAgent instance
            solver: SolverAgent instance
            verifier: VerifierAgent instance
            config: Training configuration
            graph_data: Optional PyG Data for graph-aware proposer
        """
        self.proposer = proposer
        self.solver = solver
        self.verifier = verifier
        self.graph_data = graph_data

        # Default configuration
        default_config = {
            "alpha": 0.1,  # Weight for verification reward
            "batch_size": 32,  # Windows per episode
            "lambda": 0.5,  # Balance between exploration and exploitation
            "checkpoint_every": 100,  # Episodes between checkpoints
            "log_every": 10,  # Episodes between logging
            "val_every": 50,  # Episodes between validation
            "max_windows_per_household": 10,  # Limit windows per household
            "success_threshold_mape": 20.0,  # MAPE threshold for success
            "min_episodes_before_curriculum": 50,  # Episodes before advancing curriculum
        }

        self.config = {**default_config, **(config or {})}

        # Training state
        self.episode_count = 0
        self.best_val_loss = float("inf")
        self.metrics_history = []
        self.checkpoint_dir = "data/derived/models/selfplay"

        # Performance tracking
        self.scenario_success_rates = {}
        self.solver_performance_buffer = []

        logger.info(
            f"Initialized SelfPlayTrainer with config: "
            f"alpha={self.config['alpha']}, batch_size={self.config['batch_size']}"
        )

    def train_episode(
        self,
        historical_batch: list[tuple[np.ndarray, np.ndarray]],
        current_timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """Single episode of propose→solve→verify training.

        Following AZR algorithm:
        1. PROPOSE: Generate scenario conditioned on buffer
        2. SOLVE: Forecast under proposed scenario
        3. VERIFY: Evaluate forecast plausibility
        4. UPDATE: Train both proposer and solver with rewards

        Args:
            historical_batch: List of (context, target) pairs from LCL data
            current_timestamp: Current time for scenario generation

        Returns:
            metrics: Dict with episode statistics
        """
        metrics = {
            "episode": self.episode_count,
            "scenarios": [],
            "solver_losses": [],
            "verification_rewards": [],
            "proposer_rewards": [],
            "forecast_errors": {"mae": [], "mape": [], "smape": []},
        }

        episode_start_time = time.time()

        for i, (context, ground_truth) in enumerate(historical_batch):
            # Step 1: PROPOSE scenario
            conditioning_samples = (
                self.proposer.scenario_buffer[-10:]
                if self.proposer.scenario_buffer
                else None
            )
            scenario = self.proposer.propose_scenario(
                historical_context=context,
                conditioning_samples=conditioning_samples,
                forecast_horizon=len(ground_truth),
                current_timestamp=current_timestamp,
                graph_data=self.graph_data,
            )

            # Step 2: SOLVE - forecast with scenario
            forecast = self.solver.predict(
                context_window=context, scenario=scenario, return_quantiles=True
            )
            median_forecast = forecast["0.5"]

            # Apply scenario: use per-node cascade when graph topology available
            if self.graph_data is not None and ground_truth.ndim == 2:
                modified_target = scenario.apply_to_graph_timeseries(ground_truth)
            else:
                modified_target = scenario.apply_to_timeseries(ground_truth)

            # Step 3: VERIFY - evaluate forecast
            verification_reward, details = self.verifier.evaluate(
                forecast=median_forecast, scenario=scenario, return_details=True
            )

            # Calculate forecast accuracy metrics
            mae = mean_absolute_error(modified_target, median_forecast)
            mape = mean_absolute_percentage_error(modified_target, median_forecast)
            smape = mape  # Use MAPE as proxy for SMAPE

            # Step 4a: UPDATE SOLVER
            solver_loss = self.solver.train_step(
                context=context,
                target=modified_target,
                scenario=scenario,
                verification_reward=verification_reward,
                alpha=self.config["alpha"],
            )

            # Step 4b: COMPUTE PROPOSER REWARD
            # Use solver's success as proxy for learnability
            solver_success_rate = self._compute_success_rate(
                forecast, modified_target, self.config["success_threshold_mape"]
            )
            proposer_reward = self.proposer.compute_learnability_reward(
                scenario, solver_success_rate
            )

            # Step 4c: UPDATE PROPOSER
            self.proposer.update_buffer(scenario, proposer_reward)

            # Update scenario success tracking
            scenario_key = scenario.scenario_type
            if scenario_key not in self.scenario_success_rates:
                self.scenario_success_rates[scenario_key] = []
            self.scenario_success_rates[scenario_key].append(solver_success_rate)

            # Log metrics
            metrics["scenarios"].append(scenario.scenario_type)
            metrics["solver_losses"].append(solver_loss)
            metrics["verification_rewards"].append(verification_reward)
            metrics["proposer_rewards"].append(proposer_reward)
            metrics["forecast_errors"]["mae"].append(mae)
            metrics["forecast_errors"]["mape"].append(mape)
            metrics["forecast_errors"]["smape"].append(smape)

            # Log detailed constraint violations if any
            if verification_reward < -0.5:
                logger.debug(
                    f"Severe constraint violations in window {i}: "
                    f"{[v for d in details.values() for v in d['violations']]}"
                )

        # Episode summary statistics
        metrics["episode_time"] = time.time() - episode_start_time
        metrics["avg_solver_loss"] = _safe_mean(metrics["solver_losses"], default=0.0)
        metrics["avg_verification_reward"] = _safe_mean(
            metrics["verification_rewards"], default=0.0
        )
        metrics["avg_proposer_reward"] = _safe_mean(
            metrics["proposer_rewards"], default=0.0
        )

        # Handle scenario diversity safely
        if metrics["scenarios"]:
            metrics["scenario_diversity"] = len(set(metrics["scenarios"])) / len(
                metrics["scenarios"]
            )
        else:
            metrics["scenario_diversity"] = 0.0

        for error_type in ["mae", "mape", "smape"]:
            metrics[f"avg_{error_type}"] = _safe_mean(
                metrics["forecast_errors"][error_type], default=0.0
            )

        self.episode_count += 1
        return metrics

    def train(
        self,
        num_episodes: int,
        train_data: list[dict] | None = None,
        val_data: list[dict] | None = None,
        test_data: list[dict] | None = None,
        save_checkpoints: bool = True,
    ) -> list[dict[str, Any]]:
        """Full self-play training loop with periodic validation.

        Args:
            num_episodes: Number of training episodes
            train_data: Training dataset (if None, loads from LCL)
            val_data: Validation dataset
            test_data: Test dataset for final evaluation
            save_checkpoints: Whether to save model checkpoints

        Returns:
            List of episode metrics
        """
        logger.info(f"Starting self-play training for {num_episodes} episodes")

        # Load data if not provided
        if train_data is None:
            loader = EnergyDataLoader(use_samples=self.solver.use_samples)
            train_data = self._prepare_data_windows(loader, "train")

            if val_data is None:
                val_data = self._prepare_data_windows(loader, "validation")

        # Create progress bar
        pbar = tqdm(total=num_episodes, desc="Self-play training")

        for episode in range(num_episodes):
            # Sample batch from historical data
            batch = self._sample_historical_batch(train_data)

            # Train episode
            metrics = self.train_episode(batch)
            self.metrics_history.append(metrics)

            # Update solver's historical buffer for stability
            historical_windows = [
                {"history_energy": context, "target_energy": target}
                for context, target in batch[:5]  # Keep a few for stability
            ]
            self.solver.update_historical_buffer(historical_windows)

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{metrics['avg_solver_loss']:.3f}",
                    "reward": f"{metrics['avg_verification_reward']:.3f}",
                    "mae": f"{metrics['avg_mae']:.3f}",
                }
            )

            # Periodic logging
            if episode % self.config["log_every"] == 0:
                self._print_progress(episode, metrics)

            # Periodic validation
            if episode % self.config["val_every"] == 0 and val_data is not None:
                val_metrics = self.validate(val_data[:100])  # Validate on subset
                logger.info(
                    f"Validation at episode {episode}: "
                    f"MAE={val_metrics['mae']:.3f}, "
                    f"constraint_violations={val_metrics['violation_rate']*100:.1f}%"
                )

                # Check for improvement
                if val_metrics["avg_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["avg_loss"]
                    if save_checkpoints:
                        self._save_checkpoint(f"best_model_ep{episode}.pth")

            # Periodic checkpointing
            if save_checkpoints and episode % self.config["checkpoint_every"] == 0:
                self._save_checkpoint(f"checkpoint_ep{episode}.pth")

        pbar.close()

        # Final evaluation on test set
        if test_data is not None:
            logger.info("Running final evaluation on test set")
            test_metrics = self.validate(test_data)
            logger.info(
                f"Test results: MAE={test_metrics['mae']:.3f}, "
                f"MAPE={test_metrics['mape']:.1f}%, "
                f"constraint violations={test_metrics['violation_rate']*100:.1f}%"
            )

        # Save final checkpoint
        if save_checkpoints:
            self._save_checkpoint("final_model.pth")
            self._save_training_summary()

        return self.metrics_history

    def validate(
        self, val_data: list[tuple[np.ndarray, np.ndarray]], use_scenarios: bool = False
    ) -> dict[str, float]:
        """Validate model performance on held-out data.

        Args:
            val_data: Validation windows (context, target) pairs
            use_scenarios: Whether to apply scenarios during validation

        Returns:
            Dictionary with validation metrics
        """
        logger.debug(f"Validating on {len(val_data)} windows")

        all_losses = []
        all_mae = []
        all_mape = []
        all_smape = []
        all_violations = []

        for context, target in val_data:
            # Generate scenario if requested
            scenario = None
            if use_scenarios:
                scenario = self.proposer.propose_scenario(
                    historical_context=context, forecast_horizon=len(target)
                )
                target = scenario.apply_to_timeseries(target)

            # Forecast
            forecast = self.solver.predict(context, scenario, return_quantiles=True)
            median_forecast = forecast["0.5"]

            # Compute metrics
            loss = self.solver.compute_forecast_loss(forecast, target)
            mae = mean_absolute_error(target, median_forecast)
            mape = mean_absolute_percentage_error(target, median_forecast)
            smape = mape  # Use MAPE as proxy for SMAPE

            # Check constraints
            verification_reward = self.verifier.evaluate(median_forecast, scenario)
            has_violation = verification_reward < -0.1

            all_losses.append(loss)
            all_mae.append(mae)
            all_mape.append(mape)
            all_smape.append(smape)
            all_violations.append(has_violation)

        return {
            "avg_loss": _safe_mean(all_losses, default=0.0),
            "mae": _safe_mean(all_mae, default=0.0),
            "mape": _safe_mean(all_mape, default=0.0),
            "smape": _safe_mean(all_smape, default=0.0),
            "violation_rate": _safe_mean(all_violations, default=0.0),
            "mae_std": np.std(all_mae) if all_mae else 0.0,
            "mape_std": np.std(all_mape) if all_mape else 0.0,
        }

    def _prepare_data_windows(
        self, loader: EnergyDataLoader, split: str
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Prepare data windows from dataset.

        Args:
            loader: Data loader instance
            split: Data split ("train", "validation", "test")

        Returns:
            List of (context, target) windows
        """
        logger.info(f"Preparing {split} data windows")

        if self.solver.use_samples:
            # Use sample data for fast testing
            windows_dict = loader.prepare_forecasting_windows(
                dataset="lcl", max_windows=200
            )
            windows = [(w["history_energy"], w["target_energy"]) for w in windows_dict]
        else:
            # Load full dataset
            data = loader.load_dataset("lcl", split=split)

            windows = []
            for household in data[:50]:  # Limit households for efficiency
                household_windows = create_sliding_windows(
                    data=household["energy"],
                    context_length=336,
                    forecast_horizon=48,
                    stride=48,  # Non-overlapping for diversity
                )

                # Limit windows per household
                max_windows = self.config.get("max_windows_per_household", 10)
                windows.extend(household_windows[:max_windows])

        logger.info(f"Prepared {len(windows)} {split} windows")
        return windows

    def _sample_historical_batch(
        self, data_windows: list[tuple[np.ndarray, np.ndarray]]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Sample a batch of historical windows for training.

        Args:
            data_windows: Available training windows

        Returns:
            Batch of (context, target) pairs
        """
        batch_size = min(self.config["batch_size"], len(data_windows))
        indices = np.random.choice(len(data_windows), batch_size, replace=False)
        return [data_windows[i] for i in indices]

    def _compute_success_rate(
        self,
        forecast: dict[str, np.ndarray],
        ground_truth: np.ndarray,
        threshold_mape: float = 20.0,
    ) -> float:
        """Estimate solver success rate for learnability reward.

        Success = MAPE < threshold (20% by default)

        Args:
            forecast: Quantile forecasts
            ground_truth: Actual values
            threshold_mape: MAPE threshold for success

        Returns:
            Success rate in [0, 1]
        """
        median_forecast = forecast["0.5"]
        mape = mean_absolute_percentage_error(ground_truth, median_forecast)

        # Sigmoid-like success rate based on MAPE
        if mape < threshold_mape * 0.5:
            return 1.0  # Very good
        elif mape < threshold_mape:
            return 0.5 + 0.5 * (1 - mape / threshold_mape)  # Good
        elif mape < threshold_mape * 2:
            return 0.5 * (1 - (mape - threshold_mape) / threshold_mape)  # Poor
        else:
            return 0.0  # Failed

    def _print_progress(self, episode: int, metrics: dict[str, Any]) -> None:
        """Print training progress to console.

        Args:
            episode: Current episode number
            metrics: Episode metrics
        """
        # Calculate moving averages
        window = 10
        recent_metrics = self.metrics_history[-window:]

        avg_loss = _safe_mean(
            [m["avg_solver_loss"] for m in recent_metrics], default=0.0
        )
        avg_reward = _safe_mean(
            [m["avg_verification_reward"] for m in recent_metrics], default=0.0
        )
        avg_mae = _safe_mean([m["avg_mae"] for m in recent_metrics], default=0.0)

        # Scenario distribution
        scenario_counts = {}
        for m in recent_metrics:
            for scenario in m["scenarios"]:
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

        total_scenarios = sum(scenario_counts.values())
        scenario_dist = {
            s: count / total_scenarios for s, count in scenario_counts.items()
        }

        logger.info(
            f"\nEpisode {episode} Summary (last {window} episodes):\n"
            f"  Avg Loss: {avg_loss:.4f}\n"
            f"  Avg Verification Reward: {avg_reward:.3f}\n"
            f"  Avg MAE: {avg_mae:.3f} kWh\n"
            f"  Scenario Distribution: {scenario_dist}\n"
            f"  Curriculum Level: {self.proposer.curriculum_level:.2f}\n"
            f"  Episode Time: {metrics['episode_time']:.1f}s"
        )

    def _save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save solver model
        solver_path = checkpoint_path.replace(".pth", "_solver.pth")
        self.solver.save_checkpoint(solver_path)

        # Save trainer state
        trainer_state = {
            "episode_count": self.episode_count,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "scenario_success_rates": self.scenario_success_rates,
            "proposer_stats": self.proposer.get_scenario_statistics(),
            "solver_stats": self.solver.get_training_summary(),
            "recent_metrics": self.metrics_history[-100:],  # Keep recent history
        }

        state_path = checkpoint_path.replace(".pth", "_trainer.json")
        with open(state_path, "w") as f:
            json.dump(trainer_state, f, indent=2, default=str)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _save_training_summary(self) -> None:
        """Save comprehensive training summary."""
        summary_path = os.path.join(self.checkpoint_dir, "training_summary.json")

        # Aggregate metrics across all episodes
        all_losses = [m["avg_solver_loss"] for m in self.metrics_history]
        all_rewards = [m["avg_verification_reward"] for m in self.metrics_history]
        all_mae = [m["avg_mae"] for m in self.metrics_history]

        # Scenario performance summary
        scenario_performance = {}
        for scenario_type, success_rates in self.scenario_success_rates.items():
            scenario_performance[scenario_type] = {
                "total_count": len(success_rates),
                "avg_success_rate": _safe_mean(success_rates, default=0.0),
                "std_success_rate": np.std(success_rates) if success_rates else 0.0,
            }

        summary = {
            "total_episodes": self.episode_count,
            "best_val_loss": self.best_val_loss,
            "final_metrics": {
                "avg_loss": _safe_mean(all_losses[-100:], default=0.0),
                "avg_reward": _safe_mean(all_rewards[-100:], default=0.0),
                "avg_mae": _safe_mean(all_mae[-100:], default=0.0),
            },
            "improvement": {
                "loss_reduction": (
                    ((all_losses[0] - all_losses[-1]) / all_losses[0] * 100)
                    if all_losses and all_losses[0] != 0
                    else 0.0
                ),
                "mae_reduction": (
                    ((all_mae[0] - all_mae[-1]) / all_mae[0] * 100)
                    if all_mae and all_mae[0] != 0
                    else 0.0
                ),
            },
            "scenario_performance": scenario_performance,
            "proposer_final_state": self.proposer.get_scenario_statistics(),
            "solver_final_state": self.solver.get_training_summary(),
            "training_time_hours": sum(m["episode_time"] for m in self.metrics_history)
            / 3600,
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Saved training summary to {summary_path}")

    def plot_training_curves(self, save_path: str | None = None) -> None:
        """Plot training curves for visualization.

        Args:
            save_path: Path to save plot (if None, displays instead)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
            return

        if not self.metrics_history:
            logger.warning("No metrics to plot")
            return

        episodes = [m["episode"] for m in self.metrics_history]
        losses = [m["avg_solver_loss"] for m in self.metrics_history]
        rewards = [m["avg_verification_reward"] for m in self.metrics_history]
        mae_values = [m["avg_mae"] for m in self.metrics_history]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss curve
        axes[0, 0].plot(episodes, losses, label="Solver Loss")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].grid(True)

        # Verification reward
        axes[0, 1].plot(episodes, rewards, label="Verification Reward", color="green")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].set_title("Verification Rewards")
        axes[0, 1].grid(True)

        # MAE curve
        axes[1, 0].plot(episodes, mae_values, label="MAE", color="orange")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("MAE (kWh)")
        axes[1, 0].set_title("Mean Absolute Error")
        axes[1, 0].grid(True)

        # Scenario distribution over time
        scenario_types = list({s for m in self.metrics_history for s in m["scenarios"]})
        scenario_counts = {st: [] for st in scenario_types}

        window = 10
        for i in range(0, len(self.metrics_history), window):
            window_metrics = self.metrics_history[i : i + window]
            counts = {st: 0 for st in scenario_types}
            total = 0

            for m in window_metrics:
                for s in m["scenarios"]:
                    counts[s] += 1
                    total += 1

            for st in scenario_types:
                scenario_counts[st].append(counts[st] / total if total > 0 else 0)

        x = list(range(0, len(self.metrics_history), window))
        bottom = np.zeros(len(x))

        colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_types)))
        for i, (st, counts) in enumerate(scenario_counts.items()):
            axes[1, 1].bar(
                x, counts, width=window * 0.8, bottom=bottom, label=st, color=colors[i]
            )
            bottom += np.array(counts)

        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Fraction")
        axes[1, 1].set_title("Scenario Distribution")
        axes[1, 1].legend(loc="upper right", fontsize=8)
        axes[1, 1].set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved training curves to {save_path}")
        else:
            plt.show()
