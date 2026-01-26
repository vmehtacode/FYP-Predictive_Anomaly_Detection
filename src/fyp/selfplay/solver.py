"""Solver agent for energy consumption forecasting in self-play training.

This module implements the forecasting component that learns from both historical
data and proposed scenarios, using the PatchTST architecture with quantile regression.
"""

import logging
import os
from typing import Any

import numpy as np

from fyp.data_loader import EnergyDataLoader
from fyp.selfplay.proposer import ScenarioProposal
from fyp.selfplay.utils import (
    calculate_pinball_loss,
    create_sliding_windows,
)

logger = logging.getLogger(__name__)

# Lazy import torch to avoid dependency issues during testing
try:
    import torch
except ImportError:
    torch = None
    logger.warning("PyTorch not available. Some solver functionality will be limited.")

# Lazy import PatchTST to avoid torch dependency
try:
    from fyp.models.patchtst import PatchTSTForecaster, create_patchtst_config
except ImportError:
    PatchTSTForecaster = None
    create_patchtst_config = None
    logger.warning("PatchTST not available. Solver will use fallback methods.")

# Import frequency-enhanced model for improved performance
try:
    from fyp.models.frequency_enhanced import (
        FrequencyEnhancedForecaster,
        create_frequency_enhanced_config,
    )
except ImportError:
    FrequencyEnhancedForecaster = None
    create_frequency_enhanced_config = None
    logger.debug("FrequencyEnhancedForecaster not available.")


class SolverAgent:
    """PatchTST-based forecasting agent for self-play training."""

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        historical_data_path: str | None = None,
        device: str = "cpu",
        pretrain_epochs: int = 20,
        use_samples: bool = False,
        use_frequency_enhanced: bool = True,  # NEW: Use frequency-enhanced model
    ):
        """Initialize solver with PatchTST architecture.

        Args:
            model_config: PatchTST hyperparameters (patch_len, d_model, etc.)
            historical_data_path: Path to LCL processed data
            device: Torch device
            pretrain_epochs: Supervised pretraining epochs on historical data
            use_samples: Whether to use sample data for fast testing
            use_frequency_enhanced: Whether to use FrequencyEnhancedPatchTST (default True)
        """
        self.use_frequency_enhanced = use_frequency_enhanced
        
        # Initialize model configuration
        if model_config is None:
            # Use frequency-enhanced config if enabled and available
            if use_frequency_enhanced and create_frequency_enhanced_config is not None:
                model_config = create_frequency_enhanced_config(use_samples=use_samples)
                logger.info("Using FrequencyEnhancedPatchTST configuration")
            elif create_patchtst_config is not None:
                model_config = create_patchtst_config(use_samples=use_samples)
            else:
                # Fallback config
                model_config = {
                    "patch_len": 16,
                    "d_model": 128,
                    "n_heads": 8,
                    "n_layers": 4,
                    "forecast_horizon": 48,
                    "quantiles": [0.1, 0.5, 0.9],
                    "learning_rate": 1e-3,
                    "max_epochs": 50,
                    "batch_size": 32,
                    "early_stopping_patience": 10,
                }

        self.model_config = model_config
        self.device = device
        self.use_samples = use_samples
        self.forecast_horizon = model_config.get("forecast_horizon", 48)

        # Initialize forecaster - prefer frequency-enhanced if available and enabled
        if use_frequency_enhanced and FrequencyEnhancedForecaster is not None:
            logger.info("Initializing FrequencyEnhancedForecaster for improved periodicity capture")
            self.model = FrequencyEnhancedForecaster(
                seq_len=model_config.get("seq_len", 96),
                patch_len=model_config.get("patch_len", 16),
                d_model=model_config.get("d_model", 128),
                n_heads=model_config.get("n_heads", 8),
                n_layers=model_config.get("n_layers", 4),
                forecast_horizon=self.forecast_horizon,
                quantiles=model_config.get("quantiles", [0.1, 0.5, 0.9]),
                learning_rate=model_config.get("learning_rate", 1e-3),
                max_epochs=model_config.get("max_epochs", 50),
                batch_size=model_config.get("batch_size", 32),
                early_stopping_patience=model_config.get("early_stopping_patience", 10),
                use_frequency_branch=model_config.get("use_frequency_branch", True),
                use_multiscale=model_config.get("use_multiscale", True),
                freq_weight=model_config.get("freq_weight", 0.3),
                device=device,
            )
        elif PatchTSTForecaster is not None:
            logger.info("Initializing standard PatchTSTForecaster")
            self.model = PatchTSTForecaster(
                patch_len=model_config.get("patch_len", 16),
                d_model=model_config.get("d_model", 128),
                n_heads=model_config.get("n_heads", 8),
                n_layers=model_config.get("n_layers", 4),
                forecast_horizon=self.forecast_horizon,
                quantiles=model_config.get("quantiles", [0.1, 0.5, 0.9]),
                learning_rate=model_config.get("learning_rate", 1e-3),
                max_epochs=model_config.get("max_epochs", 50),
                batch_size=model_config.get("batch_size", 32),
                early_stopping_patience=model_config.get("early_stopping_patience", 10),
                device=device,
            )
        else:
            self.model = None
            logger.warning(
                "No forecaster available. Using fallback prediction methods."
            )

        # Training state
        self.training_metrics = []
        self.scenario_adaptation_weights = {}
        self.is_pretrained = False

        # Pretrain on historical data if available
        if pretrain_epochs > 0 and historical_data_path:
            self._pretrain_on_historical_data(historical_data_path, pretrain_epochs)


    def _pretrain_on_historical_data(
        self, data_path: str, pretrain_epochs: int
    ) -> None:
        """Pretrain model on historical LCL data.

        Args:
            data_path: Path to processed LCL data
            pretrain_epochs: Number of pretraining epochs
        """
        logger.info(
            f"Pretraining solver on historical data for {pretrain_epochs} epochs"
        )

        try:
            # Load historical data
            loader = EnergyDataLoader(use_samples=self.use_samples)

            # Check if we're using the sample data or full data
            if self.use_samples:
                # Use sample data
                windows = loader.prepare_forecasting_windows(
                    dataset="lcl",
                    max_windows=100,  # Small sample for testing
                )
            else:
                # Load from processed data path
                train_data = loader.load_dataset("lcl", split="train")

                # Create training windows
                windows = []
                for household_data in train_data[
                    :100
                ]:  # Limit households for efficiency
                    household_windows = create_sliding_windows(
                        data=household_data["energy"],
                        context_length=336,  # 7 days
                        forecast_horizon=self.forecast_horizon,
                        stride=24,  # 12 hours
                    )

                    for context, target in household_windows[
                        :10
                    ]:  # Limit windows per household
                        windows.append(
                            {"history_energy": context, "target_energy": target}
                        )

                if len(windows) > 1000:  # Cap total windows
                    windows = windows[:1000]

            if not windows:
                logger.warning("No training windows found, skipping pretraining")
                return

            # Update model config for pretraining
            original_epochs = self.model.config["max_epochs"]
            self.model.config["max_epochs"] = pretrain_epochs

            # Train model
            train_metrics = self.model.fit(windows, validation_split=0.2)

            # Restore original epochs
            self.model.config["max_epochs"] = original_epochs

            self.is_pretrained = True
            logger.info(
                f"Pretraining complete. Final loss: {train_metrics['final_train_loss']:.4f}"
            )

        except Exception as e:
            logger.warning(f"Pretraining failed: {e}. Continuing without pretraining.")

    def predict(
        self,
        context_window: np.ndarray,
        scenario: ScenarioProposal | None = None,
        return_quantiles: bool = True,
    ) -> dict[str, np.ndarray]:
        """Forecast future consumption given context and optional scenario.

        Args:
            context_window: Historical data (336 intervals = 7 days)
            scenario: Optional proposed scenario to condition on
            return_quantiles: If True, return {0.1, 0.5, 0.9} quantiles

        Returns:
            Dict mapping quantile strings to forecasts
            Example: {"0.1": [...], "0.5": [...], "0.9": [...]}
        """
        if self.model is None or self.model.model is None:
            # Model not available or not trained yet, return naive forecast
            logger.warning("Model not available/trained, returning naive forecast")

            # Calculate baseline with validation
            baseline = np.median(context_window)
            if np.isnan(baseline) or baseline == 0:
                # Fallback to mean if median is invalid
                baseline = np.nanmean(context_window)
                if np.isnan(baseline) or baseline == 0:
                    # Ultimate fallback to a reasonable default (1 kWh)
                    baseline = 1.0
                    logger.warning("Context window invalid, using default baseline")

            forecast = np.full(self.forecast_horizon, baseline)

            # Apply scenario transformation to the baseline if provided
            if scenario is not None:
                forecast = scenario.apply_to_timeseries(forecast.copy())

            if return_quantiles:
                return {"0.1": forecast * 0.9, "0.5": forecast, "0.9": forecast * 1.1}
            else:
                return {"point": forecast}

        # Apply scenario transformation if provided
        if scenario is not None:
            # Create synthetic future to show model what scenario looks like
            # This helps the model understand the pattern
            scenario_hint = self._create_scenario_hint(context_window, scenario)
            modified_context = np.concatenate(
                [context_window[: -len(scenario_hint)], scenario_hint]
            )
        else:
            modified_context = context_window

        # Use PatchTST predict method
        try:
            predictions = self.model.predict(
                history=modified_context,
                steps=self.forecast_horizon,
                return_quantiles=return_quantiles,
            )

            # Apply scenario to predictions if needed
            if scenario is not None:
                predictions = self._apply_scenario_to_predictions(
                    predictions, scenario, context_window
                )

            return predictions

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return fallback forecast with validation
            baseline = np.median(context_window)
            if np.isnan(baseline) or baseline == 0:
                baseline = np.nanmean(context_window)
                if np.isnan(baseline) or baseline == 0:
                    baseline = 1.0
                    logger.warning("Context invalid in fallback, using default")

            forecast = np.full(self.forecast_horizon, baseline)

            # Apply scenario transformation to the baseline if provided
            if scenario is not None:
                forecast = scenario.apply_to_timeseries(forecast.copy())

            if return_quantiles:
                return {"0.1": forecast * 0.9, "0.5": forecast, "0.9": forecast * 1.1}
            else:
                return {"point": forecast}

    def _create_scenario_hint(
        self, context: np.ndarray, scenario: ScenarioProposal
    ) -> np.ndarray:
        """Create a hint for the model about the upcoming scenario.

        Args:
            context: Historical context
            scenario: Proposed scenario

        Returns:
            Hint array to append to context
        """
        # Create a small synthetic pattern that hints at the scenario
        hint_length = min(24, len(context) // 4)  # Up to 12 hours
        baseline = np.mean(context[-48:])  # Recent average

        if scenario.scenario_type == "EV_SPIKE":
            # Show gradual increase
            hint = np.linspace(baseline, baseline * 1.2, hint_length)
        elif scenario.scenario_type == "COLD_SNAP":
            # Show temperature-driven increase
            hint = baseline * np.linspace(1.0, scenario.magnitude * 0.7, hint_length)
        elif scenario.scenario_type == "OUTAGE":
            # Show sudden drop
            hint = np.concatenate(
                [
                    np.full(hint_length // 2, baseline),
                    np.full(hint_length // 2, baseline * 0.1),
                ]
            )
        else:
            # Default: slight variation
            hint = baseline * (1 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, hint_length)))

        return hint

    def _apply_scenario_to_predictions(
        self,
        predictions: dict[str, np.ndarray],
        scenario: ScenarioProposal,
        original_context: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Apply scenario transformation to predictions.

        Args:
            predictions: Quantile predictions
            scenario: Scenario to apply
            original_context: Original context without modifications

        Returns:
            Transformed predictions
        """
        transformed = {}

        for quantile, forecast in predictions.items():
            # Apply scenario transformation
            transformed_forecast = scenario.apply_to_timeseries(forecast)

            # Ensure physical plausibility
            transformed_forecast = np.maximum(transformed_forecast, 0.0)

            # Adapt based on scenario type
            if scenario.scenario_type == "EV_SPIKE" and quantile == "0.9":
                # Higher uncertainty for upper quantile during EV charging
                transformed_forecast *= 1.1
            elif scenario.scenario_type == "OUTAGE" and quantile == "0.1":
                # Lower bound should be near zero during outage
                outage_mask = transformed_forecast < 0.1
                transformed_forecast[outage_mask] = 0.0

            transformed[quantile] = transformed_forecast

        return transformed

    def compute_forecast_loss(
        self,
        forecast: dict[str, np.ndarray],
        ground_truth: np.ndarray,
        quantiles: list[float] | None = None,
    ) -> float:
        """Compute pinball loss for quantile forecasts.

        Following quantile regression:
        L(q) = sum_t max(q(y_t - f_q(t)), (q-1)(y_t - f_q(t)))

        Args:
            forecast: Dict of quantile forecasts
            ground_truth: Actual values
            quantiles: List of quantiles (default [0.1, 0.5, 0.9])

        Returns:
            Average pinball loss across quantiles
        """
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        total_loss = 0.0
        for q in quantiles:
            q_str = str(q)
            if q_str not in forecast:
                logger.warning(f"Quantile {q} not found in forecast")
                continue

            q_forecast = forecast[q_str]
            loss = calculate_pinball_loss(ground_truth, q_forecast, q)
            total_loss += loss

        return total_loss / len(quantiles)

    def train_step(
        self,
        context: np.ndarray,
        target: np.ndarray,
        scenario: ScenarioProposal | None = None,
        verification_reward: float = 0.0,
        alpha: float = 0.1,
    ) -> float:
        """Single training step combining forecast loss and verification reward.

        Total loss = forecast_loss + alpha * verification_penalty

        Args:
            context: Input time series
            target: Ground truth forecast
            scenario: Proposed scenario (if in self-play mode)
            verification_reward: Reward from verifier (-1 to +1)
            alpha: Weight for verification reward (default 0.1)

        Returns:
            Combined loss value
        """
        # Create training window
        window = {"history_energy": context, "target_energy": target}

        # If scenario provided, create augmented data
        if scenario is not None:
            # Apply scenario to target
            augmented_target = scenario.apply_to_timeseries(target)

            # Create scenario-aware window
            window["target_energy"] = augmented_target

            # Store scenario metadata for adaptation
            scenario_key = f"{scenario.scenario_type}_{scenario.magnitude:.1f}"
            if scenario_key not in self.scenario_adaptation_weights:
                self.scenario_adaptation_weights[scenario_key] = 1.0

            # Update adaptation weight based on verification reward
            # Good verification -> increase weight, bad -> decrease
            self.scenario_adaptation_weights[scenario_key] *= (
                1.0 + verification_reward * 0.1
            )
            self.scenario_adaptation_weights[scenario_key] = np.clip(
                self.scenario_adaptation_weights[scenario_key], 0.1, 2.0
            )

        # Prepare mini-batch for training
        windows = [window]

        # Add some historical windows for stability
        if hasattr(self, "_historical_buffer") and len(self._historical_buffer) > 0:
            # Sample some historical windows
            n_historical = min(3, len(self._historical_buffer))
            historical_indices = np.random.choice(
                len(self._historical_buffer), n_historical, replace=False
            )
            for idx in historical_indices:
                windows.append(self._historical_buffer[idx])

        # Check if model is available
        if self.model is None:
            logger.warning("Model not available, skipping training step")
            # Still track metrics even in fallback mode
            self.training_metrics.append(
                {
                    "forecast_loss": 1.0,
                    "verification_penalty": -verification_reward,
                    "combined_loss": 1.0,
                    "scenario_type": scenario.scenario_type if scenario else "baseline",
                }
            )
            return 1.0  # Return default loss

        # Train for one step
        original_epochs = self.model.config["max_epochs"]
        self.model.config["max_epochs"] = 1  # Single epoch for online learning

        try:
            # Fit model on mini-batch
            train_metrics = self.model.fit(windows, validation_split=0.0)
            forecast_loss = train_metrics["final_train_loss"]

            # Combine with verification penalty
            verification_penalty = -verification_reward  # Convert reward to penalty
            combined_loss = forecast_loss + alpha * verification_penalty

            # Store metrics
            self.training_metrics.append(
                {
                    "forecast_loss": forecast_loss,
                    "verification_penalty": verification_penalty,
                    "combined_loss": combined_loss,
                    "scenario_type": scenario.scenario_type if scenario else "baseline",
                }
            )

        except Exception as e:
            logger.error(f"Training step failed: {e}")
            combined_loss = 1.0  # Default high loss on failure

        finally:
            # Restore original epochs
            self.model.config["max_epochs"] = original_epochs

        return combined_loss

    def update_historical_buffer(self, windows: list[dict[str, np.ndarray]]) -> None:
        """Update internal buffer of historical windows for training stability.

        Args:
            windows: List of training windows with history and target
        """
        if not hasattr(self, "_historical_buffer"):
            self._historical_buffer = []

        self._historical_buffer.extend(windows)

        # Keep only recent windows
        max_buffer_size = 100
        if len(self._historical_buffer) > max_buffer_size:
            self._historical_buffer = self._historical_buffer[-max_buffer_size:]

    def get_training_summary(self) -> dict[str, Any]:
        """Get summary of training metrics.

        Returns:
            Dictionary with training statistics
        """
        if not self.training_metrics:
            return {
                "total_steps": 0,
                "avg_forecast_loss": 0.0,
                "avg_verification_penalty": 0.0,
                "avg_combined_loss": 0.0,
                "scenario_breakdown": {},
            }

        # Aggregate metrics
        forecast_losses = [m["forecast_loss"] for m in self.training_metrics]
        verification_penalties = [
            m["verification_penalty"] for m in self.training_metrics
        ]
        combined_losses = [m["combined_loss"] for m in self.training_metrics]

        # Scenario breakdown
        scenario_losses = {}
        for metric in self.training_metrics:
            st = metric["scenario_type"]
            if st not in scenario_losses:
                scenario_losses[st] = []
            scenario_losses[st].append(metric["combined_loss"])

        scenario_breakdown = {
            st: {
                "count": len(losses),
                "avg_loss": np.mean(losses),
                "std_loss": np.std(losses),
            }
            for st, losses in scenario_losses.items()
        }

        return {
            "total_steps": len(self.training_metrics),
            "avg_forecast_loss": np.mean(forecast_losses),
            "avg_verification_penalty": np.mean(verification_penalties),
            "avg_combined_loss": np.mean(combined_losses),
            "scenario_breakdown": scenario_breakdown,
            "adaptation_weights": dict(self.scenario_adaptation_weights),
            "is_pretrained": self.is_pretrained,
        }

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        if self.model is None or self.model.model is None:
            logger.warning("No model to save")
            # Save basic state without model
            checkpoint = {
                "model_config": self.model_config,
                "training_metrics": self.training_metrics,
                "scenario_adaptation_weights": self.scenario_adaptation_weights,
                "scaler_mean": None,
                "scaler_std": None,
                "is_pretrained": self.is_pretrained,
            }
        else:
            checkpoint = {
                "model_state_dict": self.model.model.state_dict(),
                "model_config": self.model_config,
                "training_metrics": self.training_metrics,
                "scenario_adaptation_weights": self.scenario_adaptation_weights,
                "scaler_mean": self.model.scaler_mean,
                "scaler_std": self.model.scaler_std,
                "is_pretrained": self.is_pretrained,
            }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        if torch is not None:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        else:
            # Fallback: save as JSON (excluding model state dict)
            import json

            checkpoint_json = {
                "model_config": self.model_config,
                "training_metrics": self.training_metrics,
                "scenario_adaptation_weights": self.scenario_adaptation_weights,
                "scaler_mean": self.model.scaler_mean if self.model else None,
                "scaler_std": self.model.scaler_std if self.model else None,
                "is_pretrained": self.is_pretrained,
            }
            json_path = checkpoint_path.replace(".pth", ".json")
            with open(json_path, "w") as f:
                json.dump(checkpoint_json, f, indent=2)
            logger.info(f"Saved checkpoint (JSON) to {json_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to load checkpoint from
        """
        if not os.path.exists(checkpoint_path):
            # Try JSON fallback
            json_path = checkpoint_path.replace(".pth", ".json")
            if os.path.exists(json_path):
                checkpoint_path = json_path
            else:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if checkpoint_path.endswith(".json"):
            # Load JSON checkpoint
            import json

            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
        else:
            if torch is None:
                raise ImportError("PyTorch required to load .pth checkpoint")
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )

        # Recreate model with saved config
        self.model_config = checkpoint["model_config"]

        if PatchTSTForecaster is not None and "model_state_dict" in checkpoint:
            self.model = PatchTSTForecaster(**self.model_config, device=self.device)
            # Load model state if model was successfully initialized
            if self.model.model is not None:
                self.model.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.scaler_mean = checkpoint["scaler_mean"]
                self.model.scaler_std = checkpoint["scaler_std"]
            else:
                logger.warning(
                    "PatchTSTForecaster model not initialized, skipping state dict load"
                )
        else:
            self.model = None

        # Load training state
        self.training_metrics = checkpoint["training_metrics"]
        self.scenario_adaptation_weights = checkpoint["scenario_adaptation_weights"]
        self.is_pretrained = checkpoint["is_pretrained"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
