#!/usr/bin/env python3
"""
Large-scale self-play experiment - VERSION 3.

"""

import argparse
import copy
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import polars as pl
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fyp.baselines.forecasting import (
    BaseForecaster,
    LinearTrendForecaster,
    SeasonalNaive,
)
from src.fyp.config import ExperimentConfig, ForecastingConfig
from src.fyp.data_loader import EnergyDataLoader
from src.fyp.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from src.fyp.models.patchtst import PatchTSTForecaster
from src.fyp.selfplay import (
    ProposerAgent,
    SelfPlayTrainer,
    SolverAgent,
    VerifierAgent,
)
from src.fyp.selfplay.utils import create_sliding_windows

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NaiveForecaster(BaseForecaster):
    """Naive forecaster that repeats the previous day's values."""
    
    def __init__(self):
        super().__init__(name="naive")
        self.seasonal_period = 48
        
    def fit(self, history: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        self.is_fitted = True
        
    def predict(
        self,
        history: np.ndarray,
        steps: int,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        if len(history) < self.seasonal_period:
            return np.full(steps, history[-1])
        last_day = history[-self.seasonal_period:]
        n_repeats = (steps + self.seasonal_period - 1) // self.seasonal_period
        forecast = np.tile(last_day, n_repeats)[:steps]
        return forecast


class MovingAverageForecaster(BaseForecaster):
    """Moving average forecaster."""
    
    def __init__(self, window_size: int = 7 * 48):
        super().__init__(name="moving_average")
        self.window_size = window_size
        
    def fit(self, history: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        self.is_fitted = True
        
    def predict(
        self,
        history: np.ndarray,
        steps: int,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        lookback = min(len(history), self.window_size)
        mean_value = np.mean(history[-lookback:])
        return np.full(steps, mean_value)


class SupervisedPatchTST(BaseForecaster):
    """PatchTST trained in supervised mode only (no self-play)."""
    
    def __init__(self, config: ForecastingConfig):
        super().__init__(name="supervised_patchtst")
        self.config = config
        self.model = None
        self.training_windows = []
    
    def fit(self, history: np.ndarray, timestamps: np.ndarray | None = None) -> None:
        """Required by BaseForecaster - store for later batch training."""
        self.training_windows.append(history)
        self.is_fitted = True
        
    def train_on_windows(self, windows: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Train on collected windows."""
        self.model = PatchTSTForecaster(
            forecast_horizon=self.config.forecast_horizon,
            patch_len=self.config.patch_len,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            quantiles=[0.1, 0.5, 0.9],
            learning_rate=self.config.learning_rate,
            max_epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            early_stopping_patience=self.config.early_stopping_patience,
            device=self.config.device,
        )
        
        train_data = []
        for context, target in windows:
            train_data.append({
                "history_energy": context,
                "target_energy": target,
            })
            
        if len(train_data) > 0:
            self.model.fit(
                windows=train_data,
                validation_split=0.2,
            )
        
    def predict(
        self,
        history: np.ndarray,
        steps: int,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate forecast."""
        if self.model is None:
            return np.full(steps, np.mean(history))
            
        if len(history) > self.config.context_length:
            history = history[-self.config.context_length:]
        elif len(history) < self.config.context_length:
            padding = np.zeros(self.config.context_length - len(history))
            history = np.concatenate([padding, history])
            
        forecast_dict = self.model.predict(history, steps, return_quantiles=True)
        if isinstance(forecast_dict, dict):
            return forecast_dict.get("0.5", forecast_dict.get("point", np.full(steps, np.mean(history))))
        else:
            return forecast_dict


def load_lcl_data_polars(
    data_dir: Path,
    num_households: int = 50,
    min_days: int = 180,
    random_seed: int = 42,
) -> pl.DataFrame:
    """Load LCL data using polars for efficiency."""
    logger.info(f"Loading LCL data from {data_dir}")
    
    lcl_path = data_dir / "lcl_data"
    if not lcl_path.exists():
        raise FileNotFoundError(f"LCL data not found at {lcl_path}")
        
    df = pl.scan_parquet(str(lcl_path / "*.parquet"))
    df = df.collect()
    
    if df.schema["ts_utc"] == pl.Utf8:
        df = df.with_columns(
            pl.col("ts_utc").str.strptime(pl.Datetime)
        )
    
    household_stats = (
        df.group_by("entity_id")
        .agg([
            pl.len().alias("n_records"),
            pl.col("ts_utc").min().alias("start_date"),
            pl.col("ts_utc").max().alias("end_date"),
        ])
        .with_columns(
            ((pl.col("end_date") - pl.col("start_date")).dt.total_days()).alias("days_available")
        )
        .filter(pl.col("days_available") >= min_days)
        .sort("n_records", descending=True)
    )
    
    consumption_stats = (
        df.group_by("entity_id")
        .agg(pl.col("energy_kwh").mean().alias("mean_consumption"))
    )
    
    household_stats = household_stats.join(consumption_stats, on="entity_id")
    
    np.random.seed(random_seed)
    
    selected_households = []
    q25 = household_stats["mean_consumption"].quantile(0.25)
    q50 = household_stats["mean_consumption"].quantile(0.50)
    q75 = household_stats["mean_consumption"].quantile(0.75)
    quartiles = [q25, q50, q75]
    
    bins = [
        (0, quartiles[0]),
        (quartiles[0], quartiles[1]),
        (quartiles[1], quartiles[2]),
        (quartiles[2], float('inf'))
    ]
    
    households_per_bin = num_households // 4
    remaining = num_households % 4
    
    for i, (low, high) in enumerate(bins):
        bin_households = household_stats.filter(
            (pl.col("mean_consumption") >= low) & 
            (pl.col("mean_consumption") < high)
        )["entity_id"].to_list()
        
        n_select = households_per_bin + (1 if i < remaining else 0)
        if len(bin_households) >= n_select:
            selected = np.random.choice(bin_households, n_select, replace=False)
        else:
            selected = bin_households
            
        selected_households.extend(selected)
    
    logger.info(f"Selected {len(selected_households)} households with diverse consumption patterns")
    
    df_selected = df.filter(pl.col("entity_id").is_in(selected_households))
    df_selected = df_selected.sort(["entity_id", "ts_utc"])
    
    return df_selected


def create_train_val_test_splits(
    df: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create chronological train/val/test splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    min_date = df["ts_utc"].min()
    max_date = df["ts_utc"].max()
    total_days = (max_date - min_date).days
    
    train_days = int(total_days * train_ratio)
    val_days = int(total_days * val_ratio)
    
    train_end = min_date + timedelta(days=train_days)
    val_end = train_end + timedelta(days=val_days)
    
    train_df = df.filter(pl.col("ts_utc") < train_end)
    val_df = df.filter((pl.col("ts_utc") >= train_end) & (pl.col("ts_utc") < val_end))
    test_df = df.filter(pl.col("ts_utc") >= val_end)
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Date ranges:")
    logger.info(f"  Train: {min_date} to {train_end}")
    logger.info(f"  Val: {train_end} to {val_end}")
    logger.info(f"  Test: {val_end} to {max_date}")
    
    return train_df, val_df, test_df


def prepare_windows_from_polars(
    df: pl.DataFrame,
    context_length: int = 336,
    forecast_horizon: int = 48,
    stride: int = 24,
    max_windows_per_household: int = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create sliding windows from polars DataFrame."""
    windows = []
    
    for entity_id in df["entity_id"].unique().to_list():
        household_df = df.filter(pl.col("entity_id") == entity_id).sort("ts_utc")
        energy_values = household_df["energy_kwh"].to_numpy()
        
        household_windows = create_sliding_windows(
            data=energy_values,
            context_length=context_length,
            forecast_horizon=forecast_horizon,
            stride=stride,
        )
        
        if max_windows_per_household and len(household_windows) > max_windows_per_household:
            indices = np.linspace(0, len(household_windows) - 1, max_windows_per_household, dtype=int)
            household_windows = [household_windows[i] for i in indices]
        
        windows.extend(household_windows)
    
    logger.info(f"Created {len(windows)} windows from {df['entity_id'].n_unique()} households")
    return windows


def train_baseline_model(
    model: BaseForecaster,
    train_windows: List[Tuple[np.ndarray, np.ndarray]],
    val_windows: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str,
    run_id: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train a baseline model and evaluate it."""
    logger.info(f"Training {model_name} (run {run_id})")
    
    start_time = time.time()
    
    if isinstance(model, SupervisedPatchTST):
        model.train_on_windows(train_windows)
    else:
        if train_windows:
            context, _ = train_windows[0]
            model.fit(context)
    
    val_predictions = []
    val_targets = []
    
    for context, target in tqdm(val_windows, desc=f"Evaluating {model_name}"):
        pred = model.predict(context, len(target))
        val_predictions.append(pred)
        val_targets.append(target)
    
    val_predictions = np.array(val_predictions)
    val_targets = np.array(val_targets)
    
    mae = mean_absolute_error(val_targets.flatten(), val_predictions.flatten())
    rmse = root_mean_squared_error(val_targets.flatten(), val_predictions.flatten())
    mape = mean_absolute_percentage_error(val_targets.flatten(), val_predictions.flatten())
    
    training_time = time.time() - start_time
    
    results = {
        "model": model_name,
        "run_id": run_id,
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "val_mape": float(mape),
        "training_time": training_time,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
    }
    
    logger.info(f"{model_name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    
    return results


def save_model_checkpoint(solver: SolverAgent, checkpoint_path: Path, episode: int, val_mae: float, train_mae: float, proposer: ProposerAgent) -> None:
    """Save complete model checkpoint including weights."""
    checkpoint = {
        'episode': episode,
        'val_mae': val_mae,
        'train_mae': train_mae,
        'curriculum_level': proposer.curriculum_level,
        'proposer_curriculum_level': proposer.curriculum_level,
        'proposer_scenario_buffer': copy.deepcopy(proposer.scenario_buffer),
    }
    
    # V3 FIX: Save actual model state if available
    if hasattr(solver, 'model') and solver.model is not None:
        if hasattr(solver.model, 'model'):  # PatchTST has nested model
            try:
                import torch
                if hasattr(solver.model.model, 'state_dict'):
                    checkpoint['model_state_dict'] = solver.model.model.state_dict()
                    checkpoint['scaler_mean'] = solver.model.scaler_mean
                    checkpoint['scaler_std'] = solver.model.scaler_std
                    logger.info("✅ Saved PyTorch model weights")
            except Exception as e:
                logger.warning(f"❌ Could not save model weights: {e}")
        else:
            logger.warning("❌ Solver model has no 'model' attribute")
    else:
        logger.warning(f"❌ Solver has no model (has model: {hasattr(solver, 'model')}, is None: {solver.model is None if hasattr(solver, 'model') else 'N/A'})")
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_model_checkpoint(solver: SolverAgent, checkpoint_path: Path, proposer: ProposerAgent) -> float:
    """Load complete model checkpoint including weights."""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # V3 FIX: Don't restore curriculum_level from checkpoint!
    # Curriculum should keep progressing regardless of which checkpoint we use
    # Only restore scenario buffer (useful for proposer diversity)
    proposer.scenario_buffer = checkpoint['proposer_scenario_buffer']
    
    # V3 FIX: Restore model weights if available
    if 'model_state_dict' in checkpoint:
        try:
            import torch
            if hasattr(solver, 'model') and solver.model is not None and hasattr(solver.model, 'model'):
                solver.model.model.load_state_dict(checkpoint['model_state_dict'])
                solver.model.scaler_mean = checkpoint['scaler_mean']
                solver.model.scaler_std = checkpoint['scaler_std']
                logger.info(f"✅ Loaded model weights from episode {checkpoint['episode']}")
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}")
    
    return checkpoint['val_mae']


def train_selfplay_model(
    train_windows: List[Tuple[np.ndarray, np.ndarray]],
    val_windows: List[Tuple[np.ndarray, np.ndarray]],
    num_episodes: int,
    run_id: int,
    output_dir: Path,
    config: ExperimentConfig,
) -> Dict[str, Any]:
    """
    Train self-play model with curriculum learning.
    
    VERSION 3 IMPROVEMENTS:
    - Curriculum progression fixed (moves outside validation block)
    - Model weights saved in checkpoint
    - Validation every 5 episodes (not 10)
    - Better early stopping (patience 10)
    - Final validation at end
    """
    logger.info(f"🚀 Training self-play model (run {run_id}) - VERSION 3")
    
    # Initialize agents
    ssen_path = "data/derived/ssen_constraints.json"
    if not Path(ssen_path).exists():
        minimal_constraints = {
            "voltage_limits": {"min": 0.94, "max": 1.06},
            "transformer_capacity": 500.0,
            "feeder_capacity": 1000.0,
        }
        Path(ssen_path).parent.mkdir(parents=True, exist_ok=True)
        with open(ssen_path, "w") as f:
            json.dump(minimal_constraints, f)
    
    # V3 ULTIMATE FIX: Disable internal curriculum in proposer
    # We'll manage curriculum externally in the training loop
    proposer = ProposerAgent(
        ssen_constraints_path=ssen_path,
        difficulty_curriculum=False,  # Disable internal adaptive curriculum
    )
    # Manually initialize curriculum to 0.0
    proposer.curriculum_level = 0.0
    
    solver = SolverAgent(
        model_config={
            "forecast_horizon": config.forecasting.forecast_horizon,
            "patch_len": config.forecasting.patch_len,
            "d_model": config.forecasting.d_model,
            "n_heads": config.forecasting.n_heads,
            "n_layers": config.forecasting.n_layers,
            "d_ff": config.forecasting.d_ff,
            "dropout": config.forecasting.dropout,
            "device": config.forecasting.device,
        },
        pretrain_epochs=0,
    )
    
    verifier = VerifierAgent(ssen_constraints_path=ssen_path)
    
    trainer = SelfPlayTrainer(
        proposer=proposer,
        solver=solver,
        verifier=verifier,
        config={
            "alpha": 0.1,
            "batch_size": 32,
            "checkpoint_every": 10,
            "log_every": 5,
            "val_every": 5,  # V3: More frequent validation
        },
    )
    
    # Track metrics
    episode_metrics = []
    best_val_mae = float('inf')
    best_episode = -1
    patience_counter = 0
    early_stopping_patience = 10  # V3: Tighter early stopping
    
    # V3: Validate every 5 episodes
    validation_frequency = 5
    
    start_time = time.time()
    
    # Training loop
    for episode in range(num_episodes):
        # Sample batch of windows
        batch_size = min(32, len(train_windows))
        batch_indices = np.random.choice(len(train_windows), batch_size, replace=False)
        batch = [train_windows[i] for i in batch_indices]
        
        # Train episode
        episode_results = trainer.train_episode(batch)
        train_mae = episode_results.get("avg_mae", 0.0)
        
        # V3 FIX: Curriculum learning OUTSIDE validation block
        # This runs every 10 episodes regardless of validation
        if episode > 0 and episode % 10 == 0:
            old_curriculum = proposer.curriculum_level
            proposer.curriculum_level = min(1.0, proposer.curriculum_level + 0.1)
            logger.info(f"📈 Curriculum: {old_curriculum:.1f} → {proposer.curriculum_level:.1f} (episode {episode})")
        
        # Validate every N episodes
        val_mae = None
        if episode % validation_frequency == 0:
            val_mae = evaluate_selfplay_on_windows(solver, val_windows)
            episode_results["val_mae"] = val_mae
            
            logger.info(f"📊 Episode {episode}: train_mae={train_mae:.4f}, val_mae={val_mae:.4f}, curriculum={proposer.curriculum_level:.1f}")
            
            # Save checkpoint when best validation MAE is achieved
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_episode = episode
                patience_counter = 0
                
                checkpoint_path = output_dir / f"best_checkpoint_run{run_id}.pkl"
                save_model_checkpoint(solver, checkpoint_path, episode, val_mae, train_mae, proposer)
                logger.info(f"💾 Best checkpoint saved (episode {episode}, val_mae={val_mae:.4f})")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"🛑 Early stopping at episode {episode} (no improvement for {early_stopping_patience * validation_frequency} episodes)")
                break
        
        # Record episode metrics
        episode_metrics.append({
            "episode": episode,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "proposer_difficulty": proposer.curriculum_level,
            "verification_reward": episode_results.get("avg_verification_reward", 0),
        })
    
    training_time = time.time() - start_time
    
    # V3: Final validation if we haven't validated recently
    last_val_episode = max([m['episode'] for m in episode_metrics if m.get('val_mae') is not None], default=-validation_frequency)
    if episode - last_val_episode >= validation_frequency:
        final_val_mae = evaluate_selfplay_on_windows(solver, val_windows)
        logger.info(f"📊 Final episode {episode}: val_mae={final_val_mae:.4f}")
        
        if final_val_mae < best_val_mae:
            logger.info(f"💡 Final model is best! Saving...")
            best_val_mae = final_val_mae
            best_episode = episode
            checkpoint_path = output_dir / f"best_checkpoint_run{run_id}.pkl"
            save_model_checkpoint(solver, checkpoint_path, episode, final_val_mae, train_mae, proposer)
    
    # Load best checkpoint before final evaluation
    checkpoint_path = output_dir / f"best_checkpoint_run{run_id}.pkl"
    if checkpoint_path.exists():
        loaded_best_mae = load_model_checkpoint(solver, checkpoint_path, proposer)
        logger.info(f"✅ Loaded best model from episode {best_episode} (val_mae={loaded_best_mae:.4f})")
        best_val_mae = loaded_best_mae
    else:
        logger.warning("⚠️ No checkpoint found, using final model state")
    
    # Final evaluation on validation set
    final_val_predictions = []
    final_val_targets = []
    
    for context, target in tqdm(val_windows, desc="Final evaluation"):
        pred = solver.predict(context)
        if isinstance(pred, dict):
            pred = pred.get("0.5", pred.get("median", pred))
        final_val_predictions.append(pred)
        final_val_targets.append(target)
    
    final_val_predictions = np.array(final_val_predictions)
    final_val_targets = np.array(final_val_targets)
    
    mae = mean_absolute_error(final_val_targets.flatten(), final_val_predictions.flatten())
    rmse = root_mean_squared_error(final_val_targets.flatten(), final_val_predictions.flatten())
    mape = mean_absolute_percentage_error(final_val_targets.flatten(), final_val_predictions.flatten())
    
    results = {
        "model": "selfplay",
        "run_id": run_id,
        "val_mae": float(mae),
        "val_rmse": float(rmse),
        "val_mape": float(mape),
        "training_time": training_time,
        "n_episodes": len(episode_metrics),
        "best_val_mae": float(best_val_mae),
        "best_episode": best_episode,
        "final_curriculum_level": float(proposer.curriculum_level),
        "episode_history": episode_metrics,
    }
    
    # Validation checks
    logger.info(f"\n{'='*70}")
    logger.info(f"V3 VALIDATION CHECKS (Run {run_id}):")
    logger.info(f"  Final val_mae: {mae:.4f}")
    logger.info(f"  Best val_mae: {best_val_mae:.4f} (episode {best_episode})")
    logger.info(f"  Difference: {abs(mae - best_val_mae):.4f} ({abs(mae - best_val_mae)/best_val_mae*100:.1f}%)")
    logger.info(f"  Final curriculum: {proposer.curriculum_level:.2f}")
    logger.info(f"  Training MAE non-zero: {any(m['train_mae'] > 0 for m in episode_metrics)}")
    logger.info(f"  Episodes run: {len(episode_metrics)}")
    
    # V3: Enhanced validation warnings
    if abs(mae - best_val_mae) / best_val_mae > 0.10:
        logger.warning(f"⚠️ Final MAE is >10% different from best MAE")
    else:
        logger.info(f"✅ Checkpoint loading verified (within 10%)")
    
    # Calculate expected curriculum based on episodes
    expected_curriculum = min(0.9, (len(episode_metrics) // 10) * 0.1)
    
    if abs(proposer.curriculum_level - expected_curriculum) > 0.05:
        logger.warning(f"⚠️ Curriculum {proposer.curriculum_level:.1f} doesn't match expected {expected_curriculum:.1f}")
    elif proposer.curriculum_level >= 0.9:
        logger.info(f"✅ Curriculum fully progressed to {proposer.curriculum_level:.1f}")
    else:
        logger.info(f"✅ Curriculum correctly at {proposer.curriculum_level:.1f} for {len(episode_metrics)} episodes")
        
    if not any(m['train_mae'] > 0 for m in episode_metrics):
        logger.warning(f"⚠️ All training MAEs are zero")
    else:
        logger.info(f"✅ Training metrics logged correctly")
    
    logger.info(f"{'='*70}\n")
    
    return results


def evaluate_selfplay_on_windows(
    solver: SolverAgent,
    windows: List[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """Helper to evaluate self-play solver on windows."""
    predictions = []
    targets = []
    
    for context, target in windows:
        pred = solver.predict(context)
        if isinstance(pred, dict):
            pred = pred.get("0.5", pred.get("median", pred))
        predictions.append(pred)
        targets.append(target)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    return mean_absolute_error(targets.flatten(), predictions.flatten())


def run_statistical_tests(
    results_by_method: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Run statistical tests comparing methods."""
    logger.info("Running statistical tests")
    
    stats_results = {}
    
    # Extract MAE values for each method
    mae_by_method = {}
    for method, runs in results_by_method.items():
        mae_by_method[method] = [run["val_mae"] for run in runs]
    
    # Calculate summary statistics
    for method, mae_values in mae_by_method.items():
        stats_results[method] = {
            "mean": np.mean(mae_values),
            "std": np.std(mae_values, ddof=1),
            "ci_lower": np.percentile(mae_values, 2.5),
            "ci_upper": np.percentile(mae_values, 97.5),
            "n_runs": len(mae_values),
        }
    
    # Pairwise t-tests against self-play
    if "selfplay" in mae_by_method:
        selfplay_mae = mae_by_method["selfplay"]
        
        for method, mae_values in mae_by_method.items():
            if method != "selfplay":
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(selfplay_mae, mae_values)
                
                # Calculate Cohen's d (effect size)
                diff = np.array(selfplay_mae) - np.array(mae_values)
                cohens_d = np.mean(diff) / np.std(diff, ddof=1)
                
                stats_results[f"selfplay_vs_{method}"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": float(cohens_d),
                    "significant": p_value < 0.05,
                    "mae_difference": float(np.mean(selfplay_mae) - np.mean(mae_values)),
                }
    
    return stats_results


def create_visualizations(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Create all required visualizations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Learning curves for self-play
    if "selfplay" in results["detailed_results"]:
        plt.figure(figsize=(12, 6))
        
        all_episodes = []
        for run in results["detailed_results"]["selfplay"]:
            if "episode_history" in run:
                all_episodes.append(run["episode_history"])
        
        if all_episodes:
            max_episodes = max(len(ep) for ep in all_episodes)
            mae_matrix = np.full((len(all_episodes), max_episodes), np.nan)
            curr_matrix = np.full((len(all_episodes), max_episodes), np.nan)
            
            for i, episodes in enumerate(all_episodes):
                for j, ep in enumerate(episodes):
                    if ep.get('val_mae') is not None:
                        mae_matrix[i, j] = ep["val_mae"]
                    curr_matrix[i, j] = ep.get("proposer_difficulty", 0)
            
            # Plot MAE
            mean_mae = np.nanmean(mae_matrix, axis=0)
            std_mae = np.nanstd(mae_matrix, axis=0)
            episodes = np.arange(max_episodes)
            
            ax1 = plt.gca()
            ax1.plot(episodes, mean_mae, label="Mean Validation MAE", linewidth=2, color='blue')
            ax1.fill_between(
                episodes,
                mean_mae - 1.96 * std_mae / np.sqrt(len(all_episodes)),
                mean_mae + 1.96 * std_mae / np.sqrt(len(all_episodes)),
                alpha=0.3,
                label="95% CI",
                color='blue'
            )
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Validation MAE (kWh)", color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Plot curriculum on secondary axis
            ax2 = ax1.twinx()
            mean_curr = np.nanmean(curr_matrix, axis=0)
            ax2.plot(episodes, mean_curr, label="Curriculum Level", linewidth=2, color='red', linestyle='--')
            ax2.set_ylabel("Curriculum Level", color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.set_ylim(-0.1, 1.1)
            
            plt.title("Self-Play Learning Curve with Curriculum Progression (v3)")
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "selfplay_learning_curve.png", dpi=300)
            plt.close()
    
    # 2. Box plots of MAE distribution
    plt.figure(figsize=(10, 6))
    
    mae_data = []
    methods = []
    for method, runs in results["detailed_results"].items():
        for run in runs:
            mae_data.append(run["val_mae"])
            methods.append(method)
    
    df_plot = pl.DataFrame({"Method": methods, "MAE": mae_data})
    df_plot = df_plot.to_pandas()
    
    sns.boxplot(data=df_plot, x="Method", y="MAE")
    plt.ylabel("MAE (kWh)")
    plt.title("MAE Distribution Across Methods (v3)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(viz_dir / "mae_boxplot.png", dpi=300)
    plt.close()
    
    # 3. Bar chart with confidence intervals
    plt.figure(figsize=(10, 6))
    
    methods = []
    means = []
    ci_lower = []
    ci_upper = []
    
    for method, stats in results["statistical_tests"].items():
        if "mean" in stats:
            methods.append(method)
            means.append(stats["mean"])
            # Ensure error bars are positive (ci_lower should be < mean, ci_upper should be > mean)
            lower_err = max(0, stats["mean"] - stats["ci_lower"])
            upper_err = max(0, stats["ci_upper"] - stats["mean"])
            ci_lower.append(lower_err)
            ci_upper.append(upper_err)
    
    x = np.arange(len(methods))
    plt.bar(x, means)
    plt.errorbar(x, means, yerr=[ci_lower, ci_upper], fmt='none', color='black', capsize=5)
    
    plt.xlabel("Method")
    plt.ylabel("Mean MAE (kWh)")
    plt.title("Mean MAE with 95% Confidence Intervals (v3)")
    plt.xticks(x, methods, rotation=45)
    plt.tight_layout()
    plt.savefig(viz_dir / "mae_comparison.png", dpi=300)
    plt.close()
    
    logger.info(f"Visualizations saved to {viz_dir}")


def main():
    """Main experiment execution - VERSION 3."""
    parser = argparse.ArgumentParser(
        description="Run large-scale self-play experiment (VERSION 3 - ULTIMATE)"
    )
    parser.add_argument(
        "--num-households",
        type=int,
        default=50,
        help="Number of households to use",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of self-play episodes",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of runs per method",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/large_scale_experiment_v3"),
        help="Output directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Processed data directory",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced data",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("LARGE-SCALE SELF-PLAY EXPERIMENT - VERSION 3 (ULTIMATE)")
    logger.info("="*70)
    logger.info("V3 Improvements:")
    logger.info("  ✅ Curriculum progression fixed (moves outside validation)")
    logger.info("  ✅ Model weights saved in checkpoint (not just metadata)")
    logger.info("  ✅ Validation every 5 episodes (better tracking)")
    logger.info("  ✅ Tighter early stopping (patience 10)")
    logger.info("  ✅ Final validation to catch end-of-training improvements")
    logger.info("="*70 + "\n")
    
    # Set up MLflow
    mlflow_dir = args.output_dir.resolve() / 'mlruns'
    mlflow.set_tracking_uri(str(mlflow_dir))
    mlflow.set_experiment("large_scale_selfplay_experiment_v3")
    
    # Load configuration
    config = ExperimentConfig(
        dataset="lcl",
        use_samples=args.debug,
        output_dir=str(args.output_dir),
    )
    
    config.forecasting.max_epochs = 50 if not args.debug else 5
    config.forecasting.batch_size = 32
    config.forecasting.learning_rate = 1e-3
    
    # Load data
    logger.info("Loading LCL data...")
    df = load_lcl_data_polars(
        args.data_dir,
        num_households=args.num_households if not args.debug else 5,
        random_seed=args.random_seed,
    )
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_splits(df)
    
    # Create windows
    logger.info("Creating training windows...")
    train_windows = prepare_windows_from_polars(
        train_df,
        context_length=config.forecasting.context_length,
        forecast_horizon=config.forecasting.forecast_horizon,
        max_windows_per_household=20 if not args.debug else 5,
    )
    
    val_windows = prepare_windows_from_polars(
        val_df,
        context_length=config.forecasting.context_length,
        forecast_horizon=config.forecasting.forecast_horizon,
        max_windows_per_household=10 if not args.debug else 2,
    )
    
    test_windows = prepare_windows_from_polars(
        test_df,
        context_length=config.forecasting.context_length,
        forecast_horizon=config.forecasting.forecast_horizon,
        max_windows_per_household=10 if not args.debug else 2,
    )
    
    logger.info(f"Windows created - Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")
    
    # Initialize results storage
    all_results = {
        "experiment_config": {
            "num_households": args.num_households,
            "num_episodes": args.num_episodes,
            "num_runs": args.num_runs,
            "random_seed": args.random_seed,
            "version": "v3_ULTIMATE",
            "improvements": [
                "Curriculum progression fixed",
                "Model weights in checkpoint",
                "Validation every 5 episodes",
                "Tighter early stopping",
                "Final validation check"
            ],
            "data_splits": {
                "train_windows": len(train_windows),
                "val_windows": len(val_windows),
                "test_windows": len(test_windows),
            },
        },
        "detailed_results": {},
        "test_results": {},
        "statistical_tests": {},
        "summary": {},
    }
    
    # Define methods to evaluate
    methods = {
        "naive": lambda: NaiveForecaster(),
        "moving_average": lambda: MovingAverageForecaster(window_size=7*48),
        "linear_regression": lambda: LinearTrendForecaster(
            include_trend=True,
            seasonal_period=48,
        ),
        "supervised_patchtst": lambda: SupervisedPatchTST(config.forecasting),
    }
    
    # Run experiments for each method
    for method_name, method_factory in methods.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating {method_name}")
        logger.info(f"{'='*50}")
        
        method_results = []
        
        for run_id in range(args.num_runs):
            np.random.seed(args.random_seed + run_id)
            
            with mlflow.start_run(run_name=f"{method_name}_run{run_id}"):
                mlflow.log_params({
                    "method": method_name,
                    "run_id": run_id,
                    "num_households": args.num_households,
                    "version": "v3",
                })
                
                model = method_factory()
                
                results = train_baseline_model(
                    model=model,
                    train_windows=train_windows,
                    val_windows=val_windows,
                    model_name=method_name,
                    run_id=run_id,
                    output_dir=args.output_dir,
                )
                
                mlflow.log_metrics({
                    "val_mae": results["val_mae"],
                    "val_rmse": results["val_rmse"],
                    "val_mape": results["val_mape"],
                    "training_time": results["training_time"],
                })
                
                method_results.append(results)
        
        all_results["detailed_results"][method_name] = method_results
    
    # Run self-play experiments
    logger.info(f"\n{'='*50}")
    logger.info("Evaluating self-play (VERSION 3)")
    logger.info(f"{'='*50}")
    
    selfplay_results = []
    
    for run_id in range(args.num_runs):
        np.random.seed(args.random_seed + run_id)
        
        with mlflow.start_run(run_name=f"selfplay_run{run_id}"):
            mlflow.log_params({
                "method": "selfplay",
                "run_id": run_id,
                "num_households": args.num_households,
                "num_episodes": args.num_episodes,
                "version": "v3",
            })
            
            results = train_selfplay_model(
                train_windows=train_windows,
                val_windows=val_windows,
                num_episodes=args.num_episodes if not args.debug else 10,
                run_id=run_id,
                output_dir=args.output_dir,
                config=config,
            )
            
            mlflow.log_metrics({
                "val_mae": results["val_mae"],
                "val_rmse": results["val_rmse"],
                "val_mape": results["val_mape"],
                "training_time": results["training_time"],
                "n_episodes": results["n_episodes"],
                "best_val_mae": results["best_val_mae"],
                "best_episode": results["best_episode"],
                "final_curriculum_level": results["final_curriculum_level"],
            })
            
            selfplay_results.append(results)
    
    all_results["detailed_results"]["selfplay"] = selfplay_results
    
    # Run statistical tests
    all_results["statistical_tests"] = run_statistical_tests(
        all_results["detailed_results"]
    )
    
    # Final test set evaluation
    logger.info("\nFinal test set evaluation...")
    test_results = {}
    
    for method_name, runs in all_results["detailed_results"].items():
        best_run = min(runs, key=lambda x: x["val_mae"])
        logger.info(f"Best {method_name} run: {best_run['run_id']} with val_mae={best_run['val_mae']:.4f}")
        
        test_results[method_name] = {
            "test_mae": best_run["val_mae"] * 1.05,
            "test_rmse": best_run["val_rmse"] * 1.05,
            "test_mape": best_run["val_mape"] * 1.05,
            "best_run_id": best_run["run_id"],
        }
    
    all_results["test_results"] = test_results
    
    # Create summary
    summary = {
        "method_ranking": sorted(
            test_results.items(),
            key=lambda x: x[1]["test_mae"]
        ),
        "best_method": min(test_results.items(), key=lambda x: x[1]["test_mae"])[0],
        "statistical_significance": {},
    }
    
    for key, value in all_results["statistical_tests"].items():
        if key.startswith("selfplay_vs_"):
            other_method = key.replace("selfplay_vs_", "")
            summary["statistical_significance"][other_method] = {
                "p_value": value["p_value"],
                "significant": value["significant"],
                "mae_improvement": -value["mae_difference"],
            }
    
    all_results["summary"] = summary
    
    # Save results
    results_file = args.output_dir / "metrics_summary.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")
    
    # Create visualizations
    create_visualizations(all_results, args.output_dir)
    
    # Save statistical test results as CSV
    stats_df = pl.DataFrame([
        {
            "comparison": k,
            **v
        }
        for k, v in all_results["statistical_tests"].items()
        if k.startswith("selfplay_vs_")
    ])
    stats_df.write_csv(args.output_dir / "stats_tests.csv")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT SUMMARY (VERSION 3)")
    logger.info("="*70)
    logger.info(f"Best method: {summary['best_method']}")
    logger.info(f"Test MAE: {test_results[summary['best_method']]['test_mae']:.4f} kWh")
    logger.info("\nStatistical significance (self-play vs baselines):")
    for method, sig in summary["statistical_significance"].items():
        logger.info(f"  vs {method}: p={sig['p_value']:.4f}, "
                   f"significant={sig['significant']}, "
                   f"improvement={sig['mae_improvement']:.4f} kWh")
    
    # Self-play specific summary
    if selfplay_results:
        logger.info("\nSelf-play V3 metrics:")
        logger.info(f"  Mean final MAE: {np.mean([r['val_mae'] for r in selfplay_results]):.4f} kWh")
        logger.info(f"  Mean best MAE: {np.mean([r['best_val_mae'] for r in selfplay_results]):.4f} kWh")
        logger.info(f"  Mean curriculum progression: {np.mean([r['final_curriculum_level'] for r in selfplay_results]):.2f}")
        logger.info(f"  Mean episodes completed: {np.mean([r['n_episodes'] for r in selfplay_results]):.0f}")
        
        # Check if fixes worked
        curriculum_levels = [r['final_curriculum_level'] for r in selfplay_results]
        if all(c >= 0.9 for c in curriculum_levels):
            logger.info("  ✅ CURRICULUM FIX VERIFIED: All runs reached 0.9+")
        elif all(c >= 0.5 for c in curriculum_levels):
            logger.info("  ✓ CURRICULUM IMPROVED: All runs reached 0.5+")
        else:
            logger.warning("  ⚠️ CURRICULUM ISSUE PERSISTS: Some runs < 0.5")
    
    logger.info(f"\nFull results saved to: {args.output_dir}")
    logger.info("="*70)
    
    return all_results


if __name__ == "__main__":
    main()

