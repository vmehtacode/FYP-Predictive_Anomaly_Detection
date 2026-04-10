"""CLI runner for baseline models."""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fyp.baselines.anomaly import create_default_detectors
from fyp.baselines.forecasting import create_default_forecasters
from fyp.config import ExperimentConfig, create_sample_config, get_config_from_env
from fyp.data_loader import EnergyDataLoader
from fyp.metrics import MetricsTracker
from fyp.utils.random import (
    get_ci_safe_config_overrides,
    set_global_seeds,
    should_use_ci_mode,
)

# Setup logger
logger = logging.getLogger(__name__)

# Optional imports for advanced models
# TODO: Migrate to canonical import paths (fyp.anomaly.autoencoder)
try:
    from fyp.models.autoencoder import AutoencoderAnomalyDetector
    from fyp.models.patchtst import PatchTSTForecaster

    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced models not available: {e}")
    ADVANCED_MODELS_AVAILABLE = False

# Optional MLflow import
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_forecasting_baselines(
    dataset: str,
    use_samples: bool = False,
    horizon: int = 48,
    output_dir: Path = Path("data/derived/evaluation"),
    model_type: str = "baseline",
    config: ExperimentConfig | None = None,
) -> dict:
    """Run forecasting baseline models."""
    logger.info(f"Running forecasting baselines for {dataset}")

    # Load data
    data_root = Path("data/samples") if use_samples else Path("data/processed")

    if use_samples:
        # Load sample CSV directly
        sample_file = data_root / f"{dataset}_sample.csv"
        if not sample_file.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_file}")

        df = pd.read_csv(sample_file)

        # Identify columns by content
        time_col = None
        entity_col = None
        energy_col = None

        for col in df.columns:
            if "timestamp" in col.lower() or "time" in col.lower():
                time_col = col
            elif (
                "id" in col.lower()
                or "household" in col.lower()
                or "feeder" in col.lower()
            ):
                entity_col = col
            elif "kwh" in col.lower() or "energy" in col.lower() or "wh" in col.lower():
                energy_col = col

        # Fallback to column positions
        if not time_col:
            time_col = df.columns[0]
        if not entity_col:
            entity_col = df.columns[1]
        if not energy_col:
            energy_col = df.columns[2]

        # Create standardized DataFrame
        df_clean = pd.DataFrame()
        df_clean["ts_utc"] = pd.to_datetime(df[time_col], utc=True)
        df_clean["entity_id"] = df[entity_col].astype(str)
        df_clean["energy_kwh"] = df[energy_col].astype(float)
        df_clean["dataset"] = dataset
        df_clean["interval_mins"] = 30
        df_clean["source"] = sample_file.name
        df_clean["extras"] = "{}"

        df = df_clean
    else:
        # Load processed Parquet data
        loader = EnergyDataLoader(data_root)
        df = loader.load_dataset(dataset)

    if df.empty:
        logger.warning(f"No data found for dataset {dataset}")
        return {}

    # Create forecasting windows (adjust for sample size)
    loader = EnergyDataLoader()

    # For samples, use smaller windows
    if use_samples:
        history_length = min(24, len(df) // 3)  # Use 1/3 of data for history
        forecast_horizon = min(horizon, len(df) // 3)  # Use 1/3 for forecast
    else:
        history_length = max(48, horizon)
        forecast_horizon = horizon

    windows = loader.create_forecasting_windows(
        df,
        history_length=history_length,
        forecast_horizon=forecast_horizon,
        step_size=max(1, forecast_horizon // 4),  # Overlapping windows
    )

    if not windows:
        logger.warning("No forecasting windows created")
        return {}

    # Initialize forecasters and metrics
    if model_type == "baseline":
        forecasters = create_default_forecasters()
        use_advanced = False
    elif model_type == "patchtst" and ADVANCED_MODELS_AVAILABLE:
        # Use PatchTST model
        if config is None:
            config = (
                create_sample_config()
                if use_samples
                else ExperimentConfig(dataset=dataset)
            )

        patchtst_forecaster = PatchTSTForecaster(
            patch_len=config.forecasting.patch_len,
            d_model=config.forecasting.d_model,
            n_heads=config.forecasting.n_heads,
            n_layers=config.forecasting.n_layers,
            forecast_horizon=config.forecasting.forecast_horizon,
            quantiles=config.forecasting.quantiles,
            learning_rate=config.forecasting.learning_rate,
            max_epochs=config.forecasting.max_epochs,
            batch_size=config.forecasting.batch_size,
            device=config.forecasting.device,
        )

        forecasters = {"patchtst": patchtst_forecaster}
        use_advanced = True
    else:
        logger.warning(
            f"Model type {model_type} not available, falling back to baselines"
        )
        forecasters = create_default_forecasters()
        use_advanced = False

    tracker = MetricsTracker()

    # Train advanced models once on all windows
    if use_advanced and "patchtst" in forecasters:
        try:
            logger.info("Training PatchTST on all windows")
            training_result = forecasters["patchtst"].fit(
                windows[:50]
            )  # Limit training data
            logger.info(f"PatchTST training complete: {training_result}")
        except Exception as e:
            logger.error(f"PatchTST training failed: {e}")
            # Fall back to baselines
            forecasters = create_default_forecasters()
            use_advanced = False

    # Run forecasting on windows
    for i, window in enumerate(windows[:10]):  # Limit for speed
        entity_id = window["entity_id"]
        history = window["history_energy"]
        target = window["target_energy"]
        timestamps = window["history_timestamps"]

        logger.info(
            f"Processing window {i + 1}/{min(len(windows), 10)} for entity {entity_id}"
        )

        for name, forecaster in forecasters.items():
            try:
                if use_advanced and name == "patchtst":
                    # Advanced model with quantiles
                    quantile_forecasts = forecaster.predict(
                        history, len(target), return_quantiles=True
                    )

                    # Add results for each quantile
                    for quantile, forecast in quantile_forecasts.items():
                        tracker.add_forecasting_result(
                            entity_id=f"{entity_id}_{name}_q{quantile}",
                            window_id=i,
                            y_true=target,
                            y_pred=forecast,
                            y_train=history,
                            quantiles={float(quantile): forecast},
                        )
                else:
                    # Baseline model
                    forecast = forecaster.predict(history, len(target), timestamps)

                    # Add to tracker
                    tracker.add_forecasting_result(
                        entity_id=f"{entity_id}_{name}",
                        window_id=i,
                        y_true=target,
                        y_pred=forecast,
                        y_train=history,
                    )

            except Exception as e:
                logger.warning(f"Forecasting failed for {name}: {e}")

    # Get summary metrics
    summary = tracker.get_forecasting_summary()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_dir / "forecast_metrics.csv"
    if tracker.forecasting_results:
        pd.DataFrame(tracker.forecasting_results).to_csv(metrics_file, index=False)

    # Save summary
    summary_file = output_dir / "forecast_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Create plots
    create_forecast_plots(tracker, output_dir, dataset)

    logger.info(f"Forecasting complete. Results saved to {output_dir}")
    return summary


def run_anomaly_baselines(
    dataset: str,
    use_samples: bool = False,
    output_dir: Path = Path("data/derived/evaluation"),
    model_type: str = "baseline",
    config: ExperimentConfig | None = None,
) -> dict:
    """Run anomaly detection baseline models."""
    logger.info(f"Running anomaly baselines for {dataset}")

    # Load data (same logic as forecasting)
    data_root = Path("data/samples") if use_samples else Path("data/processed")

    if use_samples:
        sample_file = data_root / f"{dataset}_sample.csv"
        if not sample_file.exists():
            raise FileNotFoundError(f"Sample file not found: {sample_file}")

        df = pd.read_csv(sample_file)

        # Use same column detection logic
        time_col = entity_col = energy_col = None
        for col in df.columns:
            if "timestamp" in col.lower() or "time" in col.lower():
                time_col = col
            elif (
                "id" in col.lower()
                or "household" in col.lower()
                or "feeder" in col.lower()
            ):
                entity_col = col
            elif "kwh" in col.lower() or "energy" in col.lower() or "wh" in col.lower():
                energy_col = col

        if not time_col:
            time_col = df.columns[0]
        if not entity_col:
            entity_col = df.columns[1]
        if not energy_col:
            energy_col = df.columns[2]

        df_clean = pd.DataFrame()
        df_clean["ts_utc"] = pd.to_datetime(df[time_col], utc=True)
        df_clean["entity_id"] = df[entity_col].astype(str)
        df_clean["energy_kwh"] = df[energy_col].astype(float)
        df_clean["dataset"] = dataset

        df = df_clean
    else:
        loader = EnergyDataLoader(data_root)
        df = loader.load_dataset(dataset)

    if df.empty:
        logger.warning(f"No data found for dataset {dataset}")
        return {}

    # Initialize detectors and metrics
    if model_type == "baseline":
        detectors = create_default_detectors()
    elif model_type == "autoencoder" and ADVANCED_MODELS_AVAILABLE:
        # Use autoencoder model
        if config is None:
            config = (
                create_sample_config()
                if use_samples
                else ExperimentConfig(dataset=dataset)
            )

        ae_detector = AutoencoderAnomalyDetector(
            window_size=config.anomaly.window_size,
            hidden_sizes=config.anomaly.hidden_sizes,
            learning_rate=config.anomaly.learning_rate,
            max_epochs=config.anomaly.max_epochs,
            batch_size=config.anomaly.batch_size,
            contamination=config.anomaly.contamination,
            device=config.anomaly.device,
        )

        detectors = {"autoencoder": ae_detector}
    else:
        logger.warning(
            f"Model type {model_type} not available, falling back to baselines"
        )
        detectors = create_default_detectors()

    tracker = MetricsTracker()

    # Run anomaly detection per entity
    for entity_id in df["entity_id"].unique()[:5]:  # Limit for speed
        entity_df = df[df["entity_id"] == entity_id].sort_values("ts_utc")

        if len(entity_df) < 48:  # Need minimum data
            continue

        energy_values = entity_df["energy_kwh"].values

        # Split into train/test
        split_idx = int(len(energy_values) * 0.7)
        train_data = energy_values[:split_idx]
        test_data = energy_values[split_idx:]

        logger.info(f"Processing entity {entity_id}")

        for name, detector in detectors.items():
            try:
                # Fit on training data
                detector.fit(train_data)

                # Predict on test data
                scores = detector.predict_scores(test_data)

                # Create synthetic labels for evaluation (spikes as anomalies)
                synthetic_labels = create_synthetic_anomaly_labels(test_data)

                # Add to tracker
                tracker.add_anomaly_result(
                    entity_id=f"{entity_id}_{name}",
                    y_true=synthetic_labels,
                    y_scores=scores,
                    threshold=0.5,
                )

            except Exception as e:
                logger.warning(f"Anomaly detection failed for {name}: {e}")

    # Get summary metrics
    summary = tracker.get_anomaly_summary()

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = output_dir / "anomaly_metrics.csv"
    if tracker.anomaly_results:
        pd.DataFrame(tracker.anomaly_results).to_csv(metrics_file, index=False)

    # Save summary
    summary_file = output_dir / "anomaly_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Create plots
    create_anomaly_plots(tracker, output_dir, dataset)

    logger.info(f"Anomaly detection complete. Results saved to {output_dir}")
    return summary


def create_synthetic_anomaly_labels(
    data: np.ndarray, threshold_factor: float = 2.0
) -> np.ndarray:
    """Create synthetic anomaly labels based on statistical outliers."""
    mean_val = np.mean(data)
    std_val = np.std(data)
    threshold = mean_val + threshold_factor * std_val

    # Label points above threshold as anomalies
    labels = (data > threshold).astype(int)

    # Ensure at least some anomalies exist
    if np.sum(labels) == 0:
        # Use top 5% as anomalies
        threshold = np.percentile(data, 95)
        labels = (data > threshold).astype(int)

    return labels


def create_forecast_plots(
    tracker: MetricsTracker, output_dir: Path, dataset: str
) -> None:
    """Create forecasting evaluation plots."""
    if not tracker.forecasting_results:
        return

    df = pd.DataFrame(tracker.forecasting_results)

    # Plot 1: MAE by model
    plt.figure(figsize=(10, 6))

    models = [col.split("_")[-1] for col in df["entity_id"] if "_" in col]
    if models:
        model_mae = df.groupby(df["entity_id"].str.split("_").str[-1])["mae"].mean()
        model_mae.plot(kind="bar")
        plt.title(f"Mean Absolute Error by Model - {dataset.upper()}")
        plt.ylabel("MAE (kWh)")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            output_dir / "forecast_mae_by_model.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    # Plot 2: Error distribution
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ["mae", "rmse", "mape"]
    available_metrics = [m for m in metrics_to_plot if m in df.columns]

    if available_metrics:
        df[available_metrics].boxplot()
        plt.title(f"Forecast Error Distribution - {dataset.upper()}")
        plt.ylabel("Error Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            output_dir / "forecast_error_distribution.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


def create_anomaly_plots(
    tracker: MetricsTracker, output_dir: Path, dataset: str
) -> None:
    """Create anomaly detection evaluation plots."""
    if not tracker.anomaly_results:
        return

    df = pd.DataFrame(tracker.anomaly_results)

    # Plot 1: Precision-Recall by model
    plt.figure(figsize=(10, 6))

    models = [col.split("_")[-1] for col in df["entity_id"] if "_" in col]
    if models:
        model_groups = df.groupby(df["entity_id"].str.split("_").str[-1])

        for model_name, group in model_groups:
            if len(group) > 0:
                plt.scatter(
                    group["recall"].mean(),
                    group["precision"].mean(),
                    label=model_name,
                    s=100,
                    alpha=0.7,
                )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall by Model - {dataset.upper()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "anomaly_precision_recall.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    # Plot 2: F1 scores
    plt.figure(figsize=(10, 6))

    if models:
        model_f1 = df.groupby(df["entity_id"].str.split("_").str[-1])["f1"].mean()
        model_f1.plot(kind="bar")
        plt.title(f"F1 Score by Model - {dataset.upper()}")
        plt.ylabel("F1 Score")
        plt.xlabel("Model")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            output_dir / "anomaly_f1_by_model.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run baseline models for energy forecasting and anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run forecasting baselines on samples
  python -m fyp.runner forecast --dataset lcl --use-samples

  # Run anomaly detection on real data
  python -m fyp.runner anomaly --dataset ukdale

  # Custom horizon and output directory
  python -m fyp.runner forecast --dataset ssen --horizon 96 --output-dir results/
""",
    )

    parser.add_argument(
        "mode",
        choices=["forecast", "anomaly"],
        help="Type of baseline to run",
    )
    parser.add_argument(
        "--dataset",
        choices=["lcl", "ukdale", "ssen"],
        required=True,
        help="Dataset to process",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use sample data instead of full dataset",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=48,
        help="Forecast horizon in time steps (default: 48)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/derived/evaluation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--model-type",
        choices=["baseline", "patchtst", "autoencoder"],
        default="baseline",
        help="Model type to use (default: baseline)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="energy_forecasting",
        help="MLflow experiment name",
    )

    args = parser.parse_args()

    # Set deterministic seeds for reproducibility
    if args.use_samples or should_use_ci_mode():
        set_global_seeds(42)

    # Load configuration
    if args.config and args.config.exists():
        from fyp.config import load_config

        config = load_config(args.config)
    else:
        config = get_config_from_env()
        config.dataset = args.dataset
        config.use_samples = args.use_samples

        # Apply CI-safe overrides
        if args.use_samples or should_use_ci_mode():
            ci_overrides = get_ci_safe_config_overrides()
            if "forecasting" in ci_overrides:
                for key, value in ci_overrides["forecasting"].items():
                    setattr(config.forecasting, key, value)
            if "anomaly" in ci_overrides:
                for key, value in ci_overrides["anomaly"].items():
                    setattr(config.anomaly, key, value)

    # Initialize MLflow if available
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "model_type": args.model_type,
                "use_samples": args.use_samples,
                "mode": args.mode,
            }
        )

    try:
        if args.mode == "forecast":
            summary = run_forecasting_baselines(
                dataset=args.dataset,
                use_samples=args.use_samples,
                horizon=args.horizon,
                output_dir=args.output_dir,
                model_type=args.model_type,
                config=config,
            )
        elif args.mode == "anomaly":
            summary = run_anomaly_baselines(
                dataset=args.dataset,
                use_samples=args.use_samples,
                output_dir=args.output_dir,
                model_type=args.model_type,
                config=config,
            )

        # Log results to MLflow
        if MLFLOW_AVAILABLE:
            mlflow.log_metrics(summary)
            mlflow.log_artifacts(str(args.output_dir))
            mlflow.end_run()

        print("\n=== Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    except Exception as e:
        if MLFLOW_AVAILABLE:
            mlflow.end_run(status="FAILED")
        logger.error(f"Failed to run baselines: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
