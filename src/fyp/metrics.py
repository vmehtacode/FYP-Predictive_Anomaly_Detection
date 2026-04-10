"""Metrics for forecasting and anomaly detection."""


import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 48,  # 24 hours at 30-min resolution
) -> float:
    """Calculate Mean Absolute Scaled Error."""
    if len(y_train) < seasonal_period:
        # Fallback to naive forecast if not enough training data
        naive_forecast = np.roll(y_train, 1)[1:]
        scale = mean_absolute_error(y_train[1:], naive_forecast)
    else:
        # Seasonal naive forecast
        naive_forecast = y_train[:-seasonal_period]
        scale = mean_absolute_error(y_train[seasonal_period:], naive_forecast)

    if scale == 0:
        return 0.0

    return mean_absolute_error(y_true, y_pred) / scale


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calculate pinball loss for quantile predictions."""
    residual = y_true - y_pred
    return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))


def coverage_score(
    y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray
) -> float:
    """Calculate empirical coverage for prediction intervals."""
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(in_interval)


def forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    quantiles: dict[float, np.ndarray] | None = None,
) -> dict[str, float]:
    """Calculate all standard forecasting metrics."""
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }

    # MASE if training data available
    if y_train is not None:
        metrics["mase"] = mean_absolute_scaled_error(y_true, y_pred, y_train)

    # Quantile scores if available
    if quantiles:
        for q, y_q in quantiles.items():
            metrics[f"pinball_{q:.1f}"] = pinball_loss(y_true, y_q, q)

        # Coverage scores for common intervals
        if 0.1 in quantiles and 0.9 in quantiles:
            metrics["coverage_80"] = coverage_score(
                y_true, quantiles[0.1], quantiles[0.9]
            )

        if 0.05 in quantiles and 0.95 in quantiles:
            metrics["coverage_90"] = coverage_score(
                y_true, quantiles[0.05], quantiles[0.95]
            )

    return metrics


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    if len(y_pred.shape) > 1:
        # Handle probability predictions
        y_pred_binary = (y_pred[:, 1] > threshold).astype(int)
    else:
        # Handle binary or score predictions
        y_pred_binary = (y_pred > threshold).astype(int)

    true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
    false_positives = np.sum((y_true == 0) & (y_pred_binary == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred_binary == 0))

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def detection_latency(
    true_anomaly_starts: list[int],
    predicted_anomaly_times: list[int],
    max_delay: int = 48,  # Maximum acceptable delay in time steps
) -> dict[str, float]:
    """Calculate average detection latency."""
    latencies = []

    for true_start in true_anomaly_starts:
        # Find first prediction within max_delay
        detections = [
            p
            for p in predicted_anomaly_times
            if true_start <= p <= true_start + max_delay
        ]

        if detections:
            latencies.append(min(detections) - true_start)
        else:
            latencies.append(max_delay)  # Penalty for missing detection

    return {
        "avg_latency": np.mean(latencies) if latencies else max_delay,
        "detection_rate": np.mean([latency < max_delay for latency in latencies])
        if latencies
        else 0.0,
    }


def anomaly_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    true_anomaly_starts: list[int] | None = None,
    predicted_anomaly_times: list[int] | None = None,
) -> dict[str, float]:
    """Calculate all anomaly detection metrics."""
    metrics = precision_recall_f1(y_true, y_scores, threshold)

    # Add latency metrics if available
    if true_anomaly_starts and predicted_anomaly_times:
        latency_metrics = detection_latency(
            true_anomaly_starts, predicted_anomaly_times
        )
        metrics.update(latency_metrics)

    # Add score-based metrics
    metrics.update(
        {
            "avg_anomaly_score": np.mean(y_scores[y_true == 1])
            if np.any(y_true == 1)
            else 0.0,
            "avg_normal_score": np.mean(y_scores[y_true == 0])
            if np.any(y_true == 0)
            else 0.0,
        }
    )

    return metrics


class MetricsTracker:
    """Track and aggregate metrics across multiple windows/entities."""

    def __init__(self):
        self.forecasting_results = []
        self.anomaly_results = []

    def add_forecasting_result(
        self,
        entity_id: str,
        window_id: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: np.ndarray | None = None,
        quantiles: dict[float, np.ndarray] | None = None,
    ):
        """Add forecasting result for aggregation."""
        metrics = forecasting_metrics(y_true, y_pred, y_train, quantiles)
        metrics.update(
            {
                "entity_id": entity_id,
                "window_id": window_id,
                "n_steps": len(y_true),
            }
        )
        self.forecasting_results.append(metrics)

    def add_anomaly_result(
        self,
        entity_id: str,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5,
    ):
        """Add anomaly result for aggregation."""
        metrics = anomaly_metrics(y_true, y_scores, threshold)
        metrics.update(
            {
                "entity_id": entity_id,
                "n_steps": len(y_true),
                "n_anomalies": np.sum(y_true),
            }
        )
        self.anomaly_results.append(metrics)

    def get_forecasting_summary(self) -> dict[str, float]:
        """Get aggregated forecasting metrics."""
        if not self.forecasting_results:
            return {}

        # Convert to DataFrame for easier aggregation
        import pandas as pd

        df = pd.DataFrame(self.forecasting_results)

        summary = {}
        metric_cols = [
            "mae",
            "rmse",
            "mape",
            "mase",
            "pinball_0.1",
            "pinball_0.5",
            "pinball_0.9",
            "coverage_80",
            "coverage_90",
        ]

        for col in metric_cols:
            if col in df.columns:
                summary[f"avg_{col}"] = df[col].mean()
                summary[f"std_{col}"] = df[col].std()

        summary.update(
            {
                "n_windows": len(df),
                "n_entities": df["entity_id"].nunique(),
            }
        )

        return summary

    def get_anomaly_summary(self) -> dict[str, float]:
        """Get aggregated anomaly metrics."""
        if not self.anomaly_results:
            return {}

        import pandas as pd

        df = pd.DataFrame(self.anomaly_results)

        summary = {}
        metric_cols = ["precision", "recall", "f1", "avg_latency", "detection_rate"]

        for col in metric_cols:
            if col in df.columns:
                summary[f"avg_{col}"] = df[col].mean()
                summary[f"std_{col}"] = df[col].std()

        summary.update(
            {
                "n_entities": len(df),
                "total_anomalies": df["n_anomalies"].sum(),
            }
        )

        return summary
