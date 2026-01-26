"""Ensemble forecaster combining neural models with simple baselines.

This module implements ensemble strategies that combine the strengths of:
1. Neural models (PatchTST, FrequencyEnhancedPatchTST) - complex pattern learning
2. Simple baselines (Moving Average, Naive) - exploit strong periodicity

The goal is to beat simple baselines by leveraging their strengths as part of
the prediction, similar to residual learning where the neural model learns to
correct baseline predictions.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class MovingAverageBaseline:
    """Simple Moving Average baseline that captures daily periodicity."""
    
    def __init__(self, window_size: int = 48):
        """
        Args:
            window_size: Number of periods for moving average (48 = 24 hours)
        """
        self.window_size = window_size
        
    def predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        """Generate forecast using moving average.
        
        Args:
            history: Historical time series
            steps: Number of steps to forecast
            
        Returns:
            Forecast array of length `steps`
        """
        if len(history) < self.window_size:
            return np.full(steps, np.mean(history))
        
        # Use last window_size values for averaging
        forecast = np.full(steps, np.mean(history[-self.window_size:]))
        return forecast


class SeasonalNaiveBaseline:
    """Seasonal Naive baseline that repeats the same time yesterday/last week."""
    
    def __init__(self, seasonal_period: int = 48):
        """
        Args:
            seasonal_period: Seasonal period (48 = daily for 30-min data)
        """
        self.seasonal_period = seasonal_period
        
    def predict(self, history: np.ndarray, steps: int) -> np.ndarray:
        """Generate forecast by repeating seasonal pattern.
        
        Args:
            history: Historical time series
            steps: Number of steps to forecast
            
        Returns:
            Forecast array of length `steps`
        """
        if len(history) < self.seasonal_period:
            return np.full(steps, np.mean(history))
        
        # Get the last seasonal period
        seasonal_pattern = history[-self.seasonal_period:]
        
        # Tile to cover forecast horizon
        n_repeats = (steps // self.seasonal_period) + 1
        forecast = np.tile(seasonal_pattern, n_repeats)[:steps]
        
        return forecast


class EnsembleForecaster:
    """Ensemble forecaster combining neural model with simple baselines.
    
    Uses learnable or fixed weights to combine predictions from:
    - Neural model (PatchTST or FrequencyEnhancedPatchTST)
    - Moving Average baseline
    - Seasonal Naive baseline
    
    This approach ensures we capture both learnable patterns (neural) and
    exploit strong periodicity (baselines) that simple methods leverage.
    """
    
    def __init__(
        self,
        neural_model: Optional[object] = None,
        neural_weight: float = 0.6,
        moving_avg_weight: float = 0.25,
        seasonal_weight: float = 0.15,
        moving_avg_window: int = 48,
        seasonal_period: int = 48,
        adaptive_weights: bool = False,
    ):
        """Initialize ensemble forecaster.
        
        Args:
            neural_model: Neural forecaster (PatchTSTForecaster or FrequencyEnhancedForecaster)
            neural_weight: Weight for neural model predictions
            moving_avg_weight: Weight for moving average baseline
            seasonal_weight: Weight for seasonal naive baseline
            moving_avg_window: Window size for moving average
            seasonal_period: Seasonal period for naive baseline
            adaptive_weights: Whether to adaptively adjust weights based on performance
        """
        self.neural_model = neural_model
        
        # Normalize weights to sum to 1
        total_weight = neural_weight + moving_avg_weight + seasonal_weight
        self.neural_weight = neural_weight / total_weight
        self.moving_avg_weight = moving_avg_weight / total_weight
        self.seasonal_weight = seasonal_weight / total_weight
        
        # Initialize baselines
        self.moving_avg = MovingAverageBaseline(window_size=moving_avg_window)
        self.seasonal_naive = SeasonalNaiveBaseline(seasonal_period=seasonal_period)
        
        self.adaptive_weights = adaptive_weights
        self.performance_history = []
        
        logger.info(
            f"EnsembleForecaster initialized with weights: "
            f"neural={self.neural_weight:.2f}, ma={self.moving_avg_weight:.2f}, "
            f"seasonal={self.seasonal_weight:.2f}"
        )
        
    def predict(
        self, 
        history: np.ndarray, 
        steps: int,
        return_quantiles: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate ensemble forecast.
        
        Args:
            history: Historical time series
            steps: Number of steps to forecast
            return_quantiles: Whether to return quantile predictions
            
        Returns:
            Dictionary of forecasts (quantile keys if return_quantiles, else 'point')
        """
        # Get baseline predictions
        ma_forecast = self.moving_avg.predict(history, steps)
        seasonal_forecast = self.seasonal_naive.predict(history, steps)
        
        # Get neural model predictions if available
        if self.neural_model is not None:
            try:
                neural_preds = self.neural_model.predict(
                    history=history,
                    steps=steps,
                    return_quantiles=return_quantiles,
                )
            except Exception as e:
                logger.warning(f"Neural model prediction failed: {e}, using baselines only")
                neural_preds = None
        else:
            neural_preds = None
            
        # Combine predictions
        if return_quantiles:
            results = {}
            quantiles = ["0.1", "0.5", "0.9"]
            
            for q in quantiles:
                if neural_preds is not None and q in neural_preds:
                    neural_forecast = neural_preds[q]
                else:
                    # Use median baseline estimate if neural not available
                    neural_forecast = 0.5 * ma_forecast + 0.5 * seasonal_forecast
                    
                # Weighted ensemble
                ensemble_forecast = (
                    self.neural_weight * neural_forecast +
                    self.moving_avg_weight * ma_forecast +
                    self.seasonal_weight * seasonal_forecast
                )
                
                # Add uncertainty for non-median quantiles
                if q == "0.1":
                    scale = 0.9  # Lower bound
                elif q == "0.9":
                    scale = 1.1  # Upper bound
                else:
                    scale = 1.0
                    
                results[q] = np.maximum(ensemble_forecast * scale, 0.0)
                
            return results
        else:
            # Point forecast
            if neural_preds is not None and "point" in neural_preds:
                neural_forecast = neural_preds["point"]
            elif neural_preds is not None and "0.5" in neural_preds:
                neural_forecast = neural_preds["0.5"]
            else:
                neural_forecast = 0.5 * ma_forecast + 0.5 * seasonal_forecast
                
            ensemble_forecast = (
                self.neural_weight * neural_forecast +
                self.moving_avg_weight * ma_forecast +
                self.seasonal_weight * seasonal_forecast
            )
            
            return {"point": np.maximum(ensemble_forecast, 0.0)}
            
    def update_weights_from_error(
        self, 
        neural_error: float, 
        baseline_error: float,
        learning_rate: float = 0.1,
    ) -> None:
        """Adaptively update weights based on forecast errors.
        
        Args:
            neural_error: Error (MAE) from neural model
            baseline_error: Error (MAE) from baseline (average of MA and seasonal)
            learning_rate: How quickly to adjust weights
        """
        if not self.adaptive_weights:
            return
            
        # If neural is better, increase its weight
        if neural_error < baseline_error:
            delta = learning_rate * (1 - self.neural_weight)
            self.neural_weight += delta
            self.moving_avg_weight -= delta * 0.6
            self.seasonal_weight -= delta * 0.4
        else:
            # Decrease neural weight
            delta = learning_rate * self.neural_weight * 0.5
            self.neural_weight -= delta
            self.moving_avg_weight += delta * 0.6
            self.seasonal_weight += delta * 0.4
            
        # Clamp weights to reasonable bounds
        self.neural_weight = max(0.2, min(0.9, self.neural_weight))
        self.moving_avg_weight = max(0.05, min(0.5, self.moving_avg_weight))
        self.seasonal_weight = max(0.05, min(0.4, self.seasonal_weight))
        
        # Renormalize
        total = self.neural_weight + self.moving_avg_weight + self.seasonal_weight
        self.neural_weight /= total
        self.moving_avg_weight /= total
        self.seasonal_weight /= total
        
        logger.debug(
            f"Updated ensemble weights: neural={self.neural_weight:.2f}, "
            f"ma={self.moving_avg_weight:.2f}, seasonal={self.seasonal_weight:.2f}"
        )


class ResidualEnsemble:
    """Residual ensemble where neural model predicts corrections to baseline.
    
    This is an alternative ensemble strategy where:
    1. Baseline makes initial prediction
    2. Neural model predicts residual (error) of baseline
    3. Final = Baseline + Learned_Residual
    
    This can be more stable and easier to train.
    """
    
    def __init__(
        self,
        neural_model: Optional[object] = None,
        baseline_type: str = "seasonal",  # "ma" or "seasonal"
        seasonal_period: int = 48,
        residual_scale: float = 1.0,
    ):
        """Initialize residual ensemble.
        
        Args:
            neural_model: Neural model for residual prediction
            baseline_type: Type of baseline ("ma" or "seasonal")
            seasonal_period: Seasonal period for naive baseline
            residual_scale: Scale factor for residual predictions
        """
        self.neural_model = neural_model
        self.baseline_type = baseline_type
        self.residual_scale = residual_scale
        
        if baseline_type == "ma":
            self.baseline = MovingAverageBaseline(window_size=seasonal_period)
        else:
            self.baseline = SeasonalNaiveBaseline(seasonal_period=seasonal_period)
            
    def predict(
        self, 
        history: np.ndarray, 
        steps: int,
        return_quantiles: bool = True,
    ) -> dict[str, np.ndarray]:
        """Generate residual ensemble forecast.
        
        Args:
            history: Historical time series
            steps: Number of steps to forecast
            return_quantiles: Whether to return quantile predictions
            
        Returns:
            Dictionary of forecasts
        """
        # Get baseline prediction
        baseline_forecast = self.baseline.predict(history, steps)
        
        # Get neural residual prediction
        if self.neural_model is not None:
            try:
                neural_preds = self.neural_model.predict(
                    history=history,
                    steps=steps,
                    return_quantiles=return_quantiles,
                )
                
                if return_quantiles:
                    results = {}
                    for q, neural_forecast in neural_preds.items():
                        # Compute residual (difference from baseline)
                        residual = neural_forecast - baseline_forecast
                        # Scale residual 
                        scaled_residual = residual * self.residual_scale
                        # Final prediction = baseline + residual
                        final = baseline_forecast + scaled_residual
                        results[q] = np.maximum(final, 0.0)
                    return results
                else:
                    neural_forecast = neural_preds.get("point", neural_preds.get("0.5"))
                    residual = neural_forecast - baseline_forecast
                    scaled_residual = residual * self.residual_scale
                    final = baseline_forecast + scaled_residual
                    return {"point": np.maximum(final, 0.0)}
                    
            except Exception as e:
                logger.warning(f"Neural residual prediction failed: {e}")
                
        # Fall back to baseline only
        if return_quantiles:
            return {
                "0.1": baseline_forecast * 0.9,
                "0.5": baseline_forecast,
                "0.9": baseline_forecast * 1.1,
            }
        else:
            return {"point": baseline_forecast}


def create_ensemble_config(use_samples: bool = False) -> dict:
    """Create configuration for ensemble forecaster."""
    if use_samples:
        return {
            "neural_weight": 0.5,
            "moving_avg_weight": 0.3,
            "seasonal_weight": 0.2,
            "moving_avg_window": 24,
            "seasonal_period": 16,
            "adaptive_weights": False,
        }
    else:
        return {
            "neural_weight": 0.6,
            "moving_avg_weight": 0.25,
            "seasonal_weight": 0.15,
            "moving_avg_window": 48,
            "seasonal_period": 48,
            "adaptive_weights": True,
        }
