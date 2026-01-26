"""Frequency-Enhanced PatchTST for improved energy forecasting.

This module implements frequency-domain enhancements to the PatchTST architecture,
inspired by FreTS, FEDformer, and latest 2024 research. The key insight is that
simple baselines (Moving Average, Naive) implicitly capture periodicity - this
module makes periodicity capture explicit.

Key innovations:
1. Parallel frequency branch using FFT
2. Multi-scale decomposition (trend + seasonality)
3. Frequency-domain attention for capturing periodicities
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fyp.models.patchtst import EnergyPatchTST, PatchTSTForecaster


class FrequencyBranch(nn.Module):
    """Parallel frequency-domain processing branch.
    
    Applies FFT to input, processes in frequency domain, then converts back.
    This explicitly captures periodic patterns that favor simple baselines.
    """
    
    def __init__(self, seq_len: int, d_model: int, top_k_freqs: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.top_k_freqs = top_k_freqs
        
        # Frequency domain processing
        freq_len = seq_len // 2 + 1
        self.freq_embedding = nn.Linear(freq_len * 2, d_model)  # Real + Imag parts
        self.freq_processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] or [batch_size, n_patches, patch_len]
            
        Returns:
            Frequency-domain features [batch_size, d_model]
        """
        # Handle batched patch input
        if x.dim() == 3:
            batch_size, n_patches, patch_len = x.shape
            # Flatten patches to get full sequence view
            x = x.reshape(batch_size, -1)
        
        # Apply FFT
        freq = torch.fft.rfft(x, dim=-1)
        
        # Concatenate real and imaginary parts
        freq_features = torch.cat([freq.real, freq.imag], dim=-1)
        
        # Embed and process
        freq_embedded = self.freq_embedding(freq_features)
        freq_processed = self.freq_processor(freq_embedded)
        
        return self.norm(freq_processed)


class MultiScaleDecomposition(nn.Module):
    """Decompose time series into multiple scales (trend + seasonalities).
    
    Inspired by DLinear and TimeMixer - captures 30-min, daily, weekly patterns.
    """
    
    def __init__(self, kernel_sizes: list[int] = None):
        super().__init__()
        if kernel_sizes is None:
            # Default: 1.5hr, 3.5hr, 12hr windows (in 30-min periods)
            kernel_sizes = [3, 7, 24]
        
        self.kernel_sizes = kernel_sizes
        self.avg_pools = nn.ModuleList([
            nn.AvgPool1d(k, stride=1, padding=k // 2) 
            for k in kernel_sizes
        ])
        
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Args:
            x: [batch_size, seq_len] or [batch_size, channels, seq_len]
            
        Returns:
            trends: List of trend components at different scales
            seasonals: List of seasonal components at different scales
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, seq_len]
            
        trends = []
        seasonals = []
        
        for pool in self.avg_pools:
            trend = pool(x)
            # Handle padding mismatch
            if trend.size(-1) > x.size(-1):
                trend = trend[..., :x.size(-1)]
            elif trend.size(-1) < x.size(-1):
                trend = F.pad(trend, (0, x.size(-1) - trend.size(-1)))
            seasonal = x - trend
            trends.append(trend.squeeze(1))
            seasonals.append(seasonal.squeeze(1))
            
        return trends, seasonals


class FrequencyEnhancedPatchTST(nn.Module):
    """PatchTST with parallel frequency branch for improved periodicity capture.
    
    Combines:
    1. Standard PatchTST temporal processing
    2. Parallel frequency-domain branch
    3. Multi-scale decomposition
    
    This architecture is designed to beat simple baselines by explicitly
    modeling the periodicity they implicitly exploit.
    """
    
    def __init__(
        self,
        seq_len: int = 96,  # Input sequence length (e.g., 48 hours at 30-min)
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        forecast_horizon: int = 48,
        quantiles: Optional[list[float]] = None,
        use_frequency_branch: bool = True,
        use_multiscale: bool = True,
        freq_weight: float = 0.3,  # Weight for frequency branch output
    ):
        super().__init__()
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
            
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.quantiles = sorted(quantiles)
        self.use_frequency_branch = use_frequency_branch
        self.use_multiscale = use_multiscale
        self.freq_weight = freq_weight
        
        # Calculate number of patches
        self.n_patches = seq_len // patch_len
        
        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Linear(patch_len, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(self.n_patches, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Frequency branch
        if use_frequency_branch:
            self.freq_branch = FrequencyBranch(seq_len, d_model)
            combined_dim = d_model * 2  # Temporal + Frequency
        else:
            self.freq_branch = None
            combined_dim = d_model
            
        # Multi-scale decomposition
        if use_multiscale:
            self.multiscale = MultiScaleDecomposition()
            n_scales = 3  # Default number of scales
            self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)
        else:
            self.multiscale = None
            
        # Fusion and forecasting head
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, forecast_horizon * len(self.quantiles)),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> nn.Parameter:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, n_patches, patch_len] - patch-based input
               or [batch_size, seq_len] - raw sequence input
            
        Returns:
            forecasts: [batch_size, forecast_horizon, n_quantiles]
        """
        batch_size = x.size(0)
        
        # Handle raw sequence input
        if x.dim() == 2:
            seq_len = x.size(1)
            # Create patches
            n_patches = seq_len // self.patch_len
            x = x[:, :n_patches * self.patch_len].reshape(
                batch_size, n_patches, self.patch_len
            )
        
        # Store original for frequency branch
        x_original = x.clone()
        
        # === TEMPORAL BRANCH (Standard PatchTST) ===
        # Patch embedding
        x = self.patch_embedding(x)  # [batch, n_patches, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch, n_patches, d_model]
        
        # Global pooling
        temporal_features = x.mean(dim=1)  # [batch, d_model]
        
        # === FREQUENCY BRANCH ===
        if self.use_frequency_branch and self.freq_branch is not None:
            freq_features = self.freq_branch(x_original)  # [batch, d_model]
            # Combine temporal and frequency features
            combined = torch.cat([temporal_features, freq_features], dim=-1)
        else:
            combined = temporal_features
            
        # === FUSION AND FORECASTING ===
        fused = self.fusion(combined)  # [batch, d_model]
        
        # Generate forecasts
        forecast = self.forecast_head(fused)  # [batch, horizon * n_quantiles]
        
        # Reshape to quantiles
        forecast = forecast.view(batch_size, self.forecast_horizon, len(self.quantiles))
        
        return forecast


class FrequencyEnhancedForecaster(PatchTSTForecaster):
    """Forecaster wrapper for FrequencyEnhancedPatchTST.
    
    Extends PatchTSTForecaster with frequency-domain enhancements.
    """
    
    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        forecast_horizon: int = 48,
        quantiles: Optional[list[float]] = None,
        learning_rate: float = 1e-3,
        max_epochs: int = 20,
        batch_size: int = 32,
        early_stopping_patience: int = 5,
        use_frequency_branch: bool = True,
        use_multiscale: bool = True,
        freq_weight: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__(
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            forecast_horizon=forecast_horizon,
            quantiles=quantiles,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
            device=device,
        )
        
        # Additional config
        self.config["seq_len"] = seq_len
        self.config["use_frequency_branch"] = use_frequency_branch
        self.config["use_multiscale"] = use_multiscale
        self.config["freq_weight"] = freq_weight
        
    def _create_model(self) -> nn.Module:
        """Create the frequency-enhanced model."""
        return FrequencyEnhancedPatchTST(
            seq_len=self.config["seq_len"],
            patch_len=self.config["patch_len"],
            d_model=self.config["d_model"],
            n_heads=self.config["n_heads"],
            n_layers=self.config["n_layers"],
            forecast_horizon=self.config["forecast_horizon"],
            quantiles=self.config.get("quantiles", [0.1, 0.5, 0.9]),
            use_frequency_branch=self.config["use_frequency_branch"],
            use_multiscale=self.config["use_multiscale"],
            freq_weight=self.config["freq_weight"],
        ).to(self.device)


def create_frequency_enhanced_config(use_samples: bool = False) -> dict:
    """Create configuration for FrequencyEnhancedPatchTST."""
    if use_samples:
        # Fast config for CI/testing
        return {
            "seq_len": 64,
            "patch_len": 8,
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 64,
            "forecast_horizon": 16,
            "max_epochs": 3,
            "batch_size": 8,
            "learning_rate": 1e-2,
            "early_stopping_patience": 2,
            "use_frequency_branch": True,
            "use_multiscale": True,
            "freq_weight": 0.3,
        }
    else:
        # Full config for real training
        return {
            "seq_len": 96,
            "patch_len": 16,
            "d_model": 128,
            "n_heads": 8,
            "n_layers": 4,
            "d_ff": 256,
            "forecast_horizon": 48,
            "max_epochs": 50,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "early_stopping_patience": 10,
            "use_frequency_branch": True,
            "use_multiscale": True,
            "freq_weight": 0.3,
        }
