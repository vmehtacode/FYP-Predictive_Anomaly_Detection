"""1D-Conv temporal encoder for GNN input preprocessing.

This module implements a temporal feature encoder that transforms per-node
time-window features (e.g., [current_load, avg_24h, peak_7d]) into dense
embeddings suitable for GNN processing.

The encoder uses 1D convolution for efficient local pattern extraction,
which is faster than LSTM and better captures local temporal patterns
in fixed time-window snapshots.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """1D-Conv encoder for fixed time-window features.

    Processes temporal features like [current_load, avg_24h, peak_7d]
    into a dense embedding per node. Uses 1D convolution for efficient
    local pattern extraction (faster than LSTM, per research).

    For very small feature sets (< 3 features), falls back to a simple
    linear projection since convolution requires minimum spatial extent.

    Example:
        >>> encoder = TemporalEncoder(input_features=5, embed_dim=64)
        >>> x = torch.randn(100, 5)  # 100 nodes, 5 temporal features
        >>> embeddings = encoder(x)  # [100, 64]

    Attributes:
        input_features: Number of input temporal features
        embed_dim: Output embedding dimension
        use_conv: Whether conv pipeline is used (True if input_features >= 3)
    """

    def __init__(
        self,
        input_features: int,
        embed_dim: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Initialize encoder.

        Args:
            input_features: Number of temporal features (e.g., 3 for current/avg/peak)
            embed_dim: Output embedding dimension (default 64, matches GNN hidden)
            kernel_size: Conv kernel size (default 3)
            dropout: Dropout rate (default 0.1)
        """
        super().__init__()
        self.input_features = input_features
        self.embed_dim = embed_dim
        self.use_conv = input_features >= 3

        if self.use_conv:
            # 1D convolution pipeline
            # Kernel size must not exceed input size
            effective_kernel = min(kernel_size, input_features)

            self.conv1 = nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=effective_kernel,
                padding="same",
            )
            self.conv2 = nn.Conv1d(
                in_channels=32,
                out_channels=embed_dim,
                kernel_size=effective_kernel,
                padding="same",
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.dropout = nn.Dropout(dropout)
        else:
            # Linear fallback for very small feature sets
            self.linear = nn.Linear(input_features, embed_dim)
            self.dropout = nn.Dropout(dropout)

        # LayerNorm at output for stable gradient flow
        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode temporal features.

        Args:
            x: [num_nodes, num_temporal_features]

        Returns:
            [num_nodes, embed_dim]
        """
        if not self.use_conv:
            # Linear fallback path
            h = self.linear(x)
            h = torch.relu(h)
            h = self.dropout(h)
            return self.norm(h)

        # Conv pipeline path
        # [N, F] -> [N, 1, F] for Conv1d
        h = x.unsqueeze(1)

        # First conv block
        h = self.conv1(h)
        h = torch.relu(h)
        h = self.dropout(h)

        # Second conv block
        h = self.conv2(h)
        h = torch.relu(h)
        h = self.dropout(h)

        # Adaptive pooling to single value per channel
        # [N, embed_dim, F] -> [N, embed_dim, 1]
        h = self.pool(h)

        # Squeeze last dim: [N, embed_dim, 1] -> [N, embed_dim]
        h = h.squeeze(-1)

        # Final layer normalization
        return self.norm(h)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"input_features={self.input_features}, "
            f"embed_dim={self.embed_dim}, "
            f"use_conv={self.use_conv}"
        )
