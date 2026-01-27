"""GAT-based anomaly verifier with oversmoothing prevention.

This module implements the core GNN model for topology-aware anomaly detection
in UK distribution networks. It uses GATv2Conv (not GATConv) for dynamic
attention, following research recommendations that GATv2 solves the "static
attention problem" and allows dynamic attention rankings conditional on query nodes.

The model implements GCNII-style initial residual connections combined with
LayerNorm to prevent oversmoothing in the 3-layer architecture.

Architecture:
    1. Temporal encoding (1D-Conv) per node
    2. Optional node type embedding
    3. 3x GATv2Conv layers with residuals + LayerNorm
    4. Output head -> per-node anomaly score in [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from fyp.gnn.temporal_encoder import TemporalEncoder


class GATVerifier(nn.Module):
    """GAT-based anomaly verifier with oversmoothing prevention.

    Uses GATv2Conv (not GATConv) for dynamic attention. Implements GCNII-style
    initial residual connections to prevent oversmoothing in the 3-layer
    architecture.

    The model takes temporal features per node, optionally adds learnable type
    embeddings, and produces per-node anomaly scores in [0, 1].

    Example:
        >>> model = GATVerifier(temporal_features=5, hidden_channels=64)
        >>> model.eval()
        >>> x = torch.randn(100, 5)  # 100 nodes, 5 temporal features
        >>> edge_index = torch.randint(0, 100, (2, 300))
        >>> scores = model(x, edge_index)  # [100, 1]

    Attributes:
        temporal_features: Number of input temporal features
        hidden_channels: Hidden dimension size
        num_layers: Number of GAT layers
        heads: Number of attention heads
    """

    def __init__(
        self,
        temporal_features: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        num_node_types: int = 3,
    ) -> None:
        """Initialize verifier.

        Args:
            temporal_features: Number of input temporal features
            hidden_channels: Hidden dimension (default 64 per CONTEXT.md)
            num_layers: Number of GAT layers (default 3 per CONTEXT.md)
            heads: Attention heads per layer (default 4 per research)
            dropout: Dropout rate
            num_node_types: Number of node types for embedding
                (default 3: substation/feeder/household)
        """
        super().__init__()
        self.temporal_features = temporal_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.num_node_types = num_node_types

        # Temporal encoder (1D-Conv)
        self.temporal_encoder = TemporalEncoder(
            input_features=temporal_features,
            embed_dim=hidden_channels,
            dropout=dropout,
        )

        # Node type embedding (16-dim)
        self.type_embed_dim = 16
        self.type_embed = nn.Embedding(num_node_types, self.type_embed_dim)

        # Input projection: [temporal_embed + type_embed] -> hidden_channels
        # If no type provided, just uses temporal embed
        self.input_proj = nn.Linear(hidden_channels + self.type_embed_dim, hidden_channels)
        self.input_proj_no_type = nn.Linear(hidden_channels, hidden_channels)

        # GAT layers with GATv2Conv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # GATv2Conv with residual support
            # Output dim = out_channels * heads (when concat=True)
            # So we set out_channels = hidden_channels // heads
            conv = GATv2Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                concat=True,  # Concatenate head outputs (total = hidden_channels)
                dropout=dropout,
                add_self_loops=True,
                residual=True,  # Built-in residual connection
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Learnable initial residual weight (GCNII-style)
        # alpha controls balance between current layer output and initial features
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # Output head for per-node anomaly scores
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid(),  # Anomaly score in [0, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layer weights."""
        for module in [self.input_proj, self.input_proj_no_type]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        for module in self.output_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize type embeddings
        nn.init.normal_(self.type_embed.weight, mean=0, std=0.1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node temporal features [num_nodes, temporal_features]
            edge_index: Graph connectivity [2, num_edges]
            node_type: Node type indices [num_nodes] (optional)

        Returns:
            Anomaly scores [num_nodes, 1] in range [0, 1]
        """
        # Step 1: Temporal encoding
        # [N, temporal_features] -> [N, hidden_channels]
        h = self.temporal_encoder(x)

        # Step 2: Add node type embedding if provided
        if node_type is not None:
            # [N] -> [N, type_embed_dim]
            type_features = self.type_embed(node_type)
            # Concatenate and project: [N, hidden_channels + type_embed_dim]
            h = torch.cat([h, type_features], dim=-1)
            h = self.input_proj(h)
        else:
            # Just project temporal embedding
            h = self.input_proj_no_type(h)

        # Store initial features for GCNII-style residual
        h0 = h

        # Step 3: GATv2Conv layers with oversmoothing prevention
        for conv, norm in zip(self.convs, self.norms):
            # GAT layer (has internal residual via residual=True)
            h_new = conv(h, edge_index)

            # Initial residual (GCNII-style oversmoothing prevention)
            # h = alpha * h_conv + (1 - alpha) * h0
            h_new = self.alpha * h_new + (1 - self.alpha) * h0

            # Layer normalization
            h = norm(h_new)

        # Step 4: Output head for per-node anomaly scores
        return self.output_head(h)

    def extra_repr(self) -> str:
        """Extra representation for printing."""
        return (
            f"temporal_features={self.temporal_features}, "
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"heads={self.heads}"
        )
