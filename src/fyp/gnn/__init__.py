"""Graph Neural Network module for topology-aware anomaly detection.

This module implements GNN-based anomaly detection for UK distribution networks,
using SSEN grid topology to understand spatial relationships between substations,
feeders, and households.

Architecture:
    - Three-level node hierarchy: Substations -> Feeders -> Households
    - Uses PyTorch Geometric for efficient graph operations
    - GAT (Graph Attention Network) layers for message passing
    - Produces per-node anomaly scores based on learned patterns

Key Components:
    - GridGraphBuilder: Transforms SSEN metadata into PyG Data objects
    - TemporalEncoder: 1D-Conv encoder for time-window features per node
    - GATVerifier: GAT-based anomaly scoring model with oversmoothing prevention
    - SyntheticAnomalyDataset: Generates labeled training data with grid anomalies
    - AnomalyType: Enum of supported anomaly types (SPIKE, DROPOUT, CASCADE, etc.)
"""

from fyp.gnn.gat_verifier import GATVerifier
from fyp.gnn.graph_builder import GridGraphBuilder
from fyp.gnn.synthetic_dataset import AnomalyType, SyntheticAnomalyDataset
from fyp.gnn.temporal_encoder import TemporalEncoder

__all__ = [
    "GridGraphBuilder",
    "TemporalEncoder",
    "GATVerifier",
    "SyntheticAnomalyDataset",
    "AnomalyType",
]
