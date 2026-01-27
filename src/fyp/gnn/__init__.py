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
    - (Future) GATVerifier: GAT-based anomaly scoring model
    - (Future) TemporalEncoder: Encodes time-window features per node
"""

from fyp.gnn.graph_builder import GridGraphBuilder

__all__ = ["GridGraphBuilder"]
