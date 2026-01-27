"""Synthetic anomaly dataset generator for GNN training.

This module provides a reusable dataset generator that creates labeled PyG Data
objects with realistic grid anomalies for training and evaluating the GATVerifier.

The generator supports multiple anomaly types matching SSEN grid scenarios:
- SPIKE: Sudden consumption increase (EV charging, industrial load)
- DROPOUT: Zero consumption periods (outage, meter failure)
- CASCADE: Anomaly propagating through connected nodes (cold snap effect)
- RAMP_VIOLATION: Physically impossible consumption changes
- NORMAL: Baseline non-anomalous data (negative examples)

Physics-aware generation uses SSEN-style constraints for realistic bounds,
respecting the three-level node hierarchy (substation/feeder/household).
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies matching SSEN grid scenarios.

    Each type represents a realistic failure mode or unusual consumption pattern
    that the GATVerifier should learn to detect.
    """

    NORMAL = auto()
    """Baseline non-anomalous data (negative examples)."""

    SPIKE = auto()
    """Sudden consumption increase (e.g., EV charging, industrial load)."""

    DROPOUT = auto()
    """Zero consumption periods (outage, meter failure)."""

    CASCADE = auto()
    """Anomaly propagating through connected nodes (cold snap effect)."""

    RAMP_VIOLATION = auto()
    """Physically impossible consumption changes."""


class SyntheticAnomalyDataset:
    """Generate labeled PyG Data objects for GNN anomaly detection training.

    This dataset generates graph-structured data with realistic grid anomalies,
    providing labeled training/validation data for the GATVerifier model.

    The generator creates graphs with the SSEN three-level hierarchy:
    - Type 0: Primary substations (highest level)
    - Type 1: Secondary substations / feeders (mid level)
    - Type 2: LV feeders (lowest level, connect to households)

    Example:
        >>> dataset = SyntheticAnomalyDataset(num_samples=100, num_nodes=44)
        >>> sample = dataset[0]
        >>> print(f"Nodes: {sample.x.shape[0]}, Anomaly labels: {sample.y.sum()}")

    Attributes:
        num_samples: Total number of samples to generate
        num_nodes: Nodes per graph
        anomaly_ratio: Fraction of samples containing anomalies
        temporal_features: Input feature dimension
        seed: Random seed for reproducibility
    """

    # Node type constants (match GridGraphBuilder)
    NODE_TYPE_PRIMARY = 0
    NODE_TYPE_SECONDARY = 1
    NODE_TYPE_LV_FEEDER = 2

    # Anomaly parameters
    SPIKE_MAGNITUDE_RANGE = (1.5, 3.0)
    CASCADE_DECAY = 0.7
    RAMP_VIOLATION_MAGNITUDE = 5.0

    def __init__(
        self,
        num_samples: int,
        num_nodes: int = 44,
        anomaly_ratio: float = 0.5,
        temporal_features: int = 5,
        seed: int | None = None,
    ) -> None:
        """Initialize the synthetic dataset generator.

        Args:
            num_samples: Total samples to generate
            num_nodes: Nodes per graph (default 44, matches SSEN test data)
            anomaly_ratio: Fraction of samples with anomalies (default 0.5)
            temporal_features: Input feature dimension (default 5 for time-series)
            seed: Random seed for reproducibility (default None for random)
        """
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.anomaly_ratio = anomaly_ratio
        self.temporal_features = temporal_features
        self.seed = seed

        # Set up random generator
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

        # Pre-generate samples for consistency
        self._samples: list[Data] = []
        self._generate_all_samples()

    def _generate_all_samples(self) -> None:
        """Pre-generate all samples for the dataset."""
        self._samples = []
        for _ in range(self.num_samples):
            sample = self._generate_sample()
            self._samples.append(sample)

    def _generate_sample(self) -> Data:
        """Generate one labeled sample with optional anomaly.

        Returns:
            PyG Data object with x, edge_index, y, node_type, anomaly_type
        """
        # Generate graph structure
        edge_index, node_type = self._generate_graph_structure()

        # Generate baseline features (normal consumption patterns)
        x = self._generate_baseline_features()

        # Initialize labels (all normal)
        y = torch.zeros(self.num_nodes, dtype=torch.long)

        # Determine if this sample should have anomalies
        has_anomaly = torch.rand(1, generator=self._rng).item() < self.anomaly_ratio

        if has_anomaly:
            # Select anomaly type (exclude NORMAL)
            anomaly_types = [
                AnomalyType.SPIKE,
                AnomalyType.DROPOUT,
                AnomalyType.CASCADE,
                AnomalyType.RAMP_VIOLATION,
            ]
            type_idx = int(torch.randint(0, len(anomaly_types), (1,), generator=self._rng).item())
            anomaly_type = anomaly_types[type_idx]

            # Select nodes to affect (10-30% of nodes)
            num_affected = max(1, int(self.num_nodes * (0.1 + 0.2 * torch.rand(1, generator=self._rng).item())))
            affected_indices = torch.randperm(self.num_nodes, generator=self._rng)[:num_affected]
            node_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
            node_mask[affected_indices] = True

            # Inject anomaly
            x, y = self._inject_anomaly(x, y, edge_index, node_mask, node_type, anomaly_type)
        else:
            anomaly_type = AnomalyType.NORMAL

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            node_type=node_type,
            num_nodes=self.num_nodes,
        )

        # Store anomaly type as attribute
        data.anomaly_type = anomaly_type

        return data

    def _generate_graph_structure(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a realistic grid graph structure.

        Creates a three-level hierarchy matching SSEN topology:
        - ~10% primary substations
        - ~20% secondary substations
        - ~70% LV feeders

        Returns:
            Tuple of (edge_index, node_type)
        """
        # Allocate nodes to types (roughly matching SSEN ratios)
        num_primary = max(1, int(self.num_nodes * 0.1))
        num_secondary = max(1, int(self.num_nodes * 0.2))
        num_lv = self.num_nodes - num_primary - num_secondary

        # Build node type tensor
        node_type = torch.cat([
            torch.full((num_primary,), self.NODE_TYPE_PRIMARY, dtype=torch.long),
            torch.full((num_secondary,), self.NODE_TYPE_SECONDARY, dtype=torch.long),
            torch.full((num_lv,), self.NODE_TYPE_LV_FEEDER, dtype=torch.long),
        ])

        # Build edges (bidirectional)
        edges: list[list[int]] = []

        # Primary to secondary connections
        # Each secondary connects to one primary
        for s_idx in range(num_primary, num_primary + num_secondary):
            p_idx = int(torch.randint(0, num_primary, (1,), generator=self._rng).item())
            edges.append([p_idx, s_idx])
            edges.append([s_idx, p_idx])

        # Secondary to LV feeder connections
        # Each LV feeder connects to one secondary
        lv_start = num_primary + num_secondary
        for lv_idx in range(lv_start, self.num_nodes):
            s_idx = num_primary + int(torch.randint(0, num_secondary, (1,), generator=self._rng).item())
            edges.append([s_idx, lv_idx])
            edges.append([lv_idx, s_idx])

        # Convert to COO format
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return edge_index, node_type

    def _generate_baseline_features(self) -> torch.Tensor:
        """Generate baseline (normal) consumption features.

        Returns:
            Feature tensor [num_nodes, temporal_features]
        """
        # Normal consumption: Gaussian with small variance
        # Mean around 0.5 (normalized consumption)
        x = 0.5 + 0.2 * torch.randn(self.num_nodes, self.temporal_features, generator=self._rng)

        # Clamp to realistic range [0, 1]
        x = torch.clamp(x, 0.0, 1.0)

        return x

    def _inject_anomaly(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
        node_type: torch.Tensor,
        anomaly_type: AnomalyType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject anomaly into features and set labels.

        Args:
            x: Feature tensor
            y: Label tensor
            edge_index: Graph connectivity
            node_mask: Boolean mask of affected nodes
            node_type: Node type tensor
            anomaly_type: Type of anomaly to inject

        Returns:
            Tuple of (modified_x, modified_y)
        """
        x = x.clone()
        y = y.clone()

        if anomaly_type == AnomalyType.SPIKE:
            x, y = self._inject_spike(x, y, node_mask, node_type)
        elif anomaly_type == AnomalyType.DROPOUT:
            x, y = self._inject_dropout(x, y, node_mask)
        elif anomaly_type == AnomalyType.CASCADE:
            x, y = self._inject_cascade(x, y, edge_index, node_mask)
        elif anomaly_type == AnomalyType.RAMP_VIOLATION:
            x, y = self._inject_ramp_violation(x, y, node_mask)

        return x, y

    def _inject_spike(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
        node_type: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject spike anomaly (sudden consumption increase).

        Spike magnitude varies by node type:
        - Substations: Lower magnitude (aggregated loads smooth out)
        - LV feeders: Higher magnitude (direct consumer impact)

        Args:
            x: Feature tensor
            y: Label tensor
            node_mask: Nodes to affect
            node_type: Node types for magnitude scaling

        Returns:
            Tuple of (modified_x, modified_y)
        """
        # Generate random magnitude per affected node
        num_affected = node_mask.sum().item()
        min_mag, max_mag = self.SPIKE_MAGNITUDE_RANGE
        magnitudes = min_mag + (max_mag - min_mag) * torch.rand(num_affected, generator=self._rng)

        # Scale by node type (LV feeders get higher spikes)
        affected_types = node_type[node_mask]
        type_scale = torch.where(
            affected_types == self.NODE_TYPE_LV_FEEDER,
            torch.tensor(1.2),
            torch.tensor(0.8),
        )
        magnitudes = magnitudes * type_scale

        # Apply spike (multiply features)
        x[node_mask] = x[node_mask] * magnitudes.unsqueeze(1)

        # Set labels
        y[node_mask] = 1

        return x, y

    def _inject_dropout(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject dropout anomaly (zero consumption).

        Simulates outage or meter failure by zeroing all features.

        Args:
            x: Feature tensor
            y: Label tensor
            node_mask: Nodes to affect

        Returns:
            Tuple of (modified_x, modified_y)
        """
        # Zero out features for affected nodes
        x[node_mask] = 0.0

        # Set labels
        y[node_mask] = 1

        return x, y

    def _inject_cascade(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject cascade anomaly (propagating through neighbors).

        Simulates effects like cold snaps that propagate through the network.
        Anomaly starts at seed nodes and decays as it propagates.

        Args:
            x: Feature tensor
            y: Label tensor
            edge_index: Graph connectivity
            node_mask: Starting nodes for cascade

        Returns:
            Tuple of (modified_x, modified_y)
        """
        # Build adjacency for neighbor lookup
        adj: dict[int, set[int]] = {}
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        for s, d in zip(src, dst):
            if s not in adj:
                adj[s] = set()
            adj[s].add(d)

        # Start with seed nodes
        affected = set(torch.where(node_mask)[0].tolist())
        current_magnitude = 2.0  # Initial spike magnitude

        # Propagate through 2 hops with decay
        for _ in range(2):
            new_affected: set[int] = set()
            for node in affected:
                if node in adj:
                    for neighbor in adj[node]:
                        if neighbor not in affected:
                            new_affected.add(neighbor)

            # Apply decayed anomaly to new nodes
            if new_affected:
                current_magnitude *= self.CASCADE_DECAY
                new_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
                for idx in new_affected:
                    new_mask[idx] = True
                x[new_mask] = x[new_mask] * current_magnitude

            affected.update(new_affected)

        # Apply initial spike to seed nodes
        x[node_mask] = x[node_mask] * 2.0

        # All affected nodes are anomalous
        all_affected = torch.zeros(self.num_nodes, dtype=torch.bool)
        for idx in affected:
            all_affected[idx] = True
        y[all_affected] = 1

        return x, y

    def _inject_ramp_violation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject ramp violation (impossible consumption gradient).

        Creates features that change too rapidly between time steps,
        violating physical constraints of power consumption.

        Args:
            x: Feature tensor
            y: Label tensor
            node_mask: Nodes to affect

        Returns:
            Tuple of (modified_x, modified_y)
        """
        if self.temporal_features < 2:
            # Can't create gradient with single feature
            return self._inject_spike(x, y, node_mask, torch.zeros(self.num_nodes, dtype=torch.long))

        # Create impossible gradient: large jump between consecutive features
        num_affected = node_mask.sum().item()

        # Alternate between very low and very high values
        pattern = torch.zeros(num_affected, self.temporal_features)
        for t in range(self.temporal_features):
            if t % 2 == 0:
                pattern[:, t] = 0.1  # Low
            else:
                pattern[:, t] = 0.1 + self.RAMP_VIOLATION_MAGNITUDE * torch.rand(num_affected, generator=self._rng)

        x[node_mask] = pattern

        # Set labels
        y[node_mask] = 1

        return x, y

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        """Get a sample by index.

        Args:
            idx: Sample index

        Returns:
            PyG Data object with x, edge_index, y, node_type, anomaly_type

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        return self._samples[idx]

    def get_anomaly_statistics(self) -> dict[str, int]:
        """Get statistics about anomaly types in the dataset.

        Returns:
            Dict mapping anomaly type names to counts
        """
        stats: dict[str, int] = {t.name: 0 for t in AnomalyType}
        for sample in self._samples:
            stats[sample.anomaly_type.name] += 1
        return stats

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SyntheticAnomalyDataset("
            f"num_samples={self.num_samples}, "
            f"num_nodes={self.num_nodes}, "
            f"anomaly_ratio={self.anomaly_ratio}, "
            f"temporal_features={self.temporal_features})"
        )
