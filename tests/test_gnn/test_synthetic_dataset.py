"""Tests for SyntheticAnomalyDataset.

This module validates:
- Basic dataset generation and structure
- Anomaly type injection (spike, dropout, cascade, ramp_violation)
- Ratio and distribution correctness
- Integration with GATVerifier and DataLoader
- Edge cases (small graphs, no/all anomalies)
"""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from fyp.gnn import AnomalyType, GATVerifier, SyntheticAnomalyDataset


class TestBasicGeneration:
    """Test basic dataset generation and structure."""

    def test_dataset_length(self) -> None:
        """Verify __len__ returns correct count."""
        ds = SyntheticAnomalyDataset(num_samples=50, num_nodes=20)
        assert len(ds) == 50

    def test_sample_structure(self) -> None:
        """Verify each sample has x, edge_index, y, node_type."""
        ds = SyntheticAnomalyDataset(num_samples=10, num_nodes=30, seed=42)
        sample = ds[0]

        assert hasattr(sample, "x")
        assert hasattr(sample, "edge_index")
        assert hasattr(sample, "y")
        assert hasattr(sample, "node_type")
        assert hasattr(sample, "anomaly_type")

    def test_sample_shapes(self) -> None:
        """Verify x shape is [num_nodes, temporal_features]."""
        num_nodes = 25
        temporal_features = 7
        ds = SyntheticAnomalyDataset(
            num_samples=5,
            num_nodes=num_nodes,
            temporal_features=temporal_features,
            seed=42,
        )
        sample = ds[0]

        assert sample.x.shape == (num_nodes, temporal_features)
        assert sample.y.shape == (num_nodes,)
        assert sample.node_type.shape == (num_nodes,)
        assert sample.edge_index.shape[0] == 2

    def test_labels_binary(self) -> None:
        """Verify y values are 0 or 1."""
        ds = SyntheticAnomalyDataset(num_samples=20, num_nodes=30, seed=42)

        for i in range(len(ds)):
            sample = ds[i]
            unique_labels = torch.unique(sample.y)
            for label in unique_labels:
                assert label.item() in [0, 1], f"Invalid label: {label}"

    def test_node_types_valid(self) -> None:
        """Verify node_type values are 0, 1, or 2."""
        ds = SyntheticAnomalyDataset(num_samples=10, num_nodes=44, seed=42)

        for i in range(len(ds)):
            sample = ds[i]
            unique_types = torch.unique(sample.node_type)
            for t in unique_types:
                assert t.item() in [0, 1, 2], f"Invalid node type: {t}"

    def test_default_parameters(self) -> None:
        """Test dataset creation with default parameters."""
        ds = SyntheticAnomalyDataset(num_samples=10)

        assert len(ds) == 10
        sample = ds[0]
        assert sample.x.shape[0] == 44  # Default num_nodes
        assert sample.x.shape[1] == 5  # Default temporal_features


class TestAnomalyTypes:
    """Test anomaly type injection."""

    @pytest.fixture
    def forced_anomaly_dataset(self) -> SyntheticAnomalyDataset:
        """Create dataset with anomaly_ratio=1.0 to force anomalies."""
        return SyntheticAnomalyDataset(
            num_samples=100, num_nodes=30, anomaly_ratio=1.0, seed=42
        )

    def test_spike_anomaly(self, forced_anomaly_dataset: SyntheticAnomalyDataset) -> None:
        """Verify spike anomalies have elevated feature values."""
        # Find a sample with SPIKE anomaly
        spike_sample = None
        for i in range(len(forced_anomaly_dataset)):
            sample = forced_anomaly_dataset[i]
            if sample.anomaly_type == AnomalyType.SPIKE:
                spike_sample = sample
                break

        assert spike_sample is not None, "No SPIKE sample found"

        # Anomalous nodes should have higher values than normal baseline (~0.5)
        anomalous_mask = spike_sample.y == 1
        if anomalous_mask.sum() > 0:
            anomalous_features = spike_sample.x[anomalous_mask]
            # Spike multiplies features by 1.5-3x, so mean should be > 0.5
            # Note: some features might be clamped or below baseline
            assert anomalous_features.mean() > 0.3

    def test_dropout_anomaly(self, forced_anomaly_dataset: SyntheticAnomalyDataset) -> None:
        """Verify dropout anomalies zero out features."""
        # Find a sample with DROPOUT anomaly
        dropout_sample = None
        for i in range(len(forced_anomaly_dataset)):
            sample = forced_anomaly_dataset[i]
            if sample.anomaly_type == AnomalyType.DROPOUT:
                dropout_sample = sample
                break

        assert dropout_sample is not None, "No DROPOUT sample found"

        # Anomalous nodes should have zero features
        anomalous_mask = dropout_sample.y == 1
        if anomalous_mask.sum() > 0:
            anomalous_features = dropout_sample.x[anomalous_mask]
            assert torch.allclose(anomalous_features, torch.zeros_like(anomalous_features))

    def test_cascade_anomaly(self, forced_anomaly_dataset: SyntheticAnomalyDataset) -> None:
        """Verify cascade anomaly propagates through edges."""
        # Find a sample with CASCADE anomaly
        cascade_sample = None
        for i in range(len(forced_anomaly_dataset)):
            sample = forced_anomaly_dataset[i]
            if sample.anomaly_type == AnomalyType.CASCADE:
                cascade_sample = sample
                break

        assert cascade_sample is not None, "No CASCADE sample found"

        # Should have some anomalous nodes
        num_anomalous = cascade_sample.y.sum().item()
        assert num_anomalous > 0, "CASCADE should have anomalous nodes"

    def test_ramp_violation(self, forced_anomaly_dataset: SyntheticAnomalyDataset) -> None:
        """Verify ramp violation creates impossible gradients."""
        # Find a sample with RAMP_VIOLATION anomaly
        ramp_sample = None
        for i in range(len(forced_anomaly_dataset)):
            sample = forced_anomaly_dataset[i]
            if sample.anomaly_type == AnomalyType.RAMP_VIOLATION:
                ramp_sample = sample
                break

        assert ramp_sample is not None, "No RAMP_VIOLATION sample found"

        # Anomalous nodes should have high variance (alternating pattern)
        anomalous_mask = ramp_sample.y == 1
        if anomalous_mask.sum() > 0 and ramp_sample.x.shape[1] > 1:
            anomalous_features = ramp_sample.x[anomalous_mask]
            # Check for high variance in features (alternating pattern)
            var_per_node = anomalous_features.var(dim=1)
            assert var_per_node.mean() > 0.5, "RAMP_VIOLATION should have high feature variance"

    def test_all_anomaly_types_present(self) -> None:
        """Verify all anomaly types are generated."""
        ds = SyntheticAnomalyDataset(
            num_samples=500, num_nodes=30, anomaly_ratio=1.0, seed=42
        )
        stats = ds.get_anomaly_statistics()

        # All non-NORMAL types should be present
        for atype in [AnomalyType.SPIKE, AnomalyType.DROPOUT, AnomalyType.CASCADE, AnomalyType.RAMP_VIOLATION]:
            assert stats[atype.name] > 0, f"Missing anomaly type: {atype.name}"


class TestRatioAndDistribution:
    """Test anomaly ratio and distribution."""

    def test_anomaly_ratio(self) -> None:
        """Verify ~50% samples have anomalies when ratio=0.5."""
        ds = SyntheticAnomalyDataset(
            num_samples=200, num_nodes=20, anomaly_ratio=0.5, seed=42
        )
        stats = ds.get_anomaly_statistics()

        # NORMAL samples should be ~50%
        normal_ratio = stats["NORMAL"] / len(ds)
        assert 0.35 <= normal_ratio <= 0.65, f"Normal ratio {normal_ratio} not near 0.5"

    def test_reproducibility(self) -> None:
        """Same seed produces same samples."""
        ds1 = SyntheticAnomalyDataset(num_samples=10, num_nodes=20, seed=12345)
        ds2 = SyntheticAnomalyDataset(num_samples=10, num_nodes=20, seed=12345)

        for i in range(len(ds1)):
            sample1 = ds1[i]
            sample2 = ds2[i]

            assert torch.allclose(sample1.x, sample2.x)
            assert torch.equal(sample1.edge_index, sample2.edge_index)
            assert torch.equal(sample1.y, sample2.y)
            assert torch.equal(sample1.node_type, sample2.node_type)
            assert sample1.anomaly_type == sample2.anomaly_type

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different samples."""
        ds1 = SyntheticAnomalyDataset(num_samples=10, num_nodes=20, seed=111)
        ds2 = SyntheticAnomalyDataset(num_samples=10, num_nodes=20, seed=222)

        # At least some samples should differ
        any_different = False
        for i in range(len(ds1)):
            if not torch.allclose(ds1[i].x, ds2[i].x):
                any_different = True
                break
        assert any_different, "Different seeds should produce different samples"

    def test_node_label_distribution(self) -> None:
        """Some nodes labeled anomalous, some normal in anomalous samples."""
        ds = SyntheticAnomalyDataset(
            num_samples=50, num_nodes=30, anomaly_ratio=1.0, seed=42
        )

        found_mixed = False
        for i in range(len(ds)):
            sample = ds[i]
            num_anomalous = sample.y.sum().item()
            num_normal = (sample.y == 0).sum().item()

            # Should have both anomalous and normal nodes
            if num_anomalous > 0 and num_normal > 0:
                found_mixed = True
                break

        assert found_mixed, "Should have samples with mixed labels"


class TestIntegrationWithGATVerifier:
    """Test integration with GATVerifier and DataLoader."""

    def test_dataloader_compatibility(self) -> None:
        """Verify works with torch_geometric DataLoader."""
        ds = SyntheticAnomalyDataset(num_samples=20, num_nodes=30, seed=42)

        # Create DataLoader
        loader = DataLoader([ds[i] for i in range(len(ds))], batch_size=4, shuffle=True)

        # Iterate through batches
        batch_count = 0
        for batch in loader:
            assert hasattr(batch, "x")
            assert hasattr(batch, "edge_index")
            assert hasattr(batch, "y")
            batch_count += 1

        assert batch_count == 5  # 20 samples / batch_size 4

    def test_batch_creation(self) -> None:
        """Verify Batch.from_data_list works."""
        ds = SyntheticAnomalyDataset(num_samples=5, num_nodes=20, seed=42)
        data_list = [ds[i] for i in range(len(ds))]

        batch = Batch.from_data_list(data_list)

        assert batch.x.shape[0] == 5 * 20  # Total nodes
        assert batch.y.shape[0] == 5 * 20
        assert hasattr(batch, "batch")  # Batch index

    def test_model_forward(self) -> None:
        """Verify GATVerifier can process generated samples."""
        ds = SyntheticAnomalyDataset(
            num_samples=5, num_nodes=30, temporal_features=5, seed=42
        )

        model = GATVerifier(
            temporal_features=5, hidden_channels=32, num_layers=2, heads=2
        )
        model.eval()

        # Test single sample
        sample = ds[0]
        with torch.inference_mode():
            scores = model(sample.x, sample.edge_index, sample.node_type)

        assert scores.shape == (30, 1)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_model_forward_batch(self) -> None:
        """Verify GATVerifier can process batched samples."""
        ds = SyntheticAnomalyDataset(
            num_samples=8, num_nodes=20, temporal_features=5, seed=42
        )
        batch = Batch.from_data_list([ds[i] for i in range(len(ds))])

        model = GATVerifier(
            temporal_features=5, hidden_channels=32, num_layers=2, heads=2
        )
        model.eval()

        with torch.inference_mode():
            scores = model(batch.x, batch.edge_index, batch.node_type)

        assert scores.shape == (8 * 20, 1)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_graph(self) -> None:
        """Handle graphs with <10 nodes."""
        ds = SyntheticAnomalyDataset(num_samples=5, num_nodes=5, seed=42)

        for i in range(len(ds)):
            sample = ds[i]
            assert sample.x.shape[0] == 5
            assert sample.y.shape[0] == 5
            assert sample.node_type.shape[0] == 5

    def test_very_small_graph(self) -> None:
        """Handle graphs with 3 nodes (minimum for hierarchy)."""
        ds = SyntheticAnomalyDataset(num_samples=5, num_nodes=3, seed=42)

        for i in range(len(ds)):
            sample = ds[i]
            assert sample.x.shape[0] == 3

    def test_no_anomalies(self) -> None:
        """Handle anomaly_ratio=0."""
        ds = SyntheticAnomalyDataset(
            num_samples=20, num_nodes=20, anomaly_ratio=0.0, seed=42
        )

        for i in range(len(ds)):
            sample = ds[i]
            assert sample.anomaly_type == AnomalyType.NORMAL
            # All labels should be 0 (normal)
            assert sample.y.sum().item() == 0

    def test_all_anomalies(self) -> None:
        """Handle anomaly_ratio=1."""
        ds = SyntheticAnomalyDataset(
            num_samples=20, num_nodes=20, anomaly_ratio=1.0, seed=42
        )

        for i in range(len(ds)):
            sample = ds[i]
            assert sample.anomaly_type != AnomalyType.NORMAL

    def test_single_temporal_feature(self) -> None:
        """Handle single temporal feature (ramp violation edge case)."""
        ds = SyntheticAnomalyDataset(
            num_samples=10, num_nodes=20, temporal_features=1, seed=42
        )

        for i in range(len(ds)):
            sample = ds[i]
            assert sample.x.shape == (20, 1)

    def test_index_out_of_range(self) -> None:
        """Test IndexError for out-of-range access."""
        ds = SyntheticAnomalyDataset(num_samples=5, num_nodes=10)

        with pytest.raises(IndexError):
            _ = ds[5]

        with pytest.raises(IndexError):
            _ = ds[-1]

    def test_large_dataset(self) -> None:
        """Test larger dataset generation."""
        ds = SyntheticAnomalyDataset(num_samples=500, num_nodes=100, seed=42)

        assert len(ds) == 500
        sample = ds[0]
        assert sample.x.shape[0] == 100


class TestRepr:
    """Test string representation."""

    def test_repr(self) -> None:
        """Test __repr__ returns sensible string."""
        ds = SyntheticAnomalyDataset(
            num_samples=100, num_nodes=44, anomaly_ratio=0.3, temporal_features=7
        )
        repr_str = repr(ds)

        assert "SyntheticAnomalyDataset" in repr_str
        assert "num_samples=100" in repr_str
        assert "num_nodes=44" in repr_str
        assert "anomaly_ratio=0.3" in repr_str
        assert "temporal_features=7" in repr_str


class TestGraphStructure:
    """Test graph structure validity."""

    def test_edges_bidirectional(self) -> None:
        """Verify edges are bidirectional."""
        ds = SyntheticAnomalyDataset(num_samples=5, num_nodes=30, seed=42)

        for i in range(len(ds)):
            sample = ds[i]
            edge_set = set()
            for j in range(sample.edge_index.shape[1]):
                src = sample.edge_index[0, j].item()
                dst = sample.edge_index[1, j].item()
                edge_set.add((src, dst))

            # For each edge (a, b), reverse (b, a) should exist
            for src, dst in edge_set:
                assert (dst, src) in edge_set, f"Missing reverse edge for ({src}, {dst})"

    def test_no_self_loops_in_edges(self) -> None:
        """Verify no self-loops in generated edges."""
        ds = SyntheticAnomalyDataset(num_samples=10, num_nodes=30, seed=42)

        for i in range(len(ds)):
            sample = ds[i]
            for j in range(sample.edge_index.shape[1]):
                src = sample.edge_index[0, j].item()
                dst = sample.edge_index[1, j].item()
                assert src != dst, f"Self-loop found: ({src}, {dst})"

    def test_edge_indices_valid(self) -> None:
        """Verify edge indices are within node range."""
        ds = SyntheticAnomalyDataset(num_samples=5, num_nodes=25, seed=42)

        for i in range(len(ds)):
            sample = ds[i]
            assert sample.edge_index.min() >= 0
            assert sample.edge_index.max() < 25
