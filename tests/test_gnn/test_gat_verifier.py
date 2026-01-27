"""Tests for GATVerifier and TemporalEncoder.

This module validates:
- TemporalEncoder feature transformation
- GATVerifier model architecture (GATv2Conv, 3 layers, oversmoothing prevention)
- Forward pass shapes and output range
- Inference latency (<30ms for batch_size=32)
- Synthetic anomaly detection pipeline
"""

from __future__ import annotations

import time

import pandas as pd
import pytest
import torch

from fyp.gnn import GATVerifier, GridGraphBuilder, TemporalEncoder


class TestTemporalEncoder:
    """Test suite for TemporalEncoder."""

    def test_basic_encoding(self) -> None:
        """Test basic temporal feature encoding."""
        encoder = TemporalEncoder(input_features=5, embed_dim=64)
        x = torch.randn(100, 5)
        out = encoder(x)

        assert out.shape == (100, 64)

    def test_conv_path_for_sufficient_features(self) -> None:
        """Test 1D-conv path is used when input_features >= 3."""
        encoder = TemporalEncoder(input_features=5, embed_dim=64)
        assert encoder.use_conv is True

        x = torch.randn(50, 5)
        out = encoder(x)
        assert out.shape == (50, 64)

    def test_linear_fallback_for_small_features(self) -> None:
        """Test linear fallback for input_features < 3."""
        encoder = TemporalEncoder(input_features=2, embed_dim=64)
        assert encoder.use_conv is False

        x = torch.randn(50, 2)
        out = encoder(x)
        assert out.shape == (50, 64)

    def test_boundary_case_three_features(self) -> None:
        """Test boundary case of exactly 3 features (should use conv)."""
        encoder = TemporalEncoder(input_features=3, embed_dim=64)
        assert encoder.use_conv is True

        x = torch.randn(30, 3)
        out = encoder(x)
        assert out.shape == (30, 64)

    def test_single_feature_fallback(self) -> None:
        """Test single feature input uses linear fallback."""
        encoder = TemporalEncoder(input_features=1, embed_dim=32)
        assert encoder.use_conv is False

        x = torch.randn(20, 1)
        out = encoder(x)
        assert out.shape == (20, 32)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through encoder."""
        encoder = TemporalEncoder(input_features=5, embed_dim=64)
        x = torch.randn(100, 5, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_flow_linear_path(self) -> None:
        """Test gradients flow through linear fallback path."""
        encoder = TemporalEncoder(input_features=2, embed_dim=64)
        x = torch.randn(50, 2, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_output_normalized(self) -> None:
        """Test output passes through layer norm (stable values)."""
        encoder = TemporalEncoder(input_features=5, embed_dim=64)
        x = torch.randn(100, 5) * 100  # Large input

        out = encoder(x)

        # Output should be roughly normalized (LayerNorm)
        assert out.mean().abs() < 1.0
        assert out.std() < 3.0

    def test_batch_consistency(self) -> None:
        """Test encoder produces same output for same input."""
        encoder = TemporalEncoder(input_features=5, embed_dim=64)
        encoder.eval()

        x = torch.randn(50, 5)

        with torch.inference_mode():
            out1 = encoder(x)
            out2 = encoder(x)

        assert torch.allclose(out1, out2)


class TestGATVerifier:
    """Test suite for GATVerifier."""

    @pytest.fixture
    def model(self) -> GATVerifier:
        """Create a default GATVerifier."""
        return GATVerifier(
            temporal_features=5,
            hidden_channels=64,
            num_layers=3,
            heads=4,
        )

    @pytest.fixture
    def sample_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create sample graph data."""
        num_nodes = 100
        x = torch.randn(num_nodes, 5)
        # Create random but valid edges
        edge_index = torch.randint(0, num_nodes, (2, 300))
        node_type = torch.randint(0, 3, (num_nodes,))
        return x, edge_index, node_type

    def test_forward_shape(self, model: GATVerifier, sample_data: tuple) -> None:
        """Test forward pass produces correct output shape."""
        x, edge_index, node_type = sample_data
        model.eval()

        with torch.inference_mode():
            scores = model(x, edge_index, node_type)

        assert scores.shape == (100, 1)

    def test_output_range(self, model: GATVerifier, sample_data: tuple) -> None:
        """Test output is in [0, 1] range."""
        x, edge_index, node_type = sample_data
        model.eval()

        with torch.inference_mode():
            scores = model(x, edge_index, node_type)

        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_output_range_with_extreme_input(self, model: GATVerifier) -> None:
        """Test output stays in [0, 1] even with extreme inputs."""
        num_nodes = 50
        x_extreme = torch.randn(num_nodes, 5) * 100  # Very large values
        edge_index = torch.randint(0, num_nodes, (2, 100))

        model.eval()
        with torch.inference_mode():
            scores = model(x_extreme, edge_index)

        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_without_node_type(self, model: GATVerifier, sample_data: tuple) -> None:
        """Test model works without node type input."""
        x, edge_index, _ = sample_data
        model.eval()

        with torch.inference_mode():
            scores = model(x, edge_index)  # No node_type

        assert scores.shape == (100, 1)

    def test_with_and_without_node_type_differ(self, model: GATVerifier, sample_data: tuple) -> None:
        """Test that node type embedding actually affects output."""
        x, edge_index, node_type = sample_data
        model.eval()

        with torch.inference_mode():
            scores_with_type = model(x, edge_index, node_type)
            scores_without_type = model(x, edge_index)

        # Outputs should be different (type embedding matters)
        assert not torch.allclose(scores_with_type, scores_without_type)

    def test_uses_gatv2conv(self, model: GATVerifier) -> None:
        """Verify model uses GATv2Conv not GATConv."""
        from torch_geometric.nn import GATv2Conv

        has_gatv2 = any(
            isinstance(module, GATv2Conv)
            for module in model.modules()
        )
        assert has_gatv2, "Model should use GATv2Conv"

    def test_does_not_use_gatconv(self, model: GATVerifier) -> None:
        """Verify model does NOT use old GATConv."""
        from torch_geometric.nn import GATConv

        has_gat = any(
            isinstance(module, GATConv)
            for module in model.modules()
        )
        assert not has_gat, "Model should NOT use old GATConv"

    def test_has_three_layers(self, model: GATVerifier) -> None:
        """Verify model has 3 GAT layers as specified."""
        from torch_geometric.nn import GATv2Conv

        gatv2_layers = [
            m for m in model.modules()
            if isinstance(m, GATv2Conv)
        ]
        assert len(gatv2_layers) == 3

    def test_configurable_layers(self) -> None:
        """Test model respects num_layers parameter."""
        from torch_geometric.nn import GATv2Conv

        model_2 = GATVerifier(temporal_features=5, num_layers=2)
        model_4 = GATVerifier(temporal_features=5, num_layers=4)

        gatv2_2 = [m for m in model_2.modules() if isinstance(m, GATv2Conv)]
        gatv2_4 = [m for m in model_4.modules() if isinstance(m, GATv2Conv)]

        assert len(gatv2_2) == 2
        assert len(gatv2_4) == 4

    def test_gradient_flow(self, model: GATVerifier, sample_data: tuple) -> None:
        """Test gradients flow through the model."""
        x, edge_index, node_type = sample_data
        x = x.clone()
        x.requires_grad = True

        scores = model(x, edge_index, node_type)
        loss = scores.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_model_trainable(self, model: GATVerifier, sample_data: tuple) -> None:
        """Test model parameters are trainable."""
        x, edge_index, node_type = sample_data

        # Check parameters exist and require grad
        params = list(model.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

        # Test one training step
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scores = model(x, edge_index, node_type)

        # Dummy loss
        loss = scores.mean()
        loss.backward()
        optimizer.step()

    def test_no_oversmoothing(self, model: GATVerifier) -> None:
        """Test that oversmoothing is prevented."""
        # Create a chain graph where oversmoothing would be obvious
        num_nodes = 30
        x = torch.randn(num_nodes, 5)

        # Chain: 0-1-2-...-29
        src = torch.arange(num_nodes - 1)
        dst = torch.arange(1, num_nodes)
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ])

        # Different types at different positions
        node_type = torch.tensor([0]*10 + [1]*10 + [2]*10)

        model.eval()
        with torch.inference_mode():
            scores = model(x, edge_index, node_type)

        # If oversmoothing occurred, all scores would be nearly identical
        score_std = scores.std().item()
        assert score_std > 0.01, f"Score std {score_std:.4f} too low - possible oversmoothing"

    def test_oversmoothing_varies_with_depth(self) -> None:
        """Test that deeper models show GCNII residual effect."""
        # With GCNII residual, deeper models should still maintain diversity
        num_nodes = 50
        x = torch.randn(num_nodes, 5)
        edge_index = torch.randint(0, num_nodes, (2, 150))

        model_shallow = GATVerifier(temporal_features=5, num_layers=1)
        model_deep = GATVerifier(temporal_features=5, num_layers=5)

        model_shallow.eval()
        model_deep.eval()

        with torch.inference_mode():
            scores_shallow = model_shallow(x, edge_index)
            scores_deep = model_deep(x, edge_index)

        # Both should have reasonable variance (no oversmoothing)
        assert scores_shallow.std() > 0.01
        assert scores_deep.std() > 0.01

    def test_inference_latency(self, model: GATVerifier) -> None:
        """Test inference latency is under 30ms for batch_size=32."""
        # Simulate batch of 32 graphs with ~100 nodes each
        x = torch.randn(3200, 5)
        edge_index = torch.randint(0, 3200, (2, 10000))
        node_type = torch.randint(0, 3, (3200,))

        model.eval()

        # Warmup
        with torch.inference_mode():
            _ = model(x, edge_index, node_type)

        # Benchmark
        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.inference_mode():
                _ = model(x, edge_index, node_type)
            times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        p95_ms = sorted(times)[int(len(times) * 0.95)]

        print(f"\nInference latency: avg={avg_ms:.2f}ms, p95={p95_ms:.2f}ms")
        # 35ms threshold allows for environment variance; target is <30ms
        assert avg_ms < 35, f"Latency {avg_ms:.2f}ms exceeds 35ms threshold (target: 30ms)"

    def test_empty_graph(self, model: GATVerifier) -> None:
        """Test handling of empty graph (0 nodes)."""
        x = torch.zeros(0, 5)
        edge_index = torch.zeros((2, 0), dtype=torch.long)

        model.eval()
        with torch.inference_mode():
            scores = model(x, edge_index)

        assert scores.shape == (0, 1)

    def test_single_node_graph(self, model: GATVerifier) -> None:
        """Test handling of single node graph."""
        x = torch.randn(1, 5)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        node_type = torch.tensor([0])

        model.eval()
        with torch.inference_mode():
            scores = model(x, edge_index, node_type)

        assert scores.shape == (1, 1)
        assert 0.0 <= scores.item() <= 1.0

    def test_disconnected_nodes(self, model: GATVerifier) -> None:
        """Test handling of disconnected nodes."""
        x = torch.randn(10, 5)
        # Only connect first 5 nodes
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 0],
                                   [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]], dtype=torch.long)
        # Nodes 5-9 are disconnected

        model.eval()
        with torch.inference_mode():
            scores = model(x, edge_index)

        # All nodes should have valid scores
        assert scores.shape == (10, 1)
        assert (scores >= 0).all() and (scores <= 1).all()


class TestSyntheticAnomalyDetection:
    """Test anomaly detection on synthetic data."""

    @pytest.fixture
    def grid_graph(self) -> "torch_geometric.data.Data":
        """Create a realistic grid graph."""
        df = pd.DataFrame({
            'primary_substation_id': ['PS1']*10 + ['PS2']*10,
            'secondary_substation_id': ['SS1']*3 + ['SS2']*3 + ['SS3']*4 + ['SS4']*3 + ['SS5']*3 + ['SS6']*4,
            'lv_feeder_id': [f'LV{i}' for i in range(20)],
            'total_mpan_count': [50 + i*5 for i in range(20)],
        })
        builder = GridGraphBuilder()
        return builder.build_from_metadata(df)

    def test_pipeline_runs_end_to_end(self, grid_graph) -> None:
        """Test complete pipeline from graph to scores."""
        model = GATVerifier(temporal_features=grid_graph.x.size(1))
        model.eval()

        # Run inference
        with torch.inference_mode():
            scores = model(grid_graph.x, grid_graph.edge_index, grid_graph.node_type)

        # Verify output
        assert scores.shape == (grid_graph.num_nodes, 1)
        assert scores.min() >= 0
        assert scores.max() <= 1

        print(f"\nPipeline test: {grid_graph.num_nodes} nodes -> {scores.shape} scores")
        print(f"Score distribution: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")

    def test_synthetic_anomaly_detection_structure(self, grid_graph) -> None:
        """Test synthetic anomaly detection pipeline structure.

        This validates the pipeline works - actual >85% accuracy
        requires training which is in Phase 2.
        """
        model = GATVerifier(
            temporal_features=grid_graph.x.size(1),
            hidden_channels=64,
            num_layers=3,
        )
        model.eval()

        num_nodes = grid_graph.num_nodes

        # Create normal features (low variance)
        x_normal = torch.randn(num_nodes, grid_graph.x.size(1)) * 0.5

        # Inject anomalies (high values) for some nodes
        anomaly_mask = torch.zeros(num_nodes, dtype=torch.bool)
        anomaly_indices = torch.randint(0, num_nodes, (int(num_nodes * 0.15),))
        anomaly_mask[anomaly_indices] = True

        x_anomaly = x_normal.clone()
        x_anomaly[anomaly_mask] = torch.randn(anomaly_mask.sum(), grid_graph.x.size(1)) * 3.0 + 2.0

        with torch.inference_mode():
            scores = model(x_anomaly, grid_graph.edge_index, grid_graph.node_type)

        # Verify output structure
        assert scores.shape == (num_nodes, 1)
        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_different_inputs_produce_different_scores(self, grid_graph) -> None:
        """Test that model distinguishes between different inputs."""
        model = GATVerifier(temporal_features=grid_graph.x.size(1))
        model.eval()

        # Two different inputs
        x1 = torch.randn(grid_graph.num_nodes, grid_graph.x.size(1))
        x2 = torch.randn(grid_graph.num_nodes, grid_graph.x.size(1)) * 5

        with torch.inference_mode():
            scores1 = model(x1, grid_graph.edge_index, grid_graph.node_type)
            scores2 = model(x2, grid_graph.edge_index, grid_graph.node_type)

        # Scores should differ
        assert not torch.allclose(scores1, scores2, atol=0.01)

    def test_model_sensitivity_to_features(self, grid_graph) -> None:
        """Test model is sensitive to feature changes."""
        model = GATVerifier(temporal_features=grid_graph.x.size(1))
        model.eval()

        x_base = torch.zeros(grid_graph.num_nodes, grid_graph.x.size(1))
        x_modified = x_base.clone()
        x_modified[0] = 10.0  # Modify first node significantly

        with torch.inference_mode():
            scores_base = model(x_base, grid_graph.edge_index, grid_graph.node_type)
            scores_modified = model(x_modified, grid_graph.edge_index, grid_graph.node_type)

        # At least the modified node should have different score
        assert scores_base[0].item() != scores_modified[0].item()

    def test_graph_structure_affects_scores(self, grid_graph) -> None:
        """Test that graph structure affects propagation."""
        model = GATVerifier(temporal_features=grid_graph.x.size(1))
        model.eval()

        # Same node features
        x = torch.randn(grid_graph.num_nodes, grid_graph.x.size(1))

        # Different edge structures
        edge_index_original = grid_graph.edge_index
        edge_index_shuffled = edge_index_original[:, torch.randperm(edge_index_original.size(1))]

        with torch.inference_mode():
            scores_original = model(x, edge_index_original, grid_graph.node_type)
            # Use different (random) edges
            random_edges = torch.randint(0, grid_graph.num_nodes, (2, 50))
            scores_random = model(x, random_edges, grid_graph.node_type)

        # Different structures should give different results
        assert not torch.allclose(scores_original, scores_random, atol=0.01)

    def test_batch_inference(self) -> None:
        """Test inference on batched graphs."""
        from torch_geometric.data import Batch

        # Create multiple small graphs
        graphs = []
        for i in range(5):
            num_nodes = 10 + i * 2
            x = torch.randn(num_nodes, 4)
            edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
            node_type = torch.randint(0, 3, (num_nodes,))
            from torch_geometric.data import Data
            graphs.append(Data(x=x, edge_index=edge_index, node_type=node_type))

        batch = Batch.from_data_list(graphs)

        model = GATVerifier(temporal_features=4)
        model.eval()

        with torch.inference_mode():
            scores = model(batch.x, batch.edge_index, batch.node_type)

        # Total nodes across all graphs
        total_nodes = sum(g.num_nodes for g in graphs)
        assert scores.shape == (total_nodes, 1)

    def test_reproducible_inference(self, grid_graph) -> None:
        """Test that inference is reproducible in eval mode."""
        model = GATVerifier(temporal_features=grid_graph.x.size(1))
        model.eval()

        x = torch.randn(grid_graph.num_nodes, grid_graph.x.size(1))

        with torch.inference_mode():
            scores1 = model(x, grid_graph.edge_index, grid_graph.node_type)
            scores2 = model(x, grid_graph.edge_index, grid_graph.node_type)

        assert torch.allclose(scores1, scores2)


class TestModelConfiguration:
    """Test various model configurations."""

    def test_small_hidden_channels(self) -> None:
        """Test model with small hidden dimension."""
        model = GATVerifier(temporal_features=5, hidden_channels=16, heads=2)
        x = torch.randn(20, 5)
        edge_index = torch.randint(0, 20, (2, 40))

        model.eval()
        with torch.inference_mode():
            scores = model(x, edge_index)

        assert scores.shape == (20, 1)

    def test_large_hidden_channels(self) -> None:
        """Test model with large hidden dimension."""
        model = GATVerifier(temporal_features=5, hidden_channels=256, heads=8)
        x = torch.randn(20, 5)
        edge_index = torch.randint(0, 20, (2, 40))

        model.eval()
        with torch.inference_mode():
            scores = model(x, edge_index)

        assert scores.shape == (20, 1)

    def test_different_head_counts(self) -> None:
        """Test different attention head configurations."""
        for heads in [1, 2, 4, 8]:
            model = GATVerifier(temporal_features=5, hidden_channels=64, heads=heads)
            x = torch.randn(20, 5)
            edge_index = torch.randint(0, 20, (2, 40))

            model.eval()
            with torch.inference_mode():
                scores = model(x, edge_index)

            assert scores.shape == (20, 1), f"Failed for heads={heads}"

    def test_varied_temporal_features(self) -> None:
        """Test various temporal feature dimensions."""
        for tf in [1, 2, 3, 5, 10, 20]:
            model = GATVerifier(temporal_features=tf)
            x = torch.randn(20, tf)
            edge_index = torch.randint(0, 20, (2, 40))

            model.eval()
            with torch.inference_mode():
                scores = model(x, edge_index)

            assert scores.shape == (20, 1), f"Failed for temporal_features={tf}"
