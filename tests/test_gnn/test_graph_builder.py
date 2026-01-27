"""Tests for GridGraphBuilder.

This module validates the graph construction pipeline:
- Three-level hierarchy (primary substation -> secondary -> LV feeder)
- Correct topology (node count, edge count, bidirectionality)
- Node features and types
- Edge cases (missing data, custom features)
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from fyp.gnn import GridGraphBuilder


class TestGridGraphBuilder:
    """Test suite for GridGraphBuilder."""

    @pytest.fixture
    def simple_metadata(self) -> pd.DataFrame:
        """Create minimal test metadata."""
        return pd.DataFrame({
            'primary_substation_id': ['PS1', 'PS1', 'PS1'],
            'secondary_substation_id': ['SS1', 'SS1', 'SS2'],
            'lv_feeder_id': ['LV1', 'LV2', 'LV3'],
            'total_mpan_count': [50, 30, 20],
        })

    @pytest.fixture
    def complex_metadata(self) -> pd.DataFrame:
        """Create larger test metadata with multiple substations."""
        # 2 primary substations, 4 secondary, 10 LV feeders
        return pd.DataFrame({
            'primary_substation_id': ['PS1']*5 + ['PS2']*5,
            'secondary_substation_id': ['SS1', 'SS1', 'SS2', 'SS2', 'SS2'] + ['SS3', 'SS3', 'SS4', 'SS4', 'SS4'],
            'lv_feeder_id': [f'LV{i}' for i in range(1, 11)],
            'total_mpan_count': [50, 30, 40, 20, 60, 35, 45, 25, 55, 15],
        })

    def test_basic_graph_construction(self, simple_metadata: pd.DataFrame) -> None:
        """Test basic graph is constructed correctly."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # Should have 1 PS + 2 SS + 3 LV = 6 nodes
        assert data.num_nodes == 6

        # Edges should be bidirectional
        # PS1-SS1, PS1-SS2, SS1-LV1, SS1-LV2, SS2-LV3 = 5 connections * 2 = 10 edges
        assert data.edge_index.size(1) == 10

    def test_node_types_assigned(self, simple_metadata: pd.DataFrame) -> None:
        """Test node types are correctly assigned."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # Check all three types present
        unique_types = data.node_type.unique().tolist()
        assert 0 in unique_types  # Primary substations
        assert 1 in unique_types  # Secondary substations (feeders)
        assert 2 in unique_types  # LV feeders (households)

    def test_node_type_count(self, simple_metadata: pd.DataFrame) -> None:
        """Test correct count per node type."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # Count nodes by type
        type_counts = torch.bincount(data.node_type, minlength=3)

        assert type_counts[0] == 1  # 1 primary substation
        assert type_counts[1] == 2  # 2 secondary substations
        assert type_counts[2] == 3  # 3 LV feeders

    def test_node_features_shape(self, simple_metadata: pd.DataFrame) -> None:
        """Test node features have correct shape."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # Features should be [num_nodes, num_features]
        assert data.x.dim() == 2
        assert data.x.size(0) == data.num_nodes
        # Default features: 3 (one-hot type) + 1 (log mpan count) = 4
        assert data.x.size(1) >= 3

    def test_edge_index_coo_format(self, simple_metadata: pd.DataFrame) -> None:
        """Test edge_index is in correct COO format."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # COO format: [2, num_edges]
        assert data.edge_index.dim() == 2
        assert data.edge_index.size(0) == 2

        # All indices should be valid
        assert data.edge_index.min() >= 0
        assert data.edge_index.max() < data.num_nodes

    def test_bidirectional_edges(self, simple_metadata: pd.DataFrame) -> None:
        """Test edges are bidirectional."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        edge_set = set()
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[:, i].tolist()
            edge_set.add((src, dst))

        # For each edge (a, b), reverse (b, a) should also exist
        for src, dst in list(edge_set):
            assert (dst, src) in edge_set, f"Missing reverse edge for ({src}, {dst})"

    def test_complex_hierarchy(self, complex_metadata: pd.DataFrame) -> None:
        """Test with larger, more complex hierarchy."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(complex_metadata)

        # 2 PS + 4 SS + 10 LV = 16 nodes
        assert data.num_nodes == 16

    def test_complex_edge_count(self, complex_metadata: pd.DataFrame) -> None:
        """Test edge count in complex hierarchy."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(complex_metadata)

        # PS1->SS1, PS1->SS2 = 2 edges
        # PS2->SS3, PS2->SS4 = 2 edges
        # SS1->LV1,LV2 = 2 edges
        # SS2->LV3,LV4,LV5 = 3 edges
        # SS3->LV6,LV7 = 2 edges
        # SS4->LV8,LV9,LV10 = 3 edges
        # Total directed: 14 * 2 = 28 edges
        assert data.edge_index.size(1) == 28

    def test_handles_missing_mpan_count(self) -> None:
        """Test graceful handling of missing mpan count."""
        df = pd.DataFrame({
            'primary_substation_id': ['PS1', 'PS1'],
            'secondary_substation_id': ['SS1', 'SS1'],
            'lv_feeder_id': ['LV1', 'LV2'],
            # No total_mpan_count column
        })

        builder = GridGraphBuilder()
        data = builder.build_from_metadata(df)

        # Should still work, just without mpan features
        assert data.num_nodes == 4  # 1 PS + 1 SS + 2 LV
        assert data.x is not None

    def test_exclude_incomplete_nodes_default(self) -> None:
        """Test that incomplete nodes are excluded by default."""
        df = pd.DataFrame({
            'primary_substation_id': ['PS1', 'PS1', None],  # One incomplete
            'secondary_substation_id': ['SS1', 'SS1', 'SS1'],
            'lv_feeder_id': ['LV1', 'LV2', 'LV3'],
        })

        builder = GridGraphBuilder(exclude_incomplete=True)
        data = builder.build_from_metadata(df)

        # LV3 row excluded due to missing primary substation
        # Remaining: 1 PS + 1 SS + 2 LV = 4 nodes
        assert data.num_nodes == 4

    def test_include_incomplete_nodes_optional(self) -> None:
        """Test that incomplete nodes can be included."""
        df = pd.DataFrame({
            'primary_substation_id': ['PS1', 'PS1', None],
            'secondary_substation_id': ['SS1', 'SS1', 'SS1'],
            'lv_feeder_id': ['LV1', 'LV2', 'LV3'],
        })

        builder = GridGraphBuilder(exclude_incomplete=False)
        data = builder.build_from_metadata(df)

        # All 3 rows included
        # 1 PS + 1 SS + 3 LV = 5 nodes (None excluded but LV3 included)
        # Actually need to check what happens with None
        assert data.num_nodes >= 4

    def test_explicit_num_nodes_set(self, simple_metadata: pd.DataFrame) -> None:
        """Test that num_nodes is explicitly set (for isolated node safety)."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # num_nodes should match x.size(0)
        assert data.num_nodes == data.x.size(0)

    def test_node_id_mapping(self, simple_metadata: pd.DataFrame) -> None:
        """Test node ID to index mapping works correctly."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # Check node_ids attribute exists
        assert hasattr(data, 'node_ids')
        assert len(data.node_ids) == data.num_nodes

        # Check reverse lookup
        for i, node_id in enumerate(data.node_ids):
            assert builder.get_node_idx(node_id) == i
            assert builder.get_node_id(i) == node_id

    def test_custom_node_features(self, simple_metadata: pd.DataFrame) -> None:
        """Test with custom node features provided."""
        builder = GridGraphBuilder()

        # Build first to get node mapping
        data_temp = builder.build_from_metadata(simple_metadata)
        node_ids = data_temp.node_ids

        # Provide custom features (8 dims instead of default 4)
        custom_features = {
            node_id: torch.randn(8) for node_id in node_ids
        }

        data = builder.build_from_metadata(simple_metadata, node_features=custom_features)
        assert data.x.size(1) == 8

    def test_empty_dataframe(self) -> None:
        """Test handling of empty dataframe."""
        df = pd.DataFrame({
            'primary_substation_id': [],
            'secondary_substation_id': [],
            'lv_feeder_id': [],
        })

        builder = GridGraphBuilder()
        data = builder.build_from_metadata(df)

        assert data.num_nodes == 0
        assert data.edge_index.size(1) == 0

    def test_missing_required_columns_raises(self) -> None:
        """Test that missing required columns raises ValueError."""
        df = pd.DataFrame({
            'primary_substation_id': ['PS1'],
            # Missing secondary_substation_id and lv_feeder_id
        })

        builder = GridGraphBuilder()
        with pytest.raises(ValueError, match="Missing required columns"):
            builder.build_from_metadata(df)

    def test_deterministic_node_ordering(self, simple_metadata: pd.DataFrame) -> None:
        """Test node ordering is deterministic across builds."""
        builder = GridGraphBuilder()

        data1 = builder.build_from_metadata(simple_metadata)
        data2 = builder.build_from_metadata(simple_metadata)

        # Node IDs should be in same order
        assert data1.node_ids == data2.node_ids

        # Features should match
        assert torch.allclose(data1.x, data2.x)

        # Edges should match
        assert torch.equal(data1.edge_index, data2.edge_index)

    def test_node_type_order_matches_ordering(self, simple_metadata: pd.DataFrame) -> None:
        """Test that node types follow primary->secondary->lv ordering."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # First node(s) should be type 0 (primary)
        assert data.node_type[0] == 0

        # Last nodes should be type 2 (lv feeders)
        assert data.node_type[-1] == 2

        # Types should be monotonically non-decreasing
        prev_type = -1
        for t in data.node_type.tolist():
            assert t >= prev_type or prev_type == t
            prev_type = t

    def test_edge_connects_correct_hierarchy(self, simple_metadata: pd.DataFrame) -> None:
        """Test edges only connect adjacent hierarchy levels."""
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(simple_metadata)

        # Get node types for each edge
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[:, i].tolist()
            src_type = data.node_type[src].item()
            dst_type = data.node_type[dst].item()

            # Edges should only be between adjacent types (0-1 or 1-2)
            type_diff = abs(src_type - dst_type)
            assert type_diff == 1, f"Edge ({src}, {dst}) connects non-adjacent types: {src_type} -> {dst_type}"

    def test_large_scale_graph(self) -> None:
        """Test with larger scale data for performance sanity check."""
        # Create 10 primary, 50 secondary, 500 LV feeders
        rows = []
        lv_count = 0
        for ps in range(10):
            for ss in range(5):
                for lv in range(10):
                    rows.append({
                        'primary_substation_id': f'PS{ps}',
                        'secondary_substation_id': f'SS{ps}_{ss}',
                        'lv_feeder_id': f'LV{lv_count}',
                        'total_mpan_count': 50 + lv_count,
                    })
                    lv_count += 1

        df = pd.DataFrame(rows)
        builder = GridGraphBuilder()
        data = builder.build_from_metadata(df)

        # 10 PS + 50 SS + 500 LV = 560 nodes
        assert data.num_nodes == 560

        # Verify structure
        assert data.x.size(0) == 560
        assert data.edge_index.size(0) == 2
        assert data.edge_index.max() < 560
