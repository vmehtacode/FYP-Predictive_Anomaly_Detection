"""Tests for graph-aware ProposerAgent extensions.

Tests cover:
- Graph-aware scenario generation (metadata enrichment)
- Topology-based seed node selection (LV feeder preference)
- Cascade propagation through 2-hop neighbors with 0.7 decay
- Affected nodes cap at 30% of graph
- Backward compatibility with non-graph proposals
- Scenario diversity across graph-aware proposals
- Per-node time-series application via apply_to_graph_timeseries

Requirements: SELF-01 (topology-aware generation), SELF-02 (cascade propagation)
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from fyp.selfplay.proposer import ProposerAgent, ScenarioProposal


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_constraints_file():
    """Create temporary constraints file for testing."""
    constraints = {
        "household_limits": {
            "typical_max_kwh_30min": 7.5,
            "absolute_max_kwh_30min": 50.0,
        },
        "voltage_limits": {"nominal_v": 230, "min_percent": -6, "max_percent": 10},
        "power_factor": {"min_lagging": 0.95},
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(constraints, f)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def sample_graph_data():
    """Create a PyG Data object with 20 nodes in a tree topology.

    Hierarchy:
    - 2 primary substations (type=0), indices 0-1
    - 4 secondary substations (type=1), indices 2-5
    - 14 LV feeders (type=2), indices 6-19

    Edges: primary -> secondary -> LV (bidirectional)
    """
    num_nodes = 20

    # Node types
    node_type = torch.cat([
        torch.full((2,), 0, dtype=torch.long),   # primary
        torch.full((4,), 1, dtype=torch.long),   # secondary
        torch.full((14,), 2, dtype=torch.long),  # LV feeders
    ])

    # Build edges: tree structure
    edges = []
    # Primary 0 -> Secondary 2, 3
    for s in [2, 3]:
        edges.append([0, s])
        edges.append([s, 0])
    # Primary 1 -> Secondary 4, 5
    for s in [4, 5]:
        edges.append([1, s])
        edges.append([s, 1])
    # Secondary 2 -> LV 6, 7, 8
    for lv in [6, 7, 8]:
        edges.append([2, lv])
        edges.append([lv, 2])
    # Secondary 3 -> LV 9, 10, 11
    for lv in [9, 10, 11]:
        edges.append([3, lv])
        edges.append([lv, 3])
    # Secondary 4 -> LV 12, 13, 14, 15
    for lv in [12, 13, 14, 15]:
        edges.append([4, lv])
        edges.append([lv, 4])
    # Secondary 5 -> LV 16, 17, 18, 19
    for lv in [16, 17, 18, 19]:
        edges.append([5, lv])
        edges.append([lv, 5])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.randn(num_nodes, 5)

    data = Data(
        x=x,
        edge_index=edge_index,
        node_type=node_type,
        num_nodes=num_nodes,
    )
    return data


@pytest.fixture
def linear_graph_data():
    """Create a simple linear graph: A--B--C--D--E for exact hop testing.

    All nodes are LV feeders (type=2) for simplicity.
    """
    num_nodes = 5
    node_type = torch.full((num_nodes,), 2, dtype=torch.long)
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.randn(num_nodes, 5)
    data = Data(
        x=x,
        edge_index=edge_index,
        node_type=node_type,
        num_nodes=num_nodes,
    )
    return data


@pytest.fixture
def proposer(temp_constraints_file):
    """ProposerAgent instance with fixed seed."""
    return ProposerAgent(
        temp_constraints_file, difficulty_curriculum=True, random_seed=42
    )


# ============================================================================
# TestGraphAwareProposer
# ============================================================================


class TestGraphAwareProposer:
    """Tests for graph-aware scenario proposal metadata."""

    def test_propose_with_graph_data_returns_scenario(
        self, proposer, sample_graph_data
    ):
        """propose_scenario(graph_data=graph_data) returns ScenarioProposal
        with metadata['graph_aware'] == True."""
        context = np.random.rand(336)
        scenario = proposer.propose_scenario(
            historical_context=context,
            graph_data=sample_graph_data,
            forecast_horizon=48,
        )
        assert isinstance(scenario, ScenarioProposal)
        assert scenario.metadata.get("graph_aware") is True

    def test_propose_with_graph_data_has_seed_nodes(
        self, proposer, sample_graph_data
    ):
        """metadata['seed_nodes'] is a non-empty list of ints."""
        context = np.random.rand(336)
        scenario = proposer.propose_scenario(
            historical_context=context,
            graph_data=sample_graph_data,
            forecast_horizon=48,
        )
        seed_nodes = scenario.metadata.get("seed_nodes")
        assert seed_nodes is not None
        assert isinstance(seed_nodes, list)
        assert len(seed_nodes) > 0
        assert all(isinstance(n, int) for n in seed_nodes)

    def test_propose_with_graph_data_has_affected_nodes(
        self, proposer, sample_graph_data
    ):
        """metadata['affected_nodes'] is a dict mapping int -> float."""
        context = np.random.rand(336)
        scenario = proposer.propose_scenario(
            historical_context=context,
            graph_data=sample_graph_data,
            forecast_horizon=48,
        )
        affected = scenario.metadata.get("affected_nodes")
        assert affected is not None
        assert isinstance(affected, dict)
        assert len(affected) > 0
        for k, v in affected.items():
            assert isinstance(k, int)
            assert isinstance(v, float)

    def test_propose_with_graph_data_has_cascade_params(
        self, proposer, sample_graph_data
    ):
        """metadata contains 'num_hops' == 2 and 'decay_factor' == 0.7."""
        context = np.random.rand(336)
        scenario = proposer.propose_scenario(
            historical_context=context,
            graph_data=sample_graph_data,
            forecast_horizon=48,
        )
        assert scenario.metadata.get("num_hops") == 2
        assert scenario.metadata.get("decay_factor") == 0.7


# ============================================================================
# TestSeedNodeSelection
# ============================================================================


class TestSeedNodeSelection:
    """Tests for topology-based seed node selection."""

    def test_cold_snap_prefers_lv_feeders(self, proposer, sample_graph_data):
        """For COLD_SNAP, all seed nodes have node_type == 2."""
        seeds = proposer._select_seed_nodes(sample_graph_data, "COLD_SNAP")
        for s in seeds.tolist():
            assert sample_graph_data.node_type[s].item() == 2

    def test_outage_prefers_lv_feeders(self, proposer, sample_graph_data):
        """For OUTAGE, all seed nodes have node_type == 2."""
        seeds = proposer._select_seed_nodes(sample_graph_data, "OUTAGE")
        for s in seeds.tolist():
            assert sample_graph_data.node_type[s].item() == 2

    def test_ev_spike_prefers_lv_feeders(self, proposer, sample_graph_data):
        """For EV_SPIKE, all seed nodes have node_type == 2."""
        seeds = proposer._select_seed_nodes(sample_graph_data, "EV_SPIKE")
        for s in seeds.tolist():
            assert sample_graph_data.node_type[s].item() == 2

    def test_seed_nodes_within_graph_bounds(self, proposer, sample_graph_data):
        """All seed indices are < graph_data.num_nodes."""
        for scenario_type in ["COLD_SNAP", "OUTAGE", "EV_SPIKE", "PEAK_SHIFT", "MISSING_DATA"]:
            seeds = proposer._select_seed_nodes(sample_graph_data, scenario_type)
            for s in seeds.tolist():
                assert 0 <= s < sample_graph_data.num_nodes


# ============================================================================
# TestCascadePropagation
# ============================================================================


class TestCascadePropagation:
    """Tests for cascade propagation through graph neighbors."""

    def test_affected_nodes_follow_connectivity(
        self, proposer, linear_graph_data
    ):
        """Every affected node (not a seed) is a graph neighbor of another affected node."""
        # Use node 0 as seed in linear graph A--B--C--D--E
        seed_nodes = torch.tensor([0])
        affected = proposer._propagate_through_neighbors(
            seed_nodes, linear_graph_data, num_hops=2, decay_factor=0.7
        )
        # Build adjacency for verification
        adj = ProposerAgent._build_adjacency(
            linear_graph_data.edge_index, linear_graph_data.num_nodes
        )
        seed_set = set(seed_nodes.tolist())
        for node in affected:
            if node in seed_set:
                continue
            # Must be neighbor of at least one other affected node
            neighbors = set(adj[node])
            assert neighbors & set(affected.keys()), (
                f"Node {node} is not adjacent to any other affected node"
            )

    def test_cascade_decay_applied(self, proposer, linear_graph_data):
        """Nodes at hop 1 get magnitude 0.7, nodes at hop 2 get magnitude 0.49."""
        # Linear graph: 0--1--2--3--4, seed at node 2
        seed_nodes = torch.tensor([2])
        affected = proposer._propagate_through_neighbors(
            seed_nodes, linear_graph_data, num_hops=2, decay_factor=0.7
        )
        # Seed node 2 at magnitude 1.0
        assert abs(affected[2] - 1.0) < 1e-6
        # Hop-1 neighbors: nodes 1 and 3 at 0.7
        assert abs(affected[1] - 0.7) < 1e-6
        assert abs(affected[3] - 0.7) < 1e-6
        # Hop-2 neighbors: nodes 0 and 4 at 0.49
        assert abs(affected[0] - 0.49) < 1e-6
        assert abs(affected[4] - 0.49) < 1e-6

    def test_seed_nodes_get_full_magnitude(self, proposer, linear_graph_data):
        """Seed nodes have magnitude 1.0 in affected_nodes dict."""
        seed_nodes = torch.tensor([0, 4])
        affected = proposer._propagate_through_neighbors(
            seed_nodes, linear_graph_data, num_hops=2, decay_factor=0.7
        )
        assert abs(affected[0] - 1.0) < 1e-6
        assert abs(affected[4] - 1.0) < 1e-6


# ============================================================================
# TestCascadeDecay
# ============================================================================


class TestCascadeDecay:
    """Tests for cascade decay factor and hop limits."""

    def test_decay_factor_is_0_7(self, proposer, linear_graph_data):
        """The decay between hops is exactly 0.7 (project convention)."""
        seed_nodes = torch.tensor([0])
        affected = proposer._propagate_through_neighbors(
            seed_nodes, linear_graph_data, num_hops=2, decay_factor=0.7
        )
        # Hop 0: node 0 = 1.0
        # Hop 1: node 1 = 0.7
        # Hop 2: node 2 = 0.49
        assert abs(affected[0] - 1.0) < 1e-6
        assert abs(affected[1] - 0.7) < 1e-6
        assert abs(affected[2] - 0.7 * 0.7) < 1e-6

    def test_two_hops_max(self, proposer, linear_graph_data):
        """No node beyond 2 hops from any seed is in affected_nodes."""
        # Seed at node 0 in 0--1--2--3--4
        seed_nodes = torch.tensor([0])
        affected = proposer._propagate_through_neighbors(
            seed_nodes, linear_graph_data, num_hops=2, decay_factor=0.7
        )
        # Should affect 0, 1, 2 only (not 3, 4)
        assert 0 in affected
        assert 1 in affected
        assert 2 in affected
        assert 3 not in affected
        assert 4 not in affected


# ============================================================================
# TestScenarioTypeCascade
# ============================================================================


class TestScenarioTypeCascade:
    """Tests for cascade behavior in specific scenario types."""

    def test_cold_snap_cascades(self, proposer, sample_graph_data):
        """COLD_SNAP scenarios produce affected_nodes with len > len(seed_nodes)."""
        context = np.random.rand(336)
        # Run multiple proposals to get a COLD_SNAP
        for _ in range(50):
            scenario = proposer.propose_scenario(
                historical_context=context,
                graph_data=sample_graph_data,
                forecast_horizon=48,
            )
            if scenario.scenario_type == "COLD_SNAP":
                seeds = scenario.metadata["seed_nodes"]
                affected = scenario.metadata["affected_nodes"]
                assert len(affected) > len(seeds), (
                    "COLD_SNAP cascade should affect more nodes than just seeds"
                )
                return
        # If no COLD_SNAP generated, that's okay - test is statistical
        pytest.skip("No COLD_SNAP generated in 50 attempts")

    def test_outage_cascades(self, proposer, sample_graph_data):
        """OUTAGE scenarios produce affected_nodes with len > len(seed_nodes)."""
        context = np.random.rand(336)
        for _ in range(50):
            scenario = proposer.propose_scenario(
                historical_context=context,
                graph_data=sample_graph_data,
                forecast_horizon=48,
            )
            if scenario.scenario_type == "OUTAGE":
                seeds = scenario.metadata["seed_nodes"]
                affected = scenario.metadata["affected_nodes"]
                assert len(affected) > len(seeds), (
                    "OUTAGE cascade should affect more nodes than just seeds"
                )
                return
        pytest.skip("No OUTAGE generated in 50 attempts")


# ============================================================================
# TestBackwardCompat
# ============================================================================


class TestBackwardCompat:
    """Tests for backward compatibility with non-graph proposals."""

    def test_propose_without_graph_data_works(self, proposer):
        """propose_scenario() without graph_data returns valid ScenarioProposal."""
        context = np.random.rand(336)
        scenario = proposer.propose_scenario(
            historical_context=context, forecast_horizon=48
        )
        assert isinstance(scenario, ScenarioProposal)
        assert scenario.scenario_type in ProposerAgent.SCENARIO_CONFIGS
        assert scenario.duration > 0

    def test_propose_without_graph_data_no_graph_metadata(self, proposer):
        """metadata does not contain 'graph_aware' key when no graph_data."""
        context = np.random.rand(336)
        scenario = proposer.propose_scenario(
            historical_context=context, forecast_horizon=48
        )
        assert "graph_aware" not in scenario.metadata

    def test_existing_tests_still_pass(self, proposer):
        """All original propose_scenario call patterns still work."""
        context = np.random.rand(336) * 3.0
        # Call without any optional params
        s1 = proposer.propose_scenario(historical_context=context)
        assert isinstance(s1, ScenarioProposal)

        # Call with conditioning_samples
        s2 = proposer.propose_scenario(
            historical_context=context,
            conditioning_samples=[(s1, 0.5)],
            forecast_horizon=48,
        )
        assert isinstance(s2, ScenarioProposal)

        # Call with current_timestamp
        from datetime import datetime
        s3 = proposer.propose_scenario(
            historical_context=context,
            current_timestamp=datetime(2026, 1, 15, 12, 0),
        )
        assert isinstance(s3, ScenarioProposal)


# ============================================================================
# TestScenarioDiversity
# ============================================================================


class TestScenarioDiversity:
    """Tests for diversity of graph-aware scenario generation."""

    def test_graph_proposer_produces_multiple_types(
        self, proposer, sample_graph_data
    ):
        """Running 20 proposals produces at least 2 different scenario_types."""
        context = np.random.rand(336)
        types_seen = set()
        for _ in range(20):
            scenario = proposer.propose_scenario(
                historical_context=context,
                graph_data=sample_graph_data,
                forecast_horizon=48,
            )
            types_seen.add(scenario.scenario_type)
        assert len(types_seen) >= 2, (
            f"Only saw {types_seen} in 20 proposals, expected >= 2 types"
        )


# ============================================================================
# TestAffectedNodesCap
# ============================================================================


class TestAffectedNodesCap:
    """Tests for the 30% affected nodes cap."""

    def test_affected_nodes_capped_at_30_percent(
        self, proposer, sample_graph_data
    ):
        """len(affected_nodes) <= ceil(0.3 * graph_data.num_nodes)."""
        context = np.random.rand(336)
        max_allowed = math.ceil(0.3 * sample_graph_data.num_nodes)
        for _ in range(20):
            scenario = proposer.propose_scenario(
                historical_context=context,
                graph_data=sample_graph_data,
                forecast_horizon=48,
            )
            affected = scenario.metadata.get("affected_nodes", {})
            assert len(affected) <= max_allowed, (
                f"Got {len(affected)} affected nodes, cap is {max_allowed}"
            )


# ============================================================================
# TestApplyToGraphTimeseries
# ============================================================================


class TestApplyToGraphTimeseries:
    """Tests for ScenarioProposal.apply_to_graph_timeseries()."""

    def _make_proposal(self, affected_nodes=None, graph_aware=True):
        """Helper to create a ScenarioProposal with specific metadata."""
        metadata = {"start_offset": 0}
        if graph_aware:
            metadata["graph_aware"] = True
            metadata["affected_nodes"] = affected_nodes or {}
            metadata["seed_nodes"] = [
                k for k, v in (affected_nodes or {}).items() if v >= 1.0
            ]
        return ScenarioProposal(
            scenario_type="COLD_SNAP",
            magnitude=2.0,
            duration=48,
            start_time=None,
            affected_appliances=["heating"],
            baseline_context=np.ones(336),
            difficulty_score=0.5,
            physics_valid=True,
            metadata=metadata,
        )

    def test_apply_scales_affected_node_columns(self):
        """For an affected node with magnitude 0.7, the corresponding column
        is scaled by 0.7 relative to the baseline apply_to_timeseries result."""
        affected_nodes = {0: 1.0, 1: 0.7, 2: 0.49}
        proposal = self._make_proposal(affected_nodes=affected_nodes)
        baseline = np.ones((5, 48)) * 2.0

        result = proposal.apply_to_graph_timeseries(baseline)

        # Get full transformation for reference
        full_transform = proposal.apply_to_timeseries(baseline[0])
        diff = full_transform - baseline[0]

        # Node 1 (magnitude 0.7): baseline + 0.7 * (full_transform - baseline)
        expected_node1 = baseline[1] + 0.7 * diff
        np.testing.assert_allclose(result[1], expected_node1, rtol=1e-6)

    def test_apply_unaffected_nodes_unchanged(self):
        """Columns for nodes NOT in affected_nodes match baseline (no change)."""
        affected_nodes = {0: 1.0, 1: 0.7}
        proposal = self._make_proposal(affected_nodes=affected_nodes)
        baseline = np.ones((5, 48)) * 2.0

        result = proposal.apply_to_graph_timeseries(baseline)

        # Nodes 2, 3, 4 are unaffected -- should equal original baseline
        np.testing.assert_array_equal(result[2], baseline[2])
        np.testing.assert_array_equal(result[3], baseline[3])
        np.testing.assert_array_equal(result[4], baseline[4])

    def test_apply_seed_nodes_full_magnitude(self):
        """Seed nodes (magnitude 1.0) match the baseline apply_to_timeseries output."""
        affected_nodes = {0: 1.0, 1: 0.7}
        proposal = self._make_proposal(affected_nodes=affected_nodes)
        baseline = np.ones((5, 48)) * 2.0

        result = proposal.apply_to_graph_timeseries(baseline)

        # Node 0 (seed, magnitude 1.0): full transformation
        full_transform = proposal.apply_to_timeseries(baseline[0])
        np.testing.assert_allclose(result[0], full_transform, rtol=1e-6)

    def test_apply_without_graph_metadata_falls_back(self):
        """When metadata has no graph_aware field, apply_to_graph_timeseries
        applies uniform transformation (same as apply_to_timeseries per row)."""
        proposal = self._make_proposal(graph_aware=False)
        baseline = np.ones((5, 48)) * 2.0

        result = proposal.apply_to_graph_timeseries(baseline)

        # Each row should equal apply_to_timeseries(row)
        for i in range(5):
            expected = proposal.apply_to_timeseries(baseline[i])
            np.testing.assert_allclose(result[i], expected, rtol=1e-6)
