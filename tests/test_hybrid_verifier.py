"""Comprehensive tests for the hybrid verifier ensemble.

Tests cover:
- HybridVerifierConfig Pydantic validation and YAML loading
- tolerance_band_score() three-zone scoring function
- PhysicsConstraintLayer (voltage, capacity, ramp rate)
- CascadeLogicLayer (neighbor anomaly propagation)
- Ensemble score combination (normal vs early-exit nodes)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fyp.selfplay.hybrid_verifier import (
    CascadeLogicLayer,
    HybridVerifierAgent,
    PhysicsConstraintLayer,
    _combine_scores,
    tolerance_band_score,
)
from fyp.selfplay.hybrid_verifier_config import (
    EnsembleWeightsConfig,
    HybridVerifierConfig,
    load_hybrid_verifier_config,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> HybridVerifierConfig:
    """HybridVerifierConfig with defaults."""
    return HybridVerifierConfig()


@pytest.fixture
def physics_layer(default_config: HybridVerifierConfig) -> PhysicsConstraintLayer:
    """PhysicsConstraintLayer from default config."""
    return PhysicsConstraintLayer(default_config.physics)


@pytest.fixture
def cascade_layer(default_config: HybridVerifierConfig) -> CascadeLogicLayer:
    """CascadeLogicLayer from default config."""
    return CascadeLogicLayer(default_config.cascade)


@pytest.fixture
def simple_edge_index() -> torch.LongTensor:
    """5-node line graph: 0-1-2-3-4 (bidirectional)."""
    src = [0, 1, 2, 3, 1, 2, 3, 4]
    dst = [1, 2, 3, 4, 0, 1, 2, 3]
    return torch.tensor([src, dst], dtype=torch.long)


@pytest.fixture
def hybrid_verifier(default_config: HybridVerifierConfig) -> HybridVerifierAgent:
    """HybridVerifierAgent in physics-only mode (no GNN)."""
    return HybridVerifierAgent(default_config)


# ============================================================================
# Config Tests
# ============================================================================


class TestHybridVerifierConfig:
    """Tests for HybridVerifierConfig Pydantic models."""

    def test_default_config_valid(self) -> None:
        """HybridVerifierConfig() creates valid config with expected defaults."""
        config = HybridVerifierConfig()
        assert config.early_exit_threshold == 0.9
        assert config.false_negative_penalty_ratio == 2.0
        assert config.physics is not None
        assert config.cascade is not None
        assert config.ensemble_weights is not None

    def test_config_from_yaml(self) -> None:
        """load_hybrid_verifier_config loads from YAML correctly."""
        config = load_hybrid_verifier_config("configs/hybrid_verifier.yaml")
        assert config.physics.voltage.nominal_v == 230.0
        assert config.physics.capacity.absolute_max_kw == 100.0
        assert config.cascade.propagation_threshold == 0.3

    def test_config_voltage_defaults(self) -> None:
        """Nominal voltage and limits match BS EN 50160."""
        config = HybridVerifierConfig()
        v = config.physics.voltage
        assert v.nominal_v == 230.0
        assert v.lower_limit_pct == -10.0
        assert v.upper_limit_pct == 10.0
        assert v.lower_safe_pct == -6.0
        assert v.upper_safe_pct == 8.0

    def test_config_ensemble_weights(self) -> None:
        """Default ensemble weights: physics=0.4, gnn=0.4, cascade=0.2."""
        config = HybridVerifierConfig()
        w = config.ensemble_weights
        assert w.physics == 0.4
        assert w.gnn == 0.4
        assert w.cascade == 0.2

    def test_config_invalid_threshold_rejected(self) -> None:
        """Pydantic rejects early_exit_threshold > 1 or < 0."""
        with pytest.raises(Exception):
            HybridVerifierConfig(early_exit_threshold=1.5)
        with pytest.raises(Exception):
            HybridVerifierConfig(early_exit_threshold=-0.1)


# ============================================================================
# Tolerance Band Scoring Tests
# ============================================================================


class TestToleranceBandScore:
    """Tests for the tolerance_band_score function."""

    # Zone boundaries: lower_limit=10, lower_safe=20, upper_safe=80, upper_limit=90

    def test_safe_zone_returns_zero(self) -> None:
        """Value in [lower_safe, upper_safe] -> 0.0."""
        assert tolerance_band_score(50.0, 20.0, 80.0, 10.0, 90.0) == 0.0
        assert tolerance_band_score(20.0, 20.0, 80.0, 10.0, 90.0) == 0.0
        assert tolerance_band_score(80.0, 20.0, 80.0, 10.0, 90.0) == 0.0

    def test_lower_limit_returns_one(self) -> None:
        """Value below lower_limit -> 1.0."""
        assert tolerance_band_score(5.0, 20.0, 80.0, 10.0, 90.0) == 1.0

    def test_upper_limit_returns_one(self) -> None:
        """Value above upper_limit -> 1.0."""
        assert tolerance_band_score(95.0, 20.0, 80.0, 10.0, 90.0) == 1.0

    def test_lower_warning_zone_interpolation(self) -> None:
        """Value between lower_limit and lower_safe -> linear 0-1."""
        score = tolerance_band_score(15.0, 20.0, 80.0, 10.0, 90.0)
        assert abs(score - 0.5) < 1e-6

    def test_upper_warning_zone_interpolation(self) -> None:
        """Value between upper_safe and upper_limit -> linear 0-1."""
        score = tolerance_band_score(85.0, 20.0, 80.0, 10.0, 90.0)
        assert abs(score - 0.5) < 1e-6

    def test_boundary_values(self) -> None:
        """Exact boundary values produce expected scores."""
        assert tolerance_band_score(20.0, 20.0, 80.0, 10.0, 90.0) == 0.0
        assert tolerance_band_score(80.0, 20.0, 80.0, 10.0, 90.0) == 0.0
        score_ll = tolerance_band_score(10.0, 20.0, 80.0, 10.0, 90.0)
        assert abs(score_ll - 1.0) < 1e-6
        score_ul = tolerance_band_score(90.0, 20.0, 80.0, 10.0, 90.0)
        assert abs(score_ul - 1.0) < 1e-6

    def test_midpoint_warning_zone(self) -> None:
        """Midpoint of warning zone should be ~0.5."""
        assert abs(tolerance_band_score(15.0, 20.0, 80.0, 10.0, 90.0) - 0.5) < 1e-6
        assert abs(tolerance_band_score(85.0, 20.0, 80.0, 10.0, 90.0) - 0.5) < 1e-6


# ============================================================================
# Physics Constraint Layer Tests
# ============================================================================


class TestPhysicsConstraintLayer:
    """Tests for PhysicsConstraintLayer."""

    def test_normal_forecast_low_scores(
        self, physics_layer: PhysicsConstraintLayer
    ) -> None:
        """All values at 230V -> scores near 0."""
        forecast = np.full(10, 230.0)
        scores, details = physics_layer.evaluate(forecast)
        assert np.all(scores < 0.1)

    def test_voltage_violation_high_scores(
        self, physics_layer: PhysicsConstraintLayer
    ) -> None:
        """Values at 260V -> scores near/at 1.0."""
        forecast = np.full(10, 260.0)
        scores, details = physics_layer.evaluate(forecast)
        assert np.all(scores >= 0.9)

    def test_capacity_violation(
        self, physics_layer: PhysicsConstraintLayer
    ) -> None:
        """Values exceeding absolute_max_kw -> high scores."""
        forecast = np.full(5, 230.0)
        power_values = np.full(5, 120.0)
        scores, details = physics_layer.evaluate(
            forecast, power_values=power_values
        )
        assert np.all(details["capacity_scores"] >= 0.9)

    def test_ramp_rate_violation(
        self, physics_layer: PhysicsConstraintLayer
    ) -> None:
        """Large consecutive differences -> high ramp scores."""
        forecast = np.full(5, 230.0)
        power_values = np.array([10.0, 20.0, 10.0, 20.0, 10.0])
        scores, details = physics_layer.evaluate(
            forecast, power_values=power_values
        )
        assert np.any(details["ramp_scores"] > 0.5)

    def test_output_shape_matches_input(
        self, physics_layer: PhysicsConstraintLayer
    ) -> None:
        """Scores array length == forecast length."""
        for n in [1, 5, 20, 100]:
            forecast = np.full(n, 230.0)
            scores, _ = physics_layer.evaluate(forecast)
            assert len(scores) == n

    def test_scores_in_range(
        self, physics_layer: PhysicsConstraintLayer
    ) -> None:
        """All scores in [0, 1]."""
        forecast = np.random.uniform(200.0, 260.0, size=50)
        scores, _ = physics_layer.evaluate(forecast)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_details_contains_all_constraints(
        self, physics_layer: PhysicsConstraintLayer
    ) -> None:
        """Details dict has voltage_scores, capacity_scores, ramp_scores, combined_scores."""
        forecast = np.full(5, 230.0)
        _, details = physics_layer.evaluate(forecast)
        for key in ["voltage_scores", "capacity_scores", "ramp_scores", "combined_scores"]:
            assert key in details
            assert len(details[key]) == 5


# ============================================================================
# Cascade Logic Tests
# ============================================================================


class TestCascadeLogicLayer:
    """Tests for CascadeLogicLayer."""

    def test_isolated_anomaly_low_cascade(
        self,
        cascade_layer: CascadeLogicLayer,
        simple_edge_index: torch.LongTensor,
    ) -> None:
        """Single anomalous node with normal neighbors -> low cascade."""
        physics = np.array([0.0, 0.0, 0.8, 0.0, 0.0])
        gnn = np.zeros(5)
        scores = cascade_layer.evaluate(physics, gnn, simple_edge_index)
        assert scores[2] == 0.0

    def test_propagating_anomaly_high_cascade(
        self,
        cascade_layer: CascadeLogicLayer,
        simple_edge_index: torch.LongTensor,
    ) -> None:
        """Cluster of anomalous nodes -> high cascade for connected nodes."""
        physics = np.array([0.0, 0.8, 0.8, 0.8, 0.0])
        gnn = np.zeros(5)
        scores = cascade_layer.evaluate(physics, gnn, simple_edge_index)
        assert scores[2] > 0.5

    def test_no_anomaly_zero_cascade(
        self,
        cascade_layer: CascadeLogicLayer,
        simple_edge_index: torch.LongTensor,
    ) -> None:
        """All normal nodes -> all cascade scores 0."""
        physics = np.zeros(5)
        gnn = np.zeros(5)
        scores = cascade_layer.evaluate(physics, gnn, simple_edge_index)
        assert np.all(scores == 0.0)

    def test_empty_graph_no_crash(
        self,
        cascade_layer: CascadeLogicLayer,
    ) -> None:
        """Handle graph with no edges gracefully."""
        physics = np.array([0.5, 0.5])
        gnn = np.zeros(2)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        scores = cascade_layer.evaluate(physics, gnn, edge_index)
        assert len(scores) == 2
        assert np.all(scores == 0.0)

    def test_cascade_scores_in_range(
        self,
        cascade_layer: CascadeLogicLayer,
        simple_edge_index: torch.LongTensor,
    ) -> None:
        """All scores in [0, 1]."""
        physics = np.random.uniform(0, 1, size=5)
        gnn = np.random.uniform(0, 1, size=5)
        scores = cascade_layer.evaluate(physics, gnn, simple_edge_index)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


# ============================================================================
# Ensemble Combination Tests
# ============================================================================


class TestEnsembleCombination:
    """Tests for the _combine_scores ensemble function."""

    def test_normal_nodes_weighted_average(self) -> None:
        """Non-exited nodes use configured weights."""
        physics = np.array([0.5, 0.5])
        gnn = np.array([0.3, 0.3])
        cascade = np.array([0.1, 0.1])
        mask = np.array([False, False])
        weights = EnsembleWeightsConfig()

        combined, _ = _combine_scores(physics, gnn, cascade, mask, weights)

        expected = (0.4 * 0.5 + 0.4 * 0.3 + 0.2 * 0.1) / 1.0
        assert np.allclose(combined, expected)

    def test_early_exit_nodes_physics_only(self) -> None:
        """Exited nodes get physics score, gnn and cascade ignored."""
        physics = np.array([0.95, 0.95])
        gnn = np.array([0.1, 0.1])
        cascade = np.array([0.1, 0.1])
        mask = np.array([True, True])
        weights = EnsembleWeightsConfig()

        combined, _ = _combine_scores(physics, gnn, cascade, mask, weights)
        assert np.allclose(combined, 0.95)

    def test_mixed_early_exit(self) -> None:
        """Some nodes exited, some not; verify correct per-node handling."""
        physics = np.array([0.95, 0.3])
        gnn = np.array([0.1, 0.6])
        cascade = np.array([0.0, 0.2])
        mask = np.array([True, False])
        weights = EnsembleWeightsConfig()

        combined, _ = _combine_scores(physics, gnn, cascade, mask, weights)

        assert abs(combined[0] - 0.95) < 1e-6
        expected_1 = (0.4 * 0.3 + 0.4 * 0.6 + 0.2 * 0.2) / 1.0
        assert abs(combined[1] - expected_1) < 1e-6

    def test_breakdown_contains_all_fields(self) -> None:
        """Details dict has all required fields."""
        physics = np.array([0.5])
        gnn = np.array([0.3])
        cascade = np.array([0.1])
        mask = np.array([False])
        weights = EnsembleWeightsConfig()

        _, breakdown = _combine_scores(physics, gnn, cascade, mask, weights)

        for key in [
            "physics_scores", "gnn_scores", "cascade_scores",
            "combined_scores", "early_exit_mask", "early_exit_count", "weights",
        ]:
            assert key in breakdown, f"Missing key: {key}"
