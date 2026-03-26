"""Hybrid verifier ensemble for the self-play anomaly detection loop.

This module implements the multi-layer verification ensemble that combines:
  - Layer 1: Physics constraints (voltage, capacity, ramp rate)
  - Layer 2: GNN-based pattern detection (frozen GATVerifier from Phase 1)
  - Layer 3: Cascade logic (neighbor anomaly propagation in grid topology)

The physics constraint layer produces continuous severity scores in [0, 1]
using a tolerance band approach: safe zone = 0, graduated warning zone,
full violation = 1.  All thresholds are config-driven (no hardcoded values).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from fyp.gnn.gat_verifier import GATVerifier
from fyp.selfplay.hybrid_verifier_config import (
    CascadeConfig,
    EnsembleWeightsConfig,
    HybridVerifierConfig,
    PhysicsConfig,
    load_hybrid_verifier_config,
)

if TYPE_CHECKING:
    from fyp.selfplay.proposer import ScenarioProposal

logger = logging.getLogger(__name__)


# ============================================================================
# Tolerance Band Scoring
# ============================================================================


def tolerance_band_score(
    value: float,
    lower_safe: float,
    upper_safe: float,
    lower_limit: float,
    upper_limit: float,
) -> float:
    """Compute continuous severity score using tolerance band approach.

    Three zones produce graduated responses:

        [lower_safe, upper_safe]       -> 0.0 (fully safe)
        [lower_limit, lower_safe)      -> linear 0.0 to 1.0 (approaching limit)
        (upper_safe, upper_limit]      -> linear 0.0 to 1.0 (approaching limit)
        < lower_limit or > upper_limit -> 1.0 (full violation)

    Args:
        value: Measured quantity (voltage, power, ramp rate, etc.).
        lower_safe: Lower bound of the safe zone.
        upper_safe: Upper bound of the safe zone.
        lower_limit: Absolute lower limit (full violation below this).
        upper_limit: Absolute upper limit (full violation above this).

    Returns:
        Severity score in [0.0, 1.0].
    """
    if lower_safe <= value <= upper_safe:
        return 0.0
    elif value < lower_limit or value > upper_limit:
        return 1.0
    elif value < lower_safe:
        # Lower warning zone: linear interpolation
        return (lower_safe - value) / (lower_safe - lower_limit)
    else:
        # Upper warning zone: linear interpolation
        return (value - upper_safe) / (upper_limit - upper_safe)


# ============================================================================
# Physics Constraint Layer (Layer 1 of ensemble)
# ============================================================================


class PhysicsConstraintLayer:
    """First layer of the hybrid verifier ensemble.

    Evaluates physics constraints (voltage, capacity, ramp rate) on
    per-node forecast values and returns continuous severity scores.
    All thresholds come from :class:`PhysicsConfig` (loaded from YAML).

    Attributes:
        config: Physics constraint configuration.
        voltage_lower_safe: Absolute lower safe voltage (V).
        voltage_upper_safe: Absolute upper safe voltage (V).
        voltage_lower_limit: Absolute lower voltage limit (V).
        voltage_upper_limit: Absolute upper voltage limit (V).
    """

    def __init__(self, config: PhysicsConfig) -> None:
        self.config = config

        # Pre-compute absolute voltage bounds from percentages
        v = config.voltage
        self.voltage_lower_safe = v.nominal_v * (1 + v.lower_safe_pct / 100)
        self.voltage_upper_safe = v.nominal_v * (1 + v.upper_safe_pct / 100)
        self.voltage_lower_limit = v.nominal_v * (1 + v.lower_limit_pct / 100)
        self.voltage_upper_limit = v.nominal_v * (1 + v.upper_limit_pct / 100)

        # Pre-compute capacity bounds
        c = config.capacity
        # Warning zone: between overload% and critical% of the range
        # [typical_max_kw .. absolute_max_kw]
        capacity_range = c.absolute_max_kw - c.typical_max_kw
        self.capacity_lower_safe = 0.0
        self.capacity_upper_safe = c.typical_max_kw + (
            capacity_range * c.overload_threshold_pct / 100
        )
        self.capacity_lower_limit = 0.0  # non-negative power
        self.capacity_upper_limit = c.absolute_max_kw

        # Pre-compute ramp rate bounds (symmetric around 0)
        r = config.ramp_rate
        self.ramp_warning = r.warning_ramp_kw_per_interval
        self.ramp_max = r.max_ramp_kw_per_interval

    def evaluate(
        self,
        forecast: np.ndarray,
        *,
        voltage_values: np.ndarray | None = None,
        power_values: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Evaluate physics constraints on per-node data.

        By default all three constraints are scored against *forecast*.
        Callers can provide separate arrays for voltage and power when
        the forecast represents a single quantity (e.g., power only)
        and voltage data comes from a different source.

        Args:
            forecast: 1-D array of per-node forecast values (fallback for
                all constraints when specific arrays are not provided).
            voltage_values: Optional 1-D array of per-node voltages (V).
                Falls back to *forecast* if ``None``.
            power_values: Optional 1-D array of per-node power (kW).
                Falls back to *forecast* if ``None``.

        Returns:
            A tuple of ``(severity_scores, details)`` where
            *severity_scores* is a 1-D array of combined [0, 1] scores
            (one per node) and *details* is a dict of per-constraint arrays.
        """
        forecast = np.asarray(forecast, dtype=np.float64)
        n_nodes = forecast.shape[0]

        v_data = (
            np.asarray(voltage_values, dtype=np.float64)
            if voltage_values is not None
            else forecast
        )
        p_data = (
            np.asarray(power_values, dtype=np.float64)
            if power_values is not None
            else None
        )

        # --- Voltage scoring ---
        voltage_scores = np.array(
            [
                tolerance_band_score(
                    float(v),
                    self.voltage_lower_safe,
                    self.voltage_upper_safe,
                    self.voltage_lower_limit,
                    self.voltage_upper_limit,
                )
                for v in v_data
            ]
        )

        # --- Capacity scoring ---
        # Only scored when power data is explicitly provided or the
        # forecast values are in a plausible kW range (heuristic: the
        # values are below 2x absolute_max_kw, indicating they are
        # power measurements rather than voltages).
        if p_data is not None:
            cap_input = p_data
        elif np.max(np.abs(forecast)) <= 2 * self.config.capacity.absolute_max_kw:
            cap_input = forecast
        else:
            cap_input = None

        if cap_input is not None:
            capacity_scores = np.array(
                [
                    tolerance_band_score(
                        abs(float(v)),
                        self.capacity_lower_safe,
                        self.capacity_upper_safe,
                        self.capacity_lower_limit,
                        self.capacity_upper_limit,
                    )
                    for v in cap_input
                ]
            )
        else:
            capacity_scores = np.zeros(n_nodes)

        # --- Ramp rate scoring ---
        # Ramp = difference between consecutive values (power data
        # preferred, falls back to forecast only when in kW range).
        # First node has no predecessor so ramp score = 0.
        ramp_input = cap_input  # same data source as capacity
        ramp_scores = np.zeros(n_nodes)
        if ramp_input is not None and n_nodes > 1:
            diffs = np.diff(ramp_input)
            for i, d in enumerate(diffs):
                ramp_scores[i + 1] = tolerance_band_score(
                    abs(float(d)),
                    0.0,
                    self.ramp_warning,
                    0.0,
                    self.ramp_max,
                )

        # --- Combined: worst-case (max) per node ---
        combined_scores = np.maximum(
            np.maximum(voltage_scores, capacity_scores), ramp_scores
        )

        details = {
            "voltage_scores": voltage_scores,
            "capacity_scores": capacity_scores,
            "ramp_scores": ramp_scores,
            "combined_scores": combined_scores,
        }

        return combined_scores, details


# ============================================================================
# Cascade Logic Layer (Layer 3 of ensemble)
# ============================================================================


class CascadeLogicLayer:
    """Third layer of the hybrid verifier ensemble.

    Scores nodes based on neighbor anomaly propagation patterns in the grid
    topology.  A node's cascade score reflects how many of its neighbors are
    also anomalous, indicating a potential propagation event rather than an
    isolated spike.

    The layer takes physics and GNN scores as input, builds an adjacency
    structure from the COO edge_index, and computes a per-node cascade score
    in [0, 1].

    Attributes:
        propagation_threshold: Minimum signal to trigger cascade check.
        decay_factor: Signal decay per hop (reserved for future multi-hop).
    """

    def __init__(self, config: CascadeConfig) -> None:
        self.propagation_threshold = config.propagation_threshold
        self.decay_factor = config.decay_factor

    @staticmethod
    def _build_adjacency(
        edge_index: torch.Tensor, num_nodes: int
    ) -> dict[int, list[int]]:
        """Build adjacency dict from COO edge_index.

        Args:
            edge_index: Shape [2, num_edges] COO format tensor.
            num_nodes: Total number of nodes in the graph.

        Returns:
            Dict mapping each node index to a list of neighbor indices.
        """
        adj: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            adj[s].append(d)
        return adj

    def evaluate(
        self,
        physics_scores: np.ndarray,
        gnn_scores: np.ndarray,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        """Compute per-node cascade scores.

        Args:
            physics_scores: Per-node physics severity [0, 1].
            gnn_scores: Per-node GNN anomaly scores [0, 1].
            edge_index: COO edge tensor [2, num_edges].

        Returns:
            Per-node cascade scores in [0, 1].
        """
        num_nodes = len(physics_scores)
        base_signal = np.maximum(physics_scores, gnn_scores)
        adj = self._build_adjacency(edge_index, num_nodes)

        cascade_scores = np.zeros(num_nodes)
        for node in range(num_nodes):
            if base_signal[node] <= self.propagation_threshold:
                continue
            neighbors = adj[node]
            if not neighbors:
                continue
            anomalous_neighbors = sum(
                1
                for n in neighbors
                if base_signal[n] > self.propagation_threshold
            )
            cascade_scores[node] = anomalous_neighbors / len(neighbors)

        return cascade_scores


# ============================================================================
# GNN / ensemble helper methods (used by HybridVerifierAgent)
# ============================================================================


def _build_node_features(
    forecast: np.ndarray,
    num_graph_nodes: int,
    temporal_features: int = 5,
) -> torch.Tensor:
    """Map a 1-D forecast array to per-node feature tensors for GNN input.

    LV feeder nodes (first N values of forecast) are assigned their
    forecast values directly.  Remaining upstream nodes receive the mean
    forecast value.  The single value per node is tiled across
    *temporal_features* dimensions to match the GATVerifier training
    configuration.

    Args:
        forecast: 1-D array of forecast values.
        num_graph_nodes: Total nodes in the graph.
        temporal_features: Feature dimension expected by the GNN (default 5).

    Returns:
        Tensor of shape [num_graph_nodes, temporal_features].
    """
    mean_val = float(np.mean(forecast)) if len(forecast) > 0 else 0.0
    node_values = np.full(num_graph_nodes, mean_val)

    # Assign forecast values to LV feeder nodes (first N graph nodes or
    # first N forecast values, whichever is smaller)
    n_assign = min(len(forecast), num_graph_nodes)
    node_values[:n_assign] = forecast[:n_assign]

    # Tile across temporal feature dimension
    features = np.tile(node_values[:, np.newaxis], (1, temporal_features))
    return torch.tensor(features, dtype=torch.float32)


def _run_gnn_inference(
    gnn_model: GATVerifier,
    forecast: np.ndarray,
    edge_index: torch.Tensor,
    num_graph_nodes: int,
    early_exit_mask: np.ndarray,
    node_type: torch.Tensor | None = None,
    temporal_features: int = 5,
) -> np.ndarray:
    """Run frozen GNN forward pass and return per-node anomaly scores.

    All nodes participate in the forward pass (early-exited nodes still
    contribute features to neighbors via message passing).  After inference,
    GNN scores for early-exited nodes are zeroed out (they use physics
    score only in the ensemble).

    Args:
        gnn_model: Frozen GATVerifier model.
        forecast: 1-D forecast array.
        edge_index: COO edge tensor.
        num_graph_nodes: Number of graph nodes.
        early_exit_mask: Boolean mask — True for early-exited nodes.
        node_type: Optional node type tensor for GNN.
        temporal_features: Temporal feature dimension.

    Returns:
        Per-node GNN anomaly scores in [0, 1], early-exited nodes zeroed.
    """
    x = _build_node_features(forecast, num_graph_nodes, temporal_features)

    with torch.no_grad():
        scores_tensor = gnn_model(x, edge_index, node_type=node_type)

    # scores_tensor shape: [num_graph_nodes, 1]
    gnn_scores = scores_tensor.squeeze(-1).cpu().numpy()

    # Zero out early-exited nodes — they use physics score only
    # Align masks: early_exit_mask may be shorter than gnn_scores
    n_mask = min(len(early_exit_mask), len(gnn_scores))
    gnn_scores[:n_mask][early_exit_mask[:n_mask]] = 0.0

    return gnn_scores


def _combine_scores(
    physics: np.ndarray,
    gnn: np.ndarray,
    cascade: np.ndarray,
    early_exit_mask: np.ndarray,
    weights: EnsembleWeightsConfig,
) -> tuple[np.ndarray, dict]:
    """Combine layer scores with configurable weighted average.

    Normal nodes use the full weighted average.  Early-exited nodes use
    physics score only (weights become physics=1.0, gnn=0.0, cascade=0.0).

    Args:
        physics: Per-node physics scores.
        gnn: Per-node GNN scores (already zeroed for early-exited).
        cascade: Per-node cascade scores.
        early_exit_mask: Boolean mask — True for early-exited nodes.
        weights: Ensemble weight configuration.

    Returns:
        Tuple of (combined_scores, breakdown_dict).
    """
    w_p, w_g, w_c = weights.physics, weights.gnn, weights.cascade
    total_weight = w_p + w_g + w_c
    if total_weight == 0:
        total_weight = 1.0  # safety

    # Normal weighted average
    combined = (w_p * physics + w_g * gnn + w_c * cascade) / total_weight

    # Override early-exited nodes: physics only
    combined[early_exit_mask] = physics[early_exit_mask]

    breakdown = {
        "physics_scores": physics,
        "gnn_scores": gnn,
        "cascade_scores": cascade,
        "combined_scores": combined,
        "early_exit_mask": early_exit_mask,
        "early_exit_count": int(np.sum(early_exit_mask)),
        "weights": {"physics": w_p, "gnn": w_g, "cascade": w_c},
    }
    return combined, breakdown


# ============================================================================
# HybridVerifierAgent (ensemble orchestrator) — to be added in Task 2
# ============================================================================
