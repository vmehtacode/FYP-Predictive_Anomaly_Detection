"""Hybrid verifier ensemble for the self-play anomaly detection loop.

This module implements the multi-layer verification ensemble that combines:
  - Layer 1: Physics constraints (voltage, capacity, ramp rate)
  - Layer 2: GNN-based pattern detection (GATVerifier, added in Plan 02-02)
  - Layer 3: Cascade logic (added in Plan 02-02)

The physics constraint layer produces continuous severity scores in [0, 1]
using a tolerance band approach: safe zone = 0, graduated warning zone,
full violation = 1.  All thresholds are config-driven (no hardcoded values).
"""

from __future__ import annotations

import numpy as np

from fyp.selfplay.hybrid_verifier_config import PhysicsConfig


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
# Cascade Logic Layer (Layer 3) — to be added in Plan 02-02
# ============================================================================

# ============================================================================
# HybridVerifierAgent (ensemble orchestrator) — to be added in Plan 02-02
# ============================================================================
