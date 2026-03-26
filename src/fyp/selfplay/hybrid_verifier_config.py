"""Configuration models for the hybrid verifier ensemble.

Pydantic v2 config models for physics constraints, cascade logic,
ensemble weights, and the top-level HybridVerifierConfig. All physics
thresholds are loaded from YAML config files so they can be tuned per
network region without code changes.

UK standards references:
- BS EN 50160 / UK G59/3: voltage limits for LV distribution
- BS 7671:2018: domestic supply capacity limits
- SSEN feeder metadata: transformer capacity distribution
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Voltage constraints (BS EN 50160 / UK G59/3)
# ---------------------------------------------------------------------------

class VoltageConstraintConfig(BaseModel):
    """BS EN 50160 voltage constraint configuration.

    UK statutory voltage limits for LV distribution:
    - Nominal: 230V (harmonised European standard)
    - Lower limit: -10% (207V) absolute minimum
    - Upper limit: +10% (253V) absolute maximum
    - Safe zone narrower than absolute limits for graduated warning
    """

    nominal_v: float = Field(
        default=230.0, description="Nominal voltage (V)"
    )
    lower_limit_pct: float = Field(
        default=-10.0, description="Lower absolute limit as % of nominal"
    )
    lower_safe_pct: float = Field(
        default=-6.0, description="Lower safe zone boundary as % of nominal"
    )
    upper_safe_pct: float = Field(
        default=8.0, description="Upper safe zone boundary as % of nominal"
    )
    upper_limit_pct: float = Field(
        default=10.0, description="Upper absolute limit as % of nominal"
    )


# ---------------------------------------------------------------------------
# Capacity constraints (BS 7671:2018 + SSEN feeder metadata)
# ---------------------------------------------------------------------------

class CapacityConstraintConfig(BaseModel):
    """Feeder/transformer capacity constraint configuration.

    Based on UK Domestic Supply Standards (BS 7671:2018) and SSEN feeder
    metadata analysis:
    - typical_max_kw: 15 kW typical domestic maximum demand
    - absolute_max_kw: 100 kW physical fuse limit (100A at 230V)
    - overload_threshold_pct: 80% of capacity triggers warning
    - critical_threshold_pct: 95% of capacity is near-violation
    """

    typical_max_kw: float = Field(
        default=15.0, ge=0, description="Typical max domestic load (kW)"
    )
    absolute_max_kw: float = Field(
        default=100.0, ge=0, description="Absolute max load from fuse rating (kW)"
    )
    overload_threshold_pct: float = Field(
        default=80.0, ge=0, le=100,
        description="Warning threshold as % of capacity range",
    )
    critical_threshold_pct: float = Field(
        default=95.0, ge=0, le=100,
        description="Critical threshold as % of capacity range",
    )


# ---------------------------------------------------------------------------
# Ramp rate constraints
# ---------------------------------------------------------------------------

class RampRateConstraintConfig(BaseModel):
    """Ramp rate constraint configuration.

    Limits on how fast power demand can change between consecutive
    30-minute intervals.  Rapid ramps indicate equipment switching,
    EV charging surges, or potential anomalies.
    """

    max_ramp_kw_per_interval: float = Field(
        default=5.0, ge=0,
        description="Maximum allowable change per 30-min interval (kW)",
    )
    warning_ramp_kw_per_interval: float = Field(
        default=3.5, ge=0,
        description="Warning threshold for ramp rate (kW)",
    )


# ---------------------------------------------------------------------------
# Composite physics config
# ---------------------------------------------------------------------------

class PhysicsConfig(BaseModel):
    """Combined physics constraints configuration grouping voltage,
    capacity, and ramp-rate sub-configs."""

    voltage: VoltageConstraintConfig = Field(
        default_factory=VoltageConstraintConfig,
    )
    capacity: CapacityConstraintConfig = Field(
        default_factory=CapacityConstraintConfig,
    )
    ramp_rate: RampRateConstraintConfig = Field(
        default_factory=RampRateConstraintConfig,
    )


# ---------------------------------------------------------------------------
# Cascade logic config
# ---------------------------------------------------------------------------

class CascadeConfig(BaseModel):
    """Cascade logic layer configuration.

    Controls how anomaly signals propagate through the grid topology.
    A node must exceed propagation_threshold to be considered for
    cascade scoring, and signal decays by decay_factor per hop.
    """

    propagation_threshold: float = Field(
        default=0.3, ge=0, le=1,
        description="Min anomaly signal to trigger cascade check",
    )
    decay_factor: float = Field(
        default=0.7, ge=0, le=1,
        description="Signal decay per hop in the grid topology",
    )


# ---------------------------------------------------------------------------
# Ensemble weights config
# ---------------------------------------------------------------------------

class EnsembleWeightsConfig(BaseModel):
    """Ensemble layer weights for combining physics, GNN, and cascade scores.

    Weights should sum to 1.0 for interpretable combination, though this
    is not strictly enforced to allow experimentation.
    """

    physics: float = Field(
        default=0.4, ge=0, le=1, description="Physics layer weight"
    )
    gnn: float = Field(
        default=0.4, ge=0, le=1, description="GNN layer weight"
    )
    cascade: float = Field(
        default=0.2, ge=0, le=1, description="Cascade layer weight"
    )


# ---------------------------------------------------------------------------
# Top-level hybrid verifier config
# ---------------------------------------------------------------------------

class HybridVerifierConfig(BaseModel):
    """Full hybrid verifier configuration.

    Groups physics constraints, cascade logic, ensemble weights, and
    top-level parameters (early-exit threshold, penalty ratios, model path).
    """

    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    ensemble_weights: EnsembleWeightsConfig = Field(
        default_factory=EnsembleWeightsConfig,
    )
    early_exit_threshold: float = Field(
        default=0.9, ge=0, le=1,
        description="Physics severity threshold for early exit (skip GNN)",
    )
    false_negative_penalty_ratio: float = Field(
        default=2.0, ge=1,
        description="FN penalty multiplier relative to FP penalty",
    )
    gnn_checkpoint_path: str = Field(
        default="data/derived/models/gnn/gnn_verifier_v1.pth",
        description="Path to trained GATVerifier checkpoint",
    )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_hybrid_verifier_config(
    config_path: str | Path,
) -> HybridVerifierConfig:
    """Load and validate hybrid verifier configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Validated HybridVerifierConfig instance.

    If the file does not exist, logs a warning and returns defaults.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "Config file %s not found, using defaults", config_path
        )
        return HybridVerifierConfig()

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        logger.warning(
            "Config file %s is empty, using defaults", config_path
        )
        return HybridVerifierConfig()

    return HybridVerifierConfig.model_validate(data)
