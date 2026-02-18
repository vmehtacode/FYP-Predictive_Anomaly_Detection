# Phase 2: Hybrid Verifier Integration - Research

**Researched:** 2026-02-18
**Domain:** Physics-informed ensemble verifier for self-play anomaly detection training loop
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Physics constraint bounds
- Check three physical quantities: voltage bounds, capacity/loading, and ramp rates (no power balance)
- Constraints produce continuous severity scores (0-1 based on deviation magnitude), not binary flags
- Constraint values (voltage limits, capacity ratings, ramp rates) loaded from a YAML/JSON config file
- Tolerance band approach: approaching limit produces partial score, exceeding limit produces full violation (graduated response)

#### Early-exit cascade logic
- Nodes with severe physics violations skip GNN processing entirely -- marked anomalous immediately
- Early-exit threshold is configurable in the config file (not hardcoded)
- No cascade at this layer -- early-exit is node-local only, cascade logic handled separately in ensemble
- Early-exited nodes still contribute features to GNN message passing for their neighbors (neighbors benefit from context)

#### Ensemble score combination
- Three layers combine via weighted average with configurable fixed weights (not learned)
- For early-exited nodes: GNN and cascade weights set to 0, physics weight becomes 1.0 (physics score only)
- Output includes full breakdown: combined score + individual physics/GNN/cascade scores per node (for interpretability)

#### Reward signal for training
- HybridVerifier fully replaces the existing verifier -- single source of truth in the self-play loop
- Reward is continuous confidence-based (proportional to correct confidence, penalty for confident mistakes)
- Asymmetric error penalty is configurable -- ratio set in config for tuning (false negatives typically worse in grid safety)
- GNN component weights are frozen (use Phase 1 trained weights) -- only ensemble weights and physics thresholds configured

### Claude's Discretion
- Exact config file schema and default values for physics thresholds
- Internal implementation of tolerance band scoring function
- Cascade logic layer implementation details
- How to structure the HybridVerifierAgent class hierarchy

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GNN-03 | Replace MLP Verifier with GNN Verifier in self-play training loop | HybridVerifierAgent wraps GATVerifier (frozen weights) and integrates into SelfPlayTrainer via matching `.evaluate()` interface. See Architecture Patterns: "Drop-in Replacement" and "Frozen GNN Component" |
| ENS-01 | Integrate GNN learned detector as Layer 2 with physics constraints | Three-layer ensemble architecture: Layer 1 (physics), Layer 2 (GNN/GATVerifier), Layer 3 (cascade logic), combined via configurable weighted average. See Architecture Patterns: "Three-Layer Ensemble" |
| ENS-02 | Implement cascade early exit logic (physics violations skip GNN) | Early-exit mechanism in physics layer marks severe violations; GNN forward pass uses a mask so early-exited nodes skip attention computation but still contribute features to neighbor message passing. See Architecture Patterns: "Early-Exit with Neighbor Context" |

</phase_requirements>

## Summary

This phase replaces the existing `VerifierAgent` (a purely physics-constraint-based verifier operating on 1D forecast arrays) with a `HybridVerifierAgent` that combines three verification layers into a single ensemble: physics constraints producing continuous severity scores, the frozen GATVerifier from Phase 1 providing topology-aware anomaly scores, and a cascade logic layer capturing propagation patterns. The hybrid verifier must be a drop-in replacement for the existing verifier in the `SelfPlayTrainer`.

The key technical challenge is bridging two different data paradigms. The existing self-play loop operates on 1D numpy arrays (per-household forecasts), while the GATVerifier operates on PyG graph `Data` objects with per-node features and edge connectivity. The HybridVerifierAgent must accept both a forecast array and graph-structured data, run the physics constraints per-node, apply early-exit logic, invoke the frozen GNN on non-exited nodes, compute cascade scores, and produce a single reward signal compatible with the existing training loop's `verification_reward` expectation.

The second challenge is the reward signal transformation. The existing verifier returns rewards in [-1, +1] (penalties for violations, bonuses for difficulty). The new verifier must produce a continuous confidence-based reward where the signal is proportional to correct confidence and penalizes confident mistakes, with configurable asymmetric weighting for false negatives versus false positives (false negatives being worse in grid safety contexts).

**Primary recommendation:** Build `HybridVerifierAgent` as a composition class that holds `PhysicsConstraintLayer`, frozen `GATVerifier`, and `CascadeLogicLayer` as components. Use the existing Pydantic-based config pattern from `src/fyp/config.py` extended with a new `HybridVerifierConfig` model loaded from YAML. Match the existing `VerifierAgent.evaluate()` signature exactly so `SelfPlayTrainer` requires zero changes beyond swapping the verifier instance at construction time.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | ^2.1.0 | Already in project | Base framework for GNN inference and tensor operations |
| torch-geometric | ^2.7.0 | Already in project | GATVerifier model, graph Data objects, DataLoader |
| pydantic | ^2.5.0 | Already in project | Config validation for physics thresholds and ensemble weights |
| PyYAML | (via pydantic) | Already in project | YAML config file loading |
| numpy | ^1.24.0 | Already in project | Physics constraint computations on forecast arrays |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scikit-learn | ^1.3.0 | Already in project | Metrics computation for verifier evaluation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| YAML config | JSON config | YAML is more human-readable for nested physics params; project already uses both (params.yaml + ssen_constraints.json). Use YAML for new config since it supports comments for documenting threshold rationale |
| Pydantic BaseModel | dataclass + manual validation | Pydantic provides automatic type coercion, range validation via Field(ge=, le=), and better error messages. Already used in config.py |
| Composition pattern | Inheritance from VerifierAgent | Composition is cleaner because the hybrid verifier has fundamentally different internals (graph-based) vs the existing verifier (array-based). Inheritance would create awkward method overrides |

**Installation:**
```bash
# No new dependencies required -- all libraries already in pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure
```
src/fyp/
    selfplay/
        verifier.py               # EXISTING: VerifierAgent (unchanged, kept as reference)
        hybrid_verifier.py        # NEW: HybridVerifierAgent + ensemble logic
        hybrid_verifier_config.py # NEW: Pydantic config models + YAML loading
        trainer.py                # EXISTING: SelfPlayTrainer (unchanged)
    gnn/
        gat_verifier.py           # EXISTING: GATVerifier (frozen weights, unchanged)
        graph_builder.py          # EXISTING: GridGraphBuilder (unchanged)
configs/
    hybrid_verifier.yaml          # NEW: Default physics thresholds + ensemble weights
tests/
    test_hybrid_verifier.py       # NEW: Tests for all hybrid verifier components
```

### Pattern 1: Drop-in Replacement via Matching Interface
**What:** HybridVerifierAgent matches `VerifierAgent.evaluate()` signature exactly
**When to use:** When replacing the verifier in SelfPlayTrainer
**Why critical:** The trainer calls `self.verifier.evaluate(forecast=..., scenario=..., return_details=True)` and expects `(float, dict)` or `float`. The hybrid verifier must match this.

```python
# Source: Analysis of existing src/fyp/selfplay/trainer.py lines 160-161 and 378
class HybridVerifierAgent:
    """Three-layer ensemble verifier that replaces VerifierAgent in self-play."""

    def __init__(
        self,
        config: HybridVerifierConfig,
        gnn_model: GATVerifier,
        graph_data: Data,  # Pre-built grid topology
    ) -> None:
        self.config = config
        self.physics_layer = PhysicsConstraintLayer(config.physics)
        self.gnn_model = gnn_model
        self.gnn_model.eval()  # Frozen -- no training
        for param in self.gnn_model.parameters():
            param.requires_grad = False
        self.cascade_layer = CascadeLogicLayer(config.cascade)
        self.graph_data = graph_data
        self.weights = config.ensemble_weights  # {physics: 0.4, gnn: 0.4, cascade: 0.2}

    def evaluate(
        self,
        forecast: np.ndarray,
        scenario: ScenarioProposal | None = None,
        timestamps: np.ndarray | None = None,
        return_details: bool = False,
    ) -> float | tuple[float, dict]:
        """Match VerifierAgent.evaluate() signature exactly."""
        # Step 1: Physics constraints (per-node continuous severity scores)
        physics_scores, physics_details = self.physics_layer.evaluate(forecast)

        # Step 2: Early-exit check
        early_exit_mask = physics_scores > self.config.early_exit_threshold

        # Step 3: GNN inference on non-exited nodes (with graph context)
        gnn_scores = self._run_gnn_inference(forecast, early_exit_mask)

        # Step 4: Cascade logic
        cascade_scores = self.cascade_layer.evaluate(
            physics_scores, gnn_scores, self.graph_data.edge_index
        )

        # Step 5: Ensemble combination
        combined, breakdown = self._combine_scores(
            physics_scores, gnn_scores, cascade_scores, early_exit_mask
        )

        # Step 6: Convert to reward signal
        reward = self._compute_reward(combined, scenario)

        if return_details:
            return reward, breakdown
        return reward
```

### Pattern 2: Tolerance Band Scoring Function (Claude's Discretion)
**What:** Continuous severity score [0, 1] based on how far a value deviates from safe bounds
**When to use:** All three physics constraints (voltage, capacity, ramp rate)
**Recommendation:** Piecewise linear function with three zones: safe (0), warning/tolerance (0 to threshold), violation (threshold to 1.0)

```python
def tolerance_band_score(
    value: float,
    lower_safe: float,
    upper_safe: float,
    lower_limit: float,
    upper_limit: float,
) -> float:
    """Compute continuous severity score using tolerance band approach.

    Zones:
        [lower_safe, upper_safe]     -> 0.0 (fully safe)
        [lower_limit, lower_safe)    -> linear 0.0 to 1.0 (approaching limit)
        (upper_safe, upper_limit]    -> linear 0.0 to 1.0 (approaching limit)
        < lower_limit or > upper_limit -> 1.0 (full violation)

    Args:
        value: Measured quantity
        lower_safe: Lower bound of safe zone
        upper_safe: Upper bound of safe zone
        lower_limit: Absolute lower limit
        upper_limit: Absolute upper limit

    Returns:
        Severity score in [0.0, 1.0]
    """
    if lower_safe <= value <= upper_safe:
        return 0.0
    elif value < lower_limit or value > upper_limit:
        return 1.0
    elif value < lower_safe:
        # In lower warning zone: linear interpolation
        return (lower_safe - value) / (lower_safe - lower_limit)
    else:
        # In upper warning zone: linear interpolation
        return (value - upper_safe) / (upper_limit - upper_safe)
```

**Default voltage values (BS EN 50160 / UK G59/3):**
- `lower_limit` = 207.0V (230V * 0.9, -10%)
- `lower_safe` = 216.2V (230V * 0.94, -6%)
- `upper_safe` = 248.4V (230V * 1.08, +8%)
- `upper_limit` = 253.0V (230V * 1.10, +10%)

The safe zone is narrower than the absolute limits, creating a graduated warning band. This is consistent with the SSEN constraints already in `data/derived/ssen_constraints.json` which uses `min_v: 207.0` and `max_v: 253.0`.

### Pattern 3: Early-Exit with Neighbor Context
**What:** Nodes exceeding physics threshold skip GNN scoring but still provide features to neighbors
**When to use:** When a node has clear physics violation (severity > threshold)
**Implementation detail:** Run full GNN forward pass on all nodes, but only use GNN scores for non-exited nodes. This is simpler and more correct than masking nodes out of message passing.

```python
def _run_gnn_inference(
    self,
    forecast: np.ndarray,
    early_exit_mask: np.ndarray,  # True = skip GNN scoring for this node
) -> np.ndarray:
    """Run GATVerifier inference. All nodes participate in message passing.

    Early-exited nodes contribute features to neighbors via GNN message passing
    (their features are in the graph), but their GNN scores are discarded in
    the ensemble step (physics score used instead).
    """
    # Build temporal features from forecast for all nodes
    node_features = self._build_node_features(forecast)

    # Run full GNN forward pass (all nodes participate in message passing)
    with torch.no_grad():
        gnn_scores = self.gnn_model(
            node_features,
            self.graph_data.edge_index,
            self.graph_data.node_type,
        )

    scores = gnn_scores.squeeze().numpy()

    # Zero out GNN scores for early-exited nodes
    # (they will use physics score only in ensemble)
    scores[early_exit_mask] = 0.0

    return scores
```

**Why this works:** The GATv2Conv attention mechanism naturally handles nodes with extreme features (high severity). Neighbors see these extreme features during message passing and can adjust their own attention weights accordingly. The key insight from the locked decision is that "early-exited nodes still contribute features to GNN message passing for their neighbors" -- running the full forward pass and discarding scores afterward satisfies this requirement cleanly.

### Pattern 4: Three-Layer Ensemble Score Combination
**What:** Weighted average of physics, GNN, and cascade scores with early-exit handling
**When to use:** Final score aggregation per node

```python
def _combine_scores(
    self,
    physics_scores: np.ndarray,
    gnn_scores: np.ndarray,
    cascade_scores: np.ndarray,
    early_exit_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Combine three layers via weighted average.

    For early-exited nodes: physics weight = 1.0, others = 0.0
    For normal nodes: use configured weights
    """
    w = self.weights
    combined = np.zeros_like(physics_scores)

    # Normal nodes: weighted average
    normal_mask = ~early_exit_mask
    combined[normal_mask] = (
        w.physics * physics_scores[normal_mask]
        + w.gnn * gnn_scores[normal_mask]
        + w.cascade * cascade_scores[normal_mask]
    )

    # Early-exited nodes: physics score only
    combined[early_exit_mask] = physics_scores[early_exit_mask]

    breakdown = {
        "combined_scores": combined,
        "physics_scores": physics_scores,
        "gnn_scores": gnn_scores,
        "cascade_scores": cascade_scores,
        "early_exit_mask": early_exit_mask,
        "early_exit_count": int(early_exit_mask.sum()),
        "weights": {"physics": w.physics, "gnn": w.gnn, "cascade": w.cascade},
    }

    return combined, breakdown
```

### Pattern 5: Confidence-Based Reward Signal
**What:** Transform per-node anomaly scores into a single reward for the training loop
**When to use:** Converting HybridVerifier output to verification_reward for SelfPlayTrainer

```python
def _compute_reward(
    self,
    combined_scores: np.ndarray,
    scenario: ScenarioProposal | None,
) -> float:
    """Compute continuous confidence-based reward.

    High confidence on correct predictions -> positive reward
    High confidence on incorrect predictions -> penalty
    Asymmetric: false negatives penalized more than false positives

    Returns:
        reward in [-1, +1] compatible with existing training loop
    """
    # Mean anomaly score across all nodes
    mean_score = float(np.mean(combined_scores))

    # Confidence = how far from 0.5 (uncertain)
    confidence = abs(mean_score - 0.5) * 2  # Scale to [0, 1]

    # Determine if prediction aligns with scenario
    if scenario is not None:
        # Scenario present -> expect anomaly (higher scores = correct)
        is_correct = mean_score > 0.5
    else:
        # No scenario -> expect normal (lower scores = correct)
        is_correct = mean_score <= 0.5

    if is_correct:
        reward = confidence  # Reward proportional to correct confidence
    else:
        # Penalty proportional to incorrect confidence
        # Asymmetric: false negatives (missing real anomaly) worse
        if scenario is not None and mean_score <= 0.5:
            # False negative: missed anomaly
            reward = -confidence * self.config.false_negative_penalty_ratio
        else:
            # False positive: false alarm
            reward = -confidence

    return float(np.clip(reward, -1.0, 1.0))
```

### Pattern 6: Cascade Logic Layer (Claude's Discretion)
**What:** Score that captures anomaly propagation patterns in the grid topology
**Recommendation:** Use a simple neighbor-aggregation approach that checks if a node's neighbors also show elevated anomaly signals, indicating a cascading/propagating event.

```python
class CascadeLogicLayer:
    """Detect anomaly propagation patterns in grid topology.

    Scores how many of a node's neighbors also show anomalous behavior,
    indicating a cascade rather than an isolated event.
    """

    def __init__(self, config: CascadeConfig) -> None:
        self.propagation_threshold = config.propagation_threshold  # e.g., 0.3
        self.decay_factor = config.decay_factor  # e.g., 0.7 per hop

    def evaluate(
        self,
        physics_scores: np.ndarray,
        gnn_scores: np.ndarray,
        edge_index: torch.Tensor,
    ) -> np.ndarray:
        """Compute cascade scores based on neighbor anomaly patterns.

        A node gets a high cascade score if its neighbors also have
        elevated physics or GNN scores, suggesting propagation.
        """
        # Combine physics and GNN for base signal
        base_signal = np.maximum(physics_scores, gnn_scores)

        # Build adjacency
        num_nodes = len(base_signal)
        adj = self._build_adjacency(edge_index, num_nodes)

        cascade_scores = np.zeros(num_nodes)
        for node in range(num_nodes):
            if base_signal[node] < self.propagation_threshold:
                continue  # Only score nodes with some anomaly signal

            neighbors = adj.get(node, [])
            if not neighbors:
                continue

            # Fraction of neighbors also showing anomaly
            neighbor_signals = base_signal[neighbors]
            anomalous_neighbors = np.sum(neighbor_signals > self.propagation_threshold)
            cascade_scores[node] = anomalous_neighbors / len(neighbors)

        return cascade_scores
```

### Pattern 7: Config Schema (Claude's Discretion)
**What:** Pydantic models for YAML configuration
**Recommendation:** Extend the existing pattern from `src/fyp/config.py`

```python
from pydantic import BaseModel, Field

class VoltageConstraintConfig(BaseModel):
    """BS EN 50160 voltage constraint configuration."""
    nominal_v: float = Field(default=230.0, description="Nominal voltage (V)")
    lower_limit_pct: float = Field(default=-10.0, description="Lower absolute limit (%)")
    lower_safe_pct: float = Field(default=-6.0, description="Lower safe zone (%)")
    upper_safe_pct: float = Field(default=8.0, description="Upper safe zone (%)")
    upper_limit_pct: float = Field(default=10.0, description="Upper absolute limit (%)")

class CapacityConstraintConfig(BaseModel):
    """Feeder/transformer capacity constraint configuration."""
    typical_max_kw: float = Field(default=15.0, ge=0, description="Typical max load (kW)")
    absolute_max_kw: float = Field(default=100.0, ge=0, description="Absolute max load (kW)")
    overload_threshold_pct: float = Field(default=80.0, ge=0, le=100, description="Warning threshold (%)")
    critical_threshold_pct: float = Field(default=95.0, ge=0, le=100, description="Critical threshold (%)")

class RampRateConstraintConfig(BaseModel):
    """Ramp rate constraint configuration."""
    max_ramp_kw_per_interval: float = Field(default=5.0, ge=0, description="Max change per 30-min interval")
    warning_ramp_kw_per_interval: float = Field(default=3.5, ge=0, description="Warning threshold for ramp rate")

class PhysicsConfig(BaseModel):
    """Combined physics constraints configuration."""
    voltage: VoltageConstraintConfig = Field(default_factory=VoltageConstraintConfig)
    capacity: CapacityConstraintConfig = Field(default_factory=CapacityConstraintConfig)
    ramp_rate: RampRateConstraintConfig = Field(default_factory=RampRateConstraintConfig)

class CascadeConfig(BaseModel):
    """Cascade logic configuration."""
    propagation_threshold: float = Field(default=0.3, ge=0, le=1, description="Min anomaly signal for cascade check")
    decay_factor: float = Field(default=0.7, ge=0, le=1, description="Signal decay per hop")

class EnsembleWeightsConfig(BaseModel):
    """Ensemble layer weights (must sum to 1.0)."""
    physics: float = Field(default=0.4, ge=0, le=1, description="Physics layer weight")
    gnn: float = Field(default=0.4, ge=0, le=1, description="GNN layer weight")
    cascade: float = Field(default=0.2, ge=0, le=1, description="Cascade layer weight")

class HybridVerifierConfig(BaseModel):
    """Full hybrid verifier configuration."""
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    cascade: CascadeConfig = Field(default_factory=CascadeConfig)
    ensemble_weights: EnsembleWeightsConfig = Field(default_factory=EnsembleWeightsConfig)
    early_exit_threshold: float = Field(default=0.9, ge=0, le=1, description="Physics severity threshold for early exit")
    false_negative_penalty_ratio: float = Field(default=2.0, ge=1, description="FN penalty multiplier vs FP")
    gnn_checkpoint_path: str = Field(default="data/derived/models/gnn/gnn_verifier_v1.pth", description="Path to trained GATVerifier checkpoint")
```

**Default YAML file** (`configs/hybrid_verifier.yaml`):
```yaml
# Hybrid Verifier Configuration
# Physics thresholds based on UK standards (BS EN 50160, BS 7671:2018)

physics:
  voltage:
    nominal_v: 230.0
    lower_limit_pct: -10.0   # BS EN 50160 absolute lower
    lower_safe_pct: -6.0     # BS EN 50160 normal range lower
    upper_safe_pct: 8.0      # Conservative upper safe
    upper_limit_pct: 10.0    # BS EN 50160 absolute upper
  capacity:
    typical_max_kw: 15.0     # Typical UK household fuse (BS 7671:2018)
    absolute_max_kw: 100.0   # 100A fuse @ 230V
    overload_threshold_pct: 80.0
    critical_threshold_pct: 95.0
  ramp_rate:
    max_ramp_kw_per_interval: 5.0
    warning_ramp_kw_per_interval: 3.5

cascade:
  propagation_threshold: 0.3
  decay_factor: 0.7

ensemble_weights:
  physics: 0.4
  gnn: 0.4
  cascade: 0.2

early_exit_threshold: 0.9
false_negative_penalty_ratio: 2.0
gnn_checkpoint_path: "data/derived/models/gnn/gnn_verifier_v1.pth"
```

### Anti-Patterns to Avoid
- **Modifying SelfPlayTrainer:** The hybrid verifier must be a drop-in replacement. Do NOT change trainer.py. The trainer expects `verifier.evaluate(forecast, scenario, return_details)` and that is exactly what HybridVerifierAgent must provide.
- **Training the GNN during hybrid verification:** GNN weights are frozen from Phase 1. Use `model.eval()` + `param.requires_grad = False` to guarantee no gradient computation. Attempting to train would cause catastrophic forgetting.
- **Masking nodes out of GNN message passing for early-exit:** This breaks neighbor context. Run the full forward pass and discard scores afterward for early-exited nodes.
- **Learning ensemble weights:** The locked decision specifies fixed configurable weights, not learned weights. Do not add gradient computation for weight optimization.
- **Hard-coding physics thresholds:** All values must come from the YAML config. The entire point of config-driven constraints is tuning per network region without code changes.
- **Returning binary flags from physics layer:** The locked decision requires continuous severity scores [0, 1], not binary True/False.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config validation with ranges | Manual if/else checks | Pydantic `Field(ge=, le=)` + `BaseModel` | Automatic error messages, type coercion, nested model support |
| YAML loading | Custom parser | `yaml.safe_load()` + `Model.model_validate()` | Standard pattern already used in project (config.py) |
| Graph adjacency from edge_index | Custom adjacency matrix | PyG Data already has `edge_index` in COO format | Iterate COO directly or use `torch_geometric.utils.to_dense_adj` if needed |
| GNN inference mode | Manual gradient disabling | `model.eval()` + `torch.no_grad()` context manager | Standard PyTorch pattern, handles BatchNorm/Dropout correctly |
| Classification metrics | Custom precision/recall | `sklearn.metrics` (already imported in evaluate_gnn_verifier.py) | Edge cases handled (zero division, empty predictions) |

**Key insight:** Every component of this phase has an existing pattern in the codebase. The physics constraints exist in `selfplay/verifier.py`, the GNN inference exists in `gnn/gat_verifier.py`, the ensemble pattern exists in `models/ensemble.py`, and the config pattern exists in `config.py`. The task is composing them, not inventing new infrastructure.

## Common Pitfalls

### Pitfall 1: Data Paradigm Mismatch Between Self-Play Loop and GNN
**What goes wrong:** The self-play loop passes 1D numpy arrays (forecast per household) but GNN expects graph-structured PyG Data with per-node features and edge connectivity
**Why it happens:** Phase 1 built the GNN as standalone; Phase 2 must bridge two different data representations
**How to avoid:** HybridVerifierAgent holds a pre-built `graph_data` (the SSEN grid topology from GridGraphBuilder) and a mapping from household indices to graph node indices. When `evaluate()` receives a forecast array, it maps values to node features before GNN inference.
**Warning signs:** Shape mismatches in forward pass; GNN receiving wrong number of nodes; forecast array length not matching graph node count

### Pitfall 2: Early-Exit Threshold Too Aggressive or Too Lenient
**What goes wrong:** Too aggressive (low threshold) -> too many nodes skip GNN, losing topology awareness. Too lenient (high threshold) -> no benefit from early-exit, wasted computation on obvious violations.
**Why it happens:** The threshold interacts with the tolerance band scoring function's output distribution
**How to avoid:** Start with threshold = 0.9 (only the most severe physics violations trigger early exit). This is deliberately conservative. Log early-exit rates during training to monitor.
**Warning signs:** Early exit rate > 50% suggests threshold too low; early exit rate = 0% means it never triggers

### Pitfall 3: Reward Scale Incompatibility
**What goes wrong:** Existing verifier returns rewards in [-1, +1] with a specific distribution. New verifier returns rewards with a different mean/variance, causing training instability.
**Why it happens:** The confidence-based reward formula has different statistical properties than the weighted penalty sum in the old verifier
**How to avoid:** Test reward distribution on the same scenarios. Log reward statistics (mean, std, min, max) during first few episodes and compare against old verifier baseline. May need to scale or shift.
**Warning signs:** Solver loss diverging after switching verifiers; proposer curriculum stuck at low level; all rewards clustered near 0 or near extremes

### Pitfall 4: Frozen GNN Checkpoint Loading Mismatch
**What goes wrong:** Model architecture parameters at load time don't match checkpoint (different hidden_channels, heads, layers)
**Why it happens:** Checkpoint was trained with one config, but HybridVerifier loads with different defaults
**How to avoid:** Store architecture hyperparameters in the checkpoint (Phase 1 already does this: `epoch, model_state_dict, optimizer_state_dict, metrics`). Load from checkpoint metadata, not from config defaults. Add a validation check that loaded model matches expected architecture.
**Warning signs:** `RuntimeError: size mismatch for convs.0.att_src` or similar state_dict errors

### Pitfall 5: Cascade Score Domination or Irrelevance
**What goes wrong:** Cascade scores are always near-zero (irrelevant) or always high (dominating), skewing ensemble output
**Why it happens:** The propagation threshold or the base signal combination doesn't match the data distribution
**How to avoid:** Use `max(physics, gnn)` as base signal for cascade (covers both types of anomaly detection). Start with conservative cascade weight (0.2) and validate on known cascade anomaly scenarios from SyntheticAnomalyDataset.
**Warning signs:** Cascade scores always < 0.01 or always > 0.8; ablation shows removing cascade layer has zero impact

### Pitfall 6: Graph Data Staleness
**What goes wrong:** The pre-built graph topology doesn't reflect the actual nodes being evaluated
**Why it happens:** Graph was built once from SSEN metadata but forecasts come from different household subsets
**How to avoid:** Ensure the graph includes all nodes that may appear in forecasts. For households not in the graph, fall back to physics-only scoring (like an automatic early-exit).
**Warning signs:** KeyError when mapping household IDs to node indices; some forecasts always getting physics-only scores

## Code Examples

Verified patterns from existing codebase:

### Loading GNN Checkpoint (Frozen)
```python
# Source: scripts/evaluate_gnn_verifier.py lines 257-271
import torch
from fyp.gnn import GATVerifier

def load_frozen_gnn(checkpoint_path: str, device: str = "cpu") -> GATVerifier:
    """Load GATVerifier from checkpoint with frozen weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract architecture params from checkpoint if stored, else use defaults
    model = GATVerifier(
        temporal_features=5,   # Must match training config
        hidden_channels=64,
        num_layers=3,
        heads=4,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model
```

### Building Grid Graph (Existing Pattern)
```python
# Source: src/fyp/gnn/graph_builder.py lines 75-175
from fyp.gnn import GridGraphBuilder

def build_grid_for_verifier(metadata_parquet_path: str):
    """Build grid graph for HybridVerifier from SSEN metadata."""
    builder = GridGraphBuilder(exclude_incomplete=True)
    data = builder.build_from_parquet(metadata_parquet_path)
    # data.x: [num_nodes, 4], data.edge_index: [2, num_edges]
    # data.node_type: [num_nodes], data.node_ids: list[str]
    return data, builder
```

### Pydantic Config with YAML Loading (Existing Pattern)
```python
# Source: src/fyp/config.py lines 99-106
import yaml
from pydantic import BaseModel

def load_hybrid_verifier_config(config_path: str) -> HybridVerifierConfig:
    """Load hybrid verifier config from YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return HybridVerifierConfig.model_validate(data)
```

### Existing VerifierAgent.evaluate() Signature (Must Match)
```python
# Source: src/fyp/selfplay/verifier.py lines 428-484
# CRITICAL: HybridVerifierAgent.evaluate() must accept these exact parameters
def evaluate(
    self,
    forecast: np.ndarray,
    scenario: Optional["ScenarioProposal"] = None,
    timestamps: np.ndarray | None = None,
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """Return reward float, or (reward, details_dict) if return_details=True."""
```

### Existing SelfPlayTrainer Call Sites (Must Not Change)
```python
# Source: src/fyp/selfplay/trainer.py line 160-161
# Training call:
verification_reward, details = self.verifier.evaluate(
    forecast=median_forecast, scenario=scenario, return_details=True
)

# Source: src/fyp/selfplay/trainer.py line 378
# Validation call:
verification_reward = self.verifier.evaluate(median_forecast, scenario)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Binary rule-based physics checks | Continuous severity scoring with tolerance bands | Physics-informed ML best practice (2023-2025) | Richer reward signal, better gradient for training |
| Separate physics and ML models | Physics-informed neural network ensembles | Active research area 2024-2025 | Combines domain knowledge with learned patterns |
| Equal-weight ensemble | Configurable weighted ensemble with early-exit | Efficiency optimization pattern | Reduces wasted computation on clear violations |
| Symmetric error penalties | Asymmetric false negative weighting | RLVR research (2025, arXiv:2510.00915) | Aligns with safety-critical domain requirements |

**Deprecated/outdated:**
- Binary violation flags: Use continuous severity scores instead (provides better training signal)
- Learned ensemble weights in verifier: Adds training complexity without proven benefit for this use case; fixed weights with config tuning is more interpretable and stable
- The existing `VerifierAgent` reward range [-1, +1] with penalty accumulation: Will be replaced by confidence-based reward, but the output range [-1, +1] is preserved for compatibility

## Open Questions

1. **Forecast-to-Graph Node Mapping**
   - What we know: The self-play loop processes per-household forecasts. The graph has substations, feeders, and households as nodes. Not all graph nodes have corresponding forecast data.
   - What's unclear: Exactly how to map a single household forecast into multi-node graph features. Options: (a) assign forecast to the household's LV feeder node and aggregate upward, (b) broadcast forecast features to all nodes in the subgraph, (c) use synthetic aggregation (sum household forecasts for feeder/substation nodes).
   - Recommendation: Start with option (a) -- assign to LV feeder node, use zero/mean features for upstream nodes. This is simplest and the GNN's message passing will propagate information upward. Refine in later phases if needed.

2. **Reward Distribution Calibration**
   - What we know: The existing verifier produces rewards mostly in [-0.5, +0.1] range. The new confidence-based reward will have a different distribution.
   - What's unclear: Whether the solver's learning rate and alpha parameter need adjustment for the new reward scale.
   - Recommendation: Log reward distributions during the first 50 episodes of training with the new verifier. If mean reward shifts significantly, add a normalization step or adjust `alpha` in the trainer config.

3. **Graph Data Scope at Inference Time**
   - What we know: The SyntheticAnomalyDataset generates graphs with 44 nodes. The real SSEN metadata has variable graph sizes.
   - What's unclear: Whether HybridVerifier should use a fixed reference graph or dynamically build graphs per batch.
   - Recommendation: Use a fixed reference graph (built once from SSEN metadata at initialization). Dynamic graph building is a Phase 4+ concern for production deployment.

## Sources

### Primary (HIGH confidence)
- Existing codebase analysis: `src/fyp/selfplay/verifier.py` (VerifierAgent interface, Constraint class hierarchy, reward computation)
- Existing codebase analysis: `src/fyp/selfplay/trainer.py` (SelfPlayTrainer, verifier call sites at lines 160-161 and 378)
- Existing codebase analysis: `src/fyp/gnn/gat_verifier.py` (GATVerifier architecture, forward pass signature)
- Existing codebase analysis: `src/fyp/gnn/graph_builder.py` (GridGraphBuilder, PyG Data construction)
- Existing codebase analysis: `src/fyp/config.py` (Pydantic config pattern, YAML loading)
- Existing codebase analysis: `src/fyp/models/ensemble.py` (EnsembleForecaster weighted average pattern)
- Existing codebase analysis: `data/derived/ssen_constraints.json` (UK standards values: 230V, 207-253V range, 80%/95% overload thresholds)
- Phase 1 Research: `.planning/phases/01-gnn-verifier-foundation/01-RESEARCH.md` (GATv2Conv decisions, PyG patterns)

### Secondary (MEDIUM confidence)
- [Pydantic YAML Config Validation](https://www.sarahglasmacher.com/how-to-validate-config-yaml-pydantic/) - Best practice for nested config models with Field constraints
- [BS EN 50160 Voltage Standard](https://powerquality.blog/2021/07/22/standard-en-50160-voltage-characteristics-of-public-distribution-systems/) - UK voltage tolerance bands: 230V +10%/-6%
- [UK Electrical Supply Voltages](https://www.betavalve.com/Wiki/86/UK-Electrical-Supply-Voltages-) - Confirms 216.2V to 253.0V statutory range
- [RL with Verifiable yet Noisy Rewards](https://arxiv.org/abs/2510.00915) - Asymmetric false positive/negative handling in verifier-based RL
- [Lightweight Cascading Failures via Graph Physics-Informed Attention](https://www.sciencedirect.com/science/article/abs/pii/S0957417425020871) - Physics + GNN for power grid anomaly detection
- [Optimizing PyTorch Model Averaging for Ensembles](https://blog.poespas.me/posts/2025/02/13/optimizing-pytorch-model-averaging-for-fast-and-accurate-ensembles/) - Weighted average ensemble patterns

### Tertiary (LOW confidence)
- [PyTorch model.eval() with torch.compile](https://github.com/pytorch/pytorch/issues/104984) - Edge case: eval() can affect torch.compile behavior; verify frozen model works correctly
- [Weighted Average Ensemble Tutorial](https://machinelearningmastery.com/weighted-average-ensemble-for-deep-learning-neural-networks/) - General ensemble patterns; specific implementation details may not apply to anomaly scoring

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries already in project; no new dependencies required
- Architecture: HIGH - Every component has an existing pattern in the codebase; drop-in replacement interface verified by reading exact call sites
- Config schema: HIGH - Extends existing Pydantic pattern from config.py; physics values from SSEN constraints JSON
- Tolerance band scoring: HIGH - Mathematically straightforward; UK voltage standards well-documented
- Cascade logic: MEDIUM - Custom implementation with limited prior art in this exact form; conservative defaults mitigate risk
- Reward signal: MEDIUM - Confidence-based formula is novel for this codebase; may need calibration during integration testing
- Pitfalls: HIGH - Identified from direct codebase analysis, not speculation

**Research date:** 2026-02-18
**Valid until:** 2026-04-18 (60 days - stable libraries, no expected breaking changes)
