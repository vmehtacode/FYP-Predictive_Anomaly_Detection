# Architecture Patterns for GNN-Enhanced Self-Play Anomaly Detection

**Domain:** Energy grid anomaly detection with graph topology
**Researched:** 2026-01-26
**Confidence:** MEDIUM (WebSearch verified with domain research, existing codebase HIGH)

## Executive Summary

This document synthesizes architectural patterns for integrating GNN-based verification into the existing Proposer-Solver-Verifier self-play framework for energy grid anomaly detection. The architecture must handle four key integration challenges:

1. **GNN-Verifier Integration**: Replace physics-constraint Verifier with topology-aware GNN that validates anomalies against grid structure
2. **Graph Message Passing**: Implement hierarchical message passing respecting feeder → substation → primary topology
3. **Temporal-Graph Fusion**: Combine time-series forecasting (PatchTST/FrequencyEnhanced) with spatial graph patterns
4. **Real-Time Inference**: Deploy low-latency anomaly detection on live grid data

**Key Finding**: Research shows GNN-based verifiers significantly outperform rule-based constraints for graph-structured anomaly detection (4-17% accuracy improvement), with message-passing architectures achieving sub-50ms inference on embedded platforms when properly optimized.

## Current Architecture (Baseline)

### Existing Self-Play Components

The codebase implements a working Proposer-Solver-Verifier architecture:

```
ProposerAgent (scenario generation)
    ↓ scenarios (EV_SPIKE, COLD_SNAP, PEAK_SHIFT, OUTAGE, MISSING_DATA)
SolverAgent (PatchTST/FrequencyEnhanced forecaster)
    ↓ forecasts (quantile regression)
VerifierAgent (physics constraints: non-negativity, ramp rate, voltage, power factor)
    ↓ verification reward
SelfPlayTrainer (orchestration loop)
```

**Current Verifier Architecture:**
- Rule-based constraints (6 types: non-negativity, household_max, ramp_rate, temporal_pattern, power_factor, voltage)
- Weighted composite scoring: `total_reward = Σ(constraint_score × weight)`
- No graph awareness — treats each household time series independently
- Validation against SSEN constraints loaded from JSON

**Gap**: Current verifier operates on individual time series without considering grid topology. Anomalies propagate through distribution networks (feeder → substation hierarchy), but existing constraints don't model this spatial dependency.

## Recommended Architecture: Hybrid GNN-Verifier

### Component Overview

Replace rule-based VerifierAgent with **HybridVerifierAgent** that combines:

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| **Hard Constraint Layer** | Physics validation (non-negativity, absolute limits) | Individual forecasts | Binary valid/invalid + violations list |
| **GNN Topology Layer** | Graph-aware anomaly scoring | Batch of node forecasts + adjacency matrix | Node anomaly scores [0, 1] |
| **Temporal Consistency Layer** | Time-series plausibility | Forecast sequences | Temporal coherence score |
| **Fusion & Reward** | Combine signals into unified reward | All layer outputs | Final verification reward [-1, +1] |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      HYBRID VERIFIER AGENT                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Input: Batch of forecasts [(entity_id, forecast_array)]        │
│         + Grid topology (adjacency matrix, node features)        │
│                                                                   │
│  ┌────────────────────┐     ┌────────────────────┐              │
│  │ Hard Constraint    │     │ GNN Topology       │              │
│  │ Validation         │     │ Encoder            │              │
│  ├────────────────────┤     ├────────────────────┤              │
│  │ • Non-negativity   │     │ • Node features:   │              │
│  │ • Absolute limits  │     │   - Forecast stats │              │
│  │ • Ramp rate bounds │     │   - Historical avg │              │
│  │                    │     │   - Feeder type    │              │
│  │ Returns:           │     │                    │              │
│  │ valid: bool        │     │ • Message passing: │              │
│  │ violations: []     │     │   L layers         │              │
│  └────────┬───────────┘     │                    │              │
│           │                 │ Returns:           │              │
│           │                 │ node_scores: [N]   │              │
│           │                 └────────┬───────────┘              │
│           │                          │                          │
│           │    ┌────────────────────┐│                          │
│           │    │ Temporal           ││                          │
│           │    │ Consistency        ││                          │
│           │    ├────────────────────┤│                          │
│           │    │ • Autocorrelation  ││                          │
│           │    │ • Seasonality check││                          │
│           │    │ • Trend stability  ││                          │
│           │    │                    ││                          │
│           │    │ Returns:           ││                          │
│           │    │ temporal_score: [N]││                          │
│           │    └────────┬───────────┘│                          │
│           │             │            │                          │
│           └─────────────┼────────────┘                          │
│                         │                                       │
│                    ┌────▼──────────────┐                        │
│                    │ Fusion Layer      │                        │
│                    ├───────────────────┤                        │
│                    │ reward = α·hard + │                        │
│                    │          β·graph + │                        │
│                    │          γ·temporal│                        │
│                    │                   │                        │
│                    │ α=1.0 (veto power)│                        │
│                    │ β=0.6, γ=0.4      │                        │
│                    └───────┬───────────┘                        │
│                            │                                    │
│  Output: verification_reward ∈ [-1, +1]                        │
│          per_node_details: {scores, violations}                 │
└────────────────────────────┴────────────────────────────────────┘
```

### Design Rationale

**Why Hybrid Over Pure GNN?**
1. **Safety**: Hard constraints provide veto power — physics violations immediately invalidate (prevents catastrophic errors)
2. **Interpretability**: Rule-based layer outputs human-readable violation messages for debugging
3. **Data Efficiency**: GNN learns graph patterns, but physics constraints don't require training data
4. **Graceful Degradation**: If GNN fails, hard constraints maintain minimum safety

**Fusion Strategy:**
- Hard constraint layer has veto power (α=1.0): If `valid=False`, reward = -1.0 regardless of GNN
- GNN and temporal layers weighted equally (β=0.6, γ=0.4) when hard constraints pass
- Difficulty bonus added for challenging but valid scenarios (as in current verifier)

## GNN Topology Layer Design

### Graph Construction

**Nodes**: Energy consumption entities (households, feeders, substations)

**Edges**: Two types of connectivity

| Edge Type | Connects | Weight | Rationale |
|-----------|----------|--------|-----------|
| **Physical Topology** | Household → LV Feeder → Substation (SSEN metadata) | Binary (1.0) | Actual grid connections |
| **Correlation-Based** | k-NN in consumption pattern space | Similarity score | Learned dependencies |

**Node Features** (per entity, per forecast window):
```python
node_features = [
    # Forecast statistics (from solver output)
    forecast_mean,           # Average predicted consumption
    forecast_std,            # Forecast volatility
    forecast_max,            # Peak load

    # Historical context
    historical_mean,         # Baseline consumption
    historical_std,          # Historical volatility

    # Grid metadata (from SSEN)
    feeder_type,            # Categorical: residential/commercial/mixed
    connection_capacity,     # kW capacity
    node_type,              # Categorical: household/feeder/substation

    # Temporal features
    hour_of_day,            # 0-23 (cyclical encoding)
    day_of_week,            # 0-6 (cyclical encoding)
    is_holiday,             # Binary
]
```

**Adjacency Matrix Construction:**
```python
# Combine physical + learned topology
A_total = A_physical + λ * A_learned

A_physical: Binary adjacency from SSEN metadata (feeders → substations)
A_learned: k-NN graph based on consumption similarity (cosine distance)
λ: Balance parameter (default 0.3) — tuned via validation
```

### Message Passing Architecture

**Recommended**: Graph Attention Network (GAT) with hierarchical pooling

**Why GAT?**
- Attention mechanism learns which neighbors matter most (e.g., substation attends more to overloaded feeders)
- Handles heterogeneous node types (household/feeder/substation) via multi-head attention
- Robust to graph structure variations (missing data, new connections)

**Architecture Spec:**
```python
class GNNTopologyVerifier(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 13,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        self.embedding = nn.Linear(node_feature_dim, hidden_dim)

        # Message passing layers (GAT)
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                concat=True  # Concatenate attention heads
            )
            for _ in range(num_layers)
        ])

        # Hierarchical pooling (optional — for substation-level aggregation)
        self.pool = TopKPooling(hidden_dim, ratio=0.5)

        # Anomaly scoring head
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output ∈ [0, 1]
        )

    def forward(self, x, edge_index, batch=None):
        # x: [N, node_feature_dim]
        # edge_index: [2, E] COO format
        # batch: [N] node-to-graph assignment

        # Embed node features
        h = self.embedding(x)  # [N, hidden_dim]

        # Message passing with skip connections
        for gat in self.gat_layers:
            h_new = gat(h, edge_index)  # [N, hidden_dim]
            h = h + h_new  # Residual connection
            h = F.elu(h)

        # Anomaly scoring (per-node)
        anomaly_scores = self.score_mlp(h)  # [N, 1]

        # Optional: Hierarchical aggregation
        if self.pool is not None:
            h_pooled, edge_index_pooled, _, batch_pooled, _, _ = self.pool(
                h, edge_index, batch=batch
            )
            # Aggregate substation-level scores
            substation_scores = scatter_mean(
                anomaly_scores, batch, dim=0
            )
            return anomaly_scores, substation_scores

        return anomaly_scores.squeeze(-1)  # [N]
```

**Message Passing Flow:**

1. **Layer 1**: Local neighborhood aggregation (household → feeder)
   - Each household aggregates from connected households on same feeder
   - Attention weights learn to prioritize similar consumption patterns

2. **Layer 2**: Feeder-level aggregation (feeder → substation)
   - Feeders aggregate from connected households
   - Detects feeder-level anomalies (e.g., entire feeder spike)

3. **Layer 3**: Substation-level context (substation → feeders)
   - Substations broadcast context back to feeders
   - Enables global anomaly detection (e.g., cascading failures)

**Skip Connections**: Residual connections preserve input signal through deep layers (prevents over-smoothing)

### Training Strategy

**Objective**: Learn to assign high anomaly scores to unrealistic forecasts

**Training Data Generation:**
```python
# Positive examples: Solver's valid forecasts (low anomaly score)
# Negative examples: Injected anomalies (high anomaly score)

def generate_training_batch(historical_data, proposer, solver):
    # Sample baseline windows
    batch = sample_windows(historical_data, batch_size=32)

    # Generate scenarios
    scenarios = [proposer.propose_scenario(context) for context, _ in batch]

    # Get solver forecasts
    forecasts = [solver.predict(context, scenario)
                 for (context, _), scenario in zip(batch, scenarios)]

    # Create labels
    labels = []
    for forecast, scenario in zip(forecasts, scenarios):
        # Label = 0 if physics-valid, 1 if anomalous
        is_valid = hard_constraint_check(forecast)
        labels.append(0.0 if is_valid else 1.0)

    # Construct graph batch
    graph_batch = build_graph_batch(forecasts, adjacency_matrix)

    return graph_batch, labels
```

**Loss Function:**
```python
# Binary cross-entropy with difficulty weighting
def gnn_verifier_loss(predictions, labels, difficulty_scores):
    bce = F.binary_cross_entropy(predictions, labels, reduction='none')

    # Weight by scenario difficulty (hard scenarios get more weight)
    weighted_loss = bce * (1.0 + difficulty_scores)

    return weighted_loss.mean()
```

**Training Loop:**
1. Pretrain GNN on historical data + synthetic anomalies (offline)
2. Fine-tune during self-play episodes (online learning)
3. Periodic retraining with accumulated experience buffer

## Temporal Consistency Layer

**Purpose**: Validate time-series plausibility independent of graph structure

**Implementation:**
```python
class TemporalConsistencyScorer:
    def __init__(self, window_size: int = 48):
        self.window_size = window_size

    def score(self, forecast: np.ndarray, historical_context: np.ndarray) -> float:
        """
        Returns score ∈ [0, 1] where 1 = perfect consistency
        """
        scores = []

        # 1. Autocorrelation consistency
        historical_acf = self._autocorrelation(historical_context, lag=48)
        forecast_acf = self._autocorrelation(forecast, lag=min(48, len(forecast)))
        acf_score = 1.0 - abs(historical_acf - forecast_acf)
        scores.append(acf_score)

        # 2. Seasonality preservation
        # Daily pattern (48 intervals = 24 hours)
        historical_daily = self._extract_daily_pattern(historical_context)
        forecast_daily = self._extract_daily_pattern(forecast)
        seasonality_score = cosine_similarity(historical_daily, forecast_daily)
        scores.append(seasonality_score)

        # 3. Trend stability
        historical_trend = self._linear_trend(historical_context[-168:])  # Last week
        forecast_trend = self._linear_trend(forecast)
        # Penalize drastic trend changes
        trend_diff = abs(forecast_trend - historical_trend)
        trend_score = np.exp(-trend_diff)  # Exponential decay
        scores.append(trend_score)

        # 4. Distribution similarity (KL divergence)
        # Bin values and compare distributions
        hist_bins = np.histogram(historical_context, bins=20, density=True)[0]
        forecast_bins = np.histogram(forecast, bins=20, density=True)[0]
        kl_div = kl_divergence(hist_bins, forecast_bins)
        dist_score = np.exp(-kl_div)
        scores.append(dist_score)

        # Weighted average
        weights = [0.2, 0.4, 0.2, 0.2]  # Seasonality weighted most
        return np.average(scores, weights=weights)
```

**Integration with GNN:**
- Temporal scores computed per-node
- Combined with GNN scores via weighted sum (see Fusion Layer)
- Acts as regularizer preventing GNN from ignoring time-series structure

## Proposer-Solver-Verifier Integration

### Modified Training Loop

**Current Flow:**
```
Proposer → Solver → Verifier → Reward → Update
```

**Enhanced Flow with GNN:**
```
Proposer (scenario generation)
    ↓ scenario
Solver (PatchTST forecast)
    ↓ forecast + context
GraphConstructor (build batch graph)
    ↓ graph_batch
GNN-Verifier (hybrid validation)
    ↓ verification_reward + per-node_scores
Reward & Update (RL loop)
```

### Component Interactions

| Component | Input | Output | Communicates With |
|-----------|-------|--------|------------------|
| **ProposerAgent** | Historical context, scenario buffer | ScenarioProposal | Solver, Verifier (for rewards) |
| **SolverAgent** | Context window, scenario | Quantile forecasts | Verifier, Trainer |
| **GraphConstructor** | Batch of forecasts, SSEN topology | PyTorch Geometric Data batch | GNN-Verifier |
| **GNN-Verifier** | Graph batch, node features | Node scores, global reward | Trainer (via reward signal) |
| **SelfPlayTrainer** | All components | Orchestration, checkpointing | All |

### Graph Construction Module

**New Component** (to be added):

```python
class GraphConstructor:
    """Builds PyTorch Geometric graph batches from forecast outputs."""

    def __init__(self, ssen_topology_path: str, k_nn: int = 5, lambda_learned: float = 0.3):
        self.ssen_topology = self._load_ssen_topology(ssen_topology_path)
        self.k_nn = k_nn
        self.lambda_learned = lambda_learned

        # Build physical adjacency matrix (static)
        self.A_physical = self._build_physical_adjacency()

        # Historical consumption patterns (for learned edges)
        self.consumption_embeddings = {}  # entity_id -> embedding

    def build_batch(
        self,
        forecasts: List[Tuple[str, np.ndarray]],  # [(entity_id, forecast_array)]
        contexts: List[Tuple[str, np.ndarray]],   # [(entity_id, context_array)]
        timestamps: List[datetime]
    ) -> torch_geometric.data.Batch:
        """
        Constructs batched graph from forecast outputs.

        Returns:
            Batch object with:
                - x: [N, node_feature_dim] node features
                - edge_index: [2, E] edge connectivity
                - edge_attr: [E, edge_feature_dim] edge weights
                - batch: [N] node-to-graph assignment
        """
        node_features = []
        entity_ids = []

        for (entity_id, forecast), (_, context) in zip(forecasts, contexts):
            # Extract node features
            features = self._extract_node_features(
                entity_id, forecast, context, timestamps
            )
            node_features.append(features)
            entity_ids.append(entity_id)

        x = torch.tensor(node_features, dtype=torch.float32)  # [N, feature_dim]

        # Build adjacency (physical + learned)
        edge_index, edge_weights = self._build_adjacency(entity_ids, contexts)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weights,
            entity_ids=entity_ids  # Store for mapping back
        )

        return data

    def _build_adjacency(
        self,
        entity_ids: List[str],
        contexts: List[Tuple[str, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Builds combined adjacency matrix (physical + learned).
        """
        N = len(entity_ids)

        # Physical edges (from SSEN topology)
        edge_index_physical = []
        for i, id_i in enumerate(entity_ids):
            for j, id_j in enumerate(entity_ids):
                if i != j and self._are_connected(id_i, id_j):
                    edge_index_physical.append([i, j])

        # Learned edges (k-NN in consumption space)
        embeddings = [self._get_consumption_embedding(ctx)
                     for _, ctx in contexts]
        edge_index_learned = self._knn_graph(embeddings, k=self.k_nn)

        # Combine
        edge_index_physical = torch.tensor(edge_index_physical).t()
        edge_weights_physical = torch.ones(edge_index_physical.size(1))

        edge_weights_learned = torch.full(
            (edge_index_learned.size(1),),
            self.lambda_learned
        )

        edge_index = torch.cat([edge_index_physical, edge_index_learned], dim=1)
        edge_weights = torch.cat([edge_weights_physical, edge_weights_learned])

        return edge_index, edge_weights
```

**Integration Point**: Called by `SelfPlayTrainer` after solver prediction, before verifier evaluation.

### Scenario Generation (Proposer) Modifications

**Current**: Generates scenarios per-household independently

**Enhancement**: Graph-aware scenario generation

```python
class GraphAwareProposer(ProposerAgent):
    """
    Extends ProposerAgent with graph-aware scenario generation.
    """

    def __init__(self, *args, graph_topology: nx.Graph = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_topology = graph_topology  # SSEN network graph

    def propose_graph_scenario(
        self,
        entity_ids: List[str],
        historical_contexts: Dict[str, np.ndarray],
        forecast_horizon: int = 48
    ) -> List[ScenarioProposal]:
        """
        Generates scenarios respecting graph structure.

        Example: EV_SPIKE on one household may cascade to neighbors
        """
        scenarios = []

        # Select seed node for scenario
        seed_id = np.random.choice(entity_ids)
        seed_scenario = self.propose_scenario(
            historical_context=historical_contexts[seed_id],
            forecast_horizon=forecast_horizon
        )
        scenarios.append((seed_id, seed_scenario))

        # Propagate scenario to graph neighbors (if applicable)
        if seed_scenario.scenario_type in ["COLD_SNAP", "OUTAGE"]:
            # These scenarios affect neighborhoods
            neighbors = self._get_graph_neighbors(seed_id, k=3)
            for neighbor_id in neighbors:
                # Create correlated scenario with decay
                neighbor_scenario = self._create_correlated_scenario(
                    seed_scenario,
                    distance=self._graph_distance(seed_id, neighbor_id)
                )
                scenarios.append((neighbor_id, neighbor_scenario))

        # Fill remaining entities with baseline scenarios
        for entity_id in entity_ids:
            if entity_id not in [s[0] for s in scenarios]:
                baseline = self.propose_scenario(
                    historical_context=historical_contexts[entity_id],
                    forecast_horizon=forecast_horizon
                )
                scenarios.append((entity_id, baseline))

        return scenarios

    def _create_correlated_scenario(
        self,
        seed_scenario: ScenarioProposal,
        distance: int
    ) -> ScenarioProposal:
        """
        Creates a correlated scenario with magnitude decay by graph distance.
        """
        decay_factor = 0.8 ** distance  # Exponential decay

        correlated = ScenarioProposal(
            scenario_type=seed_scenario.scenario_type,
            magnitude=seed_scenario.magnitude * decay_factor,
            duration=seed_scenario.duration,
            start_time=seed_scenario.start_time,
            affected_appliances=seed_scenario.affected_appliances,
            baseline_context=seed_scenario.baseline_context,
            difficulty_score=seed_scenario.difficulty_score * decay_factor,
            physics_valid=True,  # Will be validated
            metadata={"correlated_from": "seed", "distance": distance}
        )

        return correlated
```

**Why Graph-Aware Scenarios?**
- Real anomalies propagate through grid (e.g., transformer failure affects all downstream feeders)
- Creates realistic training signal for GNN (learns spatial correlation)
- Enables GNN to distinguish local vs. systemic anomalies

### Solver (Forecast Model) Modifications

**Minimal Changes Required**: Existing PatchTST/FrequencyEnhanced models work as-is.

**Optional Enhancement**: Add graph-aware forecasting head

```python
class GraphAwareSolver(SolverAgent):
    """
    Extends SolverAgent with optional graph-aware forecast refinement.
    """

    def __init__(self, *args, use_graph_refinement: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_graph_refinement = use_graph_refinement

        if use_graph_refinement:
            # Small GNN to refine forecasts using neighbor information
            self.graph_refiner = nn.Sequential(
                GCNConv(1, 16),  # 1D forecast values
                nn.ReLU(),
                GCNConv(16, 1)
            )

    def predict_batch(
        self,
        contexts: List[np.ndarray],
        scenarios: List[ScenarioProposal],
        adjacency: torch.Tensor = None
    ) -> Dict[str, List[np.ndarray]]:
        """
        Batch prediction with optional graph refinement.
        """
        # Standard per-node forecasts
        forecasts = [
            self.predict(ctx, scenario)
            for ctx, scenario in zip(contexts, scenarios)
        ]

        if self.use_graph_refinement and adjacency is not None:
            # Stack forecasts into graph
            forecast_tensor = torch.tensor([f["0.5"] for f in forecasts])

            # Refine using graph
            refined = self.graph_refiner(forecast_tensor, adjacency)

            # Update median forecasts
            for i, f in enumerate(forecasts):
                f["0.5"] = refined[i].detach().numpy()

        return forecasts
```

**Trade-off**: Graph refinement adds complexity. Recommend starting without it, add if baseline solver struggles.

## Real-Time Inference Pipeline

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    REAL-TIME INFERENCE PIPELINE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │ Data Ingress │─────▶│ Preprocessing│─────▶│ Feature      │  │
│  │              │      │              │      │ Engineering  │  │
│  │ • Elexon API │      │ • Validation │      │              │  │
│  │ • SSEN feed  │      │ • Resampling │      │ • Windowing  │  │
│  │ • LCL stream │      │ • Imputation │      │ • Scaling    │  │
│  └──────────────┘      └──────────────┘      └───────┬──────┘  │
│                                                       │          │
│                                                       ▼          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Parallel Inference Branches                  │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │                                                            │  │
│  │  ┌─────────────────┐              ┌─────────────────┐   │  │
│  │  │ Solver Branch   │              │ Verifier Branch │   │  │
│  │  ├─────────────────┤              ├─────────────────┤   │  │
│  │  │ • PatchTST      │              │ • Graph         │   │  │
│  │  │   forecasting   │              │   construction  │   │  │
│  │  │ • Quantile      │              │ • GNN inference │   │  │
│  │  │   outputs       │              │ • Hard          │   │  │
│  │  │                 │              │   constraints   │   │  │
│  │  │ Latency: ~20ms  │              │ Latency: ~30ms  │   │  │
│  │  └────────┬────────┘              └────────┬────────┘   │  │
│  │           │                                │             │  │
│  │           └────────────┬───────────────────┘             │  │
│  │                        ▼                                 │  │
│  │            ┌───────────────────────┐                     │  │
│  │            │ Anomaly Scoring       │                     │  │
│  │            ├───────────────────────┤                     │  │
│  │            │ • Fusion layer        │                     │  │
│  │            │ • Threshold decision  │                     │  │
│  │            │ • Confidence interval │                     │  │
│  │            │                       │                     │  │
│  │            │ Latency: ~5ms         │                     │  │
│  │            └───────────┬───────────┘                     │  │
│  └────────────────────────┼─────────────────────────────────┘  │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Output & Actions                                          │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │ • Anomaly alerts (entity_id, timestamp, score, type)     │  │
│  │ • Visualization dashboard                                 │  │
│  │ • Logging to database                                     │  │
│  │ • Optional: Trigger investigations                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Target Latency: <100ms end-to-end                             │
└─────────────────────────────────────────────────────────────────┘
```

### Component Specifications

#### 1. Data Ingress

**Elexon BMRS API Integration:**
```python
class ElexonDataStream:
    """
    Real-time data ingress from Elexon BMRS API.
    """

    def __init__(self, api_endpoint: str, polling_interval: int = 30):
        self.api_endpoint = api_endpoint
        self.polling_interval = polling_interval  # seconds
        self.buffer = deque(maxlen=1000)  # Circular buffer

    def start_polling(self):
        """
        Continuous polling loop (separate thread).
        """
        while True:
            try:
                data = self._fetch_latest_data()
                self.buffer.append(data)
                time.sleep(self.polling_interval)
            except Exception as e:
                logger.error(f"Elexon API fetch failed: {e}")

    def _fetch_latest_data(self) -> pd.DataFrame:
        """
        Fetches latest grid data from Elexon BMRS.
        """
        response = requests.get(self.api_endpoint)
        # Parse XML/JSON response
        # Return DataFrame with columns: [timestamp, entity_id, energy_kwh]
        pass

    def get_batch(self, batch_size: int = 32) -> pd.DataFrame:
        """
        Returns latest batch for inference.
        """
        return pd.DataFrame(list(self.buffer)[-batch_size:])
```

**Why Polling?**
- Elexon BMRS updates at 5-30 minute intervals (not true streaming)
- Polling allows buffering and batch processing (more efficient)

**Alternative**: WebSocket for true streaming (if API supports)

#### 2. Preprocessing Module

**Validation & Imputation:**
```python
class RealtimePreprocessor:
    """
    Validates and preprocesses incoming data streams.
    """

    def __init__(self, schema: EnergyReadingSchema):
        self.schema = schema
        self.imputer = SimpleImputer(strategy='forward_fill')

    def process_batch(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validates and prepares batch for inference.

        Steps:
        1. Schema validation (drop invalid rows)
        2. Timestamp alignment (resample to 30-min)
        3. Missing data imputation (forward fill)
        4. Outlier clipping (3-sigma)
        """
        # Validate schema
        validated = self._validate_schema(raw_data)

        # Resample to 30-min intervals
        resampled = validated.set_index('timestamp').resample('30T').mean()

        # Impute missing values
        imputed = self.imputer.transform(resampled)

        # Clip outliers (3-sigma rule)
        clipped = self._clip_outliers(imputed, n_sigma=3)

        return clipped

    def _clip_outliers(self, data: pd.DataFrame, n_sigma: int = 3) -> pd.DataFrame:
        """
        Clips values beyond n_sigma standard deviations.
        """
        mean = data['energy_kwh'].mean()
        std = data['energy_kwh'].std()

        data['energy_kwh'] = data['energy_kwh'].clip(
            lower=mean - n_sigma * std,
            upper=mean + n_sigma * std
        )
        return data
```

#### 3. Feature Engineering

**Windowing Strategy:**
```python
class RealtimeWindower:
    """
    Creates sliding windows for real-time inference.
    """

    def __init__(self, context_length: int = 336, forecast_horizon: int = 48):
        self.context_length = context_length  # 7 days
        self.forecast_horizon = forecast_horizon  # 24 hours

        # Circular buffer per entity
        self.entity_buffers = defaultdict(lambda: deque(maxlen=context_length))

    def update_and_create_windows(
        self,
        new_data: pd.DataFrame
    ) -> List[Tuple[str, np.ndarray]]:
        """
        Updates entity buffers and creates inference windows.

        Returns:
            List of (entity_id, context_window) pairs ready for inference
        """
        windows = []

        for entity_id, group in new_data.groupby('entity_id'):
            # Update buffer
            self.entity_buffers[entity_id].extend(group['energy_kwh'].values)

            # Create window if buffer full
            if len(self.entity_buffers[entity_id]) >= self.context_length:
                context = np.array(list(self.entity_buffers[entity_id]))
                windows.append((entity_id, context))

        return windows
```

**Trade-off**: Context length vs. latency
- Longer context (336 intervals = 7 days) → Better accuracy, higher latency
- Shorter context (96 intervals = 2 days) → Lower latency, may miss long-term patterns
- **Recommendation**: Start with 336, reduce if latency exceeds target

#### 4. Parallel Inference

**Solver Branch (Forecasting):**
```python
class RealtimeSolverEngine:
    """
    Optimized solver inference with batching.
    """

    def __init__(self, model: SolverAgent, device: str = "cpu"):
        self.model = model
        self.device = device

        # Optimization: Compile model for faster inference
        if torch.cuda.is_available() and device == "cuda":
            self.model.model = torch.jit.script(self.model.model)

    def predict_batch(
        self,
        windows: List[Tuple[str, np.ndarray]]
    ) -> List[Tuple[str, Dict[str, np.ndarray]]]:
        """
        Batched forecasting for low latency.

        Target: <20ms for batch_size=32
        """
        entity_ids = [eid for eid, _ in windows]
        contexts = [ctx for _, ctx in windows]

        # Stack into batch tensor
        context_batch = torch.tensor(
            np.stack(contexts),
            dtype=torch.float32,
            device=self.device
        )

        # Batch inference (no_grad for speed)
        with torch.no_grad():
            forecast_batch = self.model.model.forward(context_batch)

        # Unpack results
        results = []
        for i, entity_id in enumerate(entity_ids):
            forecast = {
                "0.5": forecast_batch[i].cpu().numpy()
            }
            results.append((entity_id, forecast))

        return results
```

**Verifier Branch (GNN Anomaly Scoring):**
```python
class RealtimeVerifierEngine:
    """
    GNN-based verifier inference with graph batching.
    """

    def __init__(
        self,
        gnn_model: GNNTopologyVerifier,
        graph_constructor: GraphConstructor,
        device: str = "cpu"
    ):
        self.gnn_model = gnn_model
        self.graph_constructor = graph_constructor
        self.device = device

        # Precompute static adjacency matrix
        self.static_adjacency = self._precompute_adjacency()

    def score_batch(
        self,
        forecasts: List[Tuple[str, np.ndarray]],
        contexts: List[Tuple[str, np.ndarray]],
        timestamps: List[datetime]
    ) -> List[Tuple[str, float]]:
        """
        Batched GNN anomaly scoring.

        Target: <30ms for batch_size=32
        """
        # Construct graph batch
        graph_batch = self.graph_constructor.build_batch(
            forecasts, contexts, timestamps
        )

        # Move to device
        graph_batch = graph_batch.to(self.device)

        # GNN inference
        with torch.no_grad():
            anomaly_scores = self.gnn_model(
                graph_batch.x,
                graph_batch.edge_index
            )

        # Map back to entity IDs
        results = [
            (entity_id, score.item())
            for entity_id, score in zip(graph_batch.entity_ids, anomaly_scores)
        ]

        return results
```

**Parallelization Strategy:**
```python
from concurrent.futures import ThreadPoolExecutor

class ParallelInferencePipeline:
    """
    Orchestrates parallel solver + verifier inference.
    """

    def __init__(self, solver_engine, verifier_engine):
        self.solver = solver_engine
        self.verifier = verifier_engine
        self.executor = ThreadPoolExecutor(max_workers=2)

    def infer_batch(
        self,
        windows: List[Tuple[str, np.ndarray]]
    ) -> Dict[str, Dict]:
        """
        Runs solver and verifier in parallel.

        Returns:
            {
                entity_id: {
                    'forecast': np.ndarray,
                    'anomaly_score': float,
                    'is_anomaly': bool
                }
            }
        """
        # Submit parallel tasks
        solver_future = self.executor.submit(self.solver.predict_batch, windows)
        verifier_future = self.executor.submit(
            self.verifier.score_batch,
            windows,  # Contexts for graph construction
            windows,  # Forecasts (will be computed by solver)
            [datetime.now()] * len(windows)
        )

        # Wait for both
        forecasts = solver_future.result()
        anomaly_scores = verifier_future.result()

        # Merge results
        results = {}
        for (entity_id, forecast), (_, score) in zip(forecasts, anomaly_scores):
            results[entity_id] = {
                'forecast': forecast['0.5'],
                'anomaly_score': score,
                'is_anomaly': score > 0.7  # Threshold
            }

        return results
```

#### 5. Anomaly Scoring & Decision

**Threshold Strategy:**
```python
class AnomalyDecisionEngine:
    """
    Applies adaptive thresholds for anomaly detection.
    """

    def __init__(self, base_threshold: float = 0.7, percentile: float = 95):
        self.base_threshold = base_threshold
        self.percentile = percentile

        # Adaptive threshold buffer (rolling window)
        self.score_history = deque(maxlen=1000)

    def decide(
        self,
        anomaly_scores: Dict[str, float]
    ) -> Dict[str, Dict]:
        """
        Applies adaptive threshold to anomaly scores.

        Returns:
            {
                entity_id: {
                    'is_anomaly': bool,
                    'confidence': float,
                    'severity': str  # 'low'/'medium'/'high'
                }
            }
        """
        # Update score history
        self.score_history.extend(anomaly_scores.values())

        # Compute adaptive threshold (95th percentile)
        if len(self.score_history) >= 100:
            adaptive_threshold = np.percentile(list(self.score_history), self.percentile)
        else:
            adaptive_threshold = self.base_threshold

        # Apply threshold
        results = {}
        for entity_id, score in anomaly_scores.items():
            is_anomaly = score > adaptive_threshold

            # Compute confidence (distance from threshold)
            confidence = abs(score - adaptive_threshold) / adaptive_threshold

            # Severity binning
            if score > adaptive_threshold * 1.5:
                severity = 'high'
            elif score > adaptive_threshold * 1.2:
                severity = 'medium'
            elif score > adaptive_threshold:
                severity = 'low'
            else:
                severity = 'normal'

            results[entity_id] = {
                'is_anomaly': is_anomaly,
                'confidence': confidence,
                'severity': severity,
                'raw_score': score
            }

        return results
```

**Why Adaptive Thresholds?**
- Fixed thresholds produce false positives when baseline shifts (e.g., seasonal changes)
- Percentile-based thresholds adapt to distribution drift
- Requires history buffer (cold-start issue mitigated by base threshold)

### Latency Optimization Strategies

| Strategy | Technique | Latency Gain | Trade-off |
|----------|-----------|--------------|-----------|
| **Model Quantization** | Convert FP32 → FP16 or INT8 | 2-4x speedup | Slight accuracy loss (<1% typically) |
| **Batch Inference** | Process 32 entities together | 10-20x vs. sequential | Requires waiting for batch |
| **Model Pruning** | Remove low-magnitude weights | 1.5-2x speedup | Retrain required |
| **ONNX Runtime** | Convert PyTorch → ONNX | 1.5-3x speedup | Limited operator support |
| **Edge Deployment** | Run on GPU (if available) | 5-10x speedup | Hardware cost |
| **Graph Caching** | Precompute static adjacency | 20-30% reduction | Memory overhead |

**Recommended Starting Point:**
1. Batch inference (easiest, highest gain)
2. FP16 quantization (one-line change in PyTorch)
3. ONNX conversion (if latency still exceeds target)

**Latency Budget Breakdown (target <100ms):**
- Data ingress & preprocessing: 20ms
- Feature engineering: 10ms
- Solver inference: 20ms
- Verifier inference: 30ms
- Anomaly scoring: 5ms
- Output formatting: 5ms
- **Buffer**: 10ms

### Deployment Architecture

**Option 1: Single-Node Deployment** (MVP)
```
[Elexon API] → [Python Service] → [PostgreSQL]
                 ├─ Flask API
                 ├─ Inference Engine
                 └─ Anomaly Logger
```

**Pros**: Simple, low latency, easy debugging
**Cons**: Limited scalability, single point of failure

**Option 2: Microservices Architecture** (Production)
```
[Elexon API] → [Kafka] → [Preprocessing Service]
                           ↓
                    [Inference Service] (multiple replicas)
                           ↓
                    [Decision Service]
                           ↓
                    [PostgreSQL + Grafana]
```

**Pros**: Scalable, fault-tolerant, parallel processing
**Cons**: Complex, higher operational overhead

**Recommendation**: Start with Option 1 for FYP demonstration, migrate to Option 2 if production deployment required.

## Integration Roadmap

### Phase 1: GNN-Verifier Foundation (Weeks 1-2)

**Objectives:**
- Implement `GNNTopologyVerifier` module
- Create `GraphConstructor` for batch building
- Train GNN on historical data + synthetic anomalies

**Deliverables:**
- `/src/fyp/selfplay/gnn_verifier.py` (GNN model)
- `/src/fyp/selfplay/graph_constructor.py` (graph builder)
- `/tests/test_gnn_verifier.py` (unit tests)
- Training script with validation metrics

**Success Criteria:**
- GNN achieves >85% accuracy on synthetic anomaly detection
- Inference latency <30ms for batch_size=32

### Phase 2: Hybrid Verifier Integration (Weeks 3-4)

**Objectives:**
- Extend `VerifierAgent` to `HybridVerifierAgent`
- Integrate hard constraints + GNN scoring + temporal layer
- Update `SelfPlayTrainer` to use hybrid verifier

**Deliverables:**
- `/src/fyp/selfplay/hybrid_verifier.py`
- Updated `/src/fyp/selfplay/trainer.py`
- Integration tests

**Success Criteria:**
- Hybrid verifier improves anomaly detection vs. baseline (measured on held-out test set)
- Self-play loop runs without errors

### Phase 3: Graph-Aware Proposer (Week 5)

**Objectives:**
- Implement `GraphAwareProposer` with correlated scenario generation
- Integrate SSEN topology for neighborhood propagation

**Deliverables:**
- `/src/fyp/selfplay/graph_proposer.py`
- SSEN topology loader
- Scenario visualization tool

**Success Criteria:**
- Proposer generates realistic multi-node scenarios
- GNN-Verifier shows improved detection on correlated anomalies

### Phase 4: Real-Time Pipeline (Weeks 6-7)

**Objectives:**
- Implement Elexon BMRS API integration
- Build preprocessing + windowing modules
- Deploy parallel inference pipeline

**Deliverables:**
- `/src/fyp/realtime/` package
  - `data_stream.py`
  - `preprocessor.py`
  - `inference_engine.py`
- Dashboard (Flask + Plotly)
- Deployment script (Docker)

**Success Criteria:**
- End-to-end latency <100ms
- Live dashboard displays anomalies in near-real-time

### Phase 5: Evaluation & Documentation (Week 8)

**Objectives:**
- Benchmark against baselines (rule-based verifier, no graph)
- Ablation studies (GNN vs. temporal vs. hybrid)
- Write FYP report architecture section

**Deliverables:**
- Evaluation notebook (`notebooks/gnn_evaluation.ipynb`)
- Performance comparison tables
- Architecture diagrams for report

**Success Criteria:**
- GNN-enhanced system outperforms baselines on real SSEN data
- Clear ablation showing each component's contribution

## Anti-Patterns to Avoid

### 1. Over-Smoothing in Deep GNNs

**Problem**: Stacking too many GNN layers causes node embeddings to converge to same value

**Symptom**: Validation accuracy plateaus/drops after 3-4 layers

**Solution**:
- Limit to 3 layers (local + feeder + substation)
- Add skip connections (residual)
- Use JK-Net (jumping knowledge) to aggregate from all layers

### 2. Ignoring Graph Heterogeneity

**Problem**: Treating households, feeders, substations as same node type

**Symptom**: GNN fails to learn meaningful feeder-level patterns

**Solution**:
- Use Heterogeneous GNN (HetGNN) with separate embeddings per node type
- OR add node type as categorical feature
- Weight message passing by node type similarity

### 3. Static Graph Assumption

**Problem**: Grid topology changes (new connections, failures) but graph is static

**Symptom**: Inference fails on unseen topologies

**Solution**:
- Periodically rebuild adjacency matrix from latest SSEN metadata
- Use inductive GNN (GraphSAINT) that generalizes to new nodes
- Add edge dropout during training (0.1-0.2) for robustness

### 4. Batch Size = 1 (No Batching)

**Problem**: Running inference on single entities sequentially

**Symptom**: Latency exceeds 100ms by 10-100x

**Solution**:
- ALWAYS batch entities (32-64 typical)
- Use `torch.jit.script` or ONNX for batch optimization
- Precompute graph structure where possible

### 5. Forgetting Temporal Ordering

**Problem**: GNN processes spatial graph, but forecasts are time-ordered

**Symptom**: Model ignores temporal dependencies, learns only spatial

**Solution**:
- Ensure Solver outputs capture temporal patterns (PatchTST handles this)
- Add temporal features to node embeddings (hour, day, season)
- Use Temporal GNN (TGN) if needed (adds recurrence to GNN)

### 6. No Hard Constraint Veto

**Problem**: Trusting GNN to learn physics constraints from data

**Symptom**: Physically impossible forecasts pass verification

**Solution**:
- Always include hard constraint layer with veto power
- GNN augments, does not replace, physics validation
- Log physics violations for debugging

## Performance Benchmarks

### Expected Metrics (Based on Literature)

| Metric | Baseline (Rule-Based Verifier) | GNN-Enhanced Verifier | Source |
|--------|-------------------------------|----------------------|--------|
| **Anomaly Detection Accuracy** | 78-82% | 85-92% | [Graph Anomaly Detection with GNNs](https://arxiv.org/abs/2209.14930) |
| **False Positive Rate** | 15-20% | 8-12% | [GNN graph structures in network anomaly detection](https://hal.science/hal-04929581v1) |
| **Inference Latency** | 10-15ms (CPU) | 25-40ms (CPU) | [Edge AI for Anomaly Detection](https://www.mdpi.com/1999-5903/17/4/179) |
| **Training Time** | N/A (rule-based) | 2-4 hours (pretraining) | Estimated from PatchTST |

### Validation Strategy

**Test Sets:**
1. **Synthetic Anomalies**: Injected scenarios with known labels
2. **SSEN Real Data**: Distribution network consumption (no labels, but physics constraints)
3. **Adversarial Examples**: Edge cases (e.g., all nodes spike simultaneously)

**Evaluation Metrics:**
- Precision, Recall, F1 for anomaly detection
- Constraint violation rate (% forecasts violating physics)
- Latency percentiles (p50, p95, p99)
- Graph-level metrics (feeder/substation anomaly detection)

**Ablation Studies:**
- GNN-only (no hard constraints)
- Hard constraints only (no GNN)
- Hybrid (GNN + constraints) ← Expected best

## Open Questions & Research Flags

### 1. Optimal Graph Construction

**Question**: Should we use physical topology only, or mix with learned correlations?

**Trade-off**:
- Physical only: Interpretable, but may miss hidden dependencies
- Learned only: Flexible, but may hallucinate non-physical connections
- Hybrid (recommended): Combine both with weight parameter λ

**Recommendation**: Start with hybrid (λ=0.3), tune via validation

### 2. Node vs. Graph-Level Anomalies

**Question**: Should verifier detect individual node anomalies or whole-graph patterns?

**Current Design**: Node-level (per-household anomaly scores)

**Alternative**: Add graph-level pooling for feeder/substation anomaly detection

**Recommendation**: Implement both, compare in ablation study

### 3. Temporal Graph Dynamics

**Question**: Do we need temporal GNN (TGNN) or is static graph + time-series solver sufficient?

**TGNN Pros**: Captures evolving graph structure (e.g., connection changes)
**TGNN Cons**: 2-3x slower, more complex training

**Recommendation**: Start with static GNN. Add TGNN only if:
- SSEN metadata shows frequent topology changes, OR
- Baseline fails to detect cascading anomalies

### 4. Cold-Start Problem

**Question**: How to handle entities without historical data (new connections)?

**Options**:
1. Inductive GNN (learns from neighbors)
2. Meta-learning (few-shot adaptation)
3. Fallback to rule-based verifier

**Recommendation**: Option 1 (inductive GNN via GraphSAINT) + Option 3 (fallback)

### 5. Scenario Diversity vs. Realism

**Question**: Should proposer generate diverse scenarios or realistic ones?

**Trade-off**:
- Diverse: Better GNN training coverage, but may create unrealistic patterns
- Realistic: Better matches deployment, but may under-explore

**Current Design**: Diversity-weighted sampling with physics pre-check

**Recommendation**: Monitor GNN performance on real data. If high false positives, bias toward realism.

## Sources

**GNN Anomaly Detection:**
- [Graph Anomaly Detection with Graph Neural Networks: Current Status and Challenges](https://arxiv.org/abs/2209.14930)
- [GNN graph structures in network anomaly detection](https://hal.science/hal-04929581v1/document)
- [Explainable Graph Neural Networks for Power Grid Fault Detection](https://ieeexplore.ieee.org/document/11088107/)
- [Graph attention and Kolmogorov–Arnold network based smart grids intrusion detection](https://www.nature.com/articles/s41598-025-88054-9)

**Power Grid GNN Applications:**
- [PowerGNN: A Topology-Aware Graph Neural Network for Electricity Grids](https://arxiv.org/html/2503.22721v1)
- [Dynamical Graph Neural Networks for Modern Power Grid Analysis](https://www.mdpi.com/2079-9292/15/3/493)
- [Physics-Informed Graph Neural Network for Dynamic Reconfiguration of Power Systems](https://arxiv.org/html/2310.00728)
- [PowerFlowNet: Power Flow Approximation Using Message Passing Graph Neural Networks](https://arxiv.org/html/2311.03415)

**Temporal Graph Networks:**
- [TempReasoner: neural temporal graph networks for event timeline construction](https://www.nature.com/articles/s41598-026-35385-w)
- [Temporal Graph Learning in 2024](https://towardsdatascience.com/temporal-graph-learning-in-2024-feaa9371b8e2/)
- [A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection](https://arxiv.org/html/2307.03759v3)

**Self-Play Architectures:**
- [SPELL: Self-Play Reinforcement Learning for evolving Long-Context Language Models](https://arxiv.org/html/2509.23863v2)
- [Propose, Solve, Verify: Self-Play Through Formal Verification](https://arxiv.org/html/2512.18160)
- [Learning to Solve and Verify: A Self-Play Framework for Code and Test Generation](https://arxiv.org/html/2502.14948v3)

**Real-Time Inference Pipelines:**
- [Context-Aware ML/NLP Pipeline for Real-Time Anomaly Detection](https://www.mdpi.com/2504-4990/8/1/25)
- [Edge AI for Real-Time Anomaly Detection in Smart Homes](https://www.mdpi.com/1999-5903/17/4/179)
- [Near Real-Time Anomaly Detection with Delta Live Tables and Databricks Machine Learning](https://www.databricks.com/blog/near-real-time-anomaly-detection-delta-live-tables-and-databricks-machine-learning)

---

**Architecture research complete.** Findings ready for roadmap creation.
