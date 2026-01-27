# Phase 1: GNN Verifier Foundation - Context

**Gathered:** 2026-01-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a topology-aware GNN verifier that transforms SSEN grid metadata into per-node anomaly scores. This phase delivers the graph construction pipeline and GNN model architecture. Training loops, hybrid verifier integration, and evaluation frameworks are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Graph Construction
- Three-level node hierarchy: Substations → Feeders → Households (matches physical grid structure)
- Edges represent binary connectivity only — no edge weights, capacity, or distance attributes
- Nodes with missing or incomplete SSEN metadata are excluded from the graph (cleaner training signal)
- Temporal features attached as fixed time-window snapshots per node (e.g., [current_load, avg_24h, peak_7d])

### GNN Architecture
- Primary layer type: GAT (Graph Attention Network) with learnable attention weights
- Network depth: 3 layers — captures substation→feeder→household in single forward pass
- Temporal processing: Encode time patterns per-node first, then GNN aggregates across spatial structure
- Embedding dimension: 64-dimensional hidden states (lightweight for faster inference)

### Claude's Discretion
- Anomaly score normalization and threshold handling
- Oversmoothing prevention techniques (skip connections, layer norm)
- Number of attention heads in GAT layers
- Specific temporal encoder architecture (LSTM vs 1D-Conv vs simple MLP)
- Exact node feature engineering from SSEN metadata
- Batch size and learning rate tuning

</decisions>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches that meet the success criteria (>85% accuracy on synthetic anomalies, <30ms inference latency).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-gnn-verifier-foundation*
*Context gathered: 2026-01-27*
