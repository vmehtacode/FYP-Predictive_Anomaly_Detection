# Roadmap: Grid Guardian

**Created:** 2026-01-27
**Core Value:** Detect anomalies in energy distribution networks without labeled data, using physics constraints and self-play learned patterns on graph-structured grid data
**Config:** depth=standard, parallelization=false

## Milestone: v1.0 — GNN-Based Anomaly Detection with Self-Play Validation

### Phase 1: GNN Verifier Foundation

**Goal:** Build a topology-aware GNN verifier that understands grid structure and can score anomalies based on spatial relationships between nodes.

**Requirements:** GNN-01, GNN-02

**Success Criteria:**
1. Graph construction pipeline transforms SSEN metadata into PyTorch Geometric Data batches with nodes (households, feeders, substations) and edges (physical topology)
2. GNN model (GAT/GraphSAGE + temporal layer) processes graph-structured input and outputs per-node anomaly scores
3. Model achieves >85% accuracy on held-out synthetic anomalies with <30ms inference latency for batch_size=32
4. Oversmoothing is prevented (node embeddings remain distinguishable across grid hierarchy levels)

**Depends on:** None

---

### Phase 2: Hybrid Verifier Integration

**Goal:** Replace the MLP Verifier with a three-layer ensemble that combines physics constraints, GNN patterns, and cascade logic into the self-play training loop.

**Requirements:** GNN-03, ENS-01, ENS-02

**Success Criteria:**
1. Physics constraint layer (hard bounds from SSEN: voltage, capacity, ramp rates) validates inputs before GNN processing
2. Cascade early-exit logic skips GNN computation for clear physics violations, reducing average inference time
3. HybridVerifierAgent integrates into existing SelfPlayTrainer, producing verification rewards for training
4. Hybrid verifier improves detection accuracy over baseline rule-based verifier on held-out test scenarios

**Depends on:** Phase 1

---

### Phase 3: Graph-Aware Proposer

**Goal:** Enhance the Proposer to generate topology-respecting anomaly scenarios where disturbances propagate through connected nodes (e.g., COLD_SNAP cascading through neighbors).

**Requirements:** SELF-01, SELF-02

**Success Criteria:**
1. GraphAwareProposer generates scenarios that respect grid topology (anomalies occur on connected subgraphs, not random nodes)
2. Cascade scenarios (COLD_SNAP, OUTAGE) propagate through physically connected neighbors with configurable decay
3. Self-play training maintains scenario diversity (each of 5 types appears in at least 15% of batches, no mode collapse)
4. Generated scenarios are visually distinguishable and pass domain plausibility review

**Depends on:** Phase 2

---

### Phase 4: Evaluation Framework

**Goal:** Validate the GNN-based approach through rigorous multi-level evaluation comparing against baselines and measuring physics compliance.

**Requirements:** EVAL-01, EVAL-02, EVAL-03

**Success Criteria:**
1. Precision, recall, and F1 scores computed against physics-violation ground truth and synthetic scenarios
2. GNN Verifier performance compared against IsolationForest, Autoencoder, and DecompositionAnomalyDetector on identical test splits
3. Physics compliance rate measured (percentage of detected anomalies that violate known physical constraints)
4. Ablation study quantifies contribution of each component (GNN-only vs. physics-only vs. hybrid ensemble)
5. Results documented with statistical significance testing and visualization for FYP report

**Depends on:** Phase 3

---

## Progress

| Phase | Name | Status | Requirements | Coverage |
|-------|------|--------|--------------|----------|
| 1 | GNN Verifier Foundation | Not started | GNN-01, GNN-02 | 2/10 |
| 2 | Hybrid Verifier Integration | Not started | GNN-03, ENS-01, ENS-02 | 3/10 |
| 3 | Graph-Aware Proposer | Not started | SELF-01, SELF-02 | 2/10 |
| 4 | Evaluation Framework | Not started | EVAL-01, EVAL-02, EVAL-03 | 3/10 |

**Total Coverage:** 10/10 requirements mapped

---
*Roadmap created: 2026-01-27*
*Last updated: 2026-01-27*
