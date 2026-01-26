# Project Research Summary

**Project:** Grid Guardian - Energy Anomaly Detection using AZR Self-Play with GNN Verifier
**Domain:** Physics-informed machine learning for power grid infrastructure
**Researched:** 2026-01-26
**Overall Confidence:** MEDIUM-HIGH

## Executive Summary

Grid Guardian operates at the intersection of graph neural networks, self-play adversarial learning, and physics-informed constraints for energy anomaly detection on UK distribution networks. Research reveals this is a novel combination — while GNN-based fault detection and self-play for adversarial training exist independently, their integration for anomaly detection in power grids represents unexplored territory.

The recommended approach uses PyTorch Geometric with temporal extensions for topology-aware verification, integrated into the existing self-play framework (Proposer-Solver-Verifier) via a hybrid architecture that combines hard physics constraints with learned GNN patterns. The system should be validated through multi-level evaluation (physics violations, synthetic scenarios, baseline comparisons, qualitative review) since no ground-truth anomaly labels exist. Real-time demonstration capability requires robust API integration with circuit breaker patterns and fallback strategies for the Elexon BMRS API.

Key risks center on three critical failure modes: (1) GNN oversmoothing causing loss of spatial resolution needed for localization, (2) self-play mode collapse where Proposer-Verifier converge to narrow behavior patterns, and (3) evaluation validity when metrics are measured only on self-generated synthetic anomalies. Each has proven mitigation strategies from recent literature (2025-2026) including residual connections, entropy-based diversity rewards, and physics-based pseudo-labeling respectively.

## Key Findings

### Recommended Stack

**Core Decision: PyTorch Geometric + Temporal Extensions + Apache Kafka + Redis**

PyTorch Geometric (PyG) is the clear choice for GNN implementation over alternatives like DGL, driven by stronger energy domain adoption (PowerGNN, gnn-powerflow projects), larger community (13.7K+ GitHub stars), and seamless integration with the existing PyTorch 2.1.0 codebase. PyTorch Geometric Temporal provides ready-made spatiotemporal architectures (TGCN, A3TGCN, DCRNN) critical for modeling time-evolving consumption patterns on static grid topology.

**Core technologies:**
- **PyTorch Geometric 2.5+**: Graph neural network framework — proven in 2025-2026 energy research, provides optimized message passing and neighbor sampling for 100K node SSEN graphs
- **PyTorch Geometric Temporal 0.56+**: Temporal GNN library — combines spatial graph structure with time-series dependencies, essential for capturing grid inertia and sequential operations
- **ElexonDataPortal Python client**: Elexon BMRS API integration — automates UK grid data ingestion with standardized parameter handling and rate limit management (5,000 req/min)
- **Apache Kafka (confluent-kafka-python)**: Real-time streaming backbone — industry standard for critical infrastructure, supports replay for debugging, proven in energy sector
- **Redis 7.x**: State store and cache — sub-millisecond latency for GNN inference results, temporal score tracking, pub/sub for real-time alerts

**Critical version requirements:**
- PyTorch Geometric requires pip-only installation (no Conda for PyTorch >2.5.0)
- Extension packages (pyg_lib, torch_scatter, torch_sparse) must match PyTorch CUDA version exactly
- Elexon BMRS API enforces 5,000 requests/minute rate limit (requires exponential backoff implementation)

**Alternative considered but rejected:** DGL (superior for >1M node graphs but overkill for SSEN's 100K nodes), Bytewax (Python-native streaming but lacks ecosystem maturity of Kafka)

### Expected Features

**Must have (table stakes):**
- **Physics-based hard constraints (UK G59/3, BS 7671:2018)**: Voltage bounds (207-253V), household capacity limits (7.5 kWh/30min), non-negativity, ramp rates — already implemented in existing VerifierAgent, validation shows these are expected in all production energy anomaly detectors
- **Baseline anomaly detection methods**: Statistical Z-score, seasonal decomposition, Isolation Forest, autoencoder — already implemented in src/fyp/baselines/anomaly.py, required for academic rigor and comparison
- **Time series handling**: 30-minute resolution windowing, temporal train/test splits, historical context windows (7 days) — infrastructure exists, prevents data leakage
- **Evaluation metrics**: Precision/Recall/F1, MAE/RMSE, physics compliance scoring, event-based detection — implemented in src/fyp/metrics.py

**Should have (competitive differentiators):**
- **Self-play anomaly generation (AZR methodology)**: Proposer generates adversarial scenarios, curriculum learning with learnability rewards, 5 scenario types (EV_SPIKE, COLD_SNAP, PEAK_SHIFT, OUTAGE, MISSING_DATA) — 70% implemented, novel research contribution for energy domain
- **GNN-based topology-aware verification**: Message passing respects feeder→substation hierarchy, spatial-temporal architecture combining graph structure with time series — NEEDS IMPLEMENTATION, research shows 4-17% accuracy improvement over rule-based verification
- **Uncertainty quantification**: Monte Carlo Dropout for epistemic uncertainty, conformal prediction for calibrated intervals — NEEDS IMPLEMENTATION, provides risk-stratified decision making
- **Three-layer ensemble**: Physics constraints (Layer 1, veto power) → GNN learned patterns (Layer 2) → Uncertainty calibration (Layer 3) — hybrid approach combines explainability with generalization

**Defer (v2+, out of FYP scope):**
- Real-time production deployment infrastructure (monitoring, alerting, high-availability architecture)
- Multi-country grid support (different voltage standards, regulatory frameworks)
- Manual anomaly labeling (expensive, defeats purpose of self-play)
- Forecasting-based anomaly detection (PROJECT.md documents this FAILED in v1-v3 experiments)

### Architecture Approach

**Recommended: Hybrid GNN-Verifier with Three-Layer Cascade**

The architecture extends the existing Proposer-Solver-Verifier self-play framework by replacing the rule-based VerifierAgent with a hybrid system that combines physics validation, graph-aware learning, and temporal consistency. The cascade design allows early exit on hard constraint violations (avoiding expensive GNN computation) while preserving physics guarantees.

**Major components:**
1. **Hard Constraint Layer**: Fast physics validation (non-negativity, absolute limits, ramp rates) with veto power — returns binary valid/invalid plus violation details, provides explainability and safety guarantees
2. **GNN Topology Layer**: Graph Attention Network (GAT) with hierarchical message passing respecting feeder→substation→primary topology — processes batch of node forecasts with adjacency matrix (physical + learned k-NN), outputs per-node anomaly scores
3. **Temporal Consistency Layer**: Time-series plausibility scoring independent of graph structure — validates autocorrelation, seasonality preservation, trend stability, distribution similarity
4. **Fusion & Reward Module**: Combines layer outputs with adaptive weighting (α=1.0 for hard constraints veto, β=0.6 for graph, γ=0.4 for temporal) — produces unified verification reward for self-play loop

**Graph construction strategy:**
- Nodes: Households, feeders, substations with features (forecast statistics, historical baselines, grid metadata, temporal context)
- Edges: Hybrid adjacency matrix combining physical topology from SSEN metadata (binary connections) with learned k-NN correlations in consumption space (weighted by λ=0.3)
- Message passing: 3 layers max to prevent oversmoothing (Layer 1: household→feeder, Layer 2: feeder→substation, Layer 3: substation context broadcast)

**Integration with existing codebase:**
- GraphConstructor module: New component to build PyTorch Geometric batches from Solver forecasts
- GraphAwareProposer: Extends ProposerAgent with correlated scenario generation respecting topology (COLD_SNAP, OUTAGE cascade to neighbors)
- SelfPlayTrainer: Minimal modifications, calls GraphConstructor after Solver prediction, before Verifier evaluation

**Real-time inference pipeline:**
- Data ingress (Elexon BMRS polling every 30s) → Preprocessing (validation, resampling, imputation) → Feature engineering (windowing, scaling) → Parallel inference (Solver + Verifier in ThreadPool) → Anomaly scoring with adaptive thresholds → Output alerts
- Target latency: <100ms end-to-end (20ms solver, 30ms GNN verifier, 50ms overhead)
- Optimization: Batch inference (32 entities), FP16 quantization, graph caching, ONNX conversion if needed

### Critical Pitfalls

1. **GNN Oversmoothing in Deep Architectures** — Node representations converge to indistinguishable vectors as layers increase, losing spatial resolution needed for anomaly localization at different grid hierarchy levels. For SSEN's 100K nodes, this means inability to distinguish whether anomaly is at LV feeder or substation. **Prevention:** Use residual connections (provably prevent oversmoothing), gated updates (effective for >10 layers), limit depth to 2-4 layers initially, apply DropEdge regularization. **Detection:** Monitor node embedding similarity via cosine metrics, check if validation performance degrades with added layers.

2. **Self-Play Mode Collapse in Anomaly Generation** — Proposer collapses to generating only 1-2 anomaly types repeatedly, Verifier overfits to those patterns, self-play feedback loop reinforces narrow behavior. This invalidates core methodology and research contribution. **Prevention:** Entropy-based advantage shaping (MSSR framework), explicit diversity metrics with minimum thresholds (each of 5 scenario types must appear in ≥15% of batches), multi-objective rewards balancing success + diversity, curriculum learning. **Detection:** Track distribution entropy of generated anomaly types, monitor Verifier confusion matrix, test on held-out physics violations.

3. **Evaluation Validity Without Ground-Truth Labels** — Reporting high metrics on self-generated synthetic anomalies that don't represent real distribution shifts, making results unverifiable. When tested on actual grid anomalies, performance collapses. **Prevention:** Multi-level validation hierarchy: (1) Physics violations as ground truth (voltage bounds, capacity limits), (2) Synthetic scenarios for relative comparison, (3) Baseline comparison (IsolationForest, DecompositionAnomalyDetector), (4) Domain expert qualitative review. Never report only synthetic scenario performance. **Detection:** Compare physics violation detection vs. synthetic performance — gap indicates brittleness.

4. **Physics Constraint Integration Brittleness** — Loss function becomes impossible to balance between data fit, physics constraints, and anomaly detection objectives. Constraints either too rigid (training fails) or too loose (no actual constraint). **Prevention:** Staged training (data fitting → soft physics constraints → hard constraints), adaptive weighting with separate loss monitoring, projection layers for hard constraint enforcement post-GNN. **Grid Guardian specific:** Use SSEN constraints (feeder capacity, ±6% voltage, ramp rates) as soft L2 penalty during training, hard projection layer at inference.

5. **Elexon BMRS API Reliability and Fallback Fragility** — FYP demo depends on live API, which fails during presentation due to downtime, rate limiting, or network issues. **Prevention:** Circuit breaker pattern with 3-tier fallback (live API → cached responses with 5min TTL → synthetic data generator → graceful degradation), pre-demo warmup 5 minutes before presentation to populate cache, DEMO_MODE=offline environment variable using pre-recorded responses. **Detection:** Monitor API success rates in development (<95% indicates need for fallback).

## Implications for Roadmap

Based on research, suggested phase structure aligns technical dependencies with academic deliverables for May 2026 FYP submission (16 weeks available from late January):

### Phase 1: GNN Verifier Foundation (Weeks 1-2)
**Rationale:** GNN-based verification is the core novel contribution and prerequisite for all downstream work. Must be built first before integration into self-play loop. Well-documented patterns in PyG reduce implementation risk.

**Delivers:**
- Functional GNN topology verifier module using PyTorch Geometric GAT layers
- Graph construction pipeline from SSEN metadata to PyG Data batches
- Trained model on historical data + synthetic anomalies achieving >85% accuracy
- Inference latency <30ms for batch_size=32

**Addresses (from FEATURES.md):** GNN-based topology-aware verification (must-have differentiator), spatial-temporal architecture

**Uses (from STACK.md):** PyTorch Geometric, PyTorch Geometric Temporal

**Avoids (from PITFALLS.md):** Oversmoothing via 2-4 layer limit with residual connections from start, overfitting via DropEdge regularization

### Phase 2: Hybrid Verifier Integration (Weeks 3-4)
**Rationale:** Extends existing VerifierAgent to combine physics constraints with GNN, preserving safety guarantees while adding learned patterns. Dependencies on Phase 1 (GNN must work) and existing codebase (physics constraints, self-play framework already implemented).

**Delivers:**
- HybridVerifierAgent with three-layer cascade (physics → GNN → temporal)
- Updated SelfPlayTrainer using hybrid verifier in training loop
- Validation showing hybrid improves detection vs. baseline rule-based verifier
- Integration tests for full self-play pipeline

**Addresses (from FEATURES.md):** Three-layer ensemble (differentiator), physics constraint integration (table stakes)

**Implements (from ARCHITECTURE.md):** Fusion module with adaptive weighting, cascade logic with early exit

**Avoids (from PITFALLS.md):** Physics constraint brittleness via staged training, separate loss monitoring, projection layers

### Phase 3: Self-Play with Diversity Safeguards (Week 5)
**Rationale:** With hybrid verifier functional, train self-play loop with explicit mode collapse prevention. GraphAwareProposer generates correlated scenarios respecting topology. Critical for research validity.

**Delivers:**
- GraphAwareProposer with correlated scenario generation (COLD_SNAP/OUTAGE cascade to neighbors)
- Entropy-based diversity tracking with minimum thresholds (≥15% per scenario type)
- Self-play training runs without mode collapse (validated via scenario distribution monitoring)
- Scenario visualization tools for qualitative assessment

**Addresses (from FEATURES.md):** Self-play anomaly generation (core differentiator), curriculum learning with learnability rewards

**Avoids (from PITFALLS.md):** Mode collapse via diversity metrics, entropy-based rewards, multi-objective optimization

**Research flag:** Monitor training carefully for collapse; have fallback to frozen verifier if needed

### Phase 4: Uncertainty Quantification (Week 6)
**Rationale:** Adds risk-stratified decision making on top of functional anomaly detector. Monte Carlo Dropout is lightweight (50 lines of code) but strengthens contribution. Conformal prediction provides theoretical grounding.

**Delivers:**
- MC Dropout epistemic uncertainty via dropout sampling at inference (50-100 forward passes)
- Conformal prediction calibration for distribution-free prediction intervals with coverage guarantees
- Risk stratification: high-confidence anomalies (require investigation) vs. low-confidence (possible false positives)
- Calibration metrics: coverage error, interval sharpness

**Addresses (from FEATURES.md):** Uncertainty quantification (should-have differentiator)

**Uses (from STACK.md):** Torch-Uncertainty or native PyTorch implementation, MAPIE for conformal prediction

**Complexity:** LOW (standard techniques, libraries available)

### Phase 5: Multi-Level Evaluation Framework (Weeks 7-8)
**Rationale:** Research validity depends on proper evaluation without ground-truth labels. Multi-level validation (physics + synthetic + baselines + qualitative) provides convergent evidence. Benchmarking against existing methods required for academic rigor.

**Delivers:**
- Four-level validation hierarchy implemented: physics violations (ground truth), synthetic scenarios, baseline comparisons, domain plausibility
- Ablation studies: GNN-only vs. physics-only vs. hybrid, effect of each component
- Benchmark comparison: IsolationForest, DecompositionAnomalyDetector, PatchTST baseline on same splits
- Performance tables, evaluation notebooks, result visualization

**Addresses (from FEATURES.md):** Evaluation metrics (table stakes), comparison vs baselines (academic requirement)

**Avoids (from PITFALLS.md):** Invalid evaluation via multi-level validation, never reporting only synthetic metrics

**Critical for FYP:** This phase validates research contribution; cannot be skipped

### Phase 6: Elexon BMRS Live Integration (Optional, Week 9-10)
**Rationale:** Real-time UK grid data demonstration adds presentation value but is not core research contribution. Only pursue if Phases 1-5 complete successfully with time remaining.

**Delivers:**
- Elexon BMRS API integration with circuit breaker pattern and 3-tier fallback
- Real-time inference pipeline: data ingress → preprocessing → parallel solver/verifier → adaptive thresholding
- Live dashboard (Flask + Plotly) displaying anomaly detections on UK grid data
- Pre-demo warmup script and DEMO_MODE=offline fallback

**Addresses (from FEATURES.md):** Live data capability (optional presentation value, PROJECT.md requirement)

**Uses (from STACK.md):** ElexonDataPortal, Apache Kafka, Redis, confluence-kafka-python

**Avoids (from PITFALLS.md):** API reliability issues via circuit breaker, cache, synthetic fallback

**Complexity:** MEDIUM (API integration straightforward but requires robust error handling)

### Phase Ordering Rationale

**Dependency-driven sequencing:**
- GNN Verifier (Phase 1) is prerequisite for Hybrid Verifier (Phase 2) and Self-Play training (Phase 3)
- Uncertainty Quantification (Phase 4) requires trained models from Phases 1-3 but can proceed in parallel with Phase 3
- Evaluation (Phase 5) needs all components functional but does not block other phases
- Elexon integration (Phase 6) is independent and optional

**Risk mitigation:**
- Critical failure modes addressed early: GNN implementation (Phase 1), mode collapse prevention (Phase 3), evaluation validity (Phase 5)
- Physics constraints already implemented; hybrid approach preserves them while adding GNN
- Validation designed from start, not added post-hoc

**Timeline fit:**
- Core research (Phases 1-5): 8 weeks focused development
- Buffer: 2 weeks for debugging, integration issues
- Writing: 4-6 weeks overlapping with Phases 4-6
- Total: 16 weeks to May 2026 submission is feasible

### Research Flags

**Phases needing deeper research during planning:**
- **Phase 3 (Self-Play):** Monitor training stability closely; mode collapse is well-documented risk but mitigation effectiveness varies. Have contingency: freeze verifier if collapse occurs.
- **Phase 6 (Elexon Integration):** API rate limits and downtime poorly documented; may need trial-and-error to tune circuit breaker thresholds.

**Phases with standard patterns (minimal additional research):**
- **Phase 1 (GNN Verifier):** PyTorch Geometric well-documented, PowerGNN provides reference implementation for power grids
- **Phase 4 (Uncertainty):** MC Dropout and conformal prediction are mature techniques with established libraries
- **Phase 5 (Evaluation):** Multi-level validation strategies documented in recent anomaly detection literature

**Recommended approach:** Execute Phases 1-2 first, then reassess Phase 3 based on observed training behavior. Phase 6 is optional; skip if timeline pressure.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | PyTorch Geometric proven in energy domain (PowerGNN, gnn-powerflow), ElexonDataPortal actively maintained, Kafka industry standard for critical infrastructure. Version compatibility verified, installation paths documented. |
| Features | HIGH | Table stakes features already 70% implemented, differentiators (GNN, self-play, uncertainty) validated in 2025-2026 literature. Clear separation of must-have vs. optional reduces scope risk. |
| Architecture | MEDIUM-HIGH | Hybrid verifier pattern combines proven techniques (physics constraints exist, GAT well-established) but specific integration is novel. Graph construction strategy documented in PowerGNN. Real-time pipeline follows standard patterns. |
| Pitfalls | HIGH | Top 5 pitfalls backed by multiple sources (oversmoothing: 8+ sources, mode collapse: 7+ sources, evaluation validity: 6+ sources). Mitigation strategies proven in recent literature. Grid Guardian-specific recommendations provided. |

**Overall confidence:** MEDIUM-HIGH

**Reasoning:**
- Core components (PyG, self-play framework, physics constraints) are well-researched with high-quality sources (arXiv, peer-reviewed, official docs)
- Novel combination (GNN + self-play for energy anomaly detection) is unexplored but built from proven building blocks
- Integration complexity is the primary risk, but staged approach (Phase 1 validates GNN, Phase 2 validates hybrid, Phase 3 validates self-play) provides incremental validation
- Timeline is tight but feasible with 16 weeks available; MVP (Phases 1-5) achievable in 10 weeks with 6-week writing overlap

### Gaps to Address

**Graph construction from SSEN metadata:**
- Research confirms SSEN data contains feeder→substation topology but specific schema parsing not documented. May require trial-and-error in Phase 1.
- **Mitigation:** Start with synthetic graph (ring topology, scale-free) for initial GNN development, then integrate real SSEN topology once schema is understood.

**Optimal hyperparameters for 100K node graphs:**
- Literature provides ranges (2-4 GNN layers, hidden_dim=64, num_heads=4 for GAT) but optimal values for SSEN-specific topology unknown.
- **Mitigation:** Hyperparameter sweep in Phase 1 validation, document what works for Grid Guardian specifically.

**Self-play training stability with GNN verifier:**
- Most self-play research uses simpler discriminators (MLPs, linear layers). GNN verifier adds complexity that may destabilize training.
- **Mitigation:** Phase 2 tests hybrid verifier in isolation before Phase 3 integrates into self-play loop. If unstable, freeze GNN and fine-tune only fusion weights.

**Elexon BMRS API rate limit behavior:**
- 5,000 requests/minute documented but actual throttling behavior (HTTP 429 response time, exponential backoff recommendations) not detailed.
- **Mitigation:** Phase 6 is optional. If pursued, implement conservative rate limiting (4,500 req/min buffer) and test thoroughly before demo dependency.

## Sources

### Primary (HIGH confidence)
**Stack Research:**
- PyTorch Geometric official docs (pytorch-geometric.readthedocs.io) — installation, API reference
- PowerGNN: A Topology-Aware Graph Neural Network for Electricity Grids (arxiv:2503.22721v1, March 2025) — domain-specific GNN architecture
- ElexonDataPortal Python client (github.com/OSUKED/ElexonDataPortal) — UK grid API integration
- Kafka Python clients comparison (Confluent docs, Quix comparative analysis) — streaming infrastructure

**Features Research:**
- Physics-Informed Convolutional Autoencoder for Cyber Anomaly Detection (arxiv:2312.04758) — physics constraints improve accuracy
- Self-play SWE-RL (alphaxiv.org/overview/2512.18552v1, December 2025) — adversarial self-play methodology
- Bayesian autoencoders with uncertainty quantification (ScienceDirect, 2022) — epistemic/aleatoric uncertainty
- PowerGraph benchmark dataset (NeurIPS 2024) — GNN validation for power grids

**Architecture Research:**
- Graph Attention Networks (Veličković et al.) — GAT layer design
- Hierarchical graph pooling (DiffPool, NeurIPS 2018) — multi-resolution embeddings
- Temporal Graph Learning 2024 (Towards Data Science) — spatiotemporal GNN survey
- Real-time ML Inference Infrastructure (Databricks) — deployment patterns

**Pitfalls Research:**
- Overcoming Policy Collapse in Deep RL (OpenReview ICLR) — mode collapse mitigation
- Solving Oversmoothing via Nonlocal Message Passing (arxiv:2512.08475, December 2025) — GNN depth issues
- Towards Unsupervised Validation of Anomaly-Detection Models (arxiv:2410.14579v1) — evaluation without labels
- Circuit Breaker Pattern (Azure Architecture Center) — fallback strategies

### Secondary (MEDIUM confidence)
- DGL vs. PyTorch Geometric benchmarks (dgl.ai, Medium articles) — GNN framework comparison
- MTAD-GAT implementation (github.com/ML4ITS/mtad-gat-pytorch) — temporal anomaly detection baseline
- Elexon BMRS API rate limits (Elexon Portal docs) — 5,000 req/min threshold
- Torch-Uncertainty framework (arxiv:2511.10282v1) — PyTorch UQ toolkit

### Tertiary (LOW confidence, needs validation)
- Optimal GNN depth for power grids — inferred from general graph literature, needs domain-specific tuning
- Self-play training stability with GNN verifier — no prior work found combining these, will require experimentation
- SSEN-specific graph construction schema — assumed from EDA, needs verification during implementation

---
*Research completed: 2026-01-26*
*Ready for roadmap: yes*

**Next Steps for Orchestrator:**
1. Use Phase 1-6 structure as starting point for roadmap creation
2. For Phase 1 (GNN Verifier), detailed requirements can be derived directly from STACK.md and ARCHITECTURE.md
3. For Phase 3 (Self-Play), flag for extra planning attention due to training stability concerns
4. Phase 6 (Elexon) should be marked optional/stretch goal given timeline constraints
5. Commit all research files together per synthesis workflow
