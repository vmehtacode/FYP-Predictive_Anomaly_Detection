# Grid Guardian: Predictive Anomaly Detection for UK Power Grids
## Aston University Final Year Project — Term 1 Report

**Student:** [Your Name]
**Supervisor:** Dr. Farzaneh Farhadi
**Date:** January 2026

---

# Table of Contents

## Abstract
- Brief overview of Grid Guardian: GNN-based anomaly detection with self-play methodology
- Key contribution: demonstrating when self-play adds value vs. when simpler approaches suffice
- Summary of current progress (Phase 1 complete: 98.33% accuracy achieved)

---

## 1. Introduction

### 1.1 Context and Motivation
- UK energy grid transition: increasing renewable penetration, distributed generation, EV adoption
- Growing need for real-time anomaly detection in distribution networks
- Limitations of traditional rule-based SCADA monitoring systems
- Gap: lack of topology-aware ML approaches for distribution-level anomaly detection

**Source:** `.planning/PROJECT.md` (Context section), `.planning/research/SUMMARY.md` (Executive Summary)

### 1.2 Problem Statement and Research Questions
- **Primary RQ:** Can self-play adversarial learning improve anomaly detection in power grids?
- **Secondary RQ1:** How does GNN topology awareness affect detection accuracy?
- **Secondary RQ2:** When does self-play methodology add value over simpler baselines?
- **Key Finding (Negative Result):** Self-play does NOT improve forecasting for highly periodic time series
- **The Pivot:** Self-play excels at detecting BREAKS in periodicity (anomalies) rather than predicting normal consumption

**Source:** `.planning/PROJECT.md` (Key Finding section)

### 1.3 Research Objectives
- **RO1:** Develop GNN-based verifier that understands SSEN grid topology
- **RO2:** Integrate self-play framework with graph-aware anomaly generation
- **RO3:** Evaluate against physics-based ground truth and baseline detectors
- **RO4:** Demonstrate methodology for identifying when self-play is (in)appropriate

**Source:** `.planning/REQUIREMENTS.md` (v1 Requirements)

### 1.4 Significance and Contributions
- Novel combination: GNN + self-play for energy anomaly detection (unexplored territory)
- Methodological contribution: characterizing when self-play doesn't work is publishable
- Practical value: topology-aware detection for UK distribution networks
- Academic value: rigorous evaluation framework for unsupervised anomaly detection

**Source:** `.planning/research/SUMMARY.md` (Key Findings)

### 1.5 Approach Overview
- Three-layer detection ensemble: Physics constraints → GNN patterns → Uncertainty calibration
- Self-play cycle: Proposer generates scenarios, Solver forecasts, Verifier validates
- Validation without labels: multi-level evaluation (physics + synthetic + baselines + domain review)

**Source:** `.planning/research/ARCHITECTURE.md` (Recommended Architecture)

**GAP:** Write formal academic introduction prose, add citations

---

## 2. Background Research

### 2.1 Related Work

#### 2.1.1 Graph Neural Networks for Power Grids
- PowerGNN: topology-aware GNN for electricity grids (arXiv:2503.22721)
- Physics-Informed GNN for dynamic reconfiguration (arXiv:2310.00728)
- PowerFlowNet: message passing for power flow approximation (arXiv:2311.03415)
- Benchmark: PowerGraph dataset (NeurIPS 2024)

#### 2.1.2 Anomaly Detection in Energy Systems
- Physics-Informed Convolutional Autoencoder for Cyber Anomaly Detection (arXiv:2312.04758)
- Graph Anomaly Detection with GNNs: Current Status and Challenges (arXiv:2209.14930)
- SCADA-based anomaly detection in smart grids (ScienceDirect 2024)

#### 2.1.3 Self-Play and Adversarial Learning
- Absolute Zero Reasoner (AZR): Proposer-Solver-Verifier methodology
- SPELL: Self-Play for evolving language models (arXiv:2509.23863)
- Propose, Solve, Verify through formal verification (arXiv:2512.18160)

**Source:** `.planning/research/SUMMARY.md` (Sources section), `.planning/research/PITFALLS.md` (Sources)

**GAP:** Expand literature review with full academic citations in BibTeX format

### 2.2 Relevant Methods and Technologies

#### 2.2.1 GNN Architectures
- Graph Attention Networks (GAT): attention mechanism learns neighbor importance
- GraphSAGE: sampling-based for scalable inference
- Temporal GNN: combines spatial graph + time-series dependencies (TGCN, A3TGCN)
- Oversmoothing mitigation: residual connections, DropEdge, depth limits

#### 2.2.2 Physics-Informed Constraints
- UK grid standards: G59/3, BS 7671:2018
- Hard constraints: voltage bounds (207-253V), capacity limits (7.5 kWh/30min), ramp rates
- Soft constraint integration: L2 penalty during training, projection layers at inference

#### 2.2.3 Evaluation Without Ground Truth
- Anomaly Separation Index (ASI), Excess-Mass curves
- Physics-based pseudo-labels as ground truth
- Multi-level validation hierarchy

**Source:** `.planning/research/ARCHITECTURE.md`, `.planning/research/PITFALLS.md`, `.planning/research/STACK.md`

### 2.3 Datasets and Resources

#### 2.3.1 Available Datasets
| Dataset | Records | Coverage | Key Features |
|---------|---------|----------|--------------|
| UK-DALE | 114M | House-level | Appliance-level disaggregation |
| LCL | 167M | 5,567 households | London smart meter trial |
| SSEN | 100K metadata + 100K consumption | Distribution network | Feeder→substation topology |

#### 2.3.2 External APIs
- Elexon BMRS: real-time UK grid data (5,000 req/min, free)
- SSEN metadata: graph structure for distribution network

**Source:** `.planning/PROJECT.md` (Technical Environment), `.planning/research/SUMMARY.md`

### 2.4 Reflection: Impact on Project Design
- Literature confirms GNN-based verifiers outperform rule-based (4-17% accuracy improvement)
- Identified 5 critical pitfalls with mitigation strategies
- Adopted hybrid architecture: physics guarantees + GNN flexibility
- Evaluation framework designed from literature (multi-level validation)

**Source:** `.planning/research/SUMMARY.md` (Implications for Roadmap)

**GAP:** Add critical analysis comparing approaches, justify design choices academically

---

## 3. Methodology

### 3.1 Requirements Analysis

#### 3.1.1 Functional Requirements
| ID | Requirement | Phase | Status |
|----|-------------|-------|--------|
| GNN-01 | PyTorch Geometric graph from SSEN topology | 1 | Complete |
| GNN-02 | Spatial-temporal GNN architecture (GAT + GRU) | 1 | Complete |
| GNN-03 | Replace MLP Verifier with GNN in self-play | 2 | Pending |
| SELF-01 | Graph-aware Proposer with topology scenarios | 3 | Pending |
| SELF-02 | Cascade propagation (COLD_SNAP, OUTAGE) | 3 | Pending |
| ENS-01 | Three-layer ensemble integration | 2 | Pending |
| ENS-02 | Cascade early-exit logic | 2 | Pending |
| EVAL-01 | Precision/recall/F1 against baselines | 4 | Pending |
| EVAL-02 | Comparison vs IsolationForest, Autoencoder | 4 | Pending |
| EVAL-03 | Physics compliance rate measurement | 4 | Pending |

#### 3.1.2 Non-Functional Requirements
- Inference latency: <30ms for batch_size=32
- Accuracy: >85% on synthetic anomalies (achieved: 98.33%)
- Reproducibility: seed-controlled experiments with DVC versioning

**Source:** `.planning/REQUIREMENTS.md`

### 3.2 System Specifications

#### 3.2.1 Technology Stack
- **Core:** Python 3.11+, PyTorch 2.1+, PyTorch Geometric 2.5+
- **Data:** Pandas, PyArrow (Parquet), Pydantic validation
- **ML:** scikit-learn baselines, MLflow tracking, DVC versioning
- **GNN:** PyTorch Geometric Temporal for spatiotemporal architectures

#### 3.2.2 Data Schema
```
EnergyReading(ts_utc, entity_id, energy_kwh, dataset,
              interval_mins, source, extras)
```

**Source:** `.planning/research/STACK.md`, `.planning/codebase/STACK.md`

### 3.3 Design Decisions

#### 3.3.1 Architecture: Hybrid GNN-Verifier
| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Hybrid over pure GNN | Physics constraints provide safety guarantees | Implemented |
| GAT over GCN | Attention learns neighbor importance | Implemented |
| 3-layer cascade | Early exit reduces latency, preserves physics | Designed |
| Elexon over SSEN API | Elexon provides true real-time data | Deferred to v2 |

#### 3.3.2 Graph Construction Strategy
- **Nodes:** Households, feeders, substations with 13 features
- **Edges:** Physical topology (SSEN) + learned k-NN correlations (λ=0.3)
- **Message passing:** 3 layers max (household→feeder→substation)

#### 3.3.3 Self-Play Enhancements
- Diversity tracking: each of 5 scenario types ≥15% of batches
- Mode collapse prevention: entropy-based advantage shaping
- Physics validation: hard constraints before training reward

**Source:** `.planning/research/ARCHITECTURE.md`, `.planning/PROJECT.md` (Key Decisions)

#### 3.3.4 Architecture Diagram

![Three-Layer Cascade Architecture](figures/three_layer_cascade.svg)

*Figure 1: Three-layer cascade architecture showing early-exit paths for physics violations (Layer 1) and high-confidence predictions (Layer 2), with uncertainty quantification (Layer 3) for ambiguous cases.*

### 3.4 Proof-of-Concept Implementation

#### 3.4.1 Phase 1 Deliverables (Complete)
- `src/fyp/selfplay/gnn/graph_builder.py`: SSEN topology → PyG Data
- `src/fyp/selfplay/gnn/gat_verifier.py`: GAT architecture with temporal encoding
- `src/fyp/selfplay/gnn/anomaly_dataset.py`: Synthetic anomaly generator
- `src/fyp/selfplay/gnn/trainer.py`: Training pipeline with oversmoothing prevention

#### 3.4.2 Achieved Results
- **Accuracy:** 98.33% on synthetic anomalies (target: >85%)
- **Inference latency:** <30ms for batch_size=32
- **Tests:** 28/28 passing

**Source:** `.planning/phases/01-gnn-verifier-foundation/*.md`, `.planning/ROADMAP.md`

**GAP:** Include code snippets as figures (optional)

---

## 4. Implementation

### 4.1 Current State
- **Phase 1 Complete:** GNN Verifier Foundation (6 plans executed)
- **Codebase:** ~15K lines Python, modular architecture
- **Testing:** 28/28 tests passing, CI/CD via pytest

### 4.2 Module Structure
```
src/fyp/
├── ingestion/      # Dataset ingestors (LCL, UK-DALE, SSEN)
├── baselines/      # SeasonalNaive, MovingAverage, IsolationForest
├── models/         # PatchTST, Autoencoder, FrequencyEnhanced
├── selfplay/       # Proposer, Solver, Verifier, Trainer
│   └── gnn/        # NEW: Graph builder, GAT verifier, training
├── metrics.py      # MAE, RMSE, MAPE, MASE, F1
└── config.py       # Pydantic-based ExperimentConfig
```

### 4.3 Key Implementation Details
- **Graph construction:** 7.6k nodes, 16.3k edges from SSEN metadata
- **GAT architecture:** 4 attention heads, 64 hidden dim, 3 layers with residuals
- **Training:** Adam optimizer, BCE loss, DropEdge regularization

**Source:** `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`

**GAP:** Limited implementation detail for Term 1 (expected per G3.pdf)

---

## 5. Evaluation Plan

### 5.1 Evaluation Strategy

#### 5.1.1 Multi-Level Validation Hierarchy
| Level | Ground Truth | Metrics | Purpose |
|-------|--------------|---------|---------|
| 1 | Physics violations | Detection rate | Hard constraint validation |
| 2 | Synthetic scenarios | Precision/Recall/F1 | Self-play effectiveness |
| 3 | Baseline comparison | Relative performance | Academic rigor |
| 4 | Domain plausibility | Qualitative review | Real-world validity |

#### 5.1.2 Baseline Comparisons
- IsolationForest (existing)
- DecompositionAnomalyDetector (existing)
- Autoencoder (existing)
- Rule-based VerifierAgent (existing)

#### 5.1.3 Ablation Studies
- GNN-only vs. physics-only vs. hybrid ensemble
- Effect of graph topology (with vs. without)
- Self-play diversity impact

**Source:** `.planning/research/PITFALLS.md` (Pitfall 4), `.planning/REQUIREMENTS.md` (EVAL-*)

### 5.2 Planned Experiments

| Experiment | Hypothesis | Metrics |
|------------|------------|---------|
| GNN vs MLP Verifier | GNN improves spatial anomaly detection | F1, physics compliance |
| Self-play vs supervised | Self-play generalizes better | OOD accuracy |
| Cascade early-exit | Reduces latency without accuracy loss | p95 latency, F1 |

### 5.3 Ethics Declaration

**Ethics Status:** No ethical approval required

**Justification:**
- All datasets are publicly available (LCL, UK-DALE) or licensed (SSEN)
- No personal data processed beyond anonymized consumption records
- No human participants involved
- System is academic demonstration, not production deployment
- API usage within free tier limits (Elexon BMRS)

**Source:** `.planning/PROJECT.md` (Out of Scope)

**GAP:** Formal ethics declaration form may be required by Aston

---

## 6. Project Management

### 6.1 Timeline and Milestones

| Phase | Status | Weeks | Requirements |
|-------|--------|-------|--------------|
| 1. GNN Verifier Foundation | Complete | 1-2 | GNN-01, GNN-02 |
| 2. Hybrid Verifier Integration | Not started | 3-4 | GNN-03, ENS-01, ENS-02 |
| 3. Graph-Aware Proposer | Not started | 5 | SELF-01, SELF-02 |
| 4. Evaluation Framework | Not started | 6-7 | EVAL-01, EVAL-02, EVAL-03 |
| 5. Report Writing | Ongoing | 8-16 | Final submission |

### 6.2 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GNN oversmoothing | Medium | High | Residual connections, 3-layer limit |
| Self-play mode collapse | Medium | Critical | Diversity tracking, entropy rewards |
| Evaluation without labels | High | Critical | Multi-level validation hierarchy |
| API reliability (demo) | Medium | Medium | Circuit breaker, cached fallback |

**Source:** `.planning/research/PITFALLS.md` (Top 3 Project-Killing Risks)

---

## 7. Conclusion and Next Steps

### 7.1 Progress Summary
- Phase 1 complete with 98.33% accuracy (exceeds 85% target)
- Core GNN infrastructure validated and tested
- Research foundation documented comprehensively

### 7.2 Immediate Next Steps
1. **Phase 2:** Integrate GNN into existing self-play loop
2. **Phase 3:** Implement graph-aware scenario generation
3. **Phase 4:** Complete evaluation framework with baselines

### 7.3 Term 2 Goals
- Complete all 4 phases
- Write full report with evaluation results
- Prepare demonstration for viva

---

## References

<!-- GAP: Add full BibTeX citations -->

[1] PowerGNN: A Topology-Aware Graph Neural Network for Electricity Grids (arXiv:2503.22721)
[2] Graph Anomaly Detection with GNNs: Current Status and Challenges (arXiv:2209.14930)
[3] Physics-Informed GNN for Dynamic Reconfiguration (arXiv:2310.00728)
[4] Towards Unsupervised Validation of Anomaly-Detection Models (arXiv:2410.14579)
[5] Circuit Breaker Pattern (Azure Architecture Center)
... [expand with full citations]

---

## Appendices

### Appendix A: Full Requirements Traceability Matrix
**Source:** `.planning/REQUIREMENTS.md`

### Appendix B: Architecture Diagrams

#### Figure B.1: Three-Layer Cascade Architecture
![Three-Layer Cascade Architecture](figures/three_layer_cascade.svg)

**LaTeX inclusion:**
```latex
\begin{figure}[htbp]
    \centering
    \includesvg[width=0.9\textwidth]{figures/three_layer_cascade}
    \caption{Three-layer cascade architecture with early-exit logic. Layer 1 (Physics) provides veto power for hard constraint violations. Layer 2 (GNN Verifier) performs topology-aware anomaly scoring. Layer 3 (Uncertainty) calibrates predictions for ambiguous cases.}
    \label{fig:cascade-architecture}
\end{figure}
```

- Self-Play Training Loop
- Real-Time Inference Pipeline

### Appendix C: Technology Stack Details
**Source:** `.planning/research/STACK.md`

### Appendix D: Code Repository Structure
**Source:** `.planning/codebase/STRUCTURE.md`

---

# Gap Analysis Summary

| Section | Existing Content | Gap | Priority |
|---------|-----------------|-----|----------|
| 1. Introduction | PROJECT.md, SUMMARY.md | Academic prose, formal citations | HIGH |
| 2. Background | research/*.md | Full literature review with BibTeX | HIGH |
| 3. Methodology | REQUIREMENTS.md, ARCHITECTURE.md | Cascade diagram done; add self-play flow | LOW |
| 4. Implementation | codebase/*.md | Code excerpts, screenshots | LOW |
| 5. Evaluation | PITFALLS.md | Formal ethics declaration form | MEDIUM |
| References | Sources listed | Convert to BibTeX format | HIGH |
| Figures | Cascade SVG created | Add self-play loop, data flow diagrams | LOW |

**Estimated New Content Needed:** ~30% (mostly converting existing docs to academic prose and adding citations)
