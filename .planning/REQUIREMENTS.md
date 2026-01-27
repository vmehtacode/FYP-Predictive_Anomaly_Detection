# Requirements: Grid Guardian

**Defined:** 2026-01-27
**Core Value:** Detect anomalies in energy distribution networks without labeled data, using physics constraints and self-play learned patterns on graph-structured grid data

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### GNN Verifier Architecture

- [ ] **GNN-01**: Build PyTorch Geometric graph from SSEN feeder/substation topology
- [ ] **GNN-02**: Implement spatial-temporal GNN architecture (GAT/GraphSAGE + GRU)
- [ ] **GNN-03**: Replace MLP Verifier with GNN Verifier in self-play training loop

### Self-Play Enhancement

- [ ] **SELF-01**: Implement graph-aware Proposer generating topology-respecting scenarios
- [ ] **SELF-02**: Scenarios cascade through neighbors (COLD_SNAP, OUTAGE propagation)

### Three-Layer Ensemble

- [ ] **ENS-01**: Integrate GNN learned detector as Layer 2 with physics constraints
- [ ] **ENS-02**: Implement cascade early exit logic (physics violations skip GNN)

### Evaluation

- [ ] **EVAL-01**: Implement precision/recall/F1 evaluation against baselines
- [ ] **EVAL-02**: Compare GNN Verifier vs IsolationForest, Autoencoder, DecompositionAnomalyDetector
- [ ] **EVAL-03**: Measure physics compliance rate of detected anomalies

## v2 Requirements

Deferred to future milestone. Tracked but not in current roadmap.

### Uncertainty Quantification

- **UQ-01**: Monte Carlo Dropout for epistemic uncertainty
- **UQ-02**: Conformal prediction for calibrated prediction intervals

### External Data Integration

- **DATA-01**: Elexon BMRS API integration for real-time UK grid data
- **DATA-02**: Live demonstration capability for FYP presentation

### Self-Play Enhancement

- **SELF-03**: Verifier weakness targeting (Proposer conditions on struggles)

### Academic Deliverables

- **ACAD-01**: FYP report following Aston University format
- **ACAD-02**: Presentation/viva preparation

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Real-time production deployment | Academic demonstration sufficient per PROJECT.md |
| Multi-country grid support | UK-only scope with SSEN data |
| Ground-truth anomaly labels | Defeats self-play purpose; use physics validation |
| Forecasting-based detection | Proven to fail in v1-v3 experiments |
| Mobile/web frontend | CLI and notebooks sufficient for FYP |
| Proprietary data sources | Public APIs and datasets only |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| GNN-01 | Phase 1 | Pending |
| GNN-02 | Phase 1 | Pending |
| GNN-03 | Phase 2 | Pending |
| SELF-01 | Phase 3 | Pending |
| SELF-02 | Phase 3 | Pending |
| ENS-01 | Phase 2 | Pending |
| ENS-02 | Phase 2 | Pending |
| EVAL-01 | Phase 4 | Pending |
| EVAL-02 | Phase 4 | Pending |
| EVAL-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-01-27*
*Last updated: 2026-01-27 after roadmap creation*
