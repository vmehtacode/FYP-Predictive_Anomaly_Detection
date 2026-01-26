# Grid Guardian

## What This Is

Grid Guardian is an energy anomaly detection system for UK distribution networks, adapting the Absolute Zero Reasoner (AZR) self-play methodology to the energy domain. It detects network-aware anomalies without labeled data by combining GNN-based verification (understanding grid topology) with physics-informed constraints (hard violation detection). This is a Final Year Project at Aston University targeting a top-tier grade through novel methodological contributions.

## Core Value

Detect anomalies in energy distribution networks without labeled data, using physics constraints and self-play learned patterns on graph-structured grid data — demonstrating when self-play methodology adds value versus when simpler approaches suffice.

## Requirements

### Validated

<!-- Existing capabilities from codebase -->

- ✓ Data ingestion pipeline for LCL, UK-DALE, SSEN datasets (281M records) — existing
- ✓ Unified Parquet schema with time-series windowing — existing
- ✓ Baseline forecasters (SeasonalNaive, MovingAverage, Ridge) — existing
- ✓ Baseline anomaly detectors (DecompositionAnomalyDetector, IsolationForest) — existing
- ✓ Neural models (PatchTST attention-based, Autoencoder, FrequencyEnhanced) — existing
- ✓ Self-play framework (Proposer, Solver, Verifier, Trainer) — existing
- ✓ Scenario generation (EV_SPIKE, COLD_SNAP, PEAK_SHIFT, OUTAGE, MISSING_DATA) — existing
- ✓ Configuration system (Pydantic-based ExperimentConfig) — existing
- ✓ Experiment tracking (MLflow logging, DVC versioning) — existing
- ✓ Test suite (28/28 tests passing) — existing
- ✓ CLI runner for forecasting and anomaly detection — existing
- ✓ EDA notebooks with physics constraints extracted — existing
- ✓ Initial self-play experiments (v1-v3) with documented findings — existing

### Active

<!-- Current scope for this milestone -->

- [ ] Elexon BMRS API integration for real-time UK grid data
- [ ] GNN-based Verifier architecture using SSEN network topology
- [ ] Graph-aware Proposer generating topology-respecting anomaly scenarios
- [ ] Physics constraint validation layer (hard constraints from SSEN metadata)
- [ ] 3-layer anomaly detection ensemble (physics + learned + confidence calibration)
- [ ] Evaluation framework against real distribution network data
- [ ] Live data demonstration capability for FYP presentation
- [ ] FYP report following Aston University format
- [ ] Presentation/viva preparation

### Out of Scope

<!-- Explicit boundaries -->

- Real-time production deployment — academic demonstration sufficient
- Mobile/web frontend — CLI and notebooks sufficient for FYP
- Multi-country grid support — UK distribution networks only
- Proprietary data sources — public APIs and datasets only
- Hardware-in-the-loop testing — simulation sufficient

## Context

### Technical Environment
- Python 3.11+ with PyTorch, scikit-learn, Pydantic stack
- 281M energy consumption records (UK-DALE 114M, LCL 167M, SSEN 100K metadata + 100K consumption)
- SSEN provides graph structure: LV feeders → secondary substations → primary substations
- Elexon BMRS provides grid-level data complementing distribution-level SSEN/LCL

### Research Foundation
- AZR (Absolute Zero Reasoner) methodology: Proposer-Solver-Verifier self-play
- GNN papers on fault detection in distribution networks (in Research folder)
- Physics constraints extracted from EDA: feeder capacity, voltage bounds, ramp rates

### Key Finding (Negative Result)
Self-play does NOT improve forecasting for highly periodic time series (daily/weekly energy cycles). After 3 experimental versions:
- v3 achieved 61% performance improvement and 99% variance reduction over v1/v2
- But still 17% worse than simple seasonal naive baseline
- Hypothesis: Self-play adds unnecessary complexity when simple pattern matching captures periodicity

This is a valuable methodological contribution — it identifies WHEN self-play is inappropriate.

### The Pivot
Self-play should excel at detecting BREAKS in periodicity (anomalies) rather than predicting normal consumption. The adversarial Proposer generates scenarios that break patterns; the Verifier learns to detect them.

## Constraints

- **Timeline**: May 2026 submission (~4 months)
- **Data splits**: Temporal train/test only (no shuffling time series)
- **Validation**: Physics-based without ground-truth anomaly labels
- **Methodology**: Publication-quality statistical rigor
- **APIs**: Elexon BMRS (free, no key needed), public datasets only

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Pivot from forecasting to anomaly detection | Self-play doesn't beat baselines for periodic forecasting | — Pending |
| GNN-based Verifier | Grid topology matters for anomaly propagation | — Pending |
| Elexon BMRS over SSEN live API | SSEN API is metadata-only; Elexon provides true real-time data | — Pending |
| 3-layer detection (physics + learned + ensemble) | Combines explainability with generalization | — Pending |
| Negative result as contribution | Characterizing when self-play doesn't work is publishable | — Pending |

---
*Last updated: 2026-01-26 after initialization*
