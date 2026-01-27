# Grid Guardian — Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-26)

**Core value:** Detect anomalies in energy distribution networks without labeled data, using physics constraints and self-play learned patterns on graph-structured grid data
**Current focus:** Phase 1 COMPLETE - Ready for Phase 2 (Self-Play Generator)

## Current Position

- **Phase:** 1 of 4 (GNN Verifier Foundation) - COMPLETE
- **Plan:** 6 of 6 (Training & Evaluation Scripts - gap closure) - COMPLETE
- **Status:** Phase 1 fully complete with all success criteria met
- **Last activity:** 2026-01-27 - Completed 01-06-PLAN.md (training/evaluation scripts + trained model)
- **Progress:** [######....] 60%

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Plans completed | 6 | 01-01, 01-02, 01-03, 01-04, 01-05, 01-06 |
| Requirements done | 1/10 | GNN-based Verifier (Phase 1 complete) |
| Phases done | 1/4 | Phase 1 complete |
| Model accuracy | 98.33% | On held-out test set (target was 85%) |

## Key Decisions

| Decision | Rationale | Plan |
|----------|-----------|------|
| Node ordering: primary -> secondary -> LV feeder | Matches node_type tensor construction, deterministic | 01-01 |
| Default features: 4-dim (3 one-hot + 1 log MPAN) | Provides baseline for GNN training | 01-01 |
| Edge deduplication via set | Avoids duplicates from multiple rows with same connection | 01-01 |
| Explicit num_nodes in Data constructor | Prevents silent failures for isolated nodes | 01-01 |
| GATv2Conv (not GATConv) | Dynamic attention, solves static attention problem | 01-02 |
| 4 attention heads per layer | Balanced expressiveness vs compute | 01-02 |
| GCNII-style initial residual | Learnable alpha prevents oversmoothing | 01-02 |
| 1D-Conv temporal encoding | Faster than LSTM, better local pattern capture | 01-02 |
| 35ms latency threshold for tests | Allows environment variance; target is 30ms | 01-03 |
| 95% test coverage target | Practical coverage excluding __repr__ methods | 01-03 |
| Pre-generate samples in __init__ | Ensures reproducibility and consistency | 01-04 |
| 10-30% nodes affected per anomaly | Realistic anomaly spread without overwhelming signal | 01-04 |
| 2-hop cascade with 0.7 decay | Simulates realistic cascade effects | 01-04 |
| BCELoss for sigmoid outputs | GATVerifier outputs [0,1] scores, BCELoss appropriate | 01-05 |
| Adam lr=1e-3, weight_decay=1e-4 | Standard for GNN training, good convergence | 01-05 |
| Early stopping patience=10 | Prevents overfitting on synthetic data | 01-05 |
| Prediction threshold=0.5 | Standard binary classification threshold | 01-05 |
| sklearn.metrics for evaluation | Reliable precision/recall/F1 computation | 01-06 |
| 2000 training samples sufficient | Achieved 98.33% accuracy, no need for more | 01-06 |
| Model converged at epoch 53 | Early stopping triggered, best at epoch 34 | 01-06 |

## Blockers

(None)

## Accumulated Context

### Technical Decisions
- COO edge format: `torch.tensor(edges).t().contiguous()` pattern
- Node ID bidirectional mapping for reverse lookup
- torch-geometric 2.7.0 as GNN foundation
- GATv2Conv with concat=True, residual=True for attention layers
- hidden_channels // heads for out_channels when concat=True
- Sigmoid output for [0,1] anomaly scores
- pytest fixtures for reusable test data
- torch.inference_mode() for test performance
- AnomalyType enum for type-safe anomaly specification
- Seeded random generator for reproducible synthetic data
- BCELoss for binary anomaly classification
- torch_geometric.loader.DataLoader for graph batching
- Checkpoint format: epoch, model_state_dict, optimizer_state_dict, metrics
- CLI scripts with argparse for all hyperparameters
- Evaluation JSON format: accuracy, precision, recall, F1, confusion matrix

### Open Questions
- ~~SSEN metadata schema for graph construction~~ RESOLVED: Works with primary_substation_id, secondary_substation_id, lv_feeder_id columns
- ~~GNN hyperparameters~~ RESOLVED: 64 hidden, 3 layers, 4 heads, latency ~30ms
- ~~Test coverage target~~ RESOLVED: 95% achieved
- ~~Synthetic training data~~ RESOLVED: SyntheticAnomalyDataset with 5 anomaly types
- ~~Training loop~~ RESOLVED: GNNTrainer with BCELoss, early stopping, checkpointing
- ~~Trained model accuracy~~ RESOLVED: 98.33% on held-out test (well above 85% target)

### Deferred Items
- Uncertainty quantification (UQ-01, UQ-02) - v2
- Elexon BMRS live integration (DATA-01, DATA-02) - v2
- FYP report and presentation (ACAD-01, ACAD-02) - v2

## Phase 1 Success Criteria Status

| Criterion | Status |
|-----------|--------|
| GridGraphBuilder transforms SSEN metadata to PyG Data | COMPLETE (01-01) |
| GATVerifier with GATv2Conv and oversmoothing prevention | COMPLETE (01-02) |
| Training/evaluation scripts with >85% accuracy model | COMPLETE (01-06, 98.33%) |
| SyntheticAnomalyDataset generates labeled training data | COMPLETE (01-04) |
| GNNTrainer with BCELoss, early stopping, checkpointing | COMPLETE (01-05) |
| Comprehensive test suite with >95% coverage | COMPLETE (01-03) |

## Recent Activity

- 2026-01-27: Project initialized, roadmap created with 4 phases
- 2026-01-27: Research synthesis completed (MEDIUM-HIGH confidence)
- 2026-01-27: Completed 01-01 Graph Construction Pipeline (3 tasks, 8 min)
- 2026-01-27: Completed 01-02 GAT Verifier Model (3 tasks, 6 min)
- 2026-01-27: Completed 01-03 Test Suite (3 tasks, 6 min) - Phase 1 complete
- 2026-01-27: Completed 01-04 Synthetic Dataset (2 tasks, 6 min) - Gap closure
- 2026-01-27: Completed 01-05 Training Pipeline (2 tasks, 5 min) - Gap closure
- 2026-01-27: Completed 01-06 Training/Evaluation Scripts (3 tasks, 11 min) - Gap closure, PHASE 1 COMPLETE

## Session Continuity

**Last session:** 2026-01-27
**Stopped at:** Completed 01-06-PLAN.md (training/evaluation scripts + trained model)
**Resume file:** None
**Next action:** Begin Phase 2 planning (Self-Play Generator)

---
*Last updated: 2026-01-27*
