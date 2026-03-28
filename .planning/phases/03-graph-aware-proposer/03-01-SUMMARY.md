---
phase: 03-graph-aware-proposer
plan: 01
subsystem: selfplay
tags: [graph-aware, proposer, cascade, topology, pyg, torch-geometric, bfs]

# Dependency graph
requires:
  - phase: 01-gnn-verifier
    provides: GridGraphBuilder (node_type constants, COO edge format)
  - phase: 02-hybrid-ensemble
    provides: CascadeLogicLayer._build_adjacency pattern, cascade decay convention
provides:
  - Graph-aware ProposerAgent with topology-based seed selection
  - 2-hop cascade propagation with 0.7 decay through grid neighbors
  - ScenarioProposal.apply_to_graph_timeseries() for per-node magnitude scaling
  - ScenarioProposal.metadata enrichment with graph_aware, seed_nodes, affected_nodes
affects: [03-02, selfplay-trainer, graph-training-loop]

# Tech tracking
tech-stack:
  added: []
  patterns: [BFS cascade propagation, per-node magnitude blending, graph_data optional parameter]

key-files:
  created:
    - tests/test_graph_proposer.py
  modified:
    - src/fyp/selfplay/proposer.py

key-decisions:
  - "1-3 seed nodes per scenario (not 10-30%); cascade expands from few seeds"
  - "LV feeder (node_type=2) preference for COLD_SNAP, OUTAGE, EV_SPIKE seed selection"
  - "30% graph cap on affected nodes with seed preservation during subsampling"
  - "Per-node blending formula: baseline + magnitude * (transform - baseline)"

patterns-established:
  - "graph_data optional parameter pattern for backward-compatible graph enrichment"
  - "BFS cascade with decay_factor per hop for anomaly propagation"
  - "apply_to_graph_timeseries per-node scaling for multi-node time series"

requirements-completed: [SELF-01, SELF-02]

# Metrics
duration: 5min
completed: 2026-03-28
---

# Phase 03 Plan 01: Graph-Aware Proposer Summary

**Graph-aware ProposerAgent with topology-based seed selection, 2-hop BFS cascade propagation (0.7 decay), and per-node time-series application via apply_to_graph_timeseries**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-28T14:05:05Z
- **Completed:** 2026-03-28T14:10:34Z
- **Tasks:** 3 (TDD: RED/GREEN for each)
- **Files modified:** 2

## Accomplishments
- ProposerAgent.propose_scenario() accepts optional graph_data parameter with full backward compatibility
- Seed nodes selected from LV feeders (node_type=2) for COLD_SNAP/OUTAGE/EV_SPIKE, all nodes for PEAK_SHIFT/MISSING_DATA
- Cascade propagates through 2-hop BFS with 0.7 decay per hop (seeds=1.0, hop1=0.7, hop2=0.49)
- Affected nodes capped at 30% of graph with seed preservation during subsampling
- apply_to_graph_timeseries() applies per-node magnitude scaling to 2-D time-series arrays
- 24 tests covering all graph-aware behavior, backward compatibility, and per-node application

## Task Commits

Each task was committed atomically (TDD flow):

1. **Task 1: Test scaffold for graph-aware proposer** - `dcb337a` (test)
2. **Task 2: Implement graph-aware proposer methods** - `37cff8a` (feat)
3. **Task 3: Implement apply_to_graph_timeseries** - `e7370d6` (feat)

## Files Created/Modified
- `tests/test_graph_proposer.py` - 24 tests across 9 test classes covering graph-aware proposer, seed selection, cascade propagation, decay, backward compat, diversity, cap, and per-node application
- `src/fyp/selfplay/proposer.py` - Extended with graph_data parameter, _enrich_with_graph_topology, _select_seed_nodes, _build_adjacency, _propagate_through_neighbors, and apply_to_graph_timeseries

## Decisions Made
- **1-3 seed nodes per scenario:** Cascade expands from few seeds rather than affecting 10-30% directly. This produces realistic propagation patterns.
- **LV feeder preference for cascading types:** COLD_SNAP, OUTAGE, EV_SPIKE affect end-consumer nodes first, matching physical grid behavior.
- **30% cap with seed preservation:** When cascade exceeds cap, seeds are always kept and non-seed nodes are randomly subsampled. Ensures cascade structure is preserved.
- **Per-node blending formula:** `baseline + magnitude * (transform - baseline)` provides smooth interpolation between untouched and fully transformed time series.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None - all methods are fully implemented with no placeholder data.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Graph-aware proposer ready for integration with SelfPlayTrainer in Plan 03-02
- apply_to_graph_timeseries() provides the per-node application method that SelfPlayTrainer will use when graph_data is available
- All 28 existing selfplay tests continue to pass (backward compatibility confirmed)

## Self-Check: PASSED

- tests/test_graph_proposer.py: FOUND
- src/fyp/selfplay/proposer.py: FOUND
- 03-01-SUMMARY.md: FOUND
- Commit dcb337a: FOUND
- Commit 37cff8a: FOUND
- Commit e7370d6: FOUND

---
*Phase: 03-graph-aware-proposer*
*Completed: 2026-03-28*
