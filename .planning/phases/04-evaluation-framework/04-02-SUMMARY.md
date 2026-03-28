---
phase: 04-evaluation-framework
plan: 02
subsystem: evaluation
tags: [ablation, weight-sweep, early-exit, physics-compliance, wilcoxon, statistical-significance, ensemble]

# Dependency graph
requires:
  - phase: 04-evaluation-framework
    provides: VerifierBenchmark class, test data generation, evaluate_configuration() protocol
  - phase: 02-hybrid-integration
    provides: HybridVerifierAgent, PhysicsConstraintLayer, CascadeLogicLayer, EnsembleWeightsConfig
provides:
  - AblationStudy class for component isolation, weight sweep, early-exit analysis
  - CLI script (run_ablation_study.py) for running ablation study and exporting results JSON
  - Physics compliance rate measurement
  - Statistical significance testing (Wilcoxon with bootstrap fallback)
affects: [04-03-visualization]

# Tech tracking
tech-stack:
  added: [scipy.stats.wilcoxon]
  patterns: [component-isolation ablation, grid-search weight sweep, threshold-sweep trade-off analysis, paired non-parametric significance test]

key-files:
  created:
    - src/fyp/evaluation/ablation.py
    - scripts/run_ablation_study.py
  modified:
    - src/fyp/evaluation/__init__.py

key-decisions:
  - "Wilcoxon signed-rank test for statistical significance (non-parametric, paired, no normality assumption)"
  - "Bootstrap CI fallback when Wilcoxon encounters insufficient non-zero differences"
  - "Physics compliance threshold 0.3 — physics layer mean score above this counts as physics-compliant detection"
  - "Per-sample correctness vectors (not aggregate F1) used for significance tests to preserve sample pairing"
  - "Lazy test data caching via properties to avoid redundant generation across analyses"

patterns-established:
  - "Ablation pattern: isolate singles, pairs, full ensemble, and baseline for systematic contribution analysis"
  - "CLI pattern: argparse with --samples/--seed/--output/--quick flags matching evaluate_verifiers.py"
  - "Weight sweep: grid search with remainder constraint (w_cascade = 1 - w_physics - w_gnn)"

requirements-completed: [EVAL-01, EVAL-03]

# Metrics
duration: 4min
completed: 2026-03-28
---

# Phase 04 Plan 02: Ablation Study Summary

**Systematic ablation framework with component isolation, weight sweep, early-exit trade-off, per-anomaly-type breakdown, physics compliance measurement, and Wilcoxon significance testing**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-28T11:38:27Z
- **Completed:** 2026-03-28T11:42:03Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- AblationStudy class orchestrating 6 analysis types: component isolation (8 configs), weight sweep (grid search), early-exit threshold sweep, per-anomaly-type breakdown, physics compliance rate, and statistical significance
- CLI script with formatted console tables showing component contributions, optimal weights, early-exit trade-off curves, and significance test results
- Wilcoxon signed-rank test for non-parametric paired significance testing with bootstrap CI fallback
- Per-anomaly-type analysis identifying which anomaly categories (SPIKE, DROPOUT, CASCADE, RAMP_VIOLATION) benefit most from each component

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AblationStudy class with component isolation and weight sweep** - `f12c50f` (feat)
2. **Task 2: Create ablation CLI script and run study** - `bdf8d72` (feat)

## Files Created/Modified
- `src/fyp/evaluation/ablation.py` - AblationStudy class with 7 analysis methods: run_component_isolation, run_weight_sweep, run_early_exit_sweep, run_per_anomaly_type_analysis, compute_physics_compliance_rate, compute_statistical_significance, run_full_ablation
- `scripts/run_ablation_study.py` - CLI script with argparse, 6 formatted output tables, JSON export
- `src/fyp/evaluation/__init__.py` - Updated exports to include AblationStudy

## Decisions Made
- Used Wilcoxon signed-rank test (non-parametric, paired) for statistical significance because sample distributions may not be normal
- Bootstrap CI (1000 iterations, seed=42) as fallback when Wilcoxon encounters insufficient non-zero differences
- Physics compliance defined as mean physics_score > 0.3 for detected anomalies, capturing meaningful physics layer contribution
- Per-sample correctness vectors stored during component isolation to enable proper paired significance testing
- Lazy property-based caching for test_data and graph_data to avoid redundant generation across analyses

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all data sources and analysis logic are fully wired.

## Next Phase Readiness
- AblationStudy and results JSON ready for Plan 04-03 (visualization)
- Results JSON contains all sections: component_isolation, weight_sweep, early_exit_sweep, per_anomaly_type, physics_compliance, significance_tests
- CLI produces formatted tables suitable for thesis inclusion

## Self-Check: PASSED

- All 3 created/modified files verified present on disk
- Both task commits (f12c50f, bdf8d72) verified in git log
- All 4 plan verifications passed (import, CLI run, JSON sections, physics compliance)

---
*Phase: 04-evaluation-framework*
*Completed: 2026-03-28*
