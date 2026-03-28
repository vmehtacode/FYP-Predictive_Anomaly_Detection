---
phase: 04-evaluation-framework
plan: 04
subsystem: evaluation
tags: [sklearn, isolation-forest, autoencoder, benchmark, baselines, eval-02]

# Dependency graph
requires:
  - phase: 04-evaluation-framework/04-01
    provides: VerifierBenchmark framework with _create_configurations() and _evaluate_decomposition()
  - phase: 04-evaluation-framework/04-03
    provides: Test suite for evaluation framework
provides:
  - IsolationForest baseline wired into VerifierBenchmark
  - AutoencoderAnomalyDetector baseline wired into VerifierBenchmark
  - 8-configuration benchmark covering all EVAL-02 required baselines
  - Figure color palette and bar categorization for new baselines
  - EVAL-02 compliance test
affects: [evaluation, report, figures]

# Tech tracking
tech-stack:
  added: [sklearn.ensemble.IsolationForest (already in deps)]
  patterns: [_evaluate_sklearn_baseline pattern for sklearn models, _evaluate_autoencoder pattern with try/except for neural models]

key-files:
  created: []
  modified:
    - src/fyp/evaluation/benchmark.py
    - scripts/generate_evaluation_figures.py
    - tests/test_evaluation.py

key-decisions:
  - "Normalize IsolationForest decision_function scores to [0,1] via negation and min-max scaling"
  - "Cap autoencoder window_size to min(48, num_nodes) for compatibility with small graphs"
  - "Re-seed RNG before autoencoder training for benchmark reproducibility"

patterns-established:
  - "_evaluate_sklearn_baseline: reshape to (-1,1), negate decision_function, normalize, threshold at 0.5"
  - "_evaluate_autoencoder: try/except wrapper returning error dict on failure, seed-before-train for reproducibility"

requirements-completed: [EVAL-01, EVAL-02, EVAL-03]

# Metrics
duration: 3min
completed: 2026-03-28
---

# Phase 04 Plan 04: Gap Closure - Baseline Wiring Summary

**IsolationForest and AutoencoderAnomalyDetector baselines wired into VerifierBenchmark, completing EVAL-02 with 8 configurations and 15 passing tests**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-28T12:13:36Z
- **Completed:** 2026-03-28T12:16:31Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Wired IsolationForest (sklearn) and AutoencoderAnomalyDetector into VerifierBenchmark as named configurations with dedicated evaluation methods
- Benchmark now produces 8 configurations with standard metrics (accuracy, precision, recall, F1, latency) for all
- EVAL-02 fully satisfied: GNN Verifier compared against IsolationForest, AutoencoderAnomalyDetector, AND DecompositionAnomalyDetector
- All 15 tests pass including new EVAL-02 compliance test

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire IsolationForest and AutoencoderAnomalyDetector into VerifierBenchmark** - `d6b082c` (feat)
2. **Task 2: Update figure colors, bar categorization, and tests for new baselines** - `4516934` (feat)

## Files Created/Modified
- `src/fyp/evaluation/benchmark.py` - Added IsolationForest and AutoencoderAnomalyDetector imports, configurations, and evaluation methods (_evaluate_sklearn_baseline, _evaluate_autoencoder)
- `scripts/generate_evaluation_figures.py` - Added color entries for isolation_forest (#ff7f0e) and autoencoder (#aec7e8), updated _bar_type() baseline category
- `tests/test_evaluation.py` - Updated assertions from >= 5 to >= 8 configs, added test_benchmark_includes_all_eval02_baselines

## Decisions Made
- Normalized IsolationForest scores by negating decision_function and applying min-max scaling to [0,1] range, consistent with the decomposition baseline pattern
- Capped autoencoder window_size to min(48, num_nodes) to handle small graph sizes without errors
- Re-seed np.random and torch.manual_seed before autoencoder training to ensure benchmark reproducibility across runs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed autoencoder reproducibility in benchmark**
- **Found during:** Task 2 (test verification)
- **Issue:** test_benchmark_reproducible_with_seed failed because autoencoder training is non-deterministic without explicit re-seeding before each training run
- **Fix:** Added np.random.seed(self.seed) and torch.manual_seed(self.seed) at the start of _evaluate_autoencoder()
- **Files modified:** src/fyp/evaluation/benchmark.py
- **Verification:** test_benchmark_reproducible_with_seed now passes
- **Committed in:** 4516934 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Auto-fix necessary for test correctness. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all baselines fully wired with real evaluation logic.

## Next Phase Readiness
- Evaluation framework complete with all EVAL-02 baselines
- Ready for report generation and final presentation
- All 15 evaluation tests passing

## Self-Check: PASSED

All files exist, all commits verified.

---
*Phase: 04-evaluation-framework*
*Completed: 2026-03-28*
