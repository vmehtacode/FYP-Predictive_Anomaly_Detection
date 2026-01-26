# Codebase Concerns

**Analysis Date:** 2026-01-26

## Tech Debt

**Import path inconsistency (Canonical paths):**
- Issue: Dual import paths exist for models (`fyp.models.autoencoder` vs `fyp.anomaly.autoencoder`), creating maintenance burden and confusion
- Files: `src/fyp/runner.py:28`, `src/fyp/anomaly/` module structure
- Impact: Code references outdated paths, refactoring becomes complex, future imports may break
- Fix approach: Consolidate model locations in one canonical location (`fyp.models.*`), deprecate legacy paths, update all imports, add migration guide to CHANGELOG

**Broad exception handling without specific error types:**
- Issue: Multiple instances of bare `except Exception:` or `except Exception as e:` that swallow all errors indiscriminately
- Files: `src/fyp/baselines/forecasting.py:124,182`, `src/fyp/baselines/anomaly.py:235,250`, `src/fyp/selfplay/solver.py:221,293,520`, `src/fyp/runner.py:187,235,383`, `src/fyp/ingestion/base.py:72,152,209,223`
- Impact: Silent failures hide real bugs; makes debugging production issues difficult; prevents proper error recovery
- Fix approach: Replace broad exceptions with specific types (`ValueError`, `FileNotFoundError`, `ValidationError`), add logging before re-raising, implement graceful degradation rather than silent failures

**Magic numbers hardcoded throughout codebase:**
- Issue: Temporal constants (48, 96, 24, 30) scattered across multiple files without configuration abstraction
- Files: `src/fyp/baselines/forecasting.py:53`, `src/fyp/metrics.py:27,136`, `src/fyp/config.py:24,119`, `src/fyp/data_loader.py:106-107`, `src/fyp/models/frequency_enhanced.py:136,143,300,305`, `src/fyp/runner.py:56,106`, and many more
- Impact: Changing time window (e.g., from 24-hour to 12-hour analysis) requires modifying 15+ locations; easy to miss a location and cause inconsistency
- Fix approach: Extract all temporal constants to `src/fyp/constants.py` (e.g., `HOURS_24_IN_MINUTES = 48` at 30-min resolution), create configuration class for time window parameters, update all references

**Inconsistent null/empty handling:**
- Issue: Mix of `return None`, `return []`, `return {}` patterns; no uniform missing data policy across the codebase
- Files: `src/fyp/ingestion/lcl_ingestor.py:166`, `src/fyp/ingestion/base.py:70`, `src/fyp/ingestion/ssen_ingestor.py:216,782`, `src/fyp/ingestion/utils.py:237,264`, `src/fyp/ingestion/ukdale_ingestor.py:294`
- Impact: Caller must handle multiple return types; downstream code prone to AttributeError or TypeError; inconsistent behavior makes testing harder
- Fix approach: Define explicit return types in all methods, use Optional[T] consistently, establish convention: None for errors, empty collections for no-results, raise exceptions for truly exceptional cases

## Known Bugs

**Potential division by zero in metrics:**
- Symptoms: MAPE calculation may fail if any y_true values are zero
- Files: `src/fyp/metrics.py:19-20`
- Trigger: When energy consumption is exactly zero in the test set (edge case but possible)
- Workaround: Currently has `mask = y_true != 0` to filter, but if all values are zero, mean of empty array causes warning/error
- Real fix: Add explicit check and return 0.0 or infinity when all masked values, document behavior clearly

**Detection latency metrics may include false negatives silently:**
- Symptoms: If true anomalies are never detected within max_delay window, they're counted as max_delay penalty
- Files: `src/fyp/metrics.py:133-159`
- Trigger: Anomalies at end of time series or very subtle anomalies below threshold
- Workaround: max_delay set to 48 by default, may mask poor performance
- Real fix: Track and report separate "missed anomalies" metric, surface detection rate visibility, consider time-weighted penalties

**Rate-limited session cache not cleared between runs:**
- Symptoms: SSEN ingestor may use stale cached API responses across multiple ingestion runs
- Files: `src/fyp/ingestion/ssen_ingestor.py:63-69` (cache_dir created but force_refresh manual)
- Trigger: Running ingestion multiple times without clearing `.cache/ssen_api`; force_refresh parameter not exposed in CLI
- Workaround: Manual cache clearing required, not documented
- Real fix: Expose cache control in CLI, add automatic cache invalidation timestamp, document cache TTL expectations

**Data loss in feeder metadata iteration:**
- Symptoms: If CSV has duplicate lv_feeder_id values, only last one is stored in lookup dictionary
- Files: `src/fyp/ingestion/ssen_ingestor.py:139-145`
- Trigger: SSEN dataset with duplicate feeder IDs or different metadata versions
- Workaround: Duplicate entries overwrite previous ones silently
- Real fix: Detect and log duplicates, store as list-of-dicts instead of single dict, raise error on duplicate with different metadata

## Security Considerations

**No input validation on CSV column names:**
- Risk: Malformed CSV files with missing/renamed columns bypass validation, cause IndexError or silent incorrect parsing
- Files: `src/fyp/runner.py:80-98` (column detection by keyword matching in CSVs)
- Current mitigation: Fallback to positional indices if names not found, but very fragile
- Recommendations:
  - Add CSV schema validation before processing (use pydantic or jsonschema)
  - Raise explicit error instead of fallback behavior
  - Log warning when fallback used
  - Add tests for malformed CSV inputs

**API key stored in environment without validation:**
- Risk: API keys in .env files or environment variables could be logged/exposed
- Files: `src/fyp/ingestion/ssen_ingestor.py:48,56` (api_key parameter)
- Current mitigation: None explicit; relies on environment isolation
- Recommendations:
  - Use SecretStr from pydantic for API keys
  - Never log configuration containing api_key
  - Add explicit check that prevents credentials in logs
  - Document secure environment variable handling

**Parquet file reading with no integrity checks:**
- Risk: Corrupted Parquet files could cause silent parsing errors or type coercion failures
- Files: `src/fyp/ingestion/ukdale_ingestor.py:313`, `src/fyp/data_loader.py:34`
- Current mitigation: None; just reads and assumes validity
- Recommendations:
  - Add try/except with specific PyArrowException handling
  - Validate schema matches UNIFIED_SCHEMA on read
  - Check for NaN percentages above threshold
  - Log data quality metrics on load

## Performance Bottlenecks

**Inefficient feeder metadata lookup construction:**
- Problem: Creates dictionary via iteration over entire DataFrame on every ingestion run
- Files: `src/fyp/ingestion/ssen_ingestor.py:140-145` (iterrows loop)
- Cause: Using pandas iterrows (slow), no indexing/caching of result
- Improvement path:
  - Replace with set_index('lv_feeder_id').to_dict('index')
  - Cache constructed dictionary to disk (pickle)
  - Lazy-load cache only if modified time is recent
  - Expected speedup: 10-100x for large feeder datasets

**Full dataset loading for windowing operation:**
- Problem: `EnergyDataLoader.load_dataset()` reads entire Parquet partition into memory before windowing
- Files: `src/fyp/data_loader.py:34`, `src/fyp/runner.py:114`
- Cause: Single `pd.read_parquet()` call; no streaming or chunking
- Improvement path:
  - Implement chunk-based loading with generator pattern
  - Create windowing logic that operates per-chunk
  - Implement memory-conscious sample selection for large datasets
  - Expected improvement: Support 10x larger datasets without OOM

**Multiple passes over data for different metrics:**
- Problem: Each metric function (MAE, RMSE, MAPE, MASE) re-processes same arrays
- Files: `src/fyp/metrics.py:59-92` (forecasting_metrics function)
- Cause: No computation reuse; separate function calls
- Improvement path:
  - Single pass computation of all metrics simultaneously
  - Pre-compute residuals once, reuse for all error metrics
  - Expected improvement: 3-4x faster metric calculation

**Lazy imports scattered throughout codebase:**
- Problem: Conditional imports (try/except) at module level slow down startup and obscure dependencies
- Files: `src/fyp/runner.py:29-36`, `src/fyp/selfplay/solver.py:24-46`, `src/fyp/selfplay/__init__.py:45`
- Cause: Attempting to handle optional dependencies dynamically
- Improvement path:
  - Create explicit "features" system (core, torch_models, mlflow_tracking)
  - Validate all dependencies exist at startup
  - Fail fast with clear error messages rather than silent degradation
  - Expected improvement: Faster startup, clearer error messages for missing deps

## Fragile Areas

**Scenario-based self-play training:**
- Files: `src/fyp/selfplay/trainer.py`, `src/fyp/selfplay/solver.py`
- Why fragile: Complex interaction between proposer, solver, and verifier agents; multiple reward signals; state passed between components
- Safe modification:
  - Add comprehensive logging at agent boundaries
  - Never modify reward calculation without unit tests for each component
  - Validate scenario diversity metrics before/after changes
  - Add regression tests for scenario proposals
- Test coverage: Partial; `tests/test_selfplay.py` (635 lines) covers main paths but missing edge cases like convergence failures, divergent agent training

**Data ingestion pipeline with multiple fallbacks:**
- Files: `src/fyp/ingestion/ssen_ingestor.py:244-271`, `src/fyp/ingestion/base.py:140-210`
- Why fragile: Multiple fallback paths (metadata-only, sample data, full time-series); hard to trace actual code path; CSV column detection uses heuristics
- Safe modification:
  - Add explicit mode selection at initialization (METADATA_ONLY, SAMPLE_DATA, FULL_DATA)
  - Log actual data source and record count at start of read_raw_data()
  - Add assertions for expected record counts after ingestion
  - Create separate test for each ingestion mode
- Test coverage: `tests/test_ssen_ingestion.py` (298 lines) tests main path but limited coverage of fallback scenarios

**Metrics calculation with synthetic anomaly labels:**
- Files: `src/fyp/runner.py:409-426` (create_synthetic_anomaly_labels)
- Why fragile: Synthetic labels created ad-hoc for testing; threshold selection heuristic (mean + 2*std) doesn't match real anomalies
- Safe modification:
  - Document synthetic labeling strategy explicitly
  - Add configurable threshold parameters
  - Add validation that at least some anomalies detected (warn if zero)
  - Never use synthetic labels in production evaluation
- Test coverage: Used in `run_anomaly_baselines()` but assumptions not validated; easy to introduce bias

## Scaling Limits

**Ingestion batch processing with chunksize hardcoding:**
- Current capacity: Processes large CSV files in chunks of typically 10,000-50,000 records
- Limit: No adaptive chunking; OOM risk on systems with <8GB RAM
- Scaling path:
  - Make chunking configurable based on available RAM
  - Add memory profiling to determine safe chunk size
  - Implement streaming Parquet writer to avoid intermediate dataframes
  - Test with 1GB+ CSV files

**Self-play training episode buffering:**
- Current capacity: Keeps historical windows in memory for stability
- Limit: No limit on buffer size; potential memory leak if training runs for many episodes
- Scaling path:
  - Implement bounded FIFO queue with configurable max size
  - Add memory monitoring and warn when approaching limits
  - Implement checkpointing to disk for long training runs
  - Add garbage collection hooks

**Metrics aggregation with large result sets:**
- Current capacity: Aggregates all results in Python lists, then converts to DataFrame
- Limit: Python list memory overhead; no streaming aggregation
- Scaling path:
  - Stream results to Parquet file directly
  - Use chunked aggregation for summary statistics
  - Implement online mean/std calculation instead of storing all values

## Dependencies at Risk

**PyTorch version constraint (>=2.1.0) without upper bound:**
- Risk: Breaking changes in PyTorch 3.0+ not tested; future dependency incompatibility
- Impact: Self-play, PatchTST models will fail on new PyTorch versions
- Migration plan: Pin PyTorch <3.0 in pyproject.toml, test against torch 2.1, 2.2, 2.3 in CI, create feature branch for PyTorch 3.0 compatibility

**Polars vs Pandas dual dependency:**
- Risk: pyproject.toml requires polars ^0.20.0 but codebase uses only pandas; polars unused
- Impact: Dead dependency increases install size, potential version conflicts
- Migration plan: Remove polars from dependencies or add actual usage, consolidate to pandas-only

**TSLearn and SkTime with limited usage:**
- Risk: tslearn ^0.6.0 and sktime ^0.24.0 imported but not used in core codebase
- Impact: Large transitive dependencies add bloat, version constraints may cause solver conflicts
- Migration plan: Audit imports; if unused, remove from dependencies; if used in examples only, move to optional dev dependencies

**DVC with S3/Azure extras unconditionally:**
- Risk: Requires boto3 and azure-storage-blob even if not using cloud storage
- Impact: Adds ~100MB to install size; unnecessary for local-only workflows
- Migration plan: Make cloud storage optional, create extras group `[cloud]`, update install instructions

## Missing Critical Features

**No validation of data consistency across ingestion runs:**
- Problem: Can't detect data drift or verify that re-ingestion produces same records
- Blocks: Reproducibility testing, data quality auditing, change detection
- Fix: Add hash/checksum validation, implement data version tagging, store metadata about ingestion parameters

**No explicit model versioning or artifact tracking:**
- Problem: Models trained and saved without version info; can't track which model produced which results
- Blocks: Reproducibility, A/B testing, rollback capability
- Fix: Implement model registry, add model cards with hyperparameters, use MLflow more systematically

**No anomaly ground truth for evaluation:**
- Problem: Synthetic anomaly labels used for testing; no way to validate against real anomalies
- Blocks: Production readiness, real-world validation, model reliability assessment
- Fix: Create annotated datasets with real anomalies, add evaluation script using ground truth, document ground truth source

**No data preprocessing validation pipeline:**
- Problem: No way to verify data quality before model training
- Blocks: Debugging model failures, understanding data issues
- Fix: Add Great Expectations or similar data quality framework, implement validation checks at ingestion stage

## Test Coverage Gaps

**Ingestion error scenarios untested:**
- What's not tested: Malformed CSV files, missing columns, corrupt Parquet, API timeouts, file permission errors
- Files: `src/fyp/ingestion/` modules
- Risk: Silent data loss or incorrect parsing in production when edge cases occur
- Priority: HIGH - ingestion is critical data pipeline

**Metrics edge cases not covered:**
- What's not tested: All-zero values in MAPE, empty arrays, NaN/Inf handling, precision/recall with no positives
- Files: `src/fyp/metrics.py`
- Risk: Crashes or misleading metrics in edge case datasets
- Priority: HIGH - metrics drive all evaluations

**Data windowing boundary conditions:**
- What's not tested: Windows at dataset boundaries, windows with incomplete data, overlapping windows edge cases
- Files: `src/fyp/selfplay/utils.py` (create_sliding_windows), `src/fyp/data_loader.py` (create_forecasting_windows)
- Risk: Training on invalid windows, leakage between train/test
- Priority: HIGH - affects model validity

**Configuration overrides interaction:**
- What's not tested: CI-safe overrides interacting with user-provided configs, env var precedence
- Files: `src/fyp/config.py`, `src/fyp/runner.py:606-613`
- Risk: Unexpected config during CI runs, hidden defaults
- Priority: MEDIUM - affects reproducibility

**Memory/resource exhaustion scenarios:**
- What's not tested: Large file handling (>1GB), long training runs (>100 episodes), OOM behavior
- Files: `src/fyp/ingestion/ssen_ingestor.py`, `src/fyp/selfplay/trainer.py`
- Risk: Silent crashes or corrupted state on production data
- Priority: MEDIUM - affects scalability

---

*Concerns audit: 2026-01-26*
