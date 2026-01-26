# Architecture

**Analysis Date:** 2026-01-26

## Pattern Overview

**Overall:** Layered modular architecture with three core domains and a novel self-play training system

**Key Characteristics:**
- **Domain separation**: Data ingestion, baseline models, advanced neural models, and self-play training are cleanly separated
- **Self-play orchestration**: Propose-Solve-Verify cycle for curriculum-based learning in anomaly detection and forecasting
- **Pluggable models**: Abstract base classes for forecasters and anomaly detectors allow swapping implementations
- **Configuration-driven**: Pydantic-based configuration system manages all model hyperparameters and experiment settings
- **Data pipeline**: Unified Parquet schema across three energy datasets (LCL, UK-DALE, SSEN) with time-series windowing

## Layers

**Data Ingestion Layer:**
- Purpose: Read raw energy consumption data from multiple sources and transform to unified Parquet format
- Location: `src/fyp/ingestion/`
- Contains: Dataset-specific ingestors (LCL, UK-DALE, SSEN), schema validation, quality metrics
- Depends on: Pandas, PyArrow, Pydantic for validation
- Used by: Data loader, evaluation scripts

**Data Loading & Preparation Layer:**
- Purpose: Load processed data, create train/test splits, generate forecasting windows
- Location: `src/fyp/data_loader.py`
- Contains: EnergyDataLoader class handling dataset loading, filtering, train/test splitting, window creation
- Depends on: Parquet files, Pandas
- Used by: All models, runner, examples

**Baseline Models Layer:**
- Purpose: Simple reference implementations for forecasting and anomaly detection
- Location: `src/fyp/baselines/`
- Contains: SeasonalNaive, MovingAverage, DecompositionAnomalyDetector (sklearn-based)
- Depends on: NumPy, Scikit-learn, Pandas
- Used by: Runner, evaluation, ensemble components

**Advanced Neural Models Layer:**
- Purpose: Deep learning implementations for enhanced forecasting and anomaly detection
- Location: `src/fyp/models/`
- Contains: PatchTST (attention-based), FrequencyEnhancedForecaster, Autoencoder, Ensemble
- Depends on: PyTorch, custom model implementations
- Used by: Self-play solver, runner, examples

**Self-Play Training Layer:**
- Purpose: Orchestrate propose-solve-verify cycle for curriculum-based learning
- Location: `src/fyp/selfplay/`
- Contains: ProposerAgent (scenario generation), SolverAgent (model training), VerifierAgent (constraint validation), SelfPlayTrainer (orchestrator)
- Depends on: All previous layers
- Used by: Examples, research scripts

**Metrics & Evaluation Layer:**
- Purpose: Calculate performance metrics for forecasting and anomaly detection
- Location: `src/fyp/metrics.py`, `src/fyp/evaluation/`
- Contains: MAE, RMSE, MAPE, MASE, pinball loss, coverage score, MetricsTracker
- Depends on: NumPy, Pandas
- Used by: All model evaluation, runner

**Configuration Layer:**
- Purpose: Centralized configuration management for experiments
- Location: `src/fyp/config.py`
- Contains: ForecastingConfig, AnomalyConfig, ExperimentConfig, config loaders
- Depends on: Pydantic, YAML
- Used by: Runner, examples, trainer

**CLI & Execution Layer:**
- Purpose: Entry point for running baselines and experiments
- Location: `src/fyp/runner.py`
- Contains: Argument parsing, dataset loading, model execution, results saving, plotting
- Depends on: All layers above
- Used by: Command line interface

## Data Flow

**Forecasting Pipeline:**

1. Load data via `EnergyDataLoader.load_dataset()` → Parquet files with schema `(ts_utc, entity_id, energy_kwh, dataset, interval_mins, source, extras)`
2. Create forecasting windows via `create_forecasting_windows()` → `{history_energy, target_energy, history_timestamps, entity_id}`
3. Select forecaster (baseline or neural) from `create_default_forecasters()` or `PatchTSTForecaster`
4. Call `forecaster.predict(history, steps)` → numpy array of predictions
5. Calculate metrics via `forecasting_metrics(y_true, y_pred, y_train)` → `{mae, rmse, mape, mase, pinball_*}`
6. Accumulate results in `MetricsTracker` → summarize via `get_forecasting_summary()`
7. Save to CSV and JSON, generate plots

**Anomaly Detection Pipeline:**

1. Load data via `EnergyDataLoader.load_dataset()`
2. Split by entity into train/test → 70/30 time-based split
3. Select detector (baseline or autoencoder) from `create_default_detectors()`
4. Fit on training data via `detector.fit(train_data)`
5. Generate anomaly scores via `detector.predict_scores(test_data)` → array of [0, 1] scores
6. Create synthetic labels (upper 2σ as anomalies)
7. Calculate metrics (precision, recall, F1) via `MetricsTracker`
8. Save results and plots

**Self-Play Training Cycle:**

1. **Propose**: `ProposerAgent.propose_scenario()` generates `ScenarioProposal` (type: EV_SPIKE, COLD_SNAP, PEAK_SHIFT, OUTAGE, MISSING_DATA)
   - Scenarios have physical constraints checked via difficulty estimation
   - Applied to baseline windows via `apply_scenario_transformation()`

2. **Solve**: `SolverAgent.train_on_batch()` trains neural model (PatchTST or FrequencyEnhanced)
   - On windows with and without scenarios
   - Computes quantile regression loss with pinball loss
   - Returns training metrics (loss, MAE, MAPE)

3. **Verify**: `VerifierAgent.verify_forecast()` checks:
   - Constraint satisfaction (physics validity)
   - Forecast quality (MAE/MAPE thresholds)
   - Assigns reward signal: `verification_reward = constraint_satisfaction * (1 - mape_normalized)`

4. **Feedback**: `SelfPlayTrainer.train()` orchestrates:
   - Episode loop with `num_episodes`
   - Curriculum advancement based on success rates
   - Checkpoint saving every `checkpoint_every` episodes
   - Validation every `val_every` episodes
   - Returns `metrics_history` with per-episode summaries

**State Management:**

- **Trainer state**: `episode_count`, `best_val_loss`, `metrics_history`, `scenario_success_rates`, `solver_performance_buffer`
- **Model state**: Solver maintains internal neural network state (weights), checkpointed to `data/derived/models/selfplay/`
- **Data state**: Windows fed to solver are stateless; metrics aggregated in `MetricsTracker` and `SelfPlayTrainer`

## Key Abstractions

**BaseForecaster:**
- Purpose: Abstract interface for all forecasting models
- Examples: `src/fyp/baselines/forecasting.py` (SeasonalNaive, MovingAverage), `src/fyp/models/patchtst.py` (PatchTSTForecaster)
- Pattern: Implement `fit(history, timestamps)` and `predict(history, steps, timestamps)` → numpy array

**BaseAnomalyDetector:**
- Purpose: Abstract interface for all anomaly detectors
- Examples: `src/fyp/baselines/anomaly.py` (DecompositionAnomalyDetector), `src/fyp/models/autoencoder.py` (AutoencoderAnomalyDetector)
- Pattern: Implement `fit(data, timestamps)` and `predict_scores(data)` → [0, 1] array

**BaseIngestor:**
- Purpose: Abstract interface for dataset-specific ingestion
- Examples: `src/fyp/ingestion/ukdale_ingestor.py`, `src/fyp/ingestion/lcl_ingestor.py`, `src/fyp/ingestion/ssen_ingestor.py`
- Pattern: Implement `read_raw_data()` and `transform_record(record)` → EnergyReading (Pydantic model)

**ScenarioProposal:**
- Purpose: Data class representing a proposed scenario for training
- Location: `src/fyp/selfplay/proposer.py`
- Pattern: Contains scenario_type, magnitude, duration, difficulty_score, apply to baseline via `apply_to_timeseries(baseline)`

**MetricsTracker:**
- Purpose: Accumulate results across windows and compute summary statistics
- Location: `src/fyp/metrics.py` (extended in runner)
- Pattern: `add_forecasting_result()` / `add_anomaly_result()` → `get_forecasting_summary()` / `get_anomaly_summary()`

**ExperimentConfig:**
- Purpose: Typed configuration for all experiment parameters
- Location: `src/fyp/config.py`
- Pattern: Pydantic BaseModel with nested ForecastingConfig and AnomalyConfig; load from YAML or create defaults

## Entry Points

**CLI Runner:**
- Location: `src/fyp/runner.py` (main function)
- Triggers: `python -m fyp.runner forecast|anomaly --dataset lcl|ukdale|ssen [options]`
- Responsibilities: Parse arguments, load config, initialize models, run pipeline, save results to CSV/JSON/plots

**Examples:**
- `examples/selfplay_quick_demo.py`: Validate self-play cycle on synthetic data
- `examples/baseline_comparison.py`: Compare baseline models across datasets
- `examples/ablation_study.py`: Test model component importance
- `examples/hebbian_stress_test.py`: Test Hebbian learning extensions
- `examples/selfplay_bdh_demo.py`: Test BDH (Boundary-Driven Hypothesis) enhancements

**Scripts:**
- `scripts/run_large_scale_experiment.py`: Run self-play on full datasets with MLflow logging
- `scripts/test_large_scale_basic.py`: Quick validation of large-scale setup
- `scripts/run_large_scale_experiment_v3.py`: Enhanced version with curriculum fixes

**Tests:**
- Location: `tests/`
- Entry: `pytest tests/` runs all tests
- Test files: `test_smoke.py` (basic), `test_data_loading.py`, `test_baselines.py`, `test_models.py`, `test_selfplay.py`, `test_ingestion*.py`

## Error Handling

**Strategy:** Graceful degradation with fallback models

**Patterns:**

1. **Missing dependencies**: Lazy imports with try/except, warning logged, fallback to baselines
   - Example: `src/fyp/runner.py` lines 29-36 attempts PatchTST import, falls back to SeasonalNaive

2. **Data validation**: Pydantic ValidationError caught in ingestion, records skipped with error count
   - Example: `src/fyp/ingestion/base.py` lines 55-73 validates EnergyReading, logs errors

3. **Training failures**: MLflow run marked FAILED, exception logged, runner exits with status 1
   - Example: `src/fyp/runner.py` lines 657-661 catches top-level exception, marks MLflow run FAILED

4. **Model inference failures**: Exception caught per-window, warning logged, metric skipped
   - Example: `src/fyp/runner.py` lines 235-236 catches forecaster exception, logs warning

5. **Data quality issues**: Empty datasets logged as warnings, returns empty result dict
   - Example: `src/fyp/runner.py` lines 116-118 checks if df.empty before processing

## Cross-Cutting Concerns

**Logging:**
- Framework: Python logging module with basicConfig setup in most modules
- Pattern: Module-level logger `logger = logging.getLogger(__name__)` with INFO/WARNING/ERROR levels
- Files: Logs to console via basicConfig in `runner.py` and ingestion base classes

**Validation:**
- Approach: Pydantic models for schema (EnergyReading in `src/fyp/ingestion/schema.py`), business logic validation in ingestors
- Example: `src/fyp/ingestion/base.py:validate_reading()` checks timestamp alignment, future dates

**Authentication:**
- Approach: Not applicable; no external auth required. Ingestion reads local files or SSEN API via environment variables
- Config: Optional env vars for API keys loaded via `python-dotenv`

**Reproducibility:**
- Approach: Seed setting via `set_global_seeds()` in `src/fyp/utils/random.py`
- Usage: Called in runner when `use_samples=True` or CI mode detected
- Files: `src/fyp/utils/random.py` exports `set_global_seeds(seed)`, `should_use_ci_mode()`, `get_ci_safe_config_overrides()`

---

*Architecture analysis: 2026-01-26*
