# Codebase Structure

**Analysis Date:** 2026-01-26

## Directory Layout

```
FYP-Predictive_Anomaly_Detection/
├── src/fyp/                     # Main package source code
│   ├── __init__.py              # Package initialization (version 0.1.0)
│   ├── config.py                # Configuration classes and loaders
│   ├── data_loader.py           # EnergyDataLoader for Parquet datasets
│   ├── runner.py                # CLI entry point for baselines and runner
│   ├── metrics.py               # Metric calculations for forecasting/anomaly
│   │
│   ├── ingestion/               # Data ingestion pipeline
│   │   ├── __init__.py
│   │   ├── base.py              # BaseIngestor abstract class
│   │   ├── schema.py            # EnergyReading Pydantic schema
│   │   ├── utils.py             # Ingestion utilities
│   │   ├── ukdale_ingestor.py   # UK-DALE specific ingestion
│   │   ├── lcl_ingestor.py      # LCL dataset ingestion
│   │   └── ssen_ingestor.py     # SSEN dataset ingestion
│   │
│   ├── baselines/               # Baseline models (simple, interpretable)
│   │   ├── __init__.py
│   │   ├── forecasting.py       # SeasonalNaive, MovingAverage, Ridge forecasters
│   │   └── anomaly.py           # DecompositionAnomalyDetector, IsolationForest detectors
│   │
│   ├── models/                  # Advanced neural models
│   │   ├── __init__.py
│   │   ├── autoencoder.py       # AutoencoderAnomalyDetector for anomaly detection
│   │   ├── patchtst.py          # PatchTSTForecaster - attention-based time series
│   │   ├── frequency_enhanced.py # FrequencyEnhancedForecaster - FFT-enhanced model
│   │   └── ensemble.py          # Ensemble combining neural and baseline models
│   │
│   ├── anomaly/                 # Legacy anomaly detection module
│   │   ├── __init__.py
│   │   └── autoencoder.py       # Mirror of models/autoencoder.py
│   │
│   ├── selfplay/                # Self-play training orchestration
│   │   ├── __init__.py          # Exports all agents and trainer
│   │   ├── proposer.py          # ProposerAgent - generates challenging scenarios
│   │   ├── solver.py            # SolverAgent - trains neural models
│   │   ├── verifier.py          # VerifierAgent - validates forecasts against constraints
│   │   ├── trainer.py           # SelfPlayTrainer - orchestrates propose-solve-verify cycle
│   │   ├── bdh_enhancements.py  # BDH (Boundary-Driven Hypothesis) improvements
│   │   ├── utils.py             # Scenario transformations, window creation
│   │   └── curriculum.py        # Curriculum learning scheduling (if exists)
│   │
│   ├── evaluation/              # Evaluation and reporting
│   │   ├── __init__.py
│   │   └── final_poster.py      # Generate final research poster visualizations
│   │
│   └── utils/                   # General utilities
│       ├── __init__.py
│       └── random.py            # Reproducibility utilities (seeds, CI detection)
│
├── tests/                       # Test suite (pytest)
│   ├── __init__.py
│   ├── test_smoke.py            # Basic sanity checks
│   ├── test_data_loading.py     # EnergyDataLoader tests
│   ├── test_baselines.py        # Baseline model tests
│   ├── test_models.py           # Neural model tests (PatchTST, Autoencoder, Ensemble)
│   ├── test_selfplay.py         # Self-play training tests
│   ├── test_ingestion.py        # Basic ingestion tests
│   ├── test_ingestion_enhanced.py  # Enhanced ingestion feature tests
│   ├── test_ingestion_complete.py  # Full pipeline ingestion tests
│   ├── test_ssen_ingestion.py   # SSEN-specific ingestion tests
│   └── test_cli_anomaly_sample.py  # CLI runner anomaly tests
│
├── examples/                    # Demonstration and research scripts
│   ├── selfplay_quick_demo.py   # Quick self-play validation (5 episodes)
│   ├── selfplay_bdh_demo.py     # BDH enhancements demonstration
│   ├── baseline_comparison.py   # Compare baseline models across datasets
│   ├── ablation_study.py        # Test component importance
│   └── hebbian_stress_test.py   # Hebbian learning extension tests
│
├── scripts/                     # Long-running experiment scripts
│   ├── run_large_scale_experiment.py      # Full self-play training with MLflow
│   ├── run_large_scale_experiment_v3.py   # Enhanced version with curriculum fixes
│   ├── run_large_scale_experiment_FIXED.py # Stable version with bug fixes
│   ├── test_large_scale_basic.py # Quick validation
│   ├── verify_completion.py     # Check experiment status
│   └── check_pytorch.py         # PyTorch setup verification
│
├── data/                        # Data directory (DVC-tracked)
│   ├── raw/                     # Original datasets
│   │   ├── lcl/
│   │   ├── ukdale/
│   │   └── ssen/
│   ├── processed/               # Cleaned, unified Parquet format
│   │   ├── dataset=lcl/         # Parquet partitioned by dataset
│   │   ├── dataset=ukdale/
│   │   ├── dataset=ssen/
│   │   └── features/            # Derived features
│   ├── samples/                 # Small CSV samples for quick testing
│   ├── derived/                 # Generated outputs
│   │   ├── models/
│   │   │   └── selfplay/        # Checkpoints from self-play training
│   │   ├── evaluation/          # Metrics CSVs, summaries, plots
│   │   ├── reports/             # Analysis reports
│   │   └── poster/              # Final poster figures
│   └── .cache/
│
├── results/                     # MLflow experiment results (local tracking)
│   ├── large_scale_experiment/
│   │   ├── mlruns/              # MLflow run artifacts
│   │   └── visualizations/      # Generated plots
│   ├── test_v3_final/
│   ├── large_scale_experiment_v3/
│   └── [other experiment dirs]/
│
├── notebooks/                   # Jupyter notebooks for exploration
│   └── [various analysis notebooks]
│
├── docs/                        # Documentation
│   └── figures/                 # Documentation images
│
├── .planning/codebase/          # GSD planning documents
│   ├── ARCHITECTURE.md
│   ├── STRUCTURE.md
│   ├── CONVENTIONS.md (if tech focus)
│   └── [other planning docs]
│
├── .github/workflows/           # GitHub Actions CI/CD
├── .dvc/                        # Data Version Control config
├── pyproject.toml              # Poetry dependency and config
├── README.md                   # Project overview
└── CHANGELOG.md                # Version history
```

## Directory Purposes

**`src/fyp/`:**
- Purpose: Main application source code organized by domain
- Contains: All business logic for ingestion, models, training, evaluation
- Key files: `config.py` (configuration), `data_loader.py` (data access), `runner.py` (CLI)

**`src/fyp/ingestion/`:**
- Purpose: Raw data transformation to unified schema
- Contains: Dataset-specific ingestion logic, validation, quality metrics
- Key files: `base.py` (abstract ingestor), `schema.py` (Pydantic EnergyReading model), `*_ingestor.py` (implementations)

**`src/fyp/baselines/`:**
- Purpose: Simple reference models (sklearn-based, interpretable)
- Contains: SeasonalNaive, MovingAverage (forecasting), DecompositionAnomalyDetector (anomaly)
- Key files: `forecasting.py`, `anomaly.py`

**`src/fyp/models/`:**
- Purpose: Advanced neural models and ensembles
- Contains: PyTorch-based PatchTST, FrequencyEnhanced, Autoencoder, Ensemble
- Key files: `patchtst.py`, `ensemble.py`, `autoencoder.py`

**`src/fyp/selfplay/`:**
- Purpose: Curriculum learning via propose-solve-verify cycle
- Contains: ProposerAgent, SolverAgent, VerifierAgent, SelfPlayTrainer orchestrator
- Key files: `proposer.py`, `solver.py`, `verifier.py`, `trainer.py`, `utils.py`

**`src/fyp/evaluation/`:**
- Purpose: Analysis and visualization of results
- Contains: Report generation, poster creation
- Key files: `final_poster.py`

**`src/fyp/utils/`:**
- Purpose: Shared utilities for reproducibility and environment detection
- Contains: Random seed setting, CI mode detection, config overrides
- Key files: `random.py`

**`tests/`:**
- Purpose: Unit and integration test suite (pytest)
- Contains: Tests for each module, mocked external dependencies
- Key files: Tests organized by module name (`test_models.py` tests `src/fyp/models/`)

**`examples/`:**
- Purpose: Demonstration scripts and research examples
- Contains: Quick demos, ablation studies, comparisons
- Key files: `selfplay_quick_demo.py` (validation), `baseline_comparison.py`, `ablation_study.py`

**`scripts/`:**
- Purpose: Long-running experimental scripts for research
- Contains: Large-scale self-play training, experiment verification
- Key files: `run_large_scale_experiment_v3.py` (main), `test_large_scale_basic.py` (quick check)

**`data/`:**
- Purpose: Datasets and generated outputs (DVC-tracked for versioning)
- Contains: Raw inputs, processed Parquet, samples, derived outputs
- Key dirs: `raw/` (original), `processed/` (standardized), `derived/` (outputs)

**`results/`:**
- Purpose: MLflow tracking outputs and visualizations
- Contains: Per-experiment mlruns, plots, logs
- Key dirs: Named by experiment (e.g., `large_scale_experiment_v3/`)

## Key File Locations

**Entry Points:**
- `src/fyp/runner.py` - CLI runner (main entry point)
- `examples/selfplay_quick_demo.py` - Self-play demo
- `examples/baseline_comparison.py` - Model comparison
- `scripts/run_large_scale_experiment_v3.py` - Full research experiment

**Configuration:**
- `src/fyp/config.py` - ForecastingConfig, AnomalyConfig, ExperimentConfig
- `pyproject.toml` - Poetry dependencies, tool configs (ruff, mypy, pytest)
- `.env` (if exists) - Environment variables for API keys

**Core Logic:**
- `src/fyp/data_loader.py` - EnergyDataLoader class
- `src/fyp/ingestion/base.py` - BaseIngestor abstract class
- `src/fyp/baselines/forecasting.py` - BaseForecaster abstract class
- `src/fyp/models/patchtst.py` - PatchTSTForecaster neural model
- `src/fyp/selfplay/trainer.py` - SelfPlayTrainer orchestrator

**Testing:**
- `tests/test_smoke.py` - Basic sanity tests
- `tests/test_models.py` - Neural model tests
- `tests/test_selfplay.py` - Self-play cycle tests
- `tests/test_baselines.py` - Baseline model tests

## Naming Conventions

**Files:**
- Python modules: `lowercase_with_underscores.py` (snake_case)
  - Example: `data_loader.py`, `anomaly.py`, `ukdale_ingestor.py`
- Classes in file: Individual files typically contain one main class
  - Example: `data_loader.py` contains `EnergyDataLoader`, `base.py` contains `BaseIngestor`
- Test files: `test_<module_name>.py` (pytest convention)
  - Example: `test_models.py` tests `src/fyp/models/`, `test_selfplay.py` tests `src/fyp/selfplay/`

**Directories:**
- Package dirs: `lowercase_with_underscores/` (snake_case)
  - Example: `ingestion/`, `baselines/`, `models/`, `selfplay/`
- Data dirs: Descriptive lowercase
  - Example: `raw/`, `processed/`, `derived/`, `samples/`

**Classes:**
- PascalCase with descriptive names
  - Example: `EnergyDataLoader`, `BaseForecaster`, `PatchTSTForecaster`, `ProposerAgent`
- Abstract base classes: Prefix with `Base`
  - Example: `BaseIngestor`, `BaseForecaster`, `BaseAnomalyDetector`
- Agent classes in selfplay: Suffix with `Agent`
  - Example: `ProposerAgent`, `SolverAgent`, `VerifierAgent`

**Functions:**
- snake_case for regular functions
  - Example: `create_default_forecasters()`, `load_dataset()`, `apply_scenario_transformation()`
- Private functions: Prefix with underscore
  - Example: `_safe_mean()`, `_decompose_simple()`

**Constants:**
- UPPER_CASE for module-level constants
  - Example: `ADVANCED_MODELS_AVAILABLE`, `MLFLOW_AVAILABLE`

## Where to Add New Code

**New Forecasting Model:**
- Implementation: `src/fyp/models/<model_name>.py`
- Class structure: Inherit from `BaseForecaster` in `src/fyp/baselines/forecasting.py`
- Required methods: `fit(history, timestamps)` → `predict(history, steps, timestamps)` → numpy array
- Integration: Update `src/fyp/runner.py:main()` to accept model type flag
- Tests: Add test class to `tests/test_models.py`
- Example: Reference `src/fyp/models/patchtst.py` for PyTorch-based model

**New Anomaly Detector:**
- Implementation: `src/fyp/models/<detector_name>.py` or add to `anomaly.py`
- Class structure: Inherit from `BaseAnomalyDetector` in `src/fyp/baselines/anomaly.py`
- Required methods: `fit(data, timestamps)` → `predict_scores(data)` → [0, 1] array
- Integration: Update `src/fyp/runner.py:main()` for CLI flag
- Tests: Add to `tests/test_models.py`
- Example: Reference `src/fyp/models/autoencoder.py`

**New Dataset Ingestion:**
- Implementation: `src/fyp/ingestion/<dataset_name>_ingestor.py`
- Class structure: Inherit from `BaseIngestor` in `src/fyp/ingestion/base.py`
- Required methods: `read_raw_data()` yields dict → `transform_record(dict)` → EnergyReading (Pydantic)
- Schema: Define columns in `EnergyReading` dataclass in `src/fyp/ingestion/schema.py`
- Output: Parquet files to `data/processed/dataset=<name>/` via `write_parquet_batch()`
- Tests: Add to `tests/test_ingestion_enhanced.py`
- Example: Reference `src/fyp/ingestion/ukdale_ingestor.py` for full implementation

**New Experiment/Script:**
- Location: `scripts/run_<experiment_name>.py` for long-running, `examples/<name>_demo.py` for quick demos
- Structure: Import agents from `src/fyp.selfplay`, load data via `EnergyDataLoader`
- MLflow: Use `mlflow.start_run()`, log params/metrics/artifacts (see `scripts/run_large_scale_experiment_v3.py`)
- Configuration: Create ExperimentConfig in script or load from YAML
- Entry: Add to examples or scripts, document in README

**New Utility Function:**
- Shared logic: `src/fyp/utils/` (create new module if not existing)
- Module-specific: Add to existing module file (e.g., `selfplay/utils.py`)
- Example: `src/fyp/utils/random.py` for reproducibility, `src/fyp/selfplay/utils.py` for scenario transformations

**New Metric:**
- Implementation: Add function to `src/fyp/metrics.py`
- Pattern: Pure function taking numpy arrays, returning float
  - Example: `mean_absolute_error(y_true, y_pred) → float`
- Integration: Update `forecasting_metrics()` or `MetricsTracker` to include
- Tests: Add test case to `tests/test_models.py` or new test file

**New Test:**
- Location: `tests/test_<module_name>.py` or update existing
- Structure: Use pytest fixtures, mock external dependencies
- Pattern: `def test_<function_name>_<scenario>()` with assertion
- Example: See `tests/test_selfplay.py` for self-play testing pattern

## Special Directories

**`.planning/codebase/`:**
- Purpose: GSD (Get Shit Done) planning documents consumed by orchestrator
- Generated: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md
- Committed: Yes, for use across Claude instances
- Update: Re-run `/gsd:map-codebase` when architecture significantly changes

**`.dvc/`:**
- Purpose: Data Version Control configuration for tracking `data/` directory
- Generated: Automatically by DVC
- Committed: Yes (config), data files have `.dvc` pointers
- Update: Manual via `dvc add` when adding new datasets

**`results/`:**
- Purpose: MLflow experiment tracking outputs
- Generated: Automatically by MLflow during `mlflow.start_run()`
- Committed: No (typically gitignored)
- Update: Automatic during experiments

**`.github/workflows/`:**
- Purpose: GitHub Actions CI/CD pipeline definitions
- Generated: Manual YAML files
- Committed: Yes
- Update: Edit when changing test or build requirements

**`htmlcov/`:**
- Purpose: Coverage report from pytest-cov
- Generated: `pytest --cov` command
- Committed: No (gitignored)
- Update: Automatic during test runs

---

*Structure analysis: 2026-01-26*
