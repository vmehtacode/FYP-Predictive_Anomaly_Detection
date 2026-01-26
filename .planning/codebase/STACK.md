# Technology Stack

**Analysis Date:** 2026-01-26

## Languages

**Primary:**
- Python 3.11+ - Core application language (specified as `>=3.11,<3.13` in `pyproject.toml`)

## Runtime

**Environment:**
- Python 3.11 - Minimum supported version
- CPython (default)

**Package Manager:**
- Poetry - Dependency management (see `pyproject.toml` lines 2-3)
- Lockfile: Present (`poetry.lock`)

## Frameworks

**Core:**
- PyTorch 2.1.0+ - Deep learning framework for neural network models (`torch`)
- Transformers 4.57.1+ - Hugging Face transformers library for pre-trained models

**Machine Learning & Statistics:**
- scikit-learn 1.3.0+ - Classical ML algorithms (Ridge regression, preprocessing)
- statsmodels 0.14.0+ - Statistical modeling and testing
- tslearn 0.6.0+ - Time series learning algorithms
- sktime 0.24.0+ - Time series analysis and forecasting utilities

**Data Processing:**
- Polars 0.20.0+ - Fast columnar data processing (modern pandas alternative)
- Pandas 2.1.0+ - Data manipulation and analysis (legacy support)
- PyArrow 14.0.0+ - Apache Arrow for data serialization
- NumPy 1.24.0+ - Numerical computing
- H5PY 3.10.0+ - HDF5 file format support
- Tables 3.9.0+ - PyTables for hierarchical data storage

**Visualization:**
- Matplotlib 3.7.0+ - Static plotting
- Seaborn 0.12.0+ - Statistical data visualization
- Plotly 5.17.0+ - Interactive web-based visualization

**Configuration & Utilities:**
- Pydantic 2.5.0+ - Data validation and settings management (used in `src/fyp/config.py`)
- Click 8.1.0+ - CLI framework for command-line interfaces
- python-dotenv 1.0.0+ - Environment variable management
- tqdm 4.66.0+ - Progress bars
- rich 13.7.0+ - Rich terminal output and formatting
- psutil 5.9.0+ - System and process utilities
- loguru 0.7.3+ - Enhanced logging (not actively used, available for advanced logging)
- YAML 3.x (via `pyyaml` dependency) - Configuration file parsing

## Experiment & Data Management

**Core:**
- MLflow 2.8.0+ - Experiment tracking and model registry (`src/fyp/runner.py` lines 40, 617-651)
- DVC 3.30.0+ - Data version control with S3 and Azure storage support (`dvc.yaml`, `dvc.lock`)

## Testing & Quality

**Testing Framework:**
- pytest 7.4.0+ - Test runner and framework
- pytest-cov 4.1.0+ - Coverage reporting for pytest
- pytest-mock 3.12.0+ - Mocking utilities
- pytest-xdist 3.4.0+ - Parallel test execution

**Code Quality:**
- ruff 0.1.0+ - Fast Python linter (see `pyproject.toml` lines 111-131)
- black 23.10.0+ - Code formatter with 88 character line length
- mypy 1.7.0+ - Static type checker with strict configuration (`pyproject.toml` lines 169-189)
- pre-commit 3.5.0+ - Git pre-commit hooks (`.pre-commit-config.yaml`)

**Documentation:**
- MkDocs 1.5.0+ - Static documentation generator
- mkdocs-material 9.4.0+ - Material Design theme for MkDocs
- mkdocstrings 0.24.0+ - Automatic API documentation from docstrings

## Development Environment

**Jupyter:**
- Jupyter 1.0.0+ - Interactive notebook environment
- IPython Kernel 6.26.0+ - IPython kernel for Jupyter
- nbstripout 0.6.0+ - Remove output from Jupyter notebooks

## Key Dependencies - Why They Matter

**PyTorch + Transformers Stack:**
- Powers the neural network models: `PatchTSTForecaster` (`src/fyp/models/patchtst.py`), `AutoencoderAnomalyDetector` (`src/fyp/models/autoencoder.py`)
- Enables GPU acceleration for training on CUDA devices
- Used for attention-based time series forecasting

**Polars + PyArrow:**
- Efficient columnar data processing for ingestion pipeline
- Used in `src/fyp/ingestion/base.py` for parquet file writing
- Modern alternative to Pandas with better performance on large datasets

**scikit-learn:**
- Provides baseline models in `src/fyp/baselines/forecasting.py` (Ridge regression) and `src/fyp/baselines/anomaly.py`
- StandardScaler for feature normalization across all models

**Pydantic:**
- Configuration management via `ExperimentConfig`, `ForecastingConfig`, `AnomalyConfig` in `src/fyp/config.py`
- Type-safe data validation for ingested energy readings in `src/fyp/ingestion/schema.py`

**MLflow:**
- Experiment tracking and metric logging in `src/fyp/runner.py` lines 617-651
- Run parameter and artifact logging
- Default experiment: "energy_forecasting" (configurable via `mlflow_experiment` in config)

**DVC:**
- Data versioning for raw and processed datasets
- Pipeline orchestration via `dvc.yaml` with stages: `ingest_lcl`, `ingest_ukdale`, `ingest_ssen`, `train_baselines`, `train_custom`, `train_selfplay`
- S3 and Azure remote storage support configured in `.dvc/config`

**Click:**
- CLI implementation in `src/fyp/ingestion/cli.py` for data ingestion with dataset selection and API key options

## Configuration

**Environment:**
- Configuration via Pydantic models: `ExperimentConfig`, `ForecastingConfig`, `AnomalyConfig` (`src/fyp/config.py`)
- YAML file support for configuration serialization
- Environment variable overrides via `get_config_from_env()` function (lines 143-159)
- CI detection: Activates sample-based fast configs when `CI` or `GITHUB_ACTIONS` env vars are set

**Key Environment Variables:**
- `CI` / `GITHUB_ACTIONS` - Triggers sample mode for CI/CD pipelines
- `USE_SAMPLES` - Override to use sample data (true/false)
- `DEVICE` - Hardware device for models (`cpu` or `cuda`)
- `SSEN_CKAN_URL` - SSEN API endpoint (default: `https://data.ssen.co.uk`)
- `MLFLOW_TRACKING_URI` - MLflow server URI (if using remote tracking)

**Build Configuration:**
- `pyproject.toml` - Poetry dependency manifest with 50+ dependencies
- `.pre-commit-config.yaml` - Pre-commit hooks for ruff linting and formatting
- `params.yaml` - DVC parameter definitions (minimal usage, mostly in `dvc.yaml`)
- `dvc.yaml` - DVC pipeline definition with 8 stages
- `dvc.lock` - Locked pipeline outputs

## Platform Requirements

**Development:**
- Python 3.11 or higher
- Poetry for dependency management
- Git (for version control)
- Optional: CUDA 11.8+ for GPU acceleration with PyTorch
- Optional: DVC remote storage (S3 or Azure) for data versioning

**Production:**
- Python 3.11+ runtime
- All dependencies from `poetry.lock`
- Optional: GPU hardware if using GPU acceleration
- Optional: MLflow server instance for centralized experiment tracking
- Optional: DVC remote storage for production data pipelines

---

*Stack analysis: 2026-01-26*
