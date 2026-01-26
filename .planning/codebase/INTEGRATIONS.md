# External Integrations

**Analysis Date:** 2026-01-26

## APIs & External Services

**SSEN Energy Network Data:**
- Service: SSEN Low Voltage Feeder Dataset (CKAN API)
- What it's used for: Feeder metadata including network hierarchy and customer counts
- SDK/Client: `requests` library with `RateLimitedSession` wrapper (`src/fyp/ingestion/utils.py`)
- Endpoint: `https://data.ssen.co.uk` (configurable via `SSEN_CKAN_URL` env var)
- Auth: Optional API key via `--api-key` flag in CLI (`src/fyp/ingestion/cli.py` line 75)
- Rate Limiting: 1.0 second delay between requests (configurable, see `src/fyp/ingestion/ssen_ingestor.py` line 49)
- Caching: Local HTTP cache in `.cache/ssen_api` directory
- Implementation: `SSENIngestor` class in `src/fyp/ingestion/ssen_ingestor.py` lines 25-70
  - Fetches package metadata via `/api/3/action/package_show`
  - Queries datastore via `/api/3/action/datastore_search`
  - Implements exponential backoff with 3 max retries
- Data Format: CSV with 11 columns (dataset_id, postcode, primary/secondary substations, HV/LV feeders, customer counts)

## Data Ingestion Sources

**London Energy Meter Data (LCL):**
- Source: CSV files in `data/samples/lcl_sample.csv` or `data/processed/lcl_data/`
- Ingestor: `LCLIngestor` class in `src/fyp/ingestion/lcl_ingestor.py`
- Schema: Unified time series format via `EnergyReading` pydantic model (`src/fyp/ingestion/schema.py`)
- Processing: Timezone-aware datetime handling, missing value detection
- Output: Parquet files (via PyArrow) for efficient columnar storage

**UK-DALE Dataset:**
- Source: CSV files in `data/samples/ukdale_sample.csv` or `data/processed/ukdale_data/`
- Ingestor: `UKDALEIngestor` class in `src/fyp/ingestion/ukdale_ingestor.py`
- Features: Optional 30-minute downsampling (configured via `--downsample-30min` flag)
- Output: Parquet files with standardized schema

**SSEN Feeder Data:**
- Source: CSV lookup files with network topology
- Ingestor: `SSENIngestor` class in `src/fyp/ingestion/ssen_ingestor.py`
- Purpose: Network hierarchy and customer distribution validation
- API Fallback: Can fetch from SSEN CKAN API if needed

## Data Storage

**Databases:**
- Not used - Project uses file-based storage only

**File Storage:**
- Local filesystem organization:
  - Raw data: `data/raw/` (source datasets)
  - Processed: `data/processed/` (standardized parquet files)
  - Derived: `data/derived/` (model outputs, metrics, reports)
  - Samples: `data/samples/` (small CSV subsets for CI/testing)
  - Cache: `data/.cache/` (HTTP response cache from SSEN API)

**Storage Format:**
- Primary: Parquet files (via PyArrow) for numerical data
- Secondary: CSV for samples and lookup tables
- HDF5/PyTables support available (h5py, tables packages)
- JSON for configuration, metrics, and summaries

**DVC Data Versioning:**
- Remote storage support: S3 and Azure (configured via DVC)
- `.dvc/config` minimal setup (only core settings)
- DVC cache: `.dvc/cache/` and `.dvc/tmp/`
- Tracked outputs in `dvc.yaml`: processed datasets, models, metrics, plots

**Caching:**
- HTTP response caching: `data/.cache/ssen_api/` for SSEN API calls
- File-based caching via `RateLimitedSession` in `src/fyp/ingestion/utils.py` lines 215-227
- Cache invalidation: `force_refresh=True` parameter in `SSENIngestor`

## Authentication & Identity

**Auth Provider:**
- Custom implementation via environment variables and CLI flags
- SSEN API Key: Optional, passed as `--api-key` CLI argument
  - Stored in environment or `.env` file (loaded via `python-dotenv`)
  - Header injection: `Authorization: {api_key}` in `src/fyp/ingestion/ssen_ingestor.py` lines 164-168
  - Currently not enforced by SSEN CKAN API
- No user authentication required (data ingestion is public)

## Monitoring & Observability

**Experiment Tracking:**
- MLflow 2.8.0+ for run tracking and metrics logging
- Implementation: `src/fyp/runner.py` lines 40-42, 617-651
- Usage:
  - `mlflow.set_experiment("energy_forecasting")` - experiment naming
  - `mlflow.start_run()` - start tracking run
  - `mlflow.log_params()` - log model hyperparameters
  - `mlflow.log_metrics()` - log evaluation metrics (MAE, RMSE, F1, etc.)
  - `mlflow.log_artifacts()` - save plots and model artifacts
  - `mlflow.end_run()` - end run with status
- Configurable via `mlflow_experiment` and `mlflow_run_name` in `ExperimentConfig`

**Logging:**
- Framework: Python standard `logging` module
- Configuration: `logging.basicConfig()` in multiple files
- Format: `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- Level: INFO by default
- Loggers created per-module using `logging.getLogger(__name__)`
- Example usage in `src/fyp/ingestion/base.py` lines 38-42, `src/fyp/runner.py` lines 47-50
- Optional: loguru 0.7.3+ available (imported but not actively used)

**Error Handling:**
- HTTP request errors in ingestion: `requests.exceptions.RequestException` handling (`src/fyp/ingestion/ssen_ingestor.py` line 187)
- Exponential backoff with max 3 retries via `RateLimitedSession` (`src/fyp/ingestion/utils.py`)
- Pydantic validation errors for data validation (`src/fyp/ingestion/schema.py`)
- Graceful degradation: Fall back to older SSEN lookup file if new file not found (`src/fyp/ingestion/ssen_ingestor.py` lines 91-99)

## CI/CD & Deployment

**Hosting:**
- Not deployed - Research/academic project
- Local development and computational cluster execution

**CI Pipeline:**
- GitHub Actions (detected by `CI` and `GITHUB_ACTIONS` env vars in `src/fyp/config.py` lines 148-149)
- Automatic sample mode activation for faster CI runs
- Configuration location: `.github/workflows/` (present but not analyzed)

**Data Pipeline Orchestration:**
- DVC for reproducible pipeline runs
- Pipeline stages in `dvc.yaml`:
  1. `ingest_lcl` - Ingest LCL household data
  2. `ingest_ukdale` - Ingest UK-DALE data with optional 30-min downsampling
  3. `ingest_ssen` - Ingest SSEN feeder metadata
  4. `feature_engineering` - Create weather, calendar, and lag features
  5. `train_baselines` - Train baseline forecasting and anomaly models
  6. `train_custom` - Train PatchTST and autoencoder models
  7. `train_selfplay` - Train self-play propose→solve→verify system
  8. `poster_numbers` - Generate poster-ready metrics
  9. `evaluate_models` - Comprehensive evaluation and reporting

## Environment Configuration

**Required env vars for SSEN API:**
- `SSEN_CKAN_URL` (optional) - SSEN API endpoint (default: `https://data.ssen.co.uk`)
- `SSEN_API_KEY` (optional) - API authentication key

**Optional env vars for runtime behavior:**
- `CI` / `GITHUB_ACTIONS` - Activate sample-based fast mode
- `USE_SAMPLES` - Override sample usage (true/false)
- `DEVICE` - PyTorch device selection (cpu/cuda)
- `MLFLOW_TRACKING_URI` - Remote MLflow server URI (if using)

**Secrets location:**
- `.env` file (git-ignored, loaded via `python-dotenv`)
- Environment variables in runtime environment
- CLI arguments for API keys (not recommended for production)

## Webhooks & Callbacks

**Incoming:**
- None detected - No webhook receivers implemented

**Outgoing:**
- None detected - No external webhook calls

## Network & API Rate Limiting

**Rate Limiting:**
- SSEN API: 1.0 second delay between requests (configurable)
- Implementation: `RateLimitedSession` wrapper in `src/fyp/ingestion/utils.py` lines 216-227
- Backoff: Exponential backoff with 0.5 factor on retry (max 3 retries)
- HTTP Adapter: `HTTPAdapter` with retry strategy for connection pooling

## Data Dependencies

**Public Datasets:**
- LCL (London Smart Meter) - Public dataset for UK household energy consumption
- UK-DALE - Public dataset for appliance-level consumption
- SSEN Low Voltage Feeder Data - Public metadata via CKAN API

**No proprietary external data sources:**
- All data is publicly available
- Suitable for academic research and open-source projects

---

*Integration audit: 2026-01-26*
