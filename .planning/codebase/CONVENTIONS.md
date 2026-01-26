# Coding Conventions

**Analysis Date:** 2026-01-26

## Naming Patterns

**Files:**
- Lowercase with underscores: `data_loader.py`, `autoencoder.py`, `base.py`
- Test files prefixed with `test_`: `test_models.py`, `test_ingestion.py`
- Module files represent their primary class/responsibility
- Configuration files named by domain: `config.py`
- Ingestion modules suffixed with ingestor: `lcl_ingestor.py`, `ukdale_ingestor.py`, `ssen_ingestor.py`

**Functions:**
- snake_case throughout: `mean_absolute_error()`, `create_sample_config()`, `write_parquet_batch()`
- Verbs for action functions: `load_config()`, `save_config()`, `calculate_data_quality_metrics()`
- Nouns for getter functions: `get_partition_keys()`, `get_ci_safe_seed()`
- Private functions prefixed with underscore: `_create_windows()`, `_init_weights()`

**Variables:**
- snake_case for variables and parameters: `y_true`, `batch_size`, `forecast_horizon`
- Type-hinted parameters: `config: Path | None = None`, `readings: list[EnergyReading]`
- Constants in UPPERCASE: `UNIFIED_SCHEMA`, `valid_intervals = {1, 5, 10, 15, 30, 60}`

**Types/Classes:**
- PascalCase for classes: `EnergyReading`, `BaseForecaster`, `AutoencoderAnomalyDetector`
- Abstract base classes prefixed with `Base`: `BaseIngestor`, `BaseForecaster`, `BaseAnomalyDetector`
- Domain-specific suffixes: `*Ingestor`, `*Detector`, `*Forecaster`, `*Trainer`, `*Agent`
- Config classes named `{Domain}Config`: `ForecastingConfig`, `AnomalyConfig`, `ExperimentConfig`

## Code Style

**Formatting:**
- Tool: Black
- Line length: 88 characters (configured in `pyproject.toml` line 94)
- Target version: Python 3.11+ (line 95)
- See `[tool.black]` section in `pyproject.toml`

**Linting:**
- Tool: Ruff with strict rule set
- Configuration: `[tool.ruff]` in `pyproject.toml` (lines 111-131)
- Selected rules: E, W, F, I, B, C4, UP, ARG001, C901, SIM101
- Max complexity: 18 (mccabe)
- Ignored rules: E501 (line too long), B008, C901, W191

**Type Checking:**
- Tool: MyPy (strict mode enabled at `pyproject.toml` line 169)
- Configuration settings (lines 170-179):
  - `check_untyped_defs = true`
  - `disallow_any_generics = true`
  - `disallow_incomplete_defs = true`
  - `disallow_untyped_defs = true`
  - `no_implicit_optional = true`
- Overrides for untyped dependencies: polars, pyarrow, mlflow, dvc, tslearn, sktime (lines 181-190)

**Pre-commit Hooks:**
- File: `.pre-commit-config.yaml`
- Trailing whitespace removal
- End-of-file fixer
- Ruff formatting and fixing with `--exit-non-zero-on-fix`
- All hooks run automatically before commits

## Import Organization

**Order:**
1. Standard library: `import os`, `from pathlib import Path`, `from datetime import datetime`
2. Third-party packages: `import numpy as np`, `import pandas as pd`, `import torch`
3. Relative imports: `from fyp.config import ExperimentConfig`, `from .schema import EnergyReading`

Example from `src/fyp/runner.py` (lines 3-22):
```python
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fyp.baselines.anomaly import create_default_detectors
from fyp.baselines.forecasting import create_default_forecasters
from fyp.config import ExperimentConfig, create_sample_config, get_config_from_env
```

**Path Aliases:**
- First-party module: `fyp` (configured at `pyproject.toml` line 167)
- All internal imports use absolute paths: `from fyp.models.autoencoder import AutoencoderAnomalyDetector`
- Barrel exports in `__init__.py` files: `src/fyp/selfplay/__init__.py` (line 164)

## Error Handling

**Patterns:**
- Pydantic validation for data schemas (BaseModel): `src/fyp/config.py`, `src/fyp/ingestion/schema.py`
- Explicit error returns: `validate_reading()` returns `str | None` error message instead of raising
- Try-except for optional features: `src/fyp/runner.py` lines 28-36 handle missing PyTorch/MLflow
- Validation methods use field validators: `@field_validator("ts_utc")` in `EnergyReading` class (line 24)
- Custom validation logic: Check timestamp alignment for 30-min intervals, reject future timestamps

**Exception Usage:**
- ValueError for invalid inputs: "Timestamp must be timezone-aware", "Interval must be one of {valid_intervals}"
- FileNotFoundError for missing resources: `if not dataset_path.exists(): raise FileNotFoundError()`
- ImportError caught for optional dependencies: `except ImportError as e: logger.warning(f"Advanced models not available: {e}")`

**Logging:**
- Module-level logger: `logger = logging.getLogger(__name__)`
- Configuration in `basicConfig()`: `level=logging.INFO`, `format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- Info level for major operations: `logger.info(f"Loading dataset: {dataset}")`
- Warning level for degraded mode: `logger.warning(f"Advanced models not available: {e}")`

## Comments

**When to Comment:**
- Docstrings for all public functions and classes (required by MyPy strict mode)
- Complex business logic: "Seasonal naive forecast" method explanation
- Non-obvious algorithmic choices: "Fallback to naive forecast if not enough training data" in `mean_absolute_scaled_error()`
- Workarounds and migrations: "TODO: Migrate to canonical import paths (fyp.anomaly.autoencoder)" in `runner.py` line 28

**Docstring Style:**
- Google style docstrings preferred (consistent across codebase)
- All functions have docstrings: `src/fyp/config.py`, `src/fyp/data_loader.py`
- Class docstrings explain purpose: `class EnergyReading(BaseModel): """Unified schema for energy consumption readings across all datasets."""`

**Example:**
```python
def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 48,  # 24 hours at 30-min resolution
) -> float:
    """Calculate Mean Absolute Scaled Error."""
    if len(y_train) < seasonal_period:
        # Fallback to naive forecast if not enough training data
        naive_forecast = np.roll(y_train, 1)[1:]
        scale = mean_absolute_error(y_train[1:], naive_forecast)
    else:
        # Seasonal naive forecast
        naive_forecast = y_train[:-seasonal_period]
        scale = mean_absolute_error(y_train[seasonal_period:], naive_forecast)
```

## Function Design

**Size:**
- Average function length: 15-30 lines for business logic
- Larger functions (50+ lines) break complex operations: `src/fyp/runner.py` line 53-100 splits data loading logic
- Single responsibility principle: Each function does one task

**Parameters:**
- Explicit typing required by MyPy: `def load_dataset(self, dataset: str, start_date: str | None = None, ...)`
- Union types for optional values: `config_path: Path | None = None`
- Collections typed: `list[EnergyReading]`, `dict[str, float]`, `list[int]`
- Default values provided for optional parameters

**Return Values:**
- Explicit return type annotations: `-> float`, `-> dict[str, float]`, `-> tuple[pd.DataFrame, pd.DataFrame]`
- None returns for validation checks: `validate_reading() -> str | None`
- Dictionary returns for multi-value results: `forecasting_metrics() -> dict[str, float]`

## Module Design

**Exports:**
- Public API defined in docstrings and type hints
- Barrel exports in `__init__.py`: `src/fyp/selfplay/__init__.py` re-exports core classes
- Private modules indicated with leading underscore: utilities module imports
- All imports in functions are explicit: `from fyp.models.autoencoder import AutoencoderAnomalyDetector`

**Barrel Files:**
- Used in `src/fyp/selfplay/__init__.py` (line 164): `from fyp.selfplay.trainer import SelfPlayTrainer`
- Enables clean imports: `from fyp.selfplay import SelfPlayTrainer` instead of `from fyp.selfplay.trainer import SelfPlayTrainer`

## Database and State

**Configuration Management:**
- Pydantic models for all config: `ForecastingConfig`, `AnomalyConfig`, `ExperimentConfig`
- YAML file loading: `def load_config(config_path: Path | None = None) -> ExperimentConfig:`
- Environment variable override: `def get_config_from_env() -> ExperimentConfig:` checks CI environment

**Data Classes:**
- EnergyReading as unified schema: `src/fyp/ingestion/schema.py` (line 10)
- Field validators for constraints: timezone-aware timestamps, valid intervals
- JSON serialization of complex fields: `extras: dict[str, Any] | None`

---

*Convention analysis: 2026-01-26*
