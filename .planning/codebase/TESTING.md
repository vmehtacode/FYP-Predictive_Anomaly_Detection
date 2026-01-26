# Testing Patterns

**Analysis Date:** 2026-01-26

## Test Framework

**Runner:**
- Framework: pytest 7.4.0+
- Config file: `pyproject.toml` (lines 192-207)
- Minimum version: 7.0

**Assertion Library:**
- Native pytest assertions
- NumPy testing utilities: `np.testing.assert_allclose()` in `tests/test_selfplay.py` line 83
- Manual assertions: `assert len(windows) > 0`, `assert forecast[-1] > forecast[0]`

**Run Commands:**
```bash
pytest                          # Run all tests
pytest -m "not slow"            # Exclude slow tests
pytest -v                       # Verbose output
pytest --cov=src               # Coverage report (configured in pyproject.toml)
pytest -x                       # Stop on first failure
```

**Configuration Details** (from `pyproject.toml`):
- `testpaths = ["tests"]` - tests directory
- `strict-markers` and `strict-config` enforced
- Filter warnings for UserWarning and DeprecationWarning (lines 199-201)

## Test File Organization

**Location:**
- Co-located in `tests/` directory at project root
- Pattern: Separate directory, not alongside source code
- One test file per major module: `test_models.py`, `test_ingestion.py`, `test_baselines.py`

**Naming:**
- Test files: `test_*.py` (pytest auto-discovery)
- Test classes: `Test*` (e.g., `TestSchema`, `TestPatchTST`, `TestForecasting`)
- Test functions: `test_*` (e.g., `test_valid_reading()`, `test_patchtst_creation()`)
- Fixtures: `*_fixture()` or simple name like `temp_constraints_file` (lines 32-49 in test_selfplay.py)

**Structure:**
```
tests/
├── test_baselines.py          # Baseline forecasting/anomaly models
├── test_cli_anomaly_sample.py # CLI integration tests
├── test_data_loading.py       # Data loader functionality
├── test_ingestion.py          # Schema and ingestion validation
├── test_ingestion_complete.py # Full ingestion pipeline
├── test_ingestion_enhanced.py # Enhanced ingestion features
├── test_models.py             # Neural models (PatchTST, Autoencoder)
├── test_selfplay.py           # Self-play training system
├── test_smoke.py              # Smoke tests for CI
├── test_ssen_ingestion.py     # SSEN-specific ingestion
└── __init__.py
```

## Test Structure

**Suite Organization:**
```python
class TestForecasting:
    """Test forecasting models."""

    def test_seasonal_naive(self):
        """Test seasonal naive forecaster."""
        # Arrange
        data = create_synthetic_sine_data(96)

        # Act
        forecaster = SeasonalNaive(seasonal_period=48)
        forecaster.fit(data)
        forecast = forecaster.predict(data, steps=48)

        # Assert
        assert len(forecast) == 48
        assert all(f >= 0 for f in forecast)
```

Pattern from `tests/test_baselines.py` (lines 56-100):
- Class groups related tests
- Docstring explains test category
- Method docstrings explain individual test purpose
- Arrange-Act-Assert pattern (implicit)

**Patterns:**
- Setup: Create synthetic data or fixtures before test
- Execution: Call function/method under test
- Assertion: Verify output with `assert` statements

**Fixture Pattern:**
```python
@pytest.fixture
def temp_constraints_file():
    """Create temporary constraints file for testing."""
    constraints = {...}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(constraints, f)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)
```
From `tests/test_selfplay.py` lines 32-49: Setup-yield-teardown pattern for resource cleanup.

## Mocking

**Framework:**
- pytest-mock (3.12.0+) available but minimal usage
- Monkeypatching for environment variables: `os.getenv("CI")`
- Optional dependency mocking with try-except

**Patterns:**

No explicit mocking in tests - instead real object instantiation:
```python
def test_patchtst_creation(self):
    """Test PatchTST model creation."""
    from fyp.models.patchtst import EnergyPatchTST
    model = EnergyPatchTST(
        patch_len=8,
        d_model=32,
        n_heads=2,
        n_layers=1,
        forecast_horizon=16,
        quantiles=[0.1, 0.5, 0.9],
    )
    x = torch.randn(batch_size, n_patches, patch_len)
    output = model(x)
```
From `tests/test_models.py` lines 90-113: Direct instantiation without mocks.

**What to Mock:**
- Optional dependencies: Check for import availability with try-except
- Environment variables: Use `os.getenv()` in conditional tests
- File I/O: Use tempfile fixtures for temporary resources

**What NOT to Mock:**
- Core business logic classes (use real instances)
- Models and forecasters (train with real data)
- Data validation (test against actual schema)

## Fixtures and Factories

**Test Data:**

Helper functions create synthetic data for tests:
```python
def create_synthetic_energy_series(
    n_points: int = 144, noise_level: float = 0.1
) -> np.ndarray:
    """Create synthetic energy consumption series."""
    t = np.linspace(0, 6 * np.pi, n_points)
    daily = 1 + 0.5 * np.sin(2 * np.pi * t / 48)
    weekly = 0.2 * np.sin(2 * np.pi * t / (48 * 7))
    noise = np.random.normal(0, noise_level, n_points)
    energy = daily + weekly + noise
    return np.maximum(energy, 0.1)
```
From `tests/test_models.py` lines 14-25: Pattern for reproducible synthetic data.

**Data with Anomalies:**
```python
def create_energy_with_anomalies(n_points: int = 144) -> tuple[np.ndarray, np.ndarray]:
    """Create energy series with injected anomalies."""
    data = create_synthetic_energy_series(n_points, noise_level=0.05)
    labels = np.zeros(n_points)

    # Inject spike anomalies
    anomaly_indices = [30, 70, 110]
    for idx in anomaly_indices:
        if idx < n_points:
            data[idx] *= 3  # Spike
            labels[idx] = 1

    return data, labels
```
From `tests/test_models.py` lines 28-44: Paired data and labels for validation testing.

**Location:**
- Helper functions defined at module level in test files (top of each test file)
- Fixtures defined with `@pytest.fixture` decorator
- Not extracted to separate fixtures module (all inline)

**pytest Fixtures:**
```python
@pytest.fixture
def temp_constraints_file():
    """Create temporary constraints file for testing."""
    # Setup
    constraints = {...}
    # ...
    yield temp_path  # Provide resource
    # Teardown
    os.unlink(temp_path)
```

## Coverage

**Requirements:**
- No explicit coverage target configured (not enforced at `pyproject.toml` 209-229)
- HTML coverage reports generated in `htmlcov/` directory

**View Coverage:**
```bash
pytest --cov=src --cov-report=html  # Generate HTML report
pytest --cov=src --cov-report=term  # Terminal summary
```

**Coverage Configuration** (from `pyproject.toml` lines 209-229):
```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

## Test Types

**Unit Tests:**
- Scope: Individual functions and classes
- Approach: Synthetic data, fast execution (< 100ms each)
- Examples: `test_seasonal_naive()`, `test_valid_reading()`, `test_pinball_loss()`
- Location: All tests in `tests/` directory

**Integration Tests:**
- Scope: Multiple components together
- Marker: `@pytest.mark.integration`
- Example: `tests/test_ingestion_enhanced.py` line 1: `@pytest.mark.integration`
- Run: `pytest -m integration`

**Slow Tests:**
- Marker: `@pytest.mark.slow`
- Skip in CI: `pytest -m "not slow"`
- Long-running: Model training, full dataset processing

**Smoke Tests:**
- Purpose: Verify basic project structure and imports
- File: `tests/test_smoke.py`
- Coverage: stdlib imports, project directories, documentation structure

**Skipped Tests:**
- Conditional: `@pytest.mark.skipif(condition, reason="message")`
- GPU tests: Skip without CUDA (lines 83-85 in test_models.py)
- Data tests: Skip when processed data unavailable (test_data_loading.py)

## Common Patterns

**Async Testing:**
Not used in codebase - all tests are synchronous.

**Error Testing:**
```python
def test_invalid_dataset(self):
    """Test invalid dataset name."""
    with pytest.raises(ValueError):
        EnergyReading(
            dataset="invalid",
            entity_id="test",
            ts_utc=datetime.now(UTC),
            interval_mins=30,
            energy_kwh=0.5,
            source="test",
        )
```
From `tests/test_ingestion.py` lines 30-40: Use `pytest.raises()` context manager for exception testing.

**Timezone Testing:**
```python
def test_naive_timestamp(self):
    """Test timezone-naive timestamp rejection."""
    with pytest.raises(ValueError, match="timezone-aware"):
        EnergyReading(
            dataset="lcl",
            entity_id="test",
            ts_utc=datetime(2023, 1, 1, 12, 0),  # No timezone
            interval_mins=30,
            energy_kwh=0.5,
            source="test",
        )
```
From `tests/test_ingestion.py` lines 54-64: Match error message patterns.

**Parametrized Tests:**
Not heavily used - single test per scenario preferred.

**Expected Behavior Tests:**
```python
def test_seasonal_naive(self):
    """Test seasonal naive forecaster."""
    data = create_synthetic_sine_data(96)
    forecaster = SeasonalNaive(seasonal_period=48)
    forecaster.fit(data)
    forecast = forecaster.predict(data, steps=48)

    # Should repeat pattern from previous day
    expected = data[-48:]
    mae = mean_absolute_error(expected, forecast)
    assert mae < 1.0  # Should be close to exact repeat
```
From `tests/test_baselines.py` lines 59-75: Clear assertions on expected behavior.

## Test Markers

**Available Markers** (from `pyproject.toml` lines 203-207):
```toml
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

**Usage:**
```python
@pytest.mark.slow
def test_end_to_end_forecasting(self):
    """Test end-to-end forecasting pipeline."""
    # Long-running test

@pytest.mark.integration
class TestIntegrationWithSamples:
    """Integration tests using sample data."""
```

## CI/Test Configuration

**Environment Variables:**
- `CI`: Set in GitHub Actions
- `GITHUB_ACTIONS`: GitHub-specific flag
- `PYTEST_CURRENT_TEST`: Set by pytest itself
- Tests check with `os.getenv("CI")` for fast configs

**Sample Configuration:**
- Automatic for CI environment: `if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"): config = create_sample_config()`
- Smaller models, fewer epochs for fast execution
- See `src/fyp/config.py` lines 148-149 and `src/fyp/utils/random.py` lines 52-75

---

*Testing analysis: 2026-01-26*
