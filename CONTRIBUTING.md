# Contributing to FYP Energy Forecasting

Thank you for your interest in contributing to the Self-Play Energy Forecasting & Anomaly Detection project! This document provides guidelines for contributing to the codebase.

## Development Workflow

### Branching Model

We use a Git flow branching model:

- **`main`**: Production-ready code, tagged releases
- **`dev`**: Integration branch for ongoing development
- **Feature branches**: `feature/your-feature-name` branched from `dev`
- **Hotfix branches**: `hotfix/issue-description` for urgent fixes

### Contribution Process

1. **Fork and Clone**
   ```bash
   git clone https://github.com/USERNAME/FYP-Predictive_Anomaly_Detection.git
   cd FYP-Predictive_Anomaly_Detection
   ```

2. **Set Up Development Environment**
   ```bash
   # Install dependencies
   poetry install

   # Activate virtual environment
   poetry shell

   # Install pre-commit hooks
   pre-commit install

   # Initialize DVC
   dvc init --no-scm
   ```

3. **Create Feature Branch**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes**
   - Write code following our coding standards
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all checks pass locally

5. **Local Testing**
   ```bash
   # Run pre-commit checks
   pre-commit run --all-files

   # Run tests
   pytest tests/ -v

   # Check DVC pipeline
   dvc repro --force
   ```

6. **Submit Pull Request**
   - Push feature branch to your fork
   - Create PR against `dev` branch
   - Fill out PR template completely
   - Address any review feedback

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- **feat**: New feature for the user
- **fix**: Bug fix for the user
- **docs**: Changes to documentation
- **style**: Formatting, missing semicolons, etc.; no code change
- **refactor**: Refactoring production code
- **test**: Adding missing tests, refactoring tests; no production code change
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to CI configuration files and scripts
- **perf**: Performance improvements
- **revert**: Reverting a previous commit

### Examples
```bash
feat(selfplay): add EV charging spike scenario generator
fix(data): handle missing values in UK-DALE preprocessing
docs(api): update docstrings for forecasting models
test(verifier): add physics constraint validation tests
ci: update Python version to 3.11 in GitHub Actions
```

### Multi-line Commits
```bash
feat(models): implement PatchTST for energy forecasting

- Add patch-based transformer architecture
- Include energy-specific modifications for seasonal patterns
- Support uncertainty quantification through quantile heads
- Add comprehensive unit tests for model components

Closes #123
```

## Coding Standards

### Python Code Style

We use **Black** for code formatting and **Ruff** for linting:

```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Fix auto-fixable issues
ruff check --fix src/ tests/
```

### Code Quality Requirements

1. **Type Hints**: All functions must include type hints
   ```python
   def process_household_data(
       data: pd.DataFrame,
       resolution: str = "30min"
   ) -> pd.DataFrame:
       """Process household energy consumption data."""
       pass
   ```

2. **Docstrings**: All public functions and classes must have docstrings
   ```python
   def train_forecasting_model(
       data: np.ndarray,
       config: ModelConfig
   ) -> ForecastingModel:
       """Train a forecasting model on household energy data.

       Args:
           data: Training data with shape (n_samples, n_features)
           config: Model configuration including hyperparameters

       Returns:
           Trained forecasting model ready for prediction

       Raises:
           ValueError: If data is empty or config is invalid
       """
       pass
   ```

3. **Error Handling**: Use specific exception types and helpful messages
   ```python
   if data.empty:
       raise ValueError("Cannot train on empty dataset")
   ```

### Testing Standards

1. **Test Coverage**: Maintain >80% test coverage for new code
2. **Test Types**: Include unit tests, integration tests, and smoke tests
3. **Test Naming**: Use descriptive test names explaining the scenario

```python
def test_ev_spike_scenario_generation_with_winter_conditions():
    """Test EV charging spike generation during winter heating season."""
    pass

def test_forecast_accuracy_on_holiday_periods():
    """Test model accuracy during holiday periods with atypical patterns."""
    pass
```

### MLflow Experiment Standards

1. **Naming Convention**: Follow experiment taxonomy in `docs/experiments.md`
2. **Required Artifacts**: Log config, model, metrics, and plots
3. **Reproducibility**: Set seeds and log all parameters

```python
with mlflow.start_run(run_name="selfplay_patchtst_ukdale_v1_42"):
    # Set reproducibility
    mlflow.log_param("random_seed", 42)

    # Log configuration
    mlflow.log_dict(config.dict(), "config.yaml")

    # Log model and metrics
    mlflow.pytorch.log_model(model, "best_model")
    mlflow.log_metrics(evaluation_metrics)
```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/fyp --cov-report=html

# Run specific test types
pytest tests/ -m "unit"        # Unit tests only
pytest tests/ -m "integration" # Integration tests only
pytest tests/ -m "not slow"    # Skip slow tests

# Run tests in parallel
pytest tests/ -n auto
```

### Writing Tests

1. **Test Structure**: Use AAA pattern (Arrange, Act, Assert)
   ```python
   def test_scenario_generator_produces_valid_spikes():
       # Arrange
       generator = EVSpikeGenerator(power_range=(3.5, 7.0))
       baseline_data = create_sample_household_data()

       # Act
       scenario = generator.generate_scenario(baseline_data)

       # Assert
       assert scenario.peak_power >= 3.5
       assert scenario.peak_power <= 7.0
       assert scenario.duration > 0
   ```

2. **Fixtures**: Use pytest fixtures for common test data
   ```python
   @pytest.fixture
   def sample_household_data():
       """Provide sample household consumption data for testing."""
       return pd.DataFrame({
           'timestamp': pd.date_range('2023-01-01', periods=48, freq='30min'),
           'consumption': np.random.normal(1.5, 0.5, 48)
       })
   ```

3. **Parameterized Tests**: Test multiple scenarios efficiently
   ```python
   @pytest.mark.parametrize("season,expected_multiplier", [
       ("winter", 1.2),
       ("summer", 0.8),
       ("spring", 1.0),
       ("autumn", 1.0)
   ])
   def test_seasonal_adjustments(season, expected_multiplier):
       """Test seasonal consumption adjustments."""
       pass
   ```

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def aggregate_households_to_feeder(
    household_forecasts: List[np.ndarray],
    diversity_factors: Dict[int, float],
    transformer_capacity: float = 500.0
) -> np.ndarray:
    """Aggregate household forecasts into realistic distribution feeder load.

    Applies diversity factors and transformer constraints to simulate
    realistic distribution network loading from household consumption.

    Args:
        household_forecasts: List of individual household load forecasts
        diversity_factors: Mapping from household count to diversity factor
        transformer_capacity: Maximum transformer capacity in kVA

    Returns:
        Aggregated feeder load profile respecting network constraints

    Raises:
        ValueError: If household_forecasts is empty
        OverloadError: If aggregated load exceeds transformer capacity

    Example:
        >>> forecasts = [np.random.rand(48) for _ in range(50)]
        >>> diversity = {50: 0.6}  # 50 households have 60% diversity
        >>> feeder_load = aggregate_households_to_feeder(forecasts, diversity)
        >>> assert len(feeder_load) == 48
    """
    pass
```

### README Updates

When adding significant features:
1. Update the main README.md if user-facing
2. Add examples to relevant documentation files
3. Update the project roadmap if applicable

## Pull Request Guidelines

### PR Title Format
```
<type>[scope]: <description>
```

Examples:
- `feat(selfplay): implement proposer scenario generation`
- `fix(data): resolve UK-DALE missing value handling`
- `docs(api): add comprehensive model usage examples`

### PR Description Template

```markdown
## Summary
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Code changes are documented with docstrings
- [ ] README updated if needed
- [ ] API documentation updated if needed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] No merge conflicts
- [ ] All CI checks pass
- [ ] Linked to relevant issues

## Screenshots/Examples
(If applicable, include screenshots or example outputs)
```

## Local Development Pipeline

### Running the Smoke Pipeline

```bash
# Full pipeline check (should complete quickly with placeholder stages)
dvc repro

# Check pipeline status
dvc status

# Visualize pipeline
dvc dag
```

### Pre-commit Checks

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hooks
pre-commit run black
pre-commit run ruff
pre-commit run mypy
```

### Development Database

For local development with actual data:

```bash
# Add sample datasets (small subsets for development)
dvc add data/raw/ukdale_sample/
dvc add data/raw/lcl_sample/

# Configure local DVC remote (optional)
dvc remote add local /path/to/local/storage
dvc push
```

## Performance Considerations

### Computational Efficiency
- Profile code with `cProfile` for performance bottlenecks
- Use vectorized operations (NumPy, Polars) instead of loops
- Consider memory usage for large datasets
- Implement batch processing for large-scale experiments

### MLflow Best Practices
- Log metrics incrementally during training (not all at once)
- Use MLflow's automatic logging when possible
- Clean up old experiment runs periodically
- Use tags for efficient filtering and organization

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Documentation**: Check `docs/` directory for detailed information
- **Code Review**: Request reviews from maintainers for significant changes

## Recognition

Contributors are recognized in:
- GitHub contributor graphs
- Release notes for significant contributions
- Project documentation acknowledgments
- Academic citations for research contributions

Thank you for contributing to advancing energy forecasting research! 🔋⚡
