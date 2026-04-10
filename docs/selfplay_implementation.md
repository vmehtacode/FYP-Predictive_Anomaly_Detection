# Self-Play Implementation for Grid Guardian

## Overview

This document describes the implementation of the self-play reinforcement learning system for Grid Guardian, adapted for energy consumption forecasting.

## Architecture

The self-play system consists of three main components that work together in a propose→solve→verify loop:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Proposer   │────>│   Solver    │────>│  Verifier   │
│   Agent     │     │   Agent     │     │   Agent     │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                        │
       └────────────────────────────────────────┘
              Learnability Reward
```

### 1. Proposer Agent (`src/fyp/selfplay/proposer.py`)

**Purpose**: Generates challenging but physically-plausible energy consumption scenarios.

**Key Features**:
- 5 scenario types: EV_SPIKE, COLD_SNAP, PEAK_SHIFT, OUTAGE, MISSING_DATA
- Curriculum learning with progressive difficulty
- Physics constraint validation
- Learnability-based reward system

**Core Classes**:
```python
@dataclass
class ScenarioProposal:
    scenario_type: str
    magnitude: float
    duration: int
    start_time: datetime
    affected_appliances: List[str]
    baseline_context: np.ndarray
    difficulty_score: float
    physics_valid: bool
```

### 2. Solver Agent (`src/fyp/selfplay/solver.py`)

**Purpose**: Forecasts energy consumption using PatchTST with scenario awareness.

**Key Features**:
- Wraps existing PatchTST implementation
- Quantile regression (0.1, 0.5, 0.9)
- Scenario-conditioned predictions
- Combined loss with verification rewards

**Integration with PatchTST**:
```python
# Uses existing model from fyp.models.patchtst
self.model = PatchTSTForecaster(
    patch_len=16,
    d_model=128,
    n_heads=8,
    forecast_horizon=48
)
```

### 3. Verifier Agent (`src/fyp/selfplay/verifier.py`)

**Purpose**: Validates forecasts against physics constraints from SSEN standards.

**Implemented Constraints**:
1. **NonNegativityConstraint**: Energy ≥ 0 (hard physics)
2. **HouseholdMaxConstraint**: Respects UK fuse ratings (BS 7671:2018)
3. **RampRateConstraint**: Realistic consumption changes
4. **TemporalPatternConstraint**: Daily/weekly patterns
5. **PowerFactorConstraint**: UK G59/3 standards
6. **VoltageConstraint**: 230V +10%/-6% limits

**Constraint Weights**:
```python
weights = {
    "non_negativity": 1.0,      # Hard constraint
    "household_max": 1.0,        # Hard constraint
    "ramp_rate": 0.5,            # Soft constraint
    "temporal_pattern": 0.3,     # Soft constraint
    "power_factor": 0.4,         # Soft constraint
    "voltage": 0.6               # Medium constraint
}
```

### 4. Training Orchestrator (`src/fyp/selfplay/trainer.py`)

**Purpose**: Coordinates the full self-play training loop.

**Training Algorithm**:
```python
for episode in range(num_episodes):
    for context, target in batch:
        # 1. PROPOSE
        scenario = proposer.propose_scenario(context)

        # 2. SOLVE
        forecast = solver.predict(context, scenario)

        # 3. VERIFY
        reward = verifier.evaluate(forecast, scenario)

        # 4. UPDATE
        solver.train_step(context, target, scenario, reward)
        proposer.update_buffer(scenario, learnability_reward)
```

## Usage Examples

### Basic Training

```python
from fyp.selfplay import ProposerAgent, SolverAgent, VerifierAgent, SelfPlayTrainer

# Initialize components
proposer = ProposerAgent(
    ssen_constraints_path="data/derived/ssen_constraints.json",
    difficulty_curriculum=True
)

solver = SolverAgent(
    model_config={"patch_len": 16, "d_model": 128},
    historical_data_path="data/processed/lcl_data/",
    pretrain_epochs=20
)

verifier = VerifierAgent(
    ssen_constraints_path="data/derived/ssen_constraints.json"
)

# Create trainer
trainer = SelfPlayTrainer(
    proposer=proposer,
    solver=solver,
    verifier=verifier,
    config={"alpha": 0.1, "batch_size": 32}
)

# Train
metrics = trainer.train(num_episodes=1000)
```

### Fast Testing with Samples

```python
# For CI/testing with sample data
solver = SolverAgent(
    model_config=create_patchtst_config(use_samples=True),
    use_samples=True,
    pretrain_epochs=2
)

# Smaller batch size for faster testing
config = {
    "alpha": 0.1,
    "batch_size": 8,
    "max_epochs": 5
}

trainer = SelfPlayTrainer(proposer, solver, verifier, config)
```

### Custom Scenario Generation

```python
# Generate specific scenario type
context = np.random.rand(336) * 2.0  # 7 days of data

# Force EV charging scenario
scenario = proposer.propose_scenario(
    historical_context=context,
    forecast_horizon=48
)

# Apply to baseline
baseline = np.ones(48) * 2.0
transformed = scenario.apply_to_timeseries(baseline)
```

### Constraint Validation

```python
# Validate forecast against specific constraints
forecast = np.array([2.0, 3.0, 60.0, -1.0, 5.0])

# Check individual constraint
score, violations = verifier.validate_single_constraint(
    "household_max", forecast
)

# Full validation with details
reward, details = verifier.evaluate(
    forecast, scenario, return_details=True
)

for constraint, result in details.items():
    print(f"{constraint}: score={result['score']}, "
          f"violations={result['violations']}")
```

## Scenario Types

### 1. EV_SPIKE
- **Description**: Electric vehicle charging events
- **Magnitude**: 1.0-2.0x base power (3.5-7 kW)
- **Duration**: 1-8 hours
- **Difficulty**: 0.3 (relatively easy to forecast)

### 2. COLD_SNAP
- **Description**: Weather-driven heating demand
- **Magnitude**: 1.5-3.0x baseline consumption
- **Duration**: 6-72 hours
- **Difficulty**: 0.4 (moderate)

### 3. PEAK_SHIFT
- **Description**: Temporal shift in peak consumption
- **Magnitude**: ±2 hours shift
- **Duration**: 2-6 hours of shifted peak
- **Difficulty**: 0.6 (challenging)

### 4. OUTAGE
- **Description**: Equipment failures or power cuts
- **Magnitude**: 0 (zero consumption)
- **Duration**: 1-12 hours
- **Difficulty**: 0.2 (easy to detect)

### 5. MISSING_DATA
- **Description**: Meter communication failures
- **Magnitude**: N/A (data gaps)
- **Duration**: 0.5-6 hours
- **Difficulty**: 0.5 (requires interpolation)

## Training Metrics

The system tracks several key metrics during training:

```python
{
    "episode": 100,
    "avg_solver_loss": 0.152,
    "avg_verification_reward": -0.021,
    "avg_proposer_reward": 0.623,
    "avg_mae": 0.287,  # kWh
    "avg_mape": 15.3,  # %
    "scenario_diversity": 0.8,
    "curriculum_level": 0.45
}
```

## Configuration Options

### Trainer Configuration

```python
config = {
    "alpha": 0.1,                    # Verification reward weight
    "batch_size": 32,                # Windows per episode
    "lambda": 0.5,                   # Exploration vs exploitation
    "checkpoint_every": 100,         # Save frequency
    "success_threshold_mape": 20.0,  # Success criterion
    "min_episodes_before_curriculum": 50  # Curriculum progression
}
```

### Model Configuration

```python
model_config = {
    "patch_len": 16,            # Patch size for time series
    "d_model": 128,             # Model dimension
    "n_heads": 8,               # Attention heads
    "n_layers": 4,              # Transformer layers
    "forecast_horizon": 48,     # 24 hours at 30-min intervals
    "quantiles": [0.1, 0.5, 0.9],  # Prediction quantiles
    "max_epochs": 50,           # Training epochs
    "learning_rate": 1e-3       # Learning rate
}
```

## Physics Constraints from SSEN

The verifier enforces real UK electrical standards:

1. **Household Limits** (BS 7671:2018)
   - Typical: 7.5 kWh per 30 minutes (15 kW continuous)
   - Absolute: 50 kWh per 30 minutes (100A @ 230V)

2. **Voltage Limits** (UK standard)
   - Nominal: 230V
   - Range: 216.2V to 253V (-6% to +10%)

3. **Power Factor** (G59/3)
   - Minimum: 0.95 lagging

4. **Ramp Rates**
   - Maximum: 5 kWh change per 30-minute interval

## Performance Optimization

### Memory Efficiency
- Process data in batches
- Sliding window approach
- Limited historical buffer (100 windows)

### CPU Optimization
- No GPU requirements
- Vectorized operations
- Efficient PyTorch CPU kernels

### Fast Testing Mode
```python
# Enable for CI/testing
use_samples = True
solver = SolverAgent(use_samples=use_samples)

# Reduces:
# - Model size (d_model: 128 → 32)
# - Training epochs (50 → 2)
# - Batch size (32 → 8)
# - Data volume (167M → 1000 records)
```

## Validation and Testing

### Unit Tests

Run comprehensive tests:
```bash
pytest tests/test_selfplay.py -v
```

Test coverage includes:
- Individual constraints
- Scenario generation
- Forecast accuracy
- Training stability
- Checkpoint save/load
- Integration flow

### Integration Testing

```python
# Full integration test
trainer = SelfPlayTrainer(proposer, solver, verifier)
metrics = trainer.train(
    num_episodes=10,
    save_checkpoints=False
)

# Validate results
assert metrics[-1]["avg_mae"] < 0.5  # kWh
assert metrics[-1]["avg_verification_reward"] > -0.1
```

## Future Enhancements

1. **Additional Scenarios**
   - Solar generation patterns
   - Battery storage behavior
   - Heat pump operations
   - Smart meter anomalies

2. **Advanced Constraints**
   - Network topology awareness
   - Multi-household coordination
   - Reactive power modeling
   - Harmonic distortion limits

3. **Algorithmic Improvements**
   - Meta-learning for faster adaptation
   - Adversarial scenario generation
   - Multi-agent self-play
   - Continual learning from deployment

## Troubleshooting

### Common Issues

1. **High Loss Values**
   - Check data normalization
   - Verify scenario parameters
   - Reduce learning rate

2. **Constraint Violations**
   - Review SSEN limits in config
   - Check scenario magnitude bounds
   - Validate input data ranges

3. **Slow Training**
   - Enable sample mode
   - Reduce batch size
   - Limit validation frequency

### Debug Mode

```python
import logging
logging.getLogger("fyp.selfplay").setLevel(logging.DEBUG)

# Detailed constraint violations
reward, details = verifier.evaluate(
    forecast, scenario, return_details=True
)
print(json.dumps(details, indent=2))
```

## References

1. Self-play reinforcement learning literature
2. UK electrical standards (BS 7671:2018)
3. SSEN network constraints (G59/3)
4. PatchTST architecture paper

---

For questions or contributions, see [CONTRIBUTING.md](../CONTRIBUTING.md)
