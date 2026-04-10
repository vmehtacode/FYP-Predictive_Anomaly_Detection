"""Unit tests for self-play training system."""

import json
import os
import tempfile

import numpy as np
import pytest

from fyp.selfplay import (
    ProposerAgent,
    ScenarioProposal,
    SelfPlayTrainer,
    SolverAgent,
    VerifierAgent,
)
from fyp.selfplay.utils import (
    apply_scenario_transformation,
    calculate_pinball_loss,
    create_sliding_windows,
    estimate_scenario_difficulty,
    normalize_consumption,
)
from fyp.selfplay.verifier import (
    HouseholdMaxConstraint,
    NonNegativityConstraint,
    RampRateConstraint,
    TemporalPatternConstraint,
)


@pytest.fixture
def temp_constraints_file():
    """Create temporary constraints file for testing."""
    constraints = {
        "household_limits": {
            "typical_max_kwh_30min": 7.5,
            "absolute_max_kwh_30min": 50.0,
        },
        "voltage_limits": {"nominal_v": 230, "min_percent": -6, "max_percent": 10},
        "power_factor": {"min_lagging": 0.95},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(constraints, f)
        temp_path = f.name

    yield temp_path
    os.unlink(temp_path)


class TestUtils:
    """Test utility functions."""

    def test_create_sliding_windows(self):
        """Test sliding window creation."""
        data = np.random.rand(400)  # 200 hours of data
        windows = create_sliding_windows(
            data, context_length=336, forecast_horizon=48, stride=24
        )

        assert len(windows) > 0
        assert all(len(context) == 336 for context, _ in windows)
        assert all(len(target) == 48 for _, target in windows)

        # Check continuity
        context, target = windows[0]
        assert np.array_equal(data[:336], context)
        assert np.array_equal(data[336:384], target)

    def test_normalize_denormalize(self):
        """Test normalization and denormalization."""
        data = np.array([1, 2, 3, 4, 5])
        normalized, stats = normalize_consumption(data, method="standard")

        assert np.abs(np.mean(normalized)) < 1e-6  # Zero mean
        assert np.abs(np.std(normalized) - 1.0) < 1e-6  # Unit variance

        # Test denormalization
        from fyp.selfplay.utils import denormalize_consumption

        denormalized = denormalize_consumption(normalized, stats, method="standard")
        np.testing.assert_allclose(data, denormalized, rtol=1e-5)

    def test_apply_scenario_transformation(self):
        """Test scenario transformations."""
        baseline = np.ones(48) * 2.0  # 2 kWh baseline

        # Test EV spike
        ev_transformed = apply_scenario_transformation(
            baseline, "EV_SPIKE", magnitude=1.5, duration=8, start_idx=10
        )
        assert np.max(ev_transformed) > np.max(baseline)
        assert ev_transformed[10] > baseline[10]  # Spike starts

        # Test outage
        outage_transformed = apply_scenario_transformation(
            baseline, "OUTAGE", magnitude=0, duration=4, start_idx=20
        )
        assert np.all(outage_transformed[20:24] == 0.0)
        assert np.all(outage_transformed[:20] == baseline[:20])

    def test_pinball_loss(self):
        """Test pinball loss calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.2, 1.8, 3.2])

        # Test median (0.5 quantile)
        loss_median = calculate_pinball_loss(y_true, y_pred, 0.5)
        assert loss_median > 0

        # Test asymmetry
        loss_low = calculate_pinball_loss(y_true, y_pred, 0.1)
        loss_high = calculate_pinball_loss(y_true, y_pred, 0.9)
        assert loss_low != loss_high  # Should be different due to asymmetry

    def test_estimate_scenario_difficulty(self):
        """Test difficulty estimation."""
        difficulty_easy = estimate_scenario_difficulty(
            "OUTAGE", magnitude=0, duration=2, historical_volatility=0.1
        )
        difficulty_hard = estimate_scenario_difficulty(
            "PEAK_SHIFT", magnitude=2.0, duration=24, historical_volatility=0.5
        )

        assert 0 <= difficulty_easy <= 1
        assert 0 <= difficulty_hard <= 1
        assert difficulty_hard > difficulty_easy  # Peak shift should be harder


class TestConstraints:
    """Test individual constraint implementations."""

    def test_non_negativity_constraint(self):
        """Test non-negativity constraint."""
        constraint = NonNegativityConstraint()

        # Test valid forecast
        valid_forecast = np.array([1.0, 2.0, 3.0])
        score, violations = constraint.evaluate(valid_forecast)
        assert score == 0.0
        assert len(violations) == 0

        # Test invalid forecast
        invalid_forecast = np.array([1.0, -0.5, 3.0])
        score, violations = constraint.evaluate(invalid_forecast)
        assert score < 0
        assert len(violations) > 0
        assert "Negative consumption" in violations[0]

    def test_household_max_constraint(self):
        """Test household maximum constraint."""
        constraint = HouseholdMaxConstraint(typical_max_kwh=7.5, absolute_max_kwh=50.0)

        # Test normal consumption
        normal = np.array([2.0, 3.0, 4.0])
        score, violations = constraint.evaluate(normal)
        assert score == 0.0

        # Test high but valid consumption
        high = np.array([8.0, 9.0, 10.0])
        score, violations = constraint.evaluate(high)
        assert score < 0  # Penalty for exceeding typical
        assert "Unusual consumption" in str(violations)

        # Test physics violation
        impossible = np.array([60.0, 70.0, 80.0])
        score, violations = constraint.evaluate(impossible)
        assert score == -1.0  # Maximum penalty
        assert "Physics violation" in str(violations)

    def test_ramp_rate_constraint(self):
        """Test ramp rate constraint."""
        constraint = RampRateConstraint(max_ramp_kwh_per_interval=5.0)

        # Test smooth changes
        smooth = np.array([1.0, 1.5, 2.0, 2.5])
        score, violations = constraint.evaluate(smooth)
        assert score == 0.0

        # Test excessive ramp
        spiky = np.array([1.0, 10.0, 2.0, 15.0])
        score, violations = constraint.evaluate(spiky)
        assert score < 0
        assert "Excessive ramp rate" in str(violations)

    def test_temporal_pattern_constraint(self):
        """Test temporal pattern constraint."""
        constraint = TemporalPatternConstraint(
            min_daily_consumption=2.0, max_daily_consumption=50.0
        )

        # Test normal daily pattern (48 intervals = 1 day)
        normal_day = np.ones(48) * 0.5  # 0.5 kWh per 30min = 24 kWh/day
        score, violations = constraint.evaluate(normal_day)
        assert score == 0.0

        # Test abnormally low consumption
        low_day = np.ones(48) * 0.03  # 1.44 kWh/day
        score, violations = constraint.evaluate(low_day)
        assert score < 0
        assert "Abnormally low" in str(violations)


class TestVerifierAgent:
    """Test VerifierAgent composite validator."""

    def test_verifier_initialization(self, temp_constraints_file):
        """Test verifier initialization."""
        verifier = VerifierAgent(temp_constraints_file)

        assert len(verifier.constraints) > 0
        assert "non_negativity" in verifier.constraints
        assert "household_max" in verifier.constraints

        summary = verifier.get_constraint_summary()
        assert all(
            c in summary for c in ["non_negativity", "household_max", "ramp_rate"]
        )

    def test_verifier_evaluate(self, temp_constraints_file):
        """Test verifier evaluation."""
        verifier = VerifierAgent(temp_constraints_file)

        # Test valid forecast
        valid_forecast = np.ones(48) * 2.0  # 2 kWh per interval
        reward = verifier.evaluate(valid_forecast)
        assert (
            reward >= -0.1
        )  # Near perfect compliance (small tolerance for soft constraints)

        # Test with violations
        invalid_forecast = np.array([-1.0] * 10 + [60.0] * 10 + [2.0] * 28)
        reward, details = verifier.evaluate(invalid_forecast, return_details=True)
        assert reward < 0
        assert details["non_negativity"]["score"] < 0
        assert details["household_max"]["score"] < 0


class TestProposerAgent:
    """Test ProposerAgent scenario generation."""

    @pytest.fixture
    def proposer(self, temp_constraints_file):
        """Create proposer instance."""
        return ProposerAgent(
            temp_constraints_file, difficulty_curriculum=True, random_seed=42
        )

    def test_proposer_initialization(self, proposer):
        """Test proposer initialization."""
        assert proposer.curriculum_level == 0.0
        assert len(proposer.scenario_buffer) == 0
        assert proposer.episode_count == 0

    def test_scenario_proposal(self, proposer):
        """Test scenario proposal generation."""
        context = np.random.rand(336) * 3.0  # 7 days of data

        scenario = proposer.propose_scenario(
            historical_context=context, forecast_horizon=48
        )

        assert isinstance(scenario, ScenarioProposal)
        assert scenario.scenario_type in ProposerAgent.SCENARIO_CONFIGS
        assert scenario.duration > 0
        assert scenario.duration <= 48
        assert 0 <= scenario.difficulty_score <= 1
        assert len(scenario.baseline_context) == 336

    def test_scenario_application(self, proposer):
        """Test scenario application to time series."""
        context = np.ones(336) * 2.0
        baseline = np.ones(48) * 2.0

        scenario = proposer.propose_scenario(context, forecast_horizon=48)
        transformed = scenario.apply_to_timeseries(baseline)

        assert len(transformed) == len(baseline)
        assert np.all(transformed >= 0)  # No negative values

        # Check that transformation actually changed something
        if scenario.scenario_type != "MISSING_DATA":
            assert not np.array_equal(transformed, baseline)

    def test_learnability_reward(self, proposer):
        """Test learnability reward computation."""
        context = np.random.rand(336)
        scenario = proposer.propose_scenario(context)

        # Test unsolvable scenario
        reward_unsolvable = proposer.compute_learnability_reward(scenario, 0.0)
        assert reward_unsolvable == 0.0

        # Test perfect scenario
        reward_perfect = proposer.compute_learnability_reward(scenario, 1.0)
        assert reward_perfect == 0.0  # Too easy

        # Test ideal difficulty
        reward_ideal = proposer.compute_learnability_reward(scenario, 0.5)
        assert reward_ideal > 0

    def test_curriculum_progression(self, proposer):
        """Test curriculum level progression."""
        context = np.random.rand(336)
        initial_level = proposer.curriculum_level

        # Simulate successful episodes
        for _ in range(20):
            scenario = proposer.propose_scenario(context)
            proposer.update_buffer(scenario, 0.7)  # Good reward

        assert proposer.curriculum_level > initial_level
        assert proposer.episode_count == 20


class TestSolverAgent:
    """Test SolverAgent forecasting."""

    @pytest.fixture
    def solver(self):
        """Create solver instance."""
        config = {
            "patch_len": 8,
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "forecast_horizon": 16,
            "max_epochs": 2,
        }
        return SolverAgent(
            model_config=config,
            device="cpu",
            pretrain_epochs=0,  # Skip pretraining for tests
            use_samples=True,
        )

    def test_solver_initialization(self, solver):
        """Test solver initialization."""
        assert solver.forecast_horizon == 16
        assert not solver.is_pretrained
        assert len(solver.training_metrics) == 0

    def test_solver_predict_untrained(self, solver):
        """Test prediction with untrained model."""
        context = np.random.rand(336)

        predictions = solver.predict(context, return_quantiles=True)

        assert "0.1" in predictions
        assert "0.5" in predictions
        assert "0.9" in predictions
        assert len(predictions["0.5"]) == solver.forecast_horizon

    def test_solver_train_step(self, solver, temp_constraints_file):
        """Test single training step."""
        context = np.random.rand(336)
        target = np.random.rand(16)

        # Create a simple scenario
        from fyp.selfplay.proposer import ProposerAgent

        proposer = ProposerAgent(temp_constraints_file, random_seed=42)
        scenario = proposer.propose_scenario(context, forecast_horizon=16)

        loss = solver.train_step(
            context=context,
            target=target,
            scenario=scenario,
            verification_reward=0.5,
            alpha=0.1,
        )

        assert isinstance(loss, float)
        assert loss >= 0
        assert len(solver.training_metrics) > 0

    def test_solver_with_scenario(self, solver, temp_constraints_file):
        """Test prediction with scenario conditioning."""
        context = np.ones(336) * 2.0

        # Create EV spike scenario
        from fyp.selfplay.proposer import ProposerAgent

        proposer = ProposerAgent(temp_constraints_file, random_seed=42)
        scenario = proposer.propose_scenario(context, forecast_horizon=16)

        # Force it to be EV_SPIKE for testing
        scenario.scenario_type = "EV_SPIKE"
        scenario.magnitude = 1.5
        scenario.duration = 8

        predictions_baseline = solver.predict(context, scenario=None)
        predictions_scenario = solver.predict(context, scenario=scenario)

        # Predictions should be different with scenario (if model is available)
        # In fallback mode, predictions may be similar
        if solver.model is not None:
            assert not np.array_equal(
                predictions_baseline["0.5"], predictions_scenario["0.5"]
            )
        else:
            # In fallback mode, just verify that predictions are returned
            assert "0.5" in predictions_baseline
            assert "0.5" in predictions_scenario


class TestSelfPlayTrainer:
    """Test integrated self-play training."""

    @pytest.fixture
    def trainer_setup(self, temp_constraints_file):
        """Create trainer with all components."""
        proposer = ProposerAgent(
            temp_constraints_file, difficulty_curriculum=True, random_seed=42
        )

        solver = SolverAgent(
            model_config={
                "patch_len": 8,
                "d_model": 16,
                "n_heads": 2,
                "n_layers": 1,
                "forecast_horizon": 16,
                "max_epochs": 1,
            },
            device="cpu",
            pretrain_epochs=0,
            use_samples=True,
        )

        verifier = VerifierAgent(temp_constraints_file)

        config = {
            "alpha": 0.1,
            "batch_size": 4,
            "log_every": 2,
            "val_every": 5,
            "checkpoint_every": 10,
        }

        trainer = SelfPlayTrainer(proposer, solver, verifier, config)

        return trainer, proposer, solver, verifier

    def test_trainer_initialization(self, trainer_setup):
        """Test trainer initialization."""
        trainer, _, _, _ = trainer_setup

        assert trainer.episode_count == 0
        assert trainer.best_val_loss == float("inf")
        assert len(trainer.metrics_history) == 0

    def test_train_episode(self, trainer_setup):
        """Test single training episode."""
        trainer, _, _, _ = trainer_setup

        # Create synthetic batch
        batch = [(np.random.rand(336), np.random.rand(16)) for _ in range(4)]

        metrics = trainer.train_episode(batch)

        assert metrics["episode"] == 0
        assert len(metrics["scenarios"]) == 4
        assert len(metrics["solver_losses"]) == 4
        assert "avg_mae" in metrics
        assert "avg_solver_loss" in metrics

    def test_compute_success_rate(self, trainer_setup):
        """Test success rate computation."""
        trainer, _, _, _ = trainer_setup

        # Perfect forecast
        ground_truth = np.array([1.0, 2.0, 3.0])
        forecast = {"0.5": ground_truth.copy()}

        success_rate = trainer._compute_success_rate(forecast, ground_truth, 20.0)
        assert success_rate == 1.0

        # Poor forecast
        bad_forecast = {"0.5": ground_truth * 2}
        success_rate = trainer._compute_success_rate(bad_forecast, ground_truth, 20.0)
        assert success_rate < 0.5

    def test_validation(self, trainer_setup):
        """Test validation functionality."""
        trainer, _, _, _ = trainer_setup

        val_data = [(np.random.rand(336), np.random.rand(16)) for _ in range(10)]

        val_metrics = trainer.validate(val_data)

        assert "mae" in val_metrics
        assert "mape" in val_metrics
        assert "violation_rate" in val_metrics
        assert val_metrics["mae"] >= 0
        assert 0 <= val_metrics["violation_rate"] <= 1

    def test_full_training_loop(self, trainer_setup):
        """Test complete training loop."""
        trainer, proposer, solver, verifier = trainer_setup

        # Create minimal training data
        train_data = [
            (np.random.rand(336) * 2, np.random.rand(16) * 2) for _ in range(20)
        ]

        val_data = [(np.random.rand(336) * 2, np.random.rand(16) * 2) for _ in range(5)]

        # Train for a few episodes
        metrics_history = trainer.train(
            num_episodes=3,
            train_data=train_data,
            val_data=val_data,
            save_checkpoints=False,
        )

        assert len(metrics_history) == 3
        assert trainer.episode_count == 3
        assert proposer.episode_count > 0
        assert len(solver.training_metrics) > 0

        # Check that metrics improve (or at least don't explode)
        first_loss = metrics_history[0]["avg_solver_loss"]
        last_loss = metrics_history[-1]["avg_solver_loss"]
        assert not np.isnan(last_loss)
        assert not np.isinf(last_loss)
        # Loss should not explode (less than 10x initial loss)
        assert last_loss < first_loss * 10


class TestIntegration:
    """End-to-end integration tests."""

    def test_scenario_to_constraint_flow(self, temp_constraints_file):
        """Test full flow from scenario proposal to constraint validation."""
        # Initialize components
        proposer = ProposerAgent(temp_constraints_file, random_seed=42)
        verifier = VerifierAgent(temp_constraints_file)

        # Generate scenario
        context = np.random.rand(336) * 2
        scenario = proposer.propose_scenario(context, forecast_horizon=48)

        # Apply scenario
        baseline_forecast = np.ones(48) * np.mean(context)
        transformed_forecast = scenario.apply_to_timeseries(baseline_forecast)

        # Verify constraints
        reward, details = verifier.evaluate(
            transformed_forecast, scenario, return_details=True
        )

        # Physics valid scenarios should not have severe violations
        if scenario.physics_valid:
            assert reward > -1.0

        # Check that appropriate constraints were evaluated
        assert "non_negativity" in details
        assert "household_max" in details

    def test_adaptive_difficulty(self, temp_constraints_file):
        """Test that difficulty adapts based on performance."""
        proposer = ProposerAgent(
            temp_constraints_file, difficulty_curriculum=True, random_seed=42
        )

        initial_level = proposer.curriculum_level
        context = np.random.rand(336)

        # Simulate poor performance
        for _ in range(15):
            scenario = proposer.propose_scenario(context)
            proposer.update_buffer(scenario, 0.1)  # Low reward

        # Curriculum should not advance (or might decrease)
        assert proposer.curriculum_level <= initial_level + 0.1

        # Now simulate good performance
        for _ in range(15):
            scenario = proposer.propose_scenario(context)
            proposer.update_buffer(scenario, 0.8)  # High reward

        # Curriculum should advance
        assert proposer.curriculum_level > initial_level

    def test_checkpoint_save_load(self, temp_constraints_file):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create solver and train
            solver = SolverAgent(
                model_config={
                    "patch_len": 8,
                    "d_model": 16,
                    "n_heads": 2,
                    "n_layers": 1,
                    "forecast_horizon": 16,
                    "max_epochs": 1,
                },
                device="cpu",
                pretrain_epochs=0,
                use_samples=True,
            )

            # Do some training
            context = np.random.rand(336)
            target = np.random.rand(16)

            for _ in range(3):
                solver.train_step(context, target, verification_reward=0.5)

            # Save checkpoint
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pth")
            solver.save_checkpoint(checkpoint_path)

            # Check that checkpoint exists (could be .pth or .json depending on torch availability)
            assert os.path.exists(checkpoint_path) or os.path.exists(
                checkpoint_path.replace(".pth", ".json")
            )

            # Create new solver and load
            new_solver = SolverAgent(
                model_config=solver.model_config, device="cpu", pretrain_epochs=0
            )

            new_solver.load_checkpoint(checkpoint_path)

            # Check state was restored
            assert len(new_solver.training_metrics) == len(solver.training_metrics)
            assert new_solver.is_pretrained == solver.is_pretrained


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
