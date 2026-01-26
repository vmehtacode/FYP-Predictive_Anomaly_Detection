#!/usr/bin/env python3
"""
Basic test for large-scale experiment components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from src.fyp.baselines.forecasting import (
            BaseForecaster,
            LinearTrendForecaster,
            SeasonalNaive,
        )
        from src.fyp.config import ExperimentConfig, ForecastingConfig
        from src.fyp.data_loader import EnergyDataLoader
        from src.fyp.metrics import (
            mean_absolute_error,
            mean_absolute_percentage_error,
            root_mean_squared_error,
        )
        from src.fyp.models.patchtst import PatchTSTForecaster
        from src.fyp.selfplay import (
            ProposerAgent,
            SelfPlayTrainer,
            SolverAgent,
            VerifierAgent,
        )
        from src.fyp.selfplay.utils import create_sliding_windows
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_data_loading():
    """Test basic data loading."""
    print("\nTesting data loading...")
    try:
        import polars as pl
        
        lcl_path = Path("data/processed/lcl_data")
        if not lcl_path.exists():
            print(f"✗ Data path not found: {lcl_path}")
            return False
            
        # Load a small sample
        df = pl.read_parquet(str(lcl_path / "batch_000001.parquet"), n_rows=100)
        print(f"✓ Loaded {len(df)} rows with columns: {list(df.columns)}")
        print(f"  Households: {df['entity_id'].n_unique()}")
        print(f"  Date range: {df['ts_utc'].min()} to {df['ts_utc'].max()}")
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return False


def test_baseline_models():
    """Test baseline model instantiation."""
    print("\nTesting baseline models...")
    try:
        import numpy as np
        from scripts.run_large_scale_experiment import (
            NaiveForecaster,
            MovingAverageForecaster,
        )
        
        # Create dummy data
        history = np.random.randn(100) + 1.0
        
        # Test naive
        naive = NaiveForecaster()
        naive.fit(history)
        pred = naive.predict(history, steps=10)
        print(f"✓ Naive forecaster: predicted shape {pred.shape}")
        
        # Test moving average
        ma = MovingAverageForecaster(window_size=48)
        ma.fit(history)
        pred = ma.predict(history, steps=10)
        print(f"✓ Moving average: predicted shape {pred.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Baseline model error: {e}")
        return False


def test_self_play_components():
    """Test self-play component instantiation."""
    print("\nTesting self-play components...")
    try:
        from src.fyp.selfplay import (
            ProposerAgent,
            SolverAgent,
            VerifierAgent,
            SelfPlayTrainer,
        )
        
        # Create agents
        proposer = ProposerAgent(
            ssen_constraints_path="data/derived/ssen_constraints.json",
            difficulty_curriculum=True,
        )
        print("✓ ProposerAgent created")
        
        solver = SolverAgent(
            model_config={"forecast_horizon": 48},
            pretrain_epochs=0,
        )
        print("✓ SolverAgent created")
        
        verifier = VerifierAgent(
            ssen_constraints_path="data/derived/ssen_constraints.json"
        )
        print("✓ VerifierAgent created")
        
        trainer = SelfPlayTrainer(proposer, solver, verifier)
        print("✓ SelfPlayTrainer created")
        
        return True
    except Exception as e:
        print(f"✗ Self-play component error: {e}")
        return False


def test_mlflow_setup():
    """Test MLflow setup."""
    print("\nTesting MLflow setup...")
    try:
        import mlflow
        from pathlib import Path
        
        mlflow_dir = Path("results/test_mlflow")
        mlflow_dir.mkdir(parents=True, exist_ok=True)
        
        mlflow.set_tracking_uri(str(mlflow_dir))
        mlflow.set_experiment("test_experiment")
        
        with mlflow.start_run():
            mlflow.log_metric("test_metric", 1.0)
            
        print("✓ MLflow setup successful")
        return True
    except Exception as e:
        print(f"✗ MLflow error: {e}")
        return False


def main():
    """Run all tests."""
    print("Running basic tests for large-scale experiment...\n")
    
    tests = [
        test_imports,
        test_data_loading,
        test_baseline_models,
        test_self_play_components,
        test_mlflow_setup,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! The experiment script should work.")
    else:
        print("\n✗ Some tests failed. Please fix the issues before running the full experiment.")
        
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
