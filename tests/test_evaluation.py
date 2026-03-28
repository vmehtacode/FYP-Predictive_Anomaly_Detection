"""Test suite for the evaluation framework (benchmark and ablation modules).

Tests cover:
  - VerifierBenchmark: configuration creation, data generation, metric
    computation, result serialization, reproducibility.
  - AblationStudy: component isolation, weight sweep, early-exit sweep,
    per-anomaly-type analysis, physics compliance, significance testing.
  - End-to-end integration: benchmark -> ablation -> results JSON.

All tests use small sample sizes (10-20) for speed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fyp.evaluation.benchmark import VerifierBenchmark
from fyp.evaluation.ablation import AblationStudy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark() -> VerifierBenchmark:
    """Create a benchmark instance with small sample size for fast tests."""
    return VerifierBenchmark(seed=42, num_samples=10, num_nodes=44)


@pytest.fixture(scope="module")
def benchmark_results(benchmark: VerifierBenchmark) -> dict:
    """Run a benchmark and cache the results for the module."""
    return benchmark.run_benchmark()


@pytest.fixture(scope="module")
def ablation_study(benchmark: VerifierBenchmark) -> AblationStudy:
    """Create an ablation study instance."""
    return AblationStudy(benchmark)


# ---------------------------------------------------------------------------
# Benchmark Tests
# ---------------------------------------------------------------------------


class TestVerifierBenchmark:
    """Tests for the VerifierBenchmark class."""

    def test_benchmark_creates_all_configurations(
        self, benchmark: VerifierBenchmark,
    ) -> None:
        """Benchmark should create at least 8 configurations."""
        configs = benchmark._create_configurations()
        assert len(configs) >= 8, (
            f"Expected at least 8 configs, got {len(configs)}: "
            f"{list(configs.keys())}"
        )

    def test_benchmark_generates_test_data(
        self, benchmark: VerifierBenchmark,
    ) -> None:
        """Test data generation returns list of dicts with required keys."""
        test_data = benchmark._generate_test_data()

        assert isinstance(test_data, list)
        assert len(test_data) == benchmark.num_samples

        required_keys = {
            "forecast", "node_labels", "has_anomaly",
            "anomaly_type", "edge_index", "node_type",
        }
        for sample in test_data:
            assert isinstance(sample, dict)
            assert required_keys.issubset(
                sample.keys()
            ), f"Missing keys: {required_keys - sample.keys()}"

    def test_benchmark_runs_all_configurations(
        self, benchmark_results: dict,
    ) -> None:
        """Benchmark should produce results for all configurations."""
        configs = benchmark_results.get("configurations", {})
        assert len(configs) >= 8, (
            f"Expected at least 8 config results, got {len(configs)}"
        )

        # Check required metric keys for non-error results
        required_metrics = {"accuracy", "precision", "recall", "f1", "mean_latency_ms"}
        for name, result in configs.items():
            if "error" in result:
                continue
            assert required_metrics.issubset(
                result.keys()
            ), f"Config {name} missing metrics: {required_metrics - result.keys()}"

    def test_benchmark_results_have_required_metrics(
        self, benchmark_results: dict,
    ) -> None:
        """Each configuration result must contain standard metrics."""
        configs = benchmark_results.get("configurations", {})
        required = {"accuracy", "precision", "recall", "f1", "mean_latency_ms"}
        ranking_metrics = {"roc_auc", "pr_auc", "optimal_threshold", "optimal_f1"}

        for name, result in configs.items():
            if "error" in result:
                continue
            for metric in required:
                assert metric in result, (
                    f"Metric '{metric}' missing in '{name}' results"
                )
            # Ranking-based metrics must be present (may be None)
            for metric in ranking_metrics:
                assert metric in result, (
                    f"Ranking metric '{metric}' missing in '{name}' results"
                )

    def test_benchmark_metrics_in_valid_range(
        self, benchmark_results: dict,
    ) -> None:
        """Classification metrics should be in [0, 1]; latency should be positive."""
        configs = benchmark_results.get("configurations", {})
        bounded_metrics = ["accuracy", "precision", "recall", "f1"]

        for name, result in configs.items():
            if "error" in result:
                continue
            for m in bounded_metrics:
                val = result.get(m, 0.0)
                assert 0.0 <= val <= 1.0, (
                    f"{name}.{m} = {val} is outside [0, 1]"
                )
            latency = result.get("mean_latency_ms", 0.0)
            assert latency >= 0.0, (
                f"{name}.mean_latency_ms = {latency} is negative"
            )
            # Ranking-based metrics: valid when not None
            roc_auc = result.get("roc_auc")
            if roc_auc is not None:
                assert 0.0 <= roc_auc <= 1.0, (
                    f"{name}.roc_auc = {roc_auc} is outside [0, 1]"
                )
            pr_auc = result.get("pr_auc")
            if pr_auc is not None:
                assert 0.0 <= pr_auc <= 1.0, (
                    f"{name}.pr_auc = {pr_auc} is outside [0, 1]"
                )
            opt_f1 = result.get("optimal_f1")
            if opt_f1 is not None:
                assert 0.0 <= opt_f1 <= 1.0, (
                    f"{name}.optimal_f1 = {opt_f1} is outside [0, 1]"
                )

    def test_benchmark_reproducible_with_seed(self) -> None:
        """Running benchmark twice with the same seed should produce identical F1."""
        bench1 = VerifierBenchmark(seed=99, num_samples=10, num_nodes=44)
        bench2 = VerifierBenchmark(seed=99, num_samples=10, num_nodes=44)

        r1 = bench1.run_benchmark()
        r2 = bench2.run_benchmark()

        configs1 = r1.get("configurations", {})
        configs2 = r2.get("configurations", {})

        for name in configs1:
            if "error" in configs1[name] or name not in configs2:
                continue
            f1_1 = configs1[name].get("f1", 0.0)
            f1_2 = configs2[name].get("f1", 0.0)
            assert abs(f1_1 - f1_2) < 1e-6, (
                f"F1 mismatch for {name}: {f1_1} vs {f1_2}"
            )

    def test_benchmark_saves_json(
        self, benchmark: VerifierBenchmark, benchmark_results: dict,
    ) -> None:
        """Saving and reloading results JSON should preserve data."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w",
        ) as f:
            tmp_path = f.name

        try:
            benchmark.save_results(benchmark_results, tmp_path)

            assert os.path.exists(tmp_path)
            with open(tmp_path) as f:
                loaded = json.load(f)

            # Verify key structure
            assert "metadata" in loaded
            assert "configurations" in loaded
            assert "test_data_stats" in loaded

            # Verify a metric round-trips
            for name in loaded["configurations"]:
                if "error" in loaded["configurations"][name]:
                    continue
                assert "f1" in loaded["configurations"][name]
        finally:
            os.unlink(tmp_path)

    def test_benchmark_includes_all_eval02_baselines(
        self, benchmark_results: dict,
    ) -> None:
        """EVAL-02: benchmark must include isolation_forest, autoencoder, and decomposition."""
        configs = benchmark_results.get("configurations", {})
        required_baselines = {"isolation_forest", "autoencoder", "decomposition"}
        found = required_baselines.intersection(configs.keys())
        assert found == required_baselines, (
            f"EVAL-02 requires {required_baselines}, found only {found}"
        )


# ---------------------------------------------------------------------------
# Ablation Tests
# ---------------------------------------------------------------------------


class TestAblationStudy:
    """Tests for the AblationStudy class."""

    def test_component_isolation_runs(
        self, ablation_study: AblationStudy,
    ) -> None:
        """Component isolation should return results for all configs."""
        results = ablation_study.run_component_isolation()

        assert isinstance(results, dict)
        # Must have baseline and at least 3 single components
        assert "baseline" in results
        single_components = {"physics_only", "gnn_only", "cascade_only"}
        found = single_components.intersection(results.keys())
        assert len(found) >= 3, (
            f"Expected at least 3 single components, found: {found}"
        )

        # Each result has F1
        for name, metrics in results.items():
            assert "f1" in metrics, f"Missing F1 in {name}"

    def test_weight_sweep_runs(
        self, ablation_study: AblationStudy,
    ) -> None:
        """Weight sweep should return grid results with valid F1 values."""
        results = ablation_study.run_weight_sweep(quick=True)

        assert "grid_results" in results
        assert "optimal" in results
        assert "num_combinations" in results

        grid = results["grid_results"]
        assert len(grid) > 0, "Weight sweep grid is empty"

        for point in grid:
            assert "physics_weight" in point
            assert "gnn_weight" in point
            assert "f1" in point
            assert 0.0 <= point["f1"] <= 1.0

    def test_early_exit_sweep_runs(
        self, ablation_study: AblationStudy,
    ) -> None:
        """Early-exit sweep should return threshold vs metrics curves."""
        results = ablation_study.run_early_exit_sweep(quick=True)

        assert "sweep_results" in results
        sweep = results["sweep_results"]
        assert len(sweep) > 0, "Early-exit sweep is empty"

        for point in sweep:
            assert "threshold" in point
            assert "f1" in point
            assert "mean_latency_ms" in point
            assert 0.0 <= point["f1"] <= 1.0

    def test_per_anomaly_type_analysis_runs(
        self, ablation_study: AblationStudy,
    ) -> None:
        """Per-anomaly-type analysis should return per-type breakdown."""
        results = ablation_study.run_per_anomaly_type_analysis()

        assert isinstance(results, dict)
        assert len(results) > 0, "No anomaly types found"

        for atype, data in results.items():
            assert "count" in data, f"Missing count for {atype}"
            assert "hybrid" in data, f"Missing hybrid results for {atype}"
            assert "baseline" in data, f"Missing baseline results for {atype}"

    def test_physics_compliance_computed(
        self, ablation_study: AblationStudy,
    ) -> None:
        """Physics compliance should return rates for multiple configs."""
        results = ablation_study.compute_physics_compliance_rate()

        assert isinstance(results, dict)
        assert len(results) > 0

        for name, data in results.items():
            assert "compliance_rate" in data, f"Missing compliance_rate in {name}"
            assert "total_detected" in data
            rate = data["compliance_rate"]
            assert 0.0 <= rate <= 1.0, (
                f"{name} compliance_rate {rate} outside [0, 1]"
            )

    def test_significance_test_returns_p_value(
        self, ablation_study: AblationStudy,
    ) -> None:
        """Significance test should return a p_value between 0 and 1."""
        # Create two different score vectors
        scores_a = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
        scores_b = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]

        result = ablation_study.compute_statistical_significance(
            scores_a, scores_b, "test_comparison",
        )

        assert "p_value" in result or "ci_lower" in result
        assert "significant" in result

        if "p_value" in result and result["p_value"] is not None:
            p = result["p_value"]
            assert 0.0 <= p <= 1.0, f"p_value {p} outside [0, 1]"


# ---------------------------------------------------------------------------
# Integration Test
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_end_to_end(self) -> None:
        """Full pipeline: benchmark -> ablation -> results JSON, all in sequence."""
        # 1. Run benchmark
        benchmark = VerifierBenchmark(seed=42, num_samples=10, num_nodes=44)
        bench_results = benchmark.run_benchmark()

        assert "configurations" in bench_results
        assert len(bench_results["configurations"]) >= 8

        # 2. Run ablation
        study = AblationStudy(benchmark)
        ablation_results = study.run_full_ablation(quick=True)

        assert "component_isolation" in ablation_results
        assert "weight_sweep" in ablation_results
        assert "early_exit_sweep" in ablation_results
        assert "per_anomaly_type" in ablation_results
        assert "physics_compliance" in ablation_results
        assert "significance_tests" in ablation_results

        # 3. Save both to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            bench_path = os.path.join(tmpdir, "benchmark.json")
            ablation_path = os.path.join(tmpdir, "ablation.json")

            benchmark.save_results(bench_results, bench_path)
            study.save_results(ablation_results, ablation_path)

            # 4. Verify JSON files are valid and loadable
            assert os.path.exists(bench_path)
            assert os.path.exists(ablation_path)

            with open(bench_path) as f:
                loaded_bench = json.load(f)
            with open(ablation_path) as f:
                loaded_ablation = json.load(f)

            assert "configurations" in loaded_bench
            assert "component_isolation" in loaded_ablation
            assert "weight_sweep" in loaded_ablation

            # 5. Verify metric values are sensible
            for name, cfg in loaded_bench["configurations"].items():
                if "error" in cfg:
                    continue
                assert 0.0 <= cfg["f1"] <= 1.0
                assert cfg["mean_latency_ms"] >= 0.0
