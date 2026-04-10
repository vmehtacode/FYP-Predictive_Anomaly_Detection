"""Ablation study framework for hybrid verifier component analysis.

This module provides systematic ablation analysis for the hybrid verifier
ensemble, quantifying each component's contribution through:
  - Component isolation: testing singles, pairs, and full ensemble
  - Weight sweep: grid search over ensemble weights
  - Early-exit threshold sweep: latency vs accuracy trade-off
  - Per-anomaly-type breakdown: identifying component strengths
  - Physics compliance rate: % of detected anomalies with physics signal
  - Statistical significance: Wilcoxon signed-rank paired test

All evaluations use the VerifierBenchmark infrastructure for consistent
test data and evaluation protocol.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from fyp.evaluation.benchmark import VerifierBenchmark
from fyp.gnn.synthetic_dataset import AnomalyType
from fyp.selfplay.hybrid_verifier import HybridVerifierAgent
from fyp.selfplay.hybrid_verifier_config import (
    EnsembleWeightsConfig,
    HybridVerifierConfig,
)
from fyp.selfplay.verifier import VerifierAgent

logger = logging.getLogger(__name__)


class AblationStudy:
    """Ablation analysis for the hybrid verifier ensemble.

    Uses a VerifierBenchmark instance to generate consistent test data
    and evaluate configurations under identical conditions.

    Attributes:
        benchmark: The underlying VerifierBenchmark for data and evaluation.
        test_data: Cached test data shared across all analyses.
        graph_data: Shared graph topology for hybrid configurations.
    """

    def __init__(self, benchmark: VerifierBenchmark) -> None:
        """Initialize with a benchmark instance.

        Args:
            benchmark: VerifierBenchmark providing test data generation
                and evaluation infrastructure.
        """
        self.benchmark = benchmark
        self._test_data: list[dict] | None = None
        self._graph_data = None

    @property
    def test_data(self) -> list[dict]:
        """Lazily generate and cache test data."""
        if self._test_data is None:
            self._test_data = self.benchmark._generate_test_data()
        return self._test_data

    @property
    def graph_data(self):
        """Lazily generate and cache graph data."""
        if self._graph_data is None:
            self._graph_data = self.benchmark._create_graph_data()
        return self._graph_data

    # ------------------------------------------------------------------
    # Helper: evaluate a single config on cached test data
    # ------------------------------------------------------------------

    def _evaluate_config(
        self,
        name: str,
        config: HybridVerifierConfig,
    ) -> dict:
        """Evaluate a hybrid config and return metrics + per-sample scores.

        Args:
            name: Configuration name for reporting.
            config: HybridVerifierConfig to evaluate.

        Returns:
            Dict with standard metrics plus per_sample_scores list.
        """
        verifier = HybridVerifierAgent(
            config=config,
            graph_data=self.graph_data,
        )
        config_dict = {
            "verifier": verifier,
            "description": f"Ablation config: {name}",
            "type": "hybrid",
        }

        # Collect per-sample scores for statistical tests
        per_sample_scores: list[float] = []
        per_sample_correct: list[int] = []

        for sample in self.test_data:
            forecast = sample["forecast"]
            has_anomaly = sample["has_anomaly"]

            result = verifier.evaluate(forecast, return_details=True)
            if isinstance(result, tuple):
                reward, details = result
            else:
                reward = result
                details = {}

            # Extract anomaly score
            if "_breakdown" in details:
                breakdown = details["_breakdown"]
                combined = breakdown.get("combined_scores", np.array([]))
                if len(combined) > 0:
                    score = float(np.mean(combined))
                else:
                    score = abs(reward)
            else:
                score = abs(reward)

            per_sample_scores.append(score)
            predicted = 1 if score > 0.5 else 0
            per_sample_correct.append(1 if predicted == has_anomaly else 0)

        metrics = self.benchmark.evaluate_configuration(
            name,
            config_dict,
            self.test_data,
        )
        metrics["per_sample_scores"] = per_sample_scores
        metrics["per_sample_correct"] = per_sample_correct
        return metrics

    def _evaluate_baseline(self) -> dict:
        """Evaluate the baseline VerifierAgent.

        Returns:
            Dict with standard metrics plus per_sample_scores list.
        """
        try:
            baseline = VerifierAgent(
                ssen_constraints_path=self.benchmark.DEFAULT_SSEN_PATH,
            )
        except Exception:
            baseline = VerifierAgent(
                ssen_constraints_path="data/derived/ssen_constraints.json",
            )

        per_sample_scores: list[float] = []
        per_sample_correct: list[int] = []

        config_dict = {
            "verifier": baseline,
            "description": "Original VerifierAgent baseline",
            "type": "baseline",
        }

        for sample in self.test_data:
            forecast = sample["forecast"]
            has_anomaly = sample["has_anomaly"]

            result = baseline.evaluate(forecast, return_details=True)
            if isinstance(result, tuple):
                reward, _ = result
            else:
                reward = result

            score = abs(reward)
            per_sample_scores.append(score)
            predicted = 1 if score > 0.1 else 0
            per_sample_correct.append(1 if predicted == has_anomaly else 0)

        metrics = self.benchmark.evaluate_configuration(
            "baseline",
            config_dict,
            self.test_data,
        )
        metrics["per_sample_scores"] = per_sample_scores
        metrics["per_sample_correct"] = per_sample_correct
        return metrics

    # ------------------------------------------------------------------
    # Component Isolation
    # ------------------------------------------------------------------

    def run_component_isolation(self) -> dict:
        """Test each component alone and in pairs.

        Configurations tested:
          - Singles: physics_only, gnn_only, cascade_only
          - Pairs: physics+gnn, physics+cascade, gnn+cascade
          - Full: hybrid_full (default weights 0.4, 0.4, 0.2)
          - Baseline: VerifierAgent

        Returns:
            Dict mapping config_name -> metrics with lift vs baseline.
        """
        logger.info("Running component isolation analysis...")

        configs = {
            "physics_only": EnsembleWeightsConfig(
                physics=1.0,
                gnn=0.0,
                cascade=0.0,
            ),
            "gnn_only": EnsembleWeightsConfig(
                physics=0.0,
                gnn=1.0,
                cascade=0.0,
            ),
            "cascade_only": EnsembleWeightsConfig(
                physics=0.0,
                gnn=0.0,
                cascade=1.0,
            ),
            "physics_gnn": EnsembleWeightsConfig(
                physics=0.5,
                gnn=0.5,
                cascade=0.0,
            ),
            "physics_cascade": EnsembleWeightsConfig(
                physics=0.5,
                gnn=0.0,
                cascade=0.5,
            ),
            "gnn_cascade": EnsembleWeightsConfig(
                physics=0.0,
                gnn=0.5,
                cascade=0.5,
            ),
            "hybrid_full": EnsembleWeightsConfig(
                physics=0.4,
                gnn=0.4,
                cascade=0.2,
            ),
        }

        results: dict[str, dict] = {}

        # Evaluate baseline first
        logger.info("  Evaluating baseline...")
        results["baseline"] = self._evaluate_baseline()

        baseline_f1 = results["baseline"]["f1"]

        baseline_roc_auc = results["baseline"].get("roc_auc", 0) or 0

        # Evaluate each hybrid configuration
        for name, weights in configs.items():
            logger.info(
                "  Evaluating %s (%.1f, %.1f, %.1f)...",
                name,
                weights.physics,
                weights.gnn,
                weights.cascade,
            )
            cfg = HybridVerifierConfig(ensemble_weights=weights)
            metrics = self._evaluate_config(name, cfg)

            # Compute lift vs baseline (using optimal F1 for meaningful comparison)
            opt_f1 = metrics.get("optimal_f1") or metrics["f1"]
            baseline_opt_f1 = results["baseline"].get("optimal_f1") or baseline_f1
            if baseline_opt_f1 > 0:
                metrics["lift_vs_baseline_pct"] = (
                    (opt_f1 - baseline_opt_f1) / baseline_opt_f1 * 100
                )
            else:
                metrics["lift_vs_baseline_pct"] = 0.0

            # ROC-AUC lift
            cfg_roc = metrics.get("roc_auc") or 0
            if baseline_roc_auc > 0:
                metrics["roc_auc_lift_pct"] = (
                    (cfg_roc - baseline_roc_auc) / baseline_roc_auc * 100
                )
            else:
                metrics["roc_auc_lift_pct"] = 0.0

            results[name] = metrics

        # Also compute lift for baseline (0%)
        results["baseline"]["lift_vs_baseline_pct"] = 0.0
        results["baseline"]["roc_auc_lift_pct"] = 0.0

        logger.info("Component isolation complete: %d configs evaluated", len(results))
        return results

    # ------------------------------------------------------------------
    # Weight Sweep
    # ------------------------------------------------------------------

    def run_weight_sweep(self, quick: bool = False) -> dict:
        """Sweep ensemble weights in a grid to find optimal combination.

        Args:
            quick: If True, use coarser grid (3 points vs 5).

        Returns:
            Dict with 'grid_results' list and 'optimal' point.
        """
        logger.info("Running weight sweep (quick=%s)...", quick)

        if quick:
            weight_values = [0.2, 0.4, 0.6]
        else:
            weight_values = [0.2, 0.3, 0.4, 0.5, 0.6]

        grid_results: list[dict] = []

        for w_physics in weight_values:
            for w_gnn in weight_values:
                w_cascade = round(1.0 - w_physics - w_gnn, 2)
                if w_cascade < 0 or w_cascade > 1.0:
                    continue

                name = f"w_{w_physics}_{w_gnn}_{w_cascade}"
                weights = EnsembleWeightsConfig(
                    physics=w_physics,
                    gnn=w_gnn,
                    cascade=w_cascade,
                )
                cfg = HybridVerifierConfig(ensemble_weights=weights)

                logger.info(
                    "  Sweep: physics=%.1f gnn=%.1f cascade=%.2f",
                    w_physics,
                    w_gnn,
                    w_cascade,
                )
                metrics = self._evaluate_config(name, cfg)

                grid_results.append(
                    {
                        "physics_weight": w_physics,
                        "gnn_weight": w_gnn,
                        "cascade_weight": w_cascade,
                        "f1": metrics["f1"],
                        "roc_auc": metrics.get("roc_auc"),
                        "pr_auc": metrics.get("pr_auc"),
                        "optimal_f1": metrics.get("optimal_f1"),
                        "optimal_threshold": metrics.get("optimal_threshold"),
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "mean_latency_ms": metrics["mean_latency_ms"],
                    }
                )

        # Find optimal by ROC-AUC (threshold-independent ranking)
        if grid_results:
            optimal = max(
                grid_results,
                key=lambda x: x.get("roc_auc") or 0,
            )
        else:
            optimal = {}

        logger.info("Weight sweep complete: %d combinations tested", len(grid_results))
        if optimal:
            logger.info(
                "  Optimal: physics=%.1f gnn=%.1f cascade=%.2f -> ROC-AUC=%.4f, Opt-F1=%.4f",
                optimal.get("physics_weight", 0),
                optimal.get("gnn_weight", 0),
                optimal.get("cascade_weight", 0),
                optimal.get("roc_auc") or 0,
                optimal.get("optimal_f1") or 0,
            )

        return {
            "grid_results": grid_results,
            "optimal": optimal,
            "num_combinations": len(grid_results),
        }

    # ------------------------------------------------------------------
    # Early-Exit Threshold Sweep
    # ------------------------------------------------------------------

    def run_early_exit_sweep(self, quick: bool = False) -> dict:
        """Sweep early_exit_threshold to measure latency vs accuracy trade-off.

        Args:
            quick: If True, use fewer threshold points.

        Returns:
            Dict with 'sweep_results' list (threshold, F1, latency, exit_rate).
        """
        logger.info("Running early-exit threshold sweep...")

        if quick:
            thresholds = [0.5, 0.7, 0.9, 1.0]
        else:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

        sweep_results: list[dict] = []

        for threshold in thresholds:
            cfg = HybridVerifierConfig(early_exit_threshold=threshold)
            name = f"exit_{threshold}"

            logger.info("  Threshold=%.2f", threshold)
            metrics = self._evaluate_config(name, cfg)

            sweep_results.append(
                {
                    "threshold": threshold,
                    "f1": metrics["f1"],
                    "roc_auc": metrics.get("roc_auc"),
                    "optimal_f1": metrics.get("optimal_f1"),
                    "accuracy": metrics["accuracy"],
                    "mean_latency_ms": metrics["mean_latency_ms"],
                    "early_exit_rate": metrics.get("early_exit_rate", 0.0),
                }
            )

        logger.info(
            "Early-exit sweep complete: %d thresholds tested", len(sweep_results)
        )
        return {"sweep_results": sweep_results}

    # ------------------------------------------------------------------
    # Per-Anomaly-Type Analysis
    # ------------------------------------------------------------------

    def run_per_anomaly_type_analysis(self) -> dict:
        """Break down results by anomaly type.

        For each anomaly type (SPIKE, DROPOUT, CASCADE, RAMP_VIOLATION, NORMAL),
        computes precision, recall, F1 for hybrid_full vs baseline.

        Returns:
            Dict mapping anomaly_type -> {hybrid: metrics, baseline: metrics}.
        """
        logger.info("Running per-anomaly-type analysis...")

        # Group test data by anomaly type
        type_groups: dict[str, list[int]] = {}
        for i, sample in enumerate(self.test_data):
            atype = sample["anomaly_type"]
            if atype not in type_groups:
                type_groups[atype] = []
            type_groups[atype].append(i)

        # Create hybrid_full and baseline verifiers
        hybrid_cfg = HybridVerifierConfig()
        hybrid = HybridVerifierAgent(
            config=hybrid_cfg,
            graph_data=self.graph_data,
        )
        try:
            baseline = VerifierAgent(
                ssen_constraints_path=self.benchmark.DEFAULT_SSEN_PATH,
            )
        except Exception:
            baseline = VerifierAgent(
                ssen_constraints_path="data/derived/ssen_constraints.json",
            )

        results: dict[str, dict] = {}

        for atype, indices in type_groups.items():
            logger.info("  Anomaly type: %s (%d samples)", atype, len(indices))

            type_samples = [self.test_data[i] for i in indices]

            # Evaluate hybrid on this type
            hybrid_metrics = self._evaluate_verifier_on_subset(
                "hybrid",
                hybrid,
                type_samples,
                verifier_type="hybrid",
            )

            # Evaluate baseline on this type
            baseline_metrics = self._evaluate_verifier_on_subset(
                "baseline",
                baseline,
                type_samples,
                verifier_type="baseline",
            )

            results[atype] = {
                "count": len(indices),
                "hybrid": hybrid_metrics,
                "baseline": baseline_metrics,
            }

        logger.info("Per-anomaly-type analysis complete: %d types", len(results))
        return results

    def _evaluate_verifier_on_subset(
        self,
        name: str,
        verifier: object,
        samples: list[dict],
        verifier_type: str = "hybrid",
    ) -> dict:
        """Evaluate a verifier on a subset of test data.

        Args:
            name: Configuration name.
            verifier: Verifier instance (HybridVerifierAgent or VerifierAgent).
            samples: Subset of test data samples.
            verifier_type: "hybrid" or "baseline".

        Returns:
            Dict with precision, recall, f1 for this subset.
        """
        true_labels: list[int] = []
        pred_labels: list[int] = []

        for sample in samples:
            forecast = sample["forecast"]
            has_anomaly = sample["has_anomaly"]

            result = verifier.evaluate(forecast, return_details=True)
            if isinstance(result, tuple):
                reward, details = result
            else:
                reward = result
                details = {}

            if verifier_type == "hybrid" and "_breakdown" in details:
                combined = details["_breakdown"].get(
                    "combined_scores",
                    np.array([]),
                )
                if len(combined) > 0:
                    score = float(np.mean(combined))
                else:
                    score = abs(reward)
                predicted = 1 if score > 0.5 else 0
            elif verifier_type == "baseline":
                score = abs(reward)
                predicted = 1 if score > 0.1 else 0
            else:
                score = abs(reward)
                predicted = 1 if score > 0.5 else 0

            true_labels.append(has_anomaly)
            pred_labels.append(predicted)

        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "total": len(samples),
        }

    # ------------------------------------------------------------------
    # Physics Compliance Rate
    # ------------------------------------------------------------------

    def compute_physics_compliance_rate(self) -> dict:
        """Measure physics compliance for detected anomalies.

        For each configuration, physics compliance = fraction of detected
        anomalies that have physics_score > 0.3 (indicating the physics
        layer agrees with the anomaly detection).

        Returns:
            Dict mapping config_name -> compliance_rate.
        """
        logger.info("Computing physics compliance rates...")

        configs = {
            "hybrid_full": HybridVerifierConfig(),
            "physics_only": HybridVerifierConfig(
                ensemble_weights=EnsembleWeightsConfig(
                    physics=1.0,
                    gnn=0.0,
                    cascade=0.0,
                ),
            ),
            "gnn_only": HybridVerifierConfig(
                ensemble_weights=EnsembleWeightsConfig(
                    physics=0.0,
                    gnn=1.0,
                    cascade=0.0,
                ),
            ),
            "cascade_only": HybridVerifierConfig(
                ensemble_weights=EnsembleWeightsConfig(
                    physics=0.0,
                    gnn=0.0,
                    cascade=1.0,
                ),
            ),
        }

        results: dict[str, dict] = {}

        for name, cfg in configs.items():
            verifier = HybridVerifierAgent(
                config=cfg,
                graph_data=self.graph_data,
            )

            total_detected = 0
            physics_compliant = 0

            for sample in self.test_data:
                forecast = sample["forecast"]
                result = verifier.evaluate(forecast, return_details=True)

                if isinstance(result, tuple):
                    _, details = result
                else:
                    details = {}

                if "_breakdown" not in details:
                    continue

                breakdown = details["_breakdown"]
                combined = breakdown.get("combined_scores", np.array([]))
                physics = breakdown.get("physics_scores", np.array([]))

                if len(combined) == 0:
                    continue

                # Check if this sample was detected as anomalous
                sample_score = float(np.mean(combined))
                if sample_score > 0.5:
                    total_detected += 1
                    # Check physics compliance: physics layer has signal
                    physics_mean = float(np.mean(physics))
                    if physics_mean > 0.3:
                        physics_compliant += 1

            compliance_rate = (
                physics_compliant / total_detected if total_detected > 0 else 0.0
            )

            results[name] = {
                "total_detected": total_detected,
                "physics_compliant": physics_compliant,
                "compliance_rate": compliance_rate,
            }

            logger.info(
                "  %s: %d/%d compliant (%.1f%%)",
                name,
                physics_compliant,
                total_detected,
                compliance_rate * 100,
            )

        return results

    # ------------------------------------------------------------------
    # Statistical Significance
    # ------------------------------------------------------------------

    def compute_statistical_significance(
        self,
        config_a_scores: list[float],
        config_b_scores: list[float],
        test_name: str = "comparison",
    ) -> dict:
        """Compute Wilcoxon signed-rank test for paired samples.

        Uses scipy.stats.wilcoxon for non-parametric paired test.

        Args:
            config_a_scores: Per-sample correctness/scores for config A.
            config_b_scores: Per-sample correctness/scores for config B.
            test_name: Label for this comparison.

        Returns:
            Dict with statistic, p_value, significant (p < 0.05).
        """
        a = np.array(config_a_scores, dtype=np.float64)
        b = np.array(config_b_scores, dtype=np.float64)

        # Need at least some non-zero differences for Wilcoxon
        diffs = a - b
        non_zero = np.sum(diffs != 0)

        if non_zero < 2:
            return {
                "test_name": test_name,
                "method": "wilcoxon",
                "statistic": None,
                "p_value": 1.0,
                "significant": False,
                "note": "Insufficient non-zero differences for Wilcoxon test",
            }

        try:
            stat, p_value = wilcoxon(a, b, alternative="two-sided")
            return {
                "test_name": test_name,
                "method": "wilcoxon",
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }
        except ValueError as e:
            # Fallback to bootstrap CI if Wilcoxon fails
            return self._bootstrap_significance(a, b, test_name, str(e))

    def _bootstrap_significance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        test_name: str,
        reason: str,
        n_bootstrap: int = 1000,
    ) -> dict:
        """Bootstrap confidence interval fallback.

        Args:
            a: Scores for config A.
            b: Scores for config B.
            test_name: Label for comparison.
            reason: Why Wilcoxon was skipped.
            n_bootstrap: Number of bootstrap iterations.

        Returns:
            Dict with CI bounds and significance assessment.
        """
        rng = np.random.default_rng(42)
        diffs = a - b
        n = len(diffs)

        boot_means = np.array(
            [
                float(np.mean(rng.choice(diffs, size=n, replace=True)))
                for _ in range(n_bootstrap)
            ]
        )

        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        # Significant if CI does not contain 0
        significant = ci_lower > 0 or ci_upper < 0

        return {
            "test_name": test_name,
            "method": "bootstrap_ci",
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean_diff": float(np.mean(diffs)),
            "significant": significant,
            "note": f"Bootstrap fallback: {reason}",
        }

    # ------------------------------------------------------------------
    # Full Ablation Study
    # ------------------------------------------------------------------

    def run_full_ablation(self, quick: bool = False) -> dict:
        """Run all ablation analyses.

        Args:
            quick: If True, use quick mode for sweeps.

        Returns:
            Combined results dict with all sub-analyses.
        """
        logger.info("Starting full ablation study (quick=%s)...", quick)
        t0 = time.perf_counter()

        # 1. Component isolation
        component_isolation = self.run_component_isolation()

        # 2. Weight sweep
        weight_sweep = self.run_weight_sweep(quick=quick)

        # 3. Early-exit threshold sweep
        early_exit_sweep = self.run_early_exit_sweep(quick=quick)

        # 4. Per-anomaly-type analysis
        per_anomaly_type = self.run_per_anomaly_type_analysis()

        # 5. Physics compliance
        physics_compliance = self.compute_physics_compliance_rate()

        # 6. Statistical significance tests
        significance_tests = {}

        # hybrid_full vs baseline
        if "hybrid_full" in component_isolation and "baseline" in component_isolation:
            hybrid_correct = component_isolation["hybrid_full"].get(
                "per_sample_correct",
                [],
            )
            baseline_correct = component_isolation["baseline"].get(
                "per_sample_correct",
                [],
            )
            if hybrid_correct and baseline_correct:
                significance_tests[
                    "hybrid_vs_baseline"
                ] = self.compute_statistical_significance(
                    hybrid_correct,
                    baseline_correct,
                    "hybrid_full vs baseline",
                )

        # hybrid_full vs physics_only
        if (
            "hybrid_full" in component_isolation
            and "physics_only" in component_isolation
        ):
            hybrid_correct = component_isolation["hybrid_full"].get(
                "per_sample_correct",
                [],
            )
            physics_correct = component_isolation["physics_only"].get(
                "per_sample_correct",
                [],
            )
            if hybrid_correct and physics_correct:
                significance_tests[
                    "hybrid_vs_physics_only"
                ] = self.compute_statistical_significance(
                    hybrid_correct,
                    physics_correct,
                    "hybrid_full vs physics_only",
                )

        # hybrid_full vs gnn_only
        if "hybrid_full" in component_isolation and "gnn_only" in component_isolation:
            hybrid_correct = component_isolation["hybrid_full"].get(
                "per_sample_correct",
                [],
            )
            gnn_correct = component_isolation["gnn_only"].get(
                "per_sample_correct",
                [],
            )
            if hybrid_correct and gnn_correct:
                significance_tests[
                    "hybrid_vs_gnn_only"
                ] = self.compute_statistical_significance(
                    hybrid_correct,
                    gnn_correct,
                    "hybrid_full vs gnn_only",
                )

        duration_s = time.perf_counter() - t0

        # Clean per_sample data from results for JSON output
        clean_isolation = {}
        for name, metrics in component_isolation.items():
            clean = {
                k: v
                for k, v in metrics.items()
                if k not in ("per_sample_scores", "per_sample_correct")
            }
            clean_isolation[name] = clean

        results = {
            "component_isolation": clean_isolation,
            "weight_sweep": weight_sweep,
            "early_exit_sweep": early_exit_sweep,
            "per_anomaly_type": per_anomaly_type,
            "physics_compliance": physics_compliance,
            "significance_tests": significance_tests,
            "metadata": {
                "seed": self.benchmark.seed,
                "num_samples": self.benchmark.num_samples,
                "num_nodes": self.benchmark.num_nodes,
                "quick_mode": quick,
                "duration_seconds": round(duration_s, 2),
            },
        }

        logger.info("Full ablation study complete in %.1fs", duration_s)
        return results

    # ------------------------------------------------------------------
    # Save Results
    # ------------------------------------------------------------------

    def save_results(self, results: dict, output_path: str) -> None:
        """Save ablation results to JSON file.

        Args:
            results: Results dict from run_full_ablation().
            output_path: Path to save JSON output.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _json_serializer(obj: object) -> object:
            """Handle numpy types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, AnomalyType):
                return obj.name
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=_json_serializer)

        logger.info("Results saved to %s", path)
