"""Verifier benchmark framework for multi-configuration evaluation.

This module provides a reproducible, config-driven comparison framework that
measures accuracy, precision, recall, F1, inference latency, and early-exit
rate across HybridVerifierAgent, baseline VerifierAgent,
DecompositionAnomalyDetector, and ablated HybridVerifier configurations.

All configurations are evaluated on identical test data (same samples, same
order, same random seed) for fair comparison.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from sklearn.ensemble import IsolationForest

from fyp.baselines.anomaly import DecompositionAnomalyDetector
from fyp.models.autoencoder import AutoencoderAnomalyDetector
from fyp.gnn.synthetic_dataset import AnomalyType, SyntheticAnomalyDataset
from fyp.selfplay.hybrid_verifier import HybridVerifierAgent
from fyp.selfplay.hybrid_verifier_config import (
    EnsembleWeightsConfig,
    HybridVerifierConfig,
)
from fyp.selfplay.verifier import VerifierAgent

logger = logging.getLogger(__name__)


class VerifierBenchmark:
    """Orchestrate multi-configuration evaluation on synthetic anomaly data.

    All configurations are evaluated on the exact same test data (generated
    once with a fixed seed) to ensure fair comparison.  Metrics include
    accuracy, precision, recall, F1, mean and p95 inference latency, and
    early-exit rate (for hybrid configurations).

    Attributes:
        seed: Random seed for reproducibility.
        num_samples: Number of test samples to generate.
        num_nodes: Number of nodes per graph sample.
    """

    # Default paths
    DEFAULT_SSEN_PATH = "data/derived/ssen_constraints.json"
    DEFAULT_GNN_CHECKPOINT = "data/derived/models/gnn/gnn_verifier_v1.pth"

    def __init__(
        self,
        seed: int = 42,
        num_samples: int = 1000,
        num_nodes: int = 44,
    ) -> None:
        """Initialize benchmark with configuration parameters.

        Args:
            seed: Random seed for reproducible test data generation.
            num_samples: Total test samples (balanced across anomaly types).
            num_nodes: Nodes per graph (must match GNN training config).
        """
        self.seed = seed
        self.num_samples = num_samples
        self.num_nodes = num_nodes

        # Fix all random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _create_configurations(self) -> dict[str, dict]:
        """Create all verifier configurations for benchmarking.

        Returns:
            Dict mapping config_name -> {"verifier": instance,
            "description": str, "type": str}.
        """
        configs: dict[str, dict] = {}

        # 1. Baseline VerifierAgent (physics constraints only, original impl)
        try:
            baseline = VerifierAgent(
                ssen_constraints_path=self.DEFAULT_SSEN_PATH,
            )
            configs["baseline"] = {
                "verifier": baseline,
                "description": "Original VerifierAgent with physics constraints",
                "type": "baseline",
            }
        except Exception as e:
            logger.warning("Could not create baseline VerifierAgent: %s", e)

        # 2. HybridVerifierAgent with full ensemble (default weights)
        hybrid_config = HybridVerifierConfig()
        gnn_checkpoint = Path(self.DEFAULT_GNN_CHECKPOINT)
        has_gnn = gnn_checkpoint.exists()

        # Generate a sample graph for graph_data (shared across hybrid configs)
        graph_data = self._create_graph_data()

        try:
            hybrid_full = HybridVerifierAgent(
                config=hybrid_config,
                graph_data=graph_data,
            )
            configs["hybrid_full"] = {
                "verifier": hybrid_full,
                "description": "Full hybrid ensemble (physics + GNN + cascade)",
                "type": "hybrid",
            }
        except Exception as e:
            logger.warning("Could not create hybrid_full: %s", e)

        # 3. Physics-only ablation
        physics_only_config = HybridVerifierConfig(
            ensemble_weights=EnsembleWeightsConfig(
                physics=1.0, gnn=0.0, cascade=0.0,
            ),
        )
        try:
            physics_only = HybridVerifierAgent(
                config=physics_only_config,
                graph_data=graph_data,
            )
            configs["physics_only"] = {
                "verifier": physics_only,
                "description": "Hybrid with physics-only weights (1.0, 0.0, 0.0)",
                "type": "hybrid",
            }
        except Exception as e:
            logger.warning("Could not create physics_only: %s", e)

        # 4. GNN-only ablation
        gnn_only_config = HybridVerifierConfig(
            ensemble_weights=EnsembleWeightsConfig(
                physics=0.0, gnn=1.0, cascade=0.0,
            ),
        )
        try:
            gnn_only = HybridVerifierAgent(
                config=gnn_only_config,
                graph_data=graph_data,
            )
            configs["gnn_only"] = {
                "verifier": gnn_only,
                "description": "Hybrid with GNN-only weights (0.0, 1.0, 0.0)",
                "type": "hybrid",
            }
        except Exception as e:
            logger.warning("Could not create gnn_only: %s", e)

        # 5. Cascade-only ablation
        cascade_only_config = HybridVerifierConfig(
            ensemble_weights=EnsembleWeightsConfig(
                physics=0.0, gnn=0.0, cascade=1.0,
            ),
        )
        try:
            cascade_only = HybridVerifierAgent(
                config=cascade_only_config,
                graph_data=graph_data,
            )
            configs["cascade_only"] = {
                "verifier": cascade_only,
                "description": "Hybrid with cascade-only weights (0.0, 0.0, 1.0)",
                "type": "hybrid",
            }
        except Exception as e:
            logger.warning("Could not create cascade_only: %s", e)

        # 6. DecompositionAnomalyDetector (statistical baseline)
        decomp = DecompositionAnomalyDetector(
            seasonal_period=48,
            contamination=0.05,
        )
        configs["decomposition"] = {
            "verifier": decomp,
            "description": "Statistical baseline (seasonal decomposition)",
            "type": "decomposition",
        }

        # 7. IsolationForest (sklearn baseline)
        configs["isolation_forest"] = {
            "verifier": None,  # Evaluated via _evaluate_sklearn_baseline
            "description": "Isolation Forest anomaly detector (sklearn)",
            "type": "isolation_forest",
        }

        # 8. AutoencoderAnomalyDetector (neural baseline)
        configs["autoencoder"] = {
            "verifier": None,  # Evaluated via _evaluate_autoencoder
            "description": "Temporal autoencoder anomaly detector",
            "type": "autoencoder",
        }

        return configs

    def _create_graph_data(self) -> torch.Tensor:
        """Create a graph data object from SyntheticAnomalyDataset.

        Uses the first sample's graph structure as the reference topology
        for all hybrid verifier configurations.

        Returns:
            PyG Data object with edge_index, node_type, and num_nodes.
        """
        dataset = SyntheticAnomalyDataset(
            num_samples=1,
            num_nodes=self.num_nodes,
            seed=self.seed,
        )
        return dataset[0]

    def _generate_test_data(self) -> list[dict]:
        """Generate labeled test data from SyntheticAnomalyDataset.

        Returns:
            List of dicts with keys: forecast, labels, anomaly_type,
            edge_index, node_type, has_anomaly.
        """
        dataset = SyntheticAnomalyDataset(
            num_samples=self.num_samples,
            num_nodes=self.num_nodes,
            anomaly_ratio=0.5,
            seed=self.seed + 1000,  # Different seed from graph construction
        )

        test_data = []
        for i in range(len(dataset)):
            sample = dataset[i]

            # Convert node features to a 1-D forecast array
            # Use mean across temporal features for each node
            forecast = sample.x.mean(dim=1).numpy()

            # Ground truth: per-sample binary label
            # A sample is anomalous if any node has anomaly label
            has_anomaly = int(sample.y.sum().item() > 0)

            # Per-node labels for more granular analysis
            node_labels = sample.y.numpy()

            test_data.append({
                "forecast": forecast,
                "node_labels": node_labels,
                "has_anomaly": has_anomaly,
                "anomaly_type": sample.anomaly_type.name,
                "edge_index": sample.edge_index,
                "node_type": sample.node_type,
            })

        return test_data

    def evaluate_configuration(
        self,
        name: str,
        config: dict,
        test_data: list[dict],
    ) -> dict:
        """Evaluate a single configuration on all test data.

        Args:
            name: Configuration name (e.g. "hybrid_full").
            config: Dict with "verifier", "description", "type".
            test_data: List of test sample dicts from _generate_test_data.

        Returns:
            Results dict with accuracy, precision, recall, f1,
            mean_latency_ms, p95_latency_ms, early_exit_rate.
        """
        verifier = config["verifier"]
        config_type = config["type"]

        all_true_labels: list[int] = []
        all_pred_labels: list[int] = []
        all_scores: list[float] = []
        latencies: list[float] = []
        early_exit_counts: list[int] = []
        total_nodes = 0

        if config_type == "isolation_forest":
            return self._evaluate_sklearn_baseline(name, test_data)

        if config_type == "autoencoder":
            return self._evaluate_autoencoder(name, test_data)

        if config_type == "decomposition":
            return self._evaluate_decomposition(
                name, verifier, test_data,
            )

        # Verifier-type configs (baseline or hybrid)
        for sample in test_data:
            forecast = sample["forecast"]
            has_anomaly = sample["has_anomaly"]

            # Time the evaluation
            t0 = time.perf_counter()
            result = verifier.evaluate(
                forecast, return_details=True,
            )
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # Convert to ms

            if isinstance(result, tuple):
                reward, details = result
            else:
                reward = result
                details = {}

            # Extract anomaly score for this sample
            if config_type == "hybrid" and "_breakdown" in details:
                breakdown = details["_breakdown"]
                combined = breakdown.get("combined_scores", np.array([]))
                if len(combined) > 0:
                    sample_score = float(np.mean(combined))
                else:
                    sample_score = abs(reward)

                # Track early exits
                early_exit_count = breakdown.get("early_exit_count", 0)
                early_exit_counts.append(early_exit_count)
                total_nodes += self.num_nodes
            elif config_type == "baseline":
                # Baseline VerifierAgent: reward in [-1, 0], 0 = compliant
                # Map to anomaly score: more negative = more anomalous
                sample_score = abs(reward)
            else:
                sample_score = abs(reward)

            all_scores.append(sample_score)
            all_true_labels.append(has_anomaly)

            # Binary prediction: score > 0.5 means anomaly detected
            # For baseline: abs(reward) > 0.1 is anomaly (since 0 = compliant)
            if config_type == "baseline":
                predicted = 1 if sample_score > 0.1 else 0
            else:
                predicted = 1 if sample_score > 0.5 else 0

            all_pred_labels.append(predicted)

        return self._compute_metrics(
            name=name,
            true_labels=all_true_labels,
            pred_labels=all_pred_labels,
            scores=all_scores,
            latencies=latencies,
            early_exit_counts=early_exit_counts,
            total_nodes=total_nodes,
            config_type=config_type,
            description=config["description"],
        )

    def _evaluate_decomposition(
        self,
        name: str,
        detector: DecompositionAnomalyDetector,
        test_data: list[dict],
    ) -> dict:
        """Evaluate DecompositionAnomalyDetector on test data.

        The detector requires fit() on normal samples first, then
        predict_scores() on all samples.

        Args:
            name: Configuration name.
            detector: DecompositionAnomalyDetector instance.
            test_data: List of test sample dicts.

        Returns:
            Results dict with standard metrics.
        """
        # Collect normal samples for fitting
        normal_forecasts = [
            s["forecast"] for s in test_data if s["has_anomaly"] == 0
        ]

        if normal_forecasts:
            # Concatenate all normal forecasts for fitting
            fit_data = np.concatenate(normal_forecasts)
            detector.fit(fit_data)
        else:
            # Fallback: fit on first few samples
            fit_data = np.concatenate([s["forecast"] for s in test_data[:10]])
            detector.fit(fit_data)

        all_true_labels: list[int] = []
        all_pred_labels: list[int] = []
        all_scores: list[float] = []
        latencies: list[float] = []

        for sample in test_data:
            forecast = sample["forecast"]
            has_anomaly = sample["has_anomaly"]

            t0 = time.perf_counter()
            scores = detector.predict_scores(forecast)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)

            # Per-sample score: mean of per-node anomaly scores
            sample_score = float(np.mean(scores))
            all_scores.append(sample_score)
            all_true_labels.append(has_anomaly)

            # Binary prediction at threshold 0.5
            predicted = 1 if sample_score > 0.5 else 0
            all_pred_labels.append(predicted)

        return self._compute_metrics(
            name=name,
            true_labels=all_true_labels,
            pred_labels=all_pred_labels,
            scores=all_scores,
            latencies=latencies,
            early_exit_counts=[],
            total_nodes=0,
            config_type="decomposition",
            description="Statistical baseline (seasonal decomposition)",
        )

    def _evaluate_sklearn_baseline(
        self,
        name: str,
        test_data: list[dict],
    ) -> dict:
        """Evaluate IsolationForest (sklearn) on test data.

        Follows the same pattern as _evaluate_decomposition: collect normal
        data, fit, score all samples with timing, return _compute_metrics().

        Args:
            name: Configuration name.
            test_data: List of test sample dicts.

        Returns:
            Results dict with standard metrics.
        """
        # Collect normal samples for fitting
        normal_forecasts = [
            s["forecast"] for s in test_data if s["has_anomaly"] == 0
        ]

        if normal_forecasts:
            fit_data = np.concatenate(normal_forecasts).reshape(-1, 1)
        else:
            fit_data = np.concatenate(
                [s["forecast"] for s in test_data[:10]],
            ).reshape(-1, 1)

        # Create and fit IsolationForest
        model = IsolationForest(
            contamination=0.05,
            n_estimators=100,
            random_state=self.seed,
        )
        model.fit(fit_data)

        all_true_labels: list[int] = []
        all_pred_labels: list[int] = []
        all_scores: list[float] = []
        latencies: list[float] = []

        for sample in test_data:
            forecast = sample["forecast"]
            has_anomaly = sample["has_anomaly"]

            t0 = time.perf_counter()

            # Reshape for sklearn API (each value is a feature)
            X = forecast.reshape(-1, 1)
            raw_scores = -model.decision_function(X)

            # Normalize to [0, 1]
            score_range = raw_scores.max() - raw_scores.min() + 1e-8
            scores = np.clip(
                (raw_scores - raw_scores.min()) / score_range, 0, 1,
            )
            sample_score = float(np.mean(scores))

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            all_scores.append(sample_score)
            all_true_labels.append(has_anomaly)

            # Binary prediction at threshold 0.5
            predicted = 1 if sample_score > 0.5 else 0
            all_pred_labels.append(predicted)

        return self._compute_metrics(
            name=name,
            true_labels=all_true_labels,
            pred_labels=all_pred_labels,
            scores=all_scores,
            latencies=latencies,
            early_exit_counts=[],
            total_nodes=0,
            config_type="isolation_forest",
            description="Isolation Forest anomaly detector (sklearn)",
        )

    def _evaluate_autoencoder(
        self,
        name: str,
        test_data: list[dict],
    ) -> dict:
        """Evaluate AutoencoderAnomalyDetector on test data.

        Follows the same pattern as _evaluate_decomposition: collect normal
        data, fit, score all samples with timing, return _compute_metrics().

        Args:
            name: Configuration name.
            test_data: List of test sample dicts.

        Returns:
            Results dict with standard metrics, or error dict if training fails.
        """
        try:
            # Fix random seeds for reproducibility before training
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Create detector with window_size capped to num_nodes
            window_size = min(48, self.num_nodes)
            detector = AutoencoderAnomalyDetector(
                window_size=window_size,
                hidden_sizes=[32, 16, 8],
                max_epochs=10,
                batch_size=32,
                contamination=0.05,
            )

            # Collect normal sample forecasts as list[np.ndarray]
            normal_forecasts = [
                s["forecast"] for s in test_data if s["has_anomaly"] == 0
            ]
            if not normal_forecasts:
                normal_forecasts = [s["forecast"] for s in test_data[:10]]

            # Fit on normal data
            detector.fit(normal_forecasts)

            all_true_labels: list[int] = []
            all_pred_labels: list[int] = []
            all_scores: list[float] = []
            latencies: list[float] = []

            for sample in test_data:
                forecast = sample["forecast"]
                has_anomaly = sample["has_anomaly"]

                t0 = time.perf_counter()
                scores = detector.predict_scores(forecast)
                sample_score = float(np.mean(scores))
                t1 = time.perf_counter()

                latencies.append((t1 - t0) * 1000)

                all_scores.append(sample_score)
                all_true_labels.append(has_anomaly)

                # Binary prediction at threshold 0.5
                predicted = 1 if sample_score > 0.5 else 0
                all_pred_labels.append(predicted)

            return self._compute_metrics(
                name=name,
                true_labels=all_true_labels,
                pred_labels=all_pred_labels,
                scores=all_scores,
                latencies=latencies,
                early_exit_counts=[],
                total_nodes=0,
                config_type="autoencoder",
                description="Temporal autoencoder anomaly detector",
            )

        except Exception as e:
            logger.warning(
                "AutoencoderAnomalyDetector evaluation failed: %s", e,
            )
            return {
                "name": name,
                "error": str(e),
                "type": "autoencoder",
                "description": "Temporal autoencoder anomaly detector",
            }

    @staticmethod
    def _compute_metrics(
        *,
        name: str,
        true_labels: list[int],
        pred_labels: list[int],
        scores: list[float],
        latencies: list[float],
        early_exit_counts: list[int],
        total_nodes: int,
        config_type: str,
        description: str,
    ) -> dict:
        """Compute classification and latency metrics.

        Args:
            name: Configuration name.
            true_labels: Ground truth binary labels.
            pred_labels: Predicted binary labels.
            scores: Raw anomaly scores.
            latencies: Per-sample latency in ms.
            early_exit_counts: Per-sample early exit counts.
            total_nodes: Total node count for early exit rate.
            config_type: Type of verifier ("hybrid", "baseline", etc.).
            description: Human-readable description.

        Returns:
            Results dict.
        """
        y_true = np.array(true_labels)
        y_pred = np.array(pred_labels)

        # Classification metrics
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))

        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Latency metrics
        latency_arr = np.array(latencies) if latencies else np.array([0.0])
        mean_latency_ms = float(np.mean(latency_arr))
        p95_latency_ms = float(np.percentile(latency_arr, 95))

        # Early exit rate
        if early_exit_counts and total_nodes > 0:
            total_early_exits = sum(early_exit_counts)
            early_exit_rate = total_early_exits / total_nodes
        else:
            early_exit_rate = None

        return {
            "name": name,
            "description": description,
            "type": config_type,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_latency_ms": mean_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "early_exit_rate": early_exit_rate,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
            "num_samples": len(y_true),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
        }

    def run_benchmark(self) -> dict:
        """Run the full benchmark across all configurations.

        Returns:
            Dict with "metadata", "configurations", "test_data_stats".
        """
        logger.info("Starting VerifierBenchmark (seed=%d, samples=%d)", self.seed, self.num_samples)

        # 1. Generate test data (once, shared across all configs)
        logger.info("Generating test data...")
        test_data = self._generate_test_data()
        logger.info("Generated %d test samples", len(test_data))

        # Compute test data statistics
        anomaly_counts: dict[str, int] = {}
        num_anomalous = sum(1 for s in test_data if s["has_anomaly"])
        for s in test_data:
            atype = s["anomaly_type"]
            anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1

        test_data_stats = {
            "total_samples": len(test_data),
            "anomalous_samples": num_anomalous,
            "normal_samples": len(test_data) - num_anomalous,
            "anomaly_type_distribution": anomaly_counts,
        }

        # 2. Create configurations
        logger.info("Creating verifier configurations...")
        configurations = self._create_configurations()
        logger.info("Created %d configurations: %s", len(configurations), list(configurations.keys()))

        # 3. Evaluate each configuration
        results: dict[str, dict] = {}
        for name, config in configurations.items():
            logger.info("Evaluating configuration: %s", name)
            try:
                result = self.evaluate_configuration(name, config, test_data)
                results[name] = result
                logger.info(
                    "  %s: accuracy=%.4f, f1=%.4f, latency=%.2fms",
                    name,
                    result["accuracy"],
                    result["f1"],
                    result["mean_latency_ms"],
                )
            except Exception as e:
                logger.error("Failed to evaluate %s: %s", name, e)
                results[name] = {
                    "name": name,
                    "error": str(e),
                    "type": config.get("type", "unknown"),
                }

        # 4. Assemble final results
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": self.seed,
            "num_samples": self.num_samples,
            "num_nodes": self.num_nodes,
            "num_configurations": len(configurations),
            "configuration_names": list(configurations.keys()),
        }

        return {
            "metadata": metadata,
            "configurations": results,
            "test_data_stats": test_data_stats,
        }

    def save_results(self, results: dict, output_path: str) -> None:
        """Save benchmark results as JSON.

        Handles numpy arrays and datetime serialization.

        Args:
            results: Benchmark results dict from run_benchmark().
            output_path: Path to write the JSON file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _json_serializer(obj: object) -> object:
            """Custom JSON serializer for numpy and datetime types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, AnomalyType):
                return obj.name
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=_json_serializer)

        logger.info("Results saved to %s", path)
