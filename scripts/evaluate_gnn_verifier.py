#!/usr/bin/env python
"""Evaluate trained GATVerifier on held-out test set.

This script loads a trained GATVerifier checkpoint and evaluates it on
a held-out test set generated with a different seed, computing comprehensive
metrics including accuracy, precision, recall, F1, and confusion matrix.

Usage:
    python scripts/evaluate_gnn_verifier.py --model data/derived/models/gnn/gnn_verifier_v1.pth

Example with custom test parameters:
    python scripts/evaluate_gnn_verifier.py \
        --model data/derived/models/gnn/gnn_verifier_v1.pth \
        --test-samples 500 \
        --seed 9999 \
        --output data/derived/models/gnn/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.loader import DataLoader

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fyp.gnn import GATVerifier, SyntheticAnomalyDataset


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for evaluation script.

    Args:
        verbose: If True, set DEBUG level; else INFO level

    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def evaluate_model(
    model: GATVerifier,
    test_loader: DataLoader,
    threshold: float = 0.5,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate model on test data and compute comprehensive metrics.

    Args:
        model: Trained GATVerifier model
        test_loader: DataLoader with test data
        threshold: Classification threshold (default 0.5)
        device: Device for inference

    Returns:
        Dictionary with accuracy, precision, recall, F1, confusion matrix, etc.
    """
    model.eval()
    model = model.to(device)

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_scores: list[float] = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            scores = model(batch.x, batch.edge_index, batch.node_type)

            # Get predictions
            preds = (scores.squeeze() > threshold).int()

            # Collect results
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch.y.cpu().numpy().tolist())
            all_scores.extend(scores.squeeze().cpu().numpy().tolist())

    # Compute metrics using sklearn for reliable implementations
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Compute additional statistics
    num_samples = len(all_labels)
    num_positive = sum(all_labels)
    num_negative = num_samples - num_positive
    num_pred_positive = sum(all_preds)
    num_pred_negative = num_samples - num_pred_positive

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
        "num_samples": num_samples,
        "num_positive": num_positive,
        "num_negative": num_negative,
        "num_pred_positive": num_pred_positive,
        "num_pred_negative": num_pred_negative,
        "true_positives": int(cm[1][1]) if cm.shape[0] > 1 else 0,
        "false_positives": int(cm[0][1]) if cm.shape[1] > 1 else 0,
        "true_negatives": int(cm[0][0]),
        "false_negatives": int(cm[1][0]) if cm.shape[0] > 1 else 0,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained GATVerifier on held-out test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )

    # Test dataset parameters
    parser.add_argument(
        "--test-samples",
        type=int,
        default=500,
        help="Number of test samples",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=44,
        help="Nodes per graph (must match training)",
    )
    parser.add_argument(
        "--temporal-features",
        type=int,
        default=5,
        help="Temporal feature dimension (must match training)",
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.5,
        help="Fraction of samples with anomalies",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=9999,
        help="Random seed for test set (should differ from train/val)",
    )

    # Evaluation parameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.85,
        help="Target accuracy threshold for success",
    )

    # Model architecture (for loading)
    parser.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="Hidden channels (must match training)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=3,
        help="Number of GAT layers (must match training)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=4,
        help="Number of attention heads (must match training)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation metrics as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args()


def main() -> int:
    """Main evaluation entry point.

    Returns:
        Exit code: 0 if accuracy >= target, 1 if below target
    """
    args = parse_args()
    logger = setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("GATVerifier Evaluation Script")
    logger.info("=" * 60)

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model checkpoint not found: {model_path}")
        return 1

    # Load checkpoint
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    # Create model with matching architecture
    model = GATVerifier(
        temporal_features=args.temporal_features,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        heads=args.heads,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model loaded successfully")

    # Log training info from checkpoint if available
    if "epoch" in checkpoint:
        logger.info(f"  Trained for {checkpoint['epoch']} epochs")
    if "metrics" in checkpoint and "best_val_accuracy" in checkpoint["metrics"]:
        logger.info(f"  Training best val accuracy: {checkpoint['metrics']['best_val_accuracy']:.4f}")

    # Create held-out test dataset
    logger.info(f"Creating test dataset with {args.test_samples} samples (seed={args.seed})")
    test_dataset = SyntheticAnomalyDataset(
        num_samples=args.test_samples,
        num_nodes=args.num_nodes,
        temporal_features=args.temporal_features,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed,
    )
    logger.info(f"  Test anomaly distribution: {test_dataset.get_anomaly_statistics()}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Evaluate
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        threshold=args.threshold,
    )

    # Add test configuration to metrics
    metrics["test_config"] = {
        "test_samples": args.test_samples,
        "num_nodes": args.num_nodes,
        "temporal_features": args.temporal_features,
        "anomaly_ratio": args.anomaly_ratio,
        "seed": args.seed,
        "model_path": str(model_path),
    }

    # Report results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info("")
    logger.info(f"Test samples: {metrics['num_samples']}")
    logger.info(f"  Positive (anomaly): {metrics['num_positive']}")
    logger.info(f"  Negative (normal): {metrics['num_negative']}")
    logger.info("")
    logger.info("Confusion Matrix:")
    logger.info(f"  TN: {metrics['true_negatives']:5d}  FP: {metrics['false_positives']:5d}")
    logger.info(f"  FN: {metrics['false_negatives']:5d}  TP: {metrics['true_positives']:5d}")
    logger.info("=" * 60)

    # Check against target
    target = args.target_accuracy
    if metrics["accuracy"] >= target:
        logger.info(f"SUCCESS: Accuracy {metrics['accuracy']:.2%} >= {target:.0%} target")
        success = True
    else:
        logger.warning(f"BELOW TARGET: Accuracy {metrics['accuracy']:.2%} < {target:.0%} target")
        success = False

    # Save metrics if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved evaluation metrics to {output_path}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
