"""GNN Training pipeline for anomaly detection.

This module implements the training infrastructure for the GATVerifier model,
enabling training on synthetic anomaly data with proper loss functions,
optimizers, and checkpointing support.

The training pipeline supports:
- BCELoss for binary node classification (anomaly vs normal)
- Adam optimizer with configurable learning rate and weight decay
- Early stopping based on validation loss
- Model checkpointing for reproducibility
- Comprehensive metrics tracking (loss, accuracy, precision, recall, F1)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from fyp.gnn.gat_verifier import GATVerifier
from fyp.gnn.synthetic_dataset import SyntheticAnomalyDataset

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GNNTrainer:
    """Training pipeline for GATVerifier model.

    Implements a complete training loop with:
    - BCELoss for binary anomaly classification
    - Adam optimizer with configurable hyperparameters
    - Validation with early stopping
    - Model checkpointing
    - Comprehensive metrics tracking

    Example:
        >>> from fyp.gnn import GATVerifier, SyntheticAnomalyDataset, GNNTrainer
        >>> model = GATVerifier(temporal_features=5)
        >>> trainer = GNNTrainer(model, learning_rate=1e-3)
        >>> train_data = SyntheticAnomalyDataset(num_samples=100)
        >>> history = trainer.train(train_data, num_epochs=10)
        >>> print(f"Final loss: {history['train_loss'][-1]:.4f}")

    Attributes:
        model: GATVerifier model to train
        device: Training device (CPU or CUDA)
        optimizer: Adam optimizer
        criterion: BCELoss for binary classification
        checkpoint_dir: Directory for saving checkpoints
    """

    def __init__(
        self,
        model: GATVerifier,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        checkpoint_dir: str = "data/derived/models/gnn",
    ) -> None:
        """Initialize the GNN trainer.

        Args:
            model: GATVerifier model to train
            learning_rate: Learning rate for Adam optimizer (default 1e-3)
            weight_decay: L2 regularization strength (default 1e-4)
            device: Device for training ("cpu" or "cuda")
            checkpoint_dir: Directory for saving model checkpoints
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir

        # Move model to device
        self.model = self.model.to(self.device)

        # Create optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Create loss function (BCELoss for binary classification)
        # Model outputs sigmoid-activated scores, so we use BCELoss not BCEWithLogitsLoss
        self.criterion = nn.BCELoss()

        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized GNNTrainer: lr={learning_rate}, "
            f"weight_decay={weight_decay}, device={device}"
        )

    def train(
        self,
        train_dataset: SyntheticAnomalyDataset,
        val_dataset: SyntheticAnomalyDataset | None = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        log_every: int = 10,
    ) -> dict:
        """Train the GATVerifier model.

        Implements the main training loop with optional validation and
        early stopping. Returns a history dict with training metrics.

        Args:
            train_dataset: Training dataset (SyntheticAnomalyDataset)
            val_dataset: Optional validation dataset for early stopping
            num_epochs: Number of training epochs (default 100)
            batch_size: Batch size for DataLoader (default 32)
            early_stopping_patience: Epochs to wait before early stopping (default 10)
            log_every: Log training progress every N epochs (default 10)

        Returns:
            dict: Training history with keys:
                - train_loss: List of training losses per epoch
                - val_loss: List of validation losses (if val_dataset provided)
                - train_accuracy: List of training accuracies
                - val_accuracy: List of validation accuracies
                - best_val_accuracy: Best validation accuracy achieved
                - best_epoch: Epoch with best validation accuracy
                - stopped_early: Whether training stopped early
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

        # Initialize history
        history: dict = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "best_val_accuracy": 0.0,
            "best_epoch": 0,
            "stopped_early": False,
        }

        # Early stopping tracking
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_precision"].append(train_metrics["precision"])
            history["train_recall"].append(train_metrics["recall"])
            history["train_f1"].append(train_metrics["f1"])

            # Validation phase
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])
                history["val_precision"].append(val_metrics["precision"])
                history["val_recall"].append(val_metrics["recall"])
                history["val_f1"].append(val_metrics["f1"])

                # Track best validation accuracy
                if val_metrics["accuracy"] > history["best_val_accuracy"]:
                    history["best_val_accuracy"] = val_metrics["accuracy"]
                    history["best_epoch"] = epoch

                # Early stopping check
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(patience={early_stopping_patience})"
                        )
                        history["stopped_early"] = True
                        break

            # Logging
            if epoch % log_every == 0 or epoch == num_epochs - 1:
                log_msg = (
                    f"Epoch {epoch}/{num_epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"train_acc={train_metrics['accuracy']:.4f}"
                )
                if val_loader is not None:
                    log_msg += (
                        f", val_loss={val_metrics['loss']:.4f}, "
                        f"val_acc={val_metrics['accuracy']:.4f}"
                    )
                logger.info(log_msg)

        return history

    def _train_epoch(self, train_loader: DataLoader) -> dict:
        """Run one training epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            dict: Metrics for this epoch (loss, accuracy, precision, recall, f1)
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            # Move batch to device
            batch = batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            scores = self.model(batch.x, batch.edge_index, batch.node_type)

            # Compute loss
            # scores: [num_nodes, 1], batch.y: [num_nodes]
            loss = self.criterion(scores.squeeze(), batch.y.float())

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()

            # Get predictions (threshold at 0.5)
            preds = (scores.squeeze() > 0.5).long()
            all_preds.append(preds.detach().cpu())
            all_labels.append(batch.y.detach().cpu())

        # Compute epoch metrics
        avg_loss = total_loss / len(train_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = avg_loss

        return metrics

    def _validate_epoch(self, val_loader: DataLoader) -> dict:
        """Run one validation epoch.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            dict: Metrics for this epoch (loss, accuracy, precision, recall, f1)
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                scores = self.model(batch.x, batch.edge_index, batch.node_type)

                # Compute loss
                loss = self.criterion(scores.squeeze(), batch.y.float())

                # Track metrics
                total_loss += loss.item()

                # Get predictions
                preds = (scores.squeeze() > 0.5).long()
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())

        # Compute epoch metrics
        avg_loss = total_loss / len(val_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics["loss"] = avg_loss

        return metrics

    def _compute_metrics(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """Compute classification metrics.

        Args:
            preds: Predicted labels [N]
            labels: Ground truth labels [N]

        Returns:
            dict: Metrics (accuracy, precision, recall, f1)
        """
        # Accuracy
        correct = (preds == labels).sum().item()
        total = len(labels)
        accuracy = correct / total if total > 0 else 0.0

        # Precision, Recall, F1 for anomaly class (label=1)
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: dict,
    ) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            metrics: Current training metrics
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> dict:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            dict: Checkpoint data containing epoch, metrics, etc.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint

    def get_model(self) -> GATVerifier:
        """Get the trained model.

        Returns:
            The GATVerifier model (may be on GPU if trained on GPU)
        """
        return self.model


def train_gnn_verifier(
    temporal_features: int = 5,
    hidden_channels: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    num_epochs: int = 100,
    num_train_samples: int = 1000,
    num_val_samples: int = 200,
    num_nodes: int = 44,
    anomaly_ratio: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    device: str = "cpu",
    seed: int | None = None,
    **kwargs,
) -> tuple[GATVerifier, dict]:
    """Train a GATVerifier model from scratch.

    Convenience function that creates model, datasets, and trainer,
    then runs training and returns the trained model with metrics.

    Example:
        >>> model, history = train_gnn_verifier(
        ...     num_epochs=50,
        ...     num_train_samples=500,
        ...     num_val_samples=100,
        ... )
        >>> print(f"Best accuracy: {history['best_val_accuracy']:.2%}")

    Args:
        temporal_features: Number of input temporal features (default 5)
        hidden_channels: Hidden dimension size (default 64)
        num_layers: Number of GAT layers (default 3)
        heads: Number of attention heads (default 4)
        num_epochs: Number of training epochs (default 100)
        num_train_samples: Number of training samples (default 1000)
        num_val_samples: Number of validation samples (default 200)
        num_nodes: Nodes per graph (default 44, matches SSEN test data)
        anomaly_ratio: Fraction of samples with anomalies (default 0.5)
        learning_rate: Learning rate for Adam (default 1e-3)
        weight_decay: L2 regularization (default 1e-4)
        batch_size: Batch size for training (default 32)
        early_stopping_patience: Patience for early stopping (default 10)
        device: Training device ("cpu" or "cuda")
        seed: Random seed for reproducibility
        **kwargs: Additional arguments (ignored)

    Returns:
        tuple: (trained_model, training_history)
    """
    logger.info(
        f"Training GATVerifier: {num_train_samples} train samples, "
        f"{num_val_samples} val samples, {num_epochs} epochs"
    )

    # Create model
    model = GATVerifier(
        temporal_features=temporal_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
    )

    # Create datasets
    train_dataset = SyntheticAnomalyDataset(
        num_samples=num_train_samples,
        num_nodes=num_nodes,
        anomaly_ratio=anomaly_ratio,
        temporal_features=temporal_features,
        seed=seed,
    )

    val_dataset = SyntheticAnomalyDataset(
        num_samples=num_val_samples,
        num_nodes=num_nodes,
        anomaly_ratio=anomaly_ratio,
        temporal_features=temporal_features,
        seed=seed + 1 if seed is not None else None,
    )

    # Create trainer
    trainer = GNNTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    # Train
    history = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
    )

    return trainer.get_model(), history
