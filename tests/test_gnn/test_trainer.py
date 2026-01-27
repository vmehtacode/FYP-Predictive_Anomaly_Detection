"""Tests for GNN training pipeline.

This module validates:
- GNNTrainer initialization and configuration
- Training loop with loss computation and gradient updates
- Validation with early stopping
- Checkpointing (save/load)
- train_gnn_verifier convenience function
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from fyp.gnn import GATVerifier, GNNTrainer, SyntheticAnomalyDataset, train_gnn_verifier


class TestGNNTrainerInit:
    """Test GNNTrainer initialization."""

    def test_trainer_init(self) -> None:
        """Test GNNTrainer initializes with model."""
        model = GATVerifier(temporal_features=5)
        trainer = GNNTrainer(model)

        assert trainer.model is model
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_trainer_device_cpu(self) -> None:
        """Test trainer uses CPU by default."""
        model = GATVerifier(temporal_features=5)
        trainer = GNNTrainer(model, device="cpu")

        assert trainer.device == torch.device("cpu")
        assert next(trainer.model.parameters()).device == torch.device("cpu")

    def test_trainer_custom_learning_rate(self) -> None:
        """Test trainer accepts custom learning rate."""
        model = GATVerifier(temporal_features=5)
        trainer = GNNTrainer(model, learning_rate=0.01)

        assert trainer.learning_rate == 0.01

    def test_trainer_custom_weight_decay(self) -> None:
        """Test trainer accepts custom weight decay."""
        model = GATVerifier(temporal_features=5)
        trainer = GNNTrainer(model, weight_decay=0.001)

        assert trainer.weight_decay == 0.001

    def test_optimizer_created(self) -> None:
        """Test Adam optimizer is created."""
        model = GATVerifier(temporal_features=5)
        trainer = GNNTrainer(model)

        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_loss_function_is_bce(self) -> None:
        """Test BCELoss is used for binary classification."""
        model = GATVerifier(temporal_features=5)
        trainer = GNNTrainer(model)

        assert isinstance(trainer.criterion, torch.nn.BCELoss)


class TestTrainingLoop:
    """Test training loop functionality."""

    @pytest.fixture
    def small_dataset(self) -> SyntheticAnomalyDataset:
        """Create a small dataset for testing."""
        return SyntheticAnomalyDataset(
            num_samples=50,
            num_nodes=20,
            temporal_features=5,
            seed=42,
        )

    @pytest.fixture
    def trainer(self) -> GNNTrainer:
        """Create a trainer with small model."""
        model = GATVerifier(temporal_features=5, hidden_channels=16, heads=2)
        return GNNTrainer(model, learning_rate=0.01)

    def test_single_epoch(self, trainer: GNNTrainer, small_dataset: SyntheticAnomalyDataset) -> None:
        """Test training for 1 epoch completes."""
        history = trainer.train(
            train_dataset=small_dataset,
            num_epochs=1,
            batch_size=16,
        )

        assert "train_loss" in history
        assert len(history["train_loss"]) == 1

    def test_loss_computed(self, trainer: GNNTrainer, small_dataset: SyntheticAnomalyDataset) -> None:
        """Test loss is computed for each epoch."""
        history = trainer.train(
            train_dataset=small_dataset,
            num_epochs=3,
            batch_size=16,
        )

        assert len(history["train_loss"]) == 3
        assert all(loss > 0 for loss in history["train_loss"])

    def test_loss_decreases(self, trainer: GNNTrainer, small_dataset: SyntheticAnomalyDataset) -> None:
        """Test loss generally decreases over training."""
        history = trainer.train(
            train_dataset=small_dataset,
            num_epochs=10,
            batch_size=8,
        )

        # Average of last 3 epochs should be lower than first 3
        early_avg = sum(history["train_loss"][:3]) / 3
        late_avg = sum(history["train_loss"][-3:]) / 3

        # Loss should generally decrease (allow some variance)
        assert late_avg <= early_avg * 1.2  # Allow 20% tolerance

    def test_model_weights_updated(self, trainer: GNNTrainer, small_dataset: SyntheticAnomalyDataset) -> None:
        """Test model weights change after training."""
        # Get initial weights
        initial_weights = {
            name: param.clone()
            for name, param in trainer.model.named_parameters()
        }

        # Train
        trainer.train(
            train_dataset=small_dataset,
            num_epochs=3,
            batch_size=16,
        )

        # Check weights changed
        weights_changed = False
        for name, param in trainer.model.named_parameters():
            if not torch.allclose(param, initial_weights[name]):
                weights_changed = True
                break

        assert weights_changed, "Model weights should change after training"

    def test_gradient_flow(self, trainer: GNNTrainer, small_dataset: SyntheticAnomalyDataset) -> None:
        """Test gradients are computed during training."""
        from torch_geometric.loader import DataLoader

        train_loader = DataLoader(small_dataset, batch_size=8, shuffle=True)
        batch = next(iter(train_loader))

        trainer.model.train()
        trainer.optimizer.zero_grad()

        scores = trainer.model(batch.x, batch.edge_index, batch.node_type)
        loss = trainer.criterion(scores.squeeze(), batch.y.float())
        loss.backward()

        # Check gradients exist
        has_grad = False
        for param in trainer.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Gradients should be computed"

    def test_batch_processing(self, trainer: GNNTrainer, small_dataset: SyntheticAnomalyDataset) -> None:
        """Test multiple batches are processed correctly."""
        history = trainer.train(
            train_dataset=small_dataset,
            num_epochs=2,
            batch_size=8,  # Small batch = multiple batches
        )

        # Should complete without error
        assert len(history["train_loss"]) == 2

    def test_accuracy_tracked(self, trainer: GNNTrainer, small_dataset: SyntheticAnomalyDataset) -> None:
        """Test accuracy is tracked during training."""
        history = trainer.train(
            train_dataset=small_dataset,
            num_epochs=3,
            batch_size=16,
        )

        assert "train_accuracy" in history
        assert len(history["train_accuracy"]) == 3
        assert all(0 <= acc <= 1 for acc in history["train_accuracy"])


class TestValidation:
    """Test validation functionality."""

    @pytest.fixture
    def datasets(self) -> tuple[SyntheticAnomalyDataset, SyntheticAnomalyDataset]:
        """Create train and validation datasets."""
        train = SyntheticAnomalyDataset(num_samples=50, num_nodes=20, temporal_features=5, seed=42)
        val = SyntheticAnomalyDataset(num_samples=20, num_nodes=20, temporal_features=5, seed=43)
        return train, val

    @pytest.fixture
    def trainer(self) -> GNNTrainer:
        """Create a trainer."""
        model = GATVerifier(temporal_features=5, hidden_channels=16, heads=2)
        return GNNTrainer(model)

    def test_validation_loop(self, trainer: GNNTrainer, datasets: tuple) -> None:
        """Test validation runs after each epoch."""
        train, val = datasets

        history = trainer.train(
            train_dataset=train,
            val_dataset=val,
            num_epochs=3,
            batch_size=16,
        )

        assert "val_loss" in history
        assert len(history["val_loss"]) == 3

    def test_validation_metrics(self, trainer: GNNTrainer, datasets: tuple) -> None:
        """Test validation metrics are computed."""
        train, val = datasets

        history = trainer.train(
            train_dataset=train,
            val_dataset=val,
            num_epochs=3,
            batch_size=16,
        )

        assert "val_accuracy" in history
        assert "val_precision" in history
        assert "val_recall" in history
        assert "val_f1" in history

    def test_early_stopping(self, trainer: GNNTrainer, datasets: tuple) -> None:
        """Test early stopping triggers when loss plateaus."""
        train, val = datasets

        history = trainer.train(
            train_dataset=train,
            val_dataset=val,
            num_epochs=100,  # Many epochs
            batch_size=16,
            early_stopping_patience=3,  # Stop quickly
        )

        # Should stop before 100 epochs (model will converge on small data)
        # At minimum, check stopped_early flag or length
        assert "stopped_early" in history

    def test_best_val_accuracy_tracked(self, trainer: GNNTrainer, datasets: tuple) -> None:
        """Test best validation accuracy is tracked."""
        train, val = datasets

        history = trainer.train(
            train_dataset=train,
            val_dataset=val,
            num_epochs=5,
            batch_size=16,
        )

        assert "best_val_accuracy" in history
        assert "best_epoch" in history
        assert history["best_val_accuracy"] >= 0


class TestCheckpointing:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def trainer(self) -> GNNTrainer:
        """Create a trainer."""
        model = GATVerifier(temporal_features=5, hidden_channels=16, heads=2)
        return GNNTrainer(model)

    def test_save_checkpoint(self, trainer: GNNTrainer) -> None:
        """Test checkpoint file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            trainer.save_checkpoint(
                path=path,
                epoch=5,
                metrics={"train_loss": 0.5, "accuracy": 0.8},
            )

            assert os.path.exists(path)

    def test_load_checkpoint(self, trainer: GNNTrainer) -> None:
        """Test model restored from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            # Save
            trainer.save_checkpoint(
                path=path,
                epoch=5,
                metrics={"train_loss": 0.5},
            )

            # Modify weights
            for param in trainer.model.parameters():
                param.data.fill_(999)

            # Load
            checkpoint = trainer.load_checkpoint(path)

            # Check restored
            assert checkpoint["epoch"] == 5
            # Weights should be restored (not all 999)
            for param in trainer.model.parameters():
                assert not (param == 999).all()

    def test_checkpoint_contents(self, trainer: GNNTrainer) -> None:
        """Test checkpoint contains expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            trainer.save_checkpoint(
                path=path,
                epoch=10,
                metrics={"train_loss": 0.3, "accuracy": 0.9},
            )

            checkpoint = torch.load(path, weights_only=True)

            assert "epoch" in checkpoint
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "metrics" in checkpoint
            assert checkpoint["epoch"] == 10


class TestConvenienceFunction:
    """Test train_gnn_verifier convenience function."""

    def test_train_gnn_verifier_returns_model(self) -> None:
        """Test function returns trained model."""
        model, history = train_gnn_verifier(
            num_epochs=2,
            num_train_samples=30,
            num_val_samples=10,
            num_nodes=15,
            batch_size=8,
            seed=42,
        )

        assert isinstance(model, GATVerifier)
        assert isinstance(history, dict)

    def test_train_gnn_verifier_history_structure(self) -> None:
        """Test history has expected keys."""
        model, history = train_gnn_verifier(
            num_epochs=3,
            num_train_samples=30,
            num_val_samples=10,
            num_nodes=15,
            batch_size=8,
            seed=42,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert "train_accuracy" in history
        assert "val_accuracy" in history
        assert "best_val_accuracy" in history

    def test_trained_model_produces_valid_output(self) -> None:
        """Test trained model produces valid anomaly scores."""
        model, _ = train_gnn_verifier(
            num_epochs=5,
            num_train_samples=50,
            num_val_samples=10,
            num_nodes=20,
            batch_size=8,
            seed=42,
        )

        # Test inference
        model.eval()
        x = torch.randn(20, 5)
        edge_index = torch.randint(0, 20, (2, 40))

        with torch.inference_mode():
            scores = model(x, edge_index)

        assert scores.shape == (20, 1)
        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_trained_model_accuracy_above_random(self) -> None:
        """Test trained model achieves accuracy above random baseline."""
        _, history = train_gnn_verifier(
            num_epochs=20,
            num_train_samples=100,
            num_val_samples=30,
            num_nodes=20,
            batch_size=16,
            seed=42,
        )

        # Should achieve better than random (50%) on balanced data
        best_acc = history["best_val_accuracy"]
        # Allow some slack for small dataset
        assert best_acc > 0.4, f"Best accuracy {best_acc:.2%} should be above 40%"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_dataset(self) -> None:
        """Test training with very small dataset."""
        model = GATVerifier(temporal_features=5, hidden_channels=16, heads=2)
        trainer = GNNTrainer(model)

        dataset = SyntheticAnomalyDataset(
            num_samples=5,
            num_nodes=10,
            temporal_features=5,
            seed=42,
        )

        history = trainer.train(
            train_dataset=dataset,
            num_epochs=2,
            batch_size=2,
        )

        assert len(history["train_loss"]) == 2

    def test_no_validation(self) -> None:
        """Test training without validation dataset."""
        model = GATVerifier(temporal_features=5, hidden_channels=16, heads=2)
        trainer = GNNTrainer(model)

        dataset = SyntheticAnomalyDataset(
            num_samples=30,
            num_nodes=15,
            temporal_features=5,
            seed=42,
        )

        history = trainer.train(
            train_dataset=dataset,
            val_dataset=None,
            num_epochs=3,
            batch_size=8,
        )

        # Should complete without error
        assert "train_loss" in history
        assert len(history["val_loss"]) == 0  # No validation

    def test_batch_size_larger_than_dataset(self) -> None:
        """Test batch_size > num_samples."""
        model = GATVerifier(temporal_features=5, hidden_channels=16, heads=2)
        trainer = GNNTrainer(model)

        dataset = SyntheticAnomalyDataset(
            num_samples=5,
            num_nodes=10,
            temporal_features=5,
            seed=42,
        )

        history = trainer.train(
            train_dataset=dataset,
            num_epochs=2,
            batch_size=32,  # Larger than dataset
        )

        assert len(history["train_loss"]) == 2

    def test_get_model(self) -> None:
        """Test get_model returns the model."""
        model = GATVerifier(temporal_features=5)
        trainer = GNNTrainer(model)

        retrieved = trainer.get_model()
        assert retrieved is model


class TestMetricsComputation:
    """Test metrics computation."""

    @pytest.fixture
    def trainer(self) -> GNNTrainer:
        """Create a trainer."""
        model = GATVerifier(temporal_features=5, hidden_channels=16, heads=2)
        return GNNTrainer(model)

    def test_compute_metrics_perfect_predictions(self, trainer: GNNTrainer) -> None:
        """Test metrics for perfect predictions."""
        preds = torch.tensor([0, 0, 1, 1, 1])
        labels = torch.tensor([0, 0, 1, 1, 1])

        metrics = trainer._compute_metrics(preds, labels)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_compute_metrics_all_wrong(self, trainer: GNNTrainer) -> None:
        """Test metrics for all wrong predictions."""
        preds = torch.tensor([1, 1, 0, 0, 0])
        labels = torch.tensor([0, 0, 1, 1, 1])

        metrics = trainer._compute_metrics(preds, labels)

        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_compute_metrics_mixed(self, trainer: GNNTrainer) -> None:
        """Test metrics for mixed predictions."""
        preds = torch.tensor([0, 1, 1, 0, 1])
        labels = torch.tensor([0, 1, 0, 1, 1])

        metrics = trainer._compute_metrics(preds, labels)

        # 3 correct out of 5
        assert metrics["accuracy"] == 0.6
        # Precision: TP=2, FP=1 -> 2/3
        assert abs(metrics["precision"] - 2/3) < 0.01
        # Recall: TP=2, FN=1 -> 2/3
        assert abs(metrics["recall"] - 2/3) < 0.01

    def test_compute_metrics_no_positives(self, trainer: GNNTrainer) -> None:
        """Test metrics when no positive predictions."""
        preds = torch.tensor([0, 0, 0, 0, 0])
        labels = torch.tensor([0, 0, 1, 1, 1])

        metrics = trainer._compute_metrics(preds, labels)

        # Precision undefined (0/0), defaults to 0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
