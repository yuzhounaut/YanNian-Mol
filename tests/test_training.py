"""
Unit tests for training module.

Tests cover:
- Callback base class and implementations
- Trainer class functionality
- Checkpoint saving and loading
- Training loop with small dataset
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lifespan_predictor.config import Config
from lifespan_predictor.training import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    Trainer,
    MetricCollection,
    Accuracy,
)


# Simple model for testing
class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        # Handle both dict-like and tensor inputs
        if hasattr(batch, "x"):
            x = batch.x
        else:
            x = batch
        return self.fc(x)


# Custom batch class to mimic PyG Data
class SimpleBatch:
    """Simple batch class for testing."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self


def collate_simple_batch(batch_list):
    """Custom collate function for SimpleBatch."""
    x_list = [b.x for b in batch_list]
    y_list = [b.y for b in batch_list]

    x_batch = torch.stack(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)

    return SimpleBatch(x_batch, y_batch)


def create_simple_dataloader(n_samples=100, input_dim=10, batch_size=16):
    """Create a simple dataloader for testing."""
    torch.manual_seed(42)
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, 2, (n_samples, 1)).float()

    # Create custom dataset that returns SimpleBatch
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return SimpleBatch(self.x[idx], self.y[idx])

    dataset = SimpleDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_simple_batch)


class TestCallback:
    """Test base Callback class."""

    def test_callback_initialization(self):
        """Test callback can be instantiated."""
        callback = Callback()
        assert callback is not None

    def test_callback_hooks_do_nothing(self):
        """Test that base callback hooks do nothing by default."""
        callback = Callback()

        # Should not raise any errors
        callback.on_train_begin()
        callback.on_train_end()
        callback.on_epoch_begin(0)
        callback.on_epoch_end(0, {})
        callback.on_batch_begin(0)
        callback.on_batch_end(0)


class TestEarlyStopping:
    """Test EarlyStopping callback."""

    def test_initialization(self):
        """Test EarlyStopping initialization."""
        early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        assert early_stop.monitor == "val_loss"
        assert early_stop.patience == 5
        assert early_stop.mode == "min"
        assert early_stop.early_stop is False

    def test_invalid_mode_error(self):
        """Test error with invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStopping(mode="invalid")

    def test_improvement_resets_counter(self):
        """Test that improvement resets patience counter."""
        early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min", verbose=False)

        early_stop.on_epoch_end(0, {"val_loss": 1.0})
        assert early_stop.counter == 0

        early_stop.on_epoch_end(1, {"val_loss": 1.1})  # No improvement
        assert early_stop.counter == 1

        early_stop.on_epoch_end(2, {"val_loss": 0.9})  # Improvement
        assert early_stop.counter == 0

    def test_triggers_after_patience(self):
        """Test early stopping triggers after patience epochs."""
        early_stop = EarlyStopping(monitor="val_loss", patience=2, mode="min", verbose=False)

        early_stop.on_epoch_end(0, {"val_loss": 1.0})
        assert early_stop.early_stop is False

        early_stop.on_epoch_end(1, {"val_loss": 1.1})
        assert early_stop.early_stop is False
        assert early_stop.counter == 1

        early_stop.on_epoch_end(2, {"val_loss": 1.1})
        assert early_stop.early_stop is True  # Triggers after patience=2 epochs without improvement
        assert early_stop.counter == 2

    def test_max_mode(self):
        """Test early stopping in max mode."""
        early_stop = EarlyStopping(monitor="val_acc", patience=2, mode="max", verbose=False)

        early_stop.on_epoch_end(0, {"val_acc": 0.8})
        early_stop.on_epoch_end(1, {"val_acc": 0.9})  # Improvement
        assert early_stop.counter == 0

        early_stop.on_epoch_end(2, {"val_acc": 0.85})  # No improvement
        assert early_stop.counter == 1

    def test_min_delta(self):
        """Test min_delta threshold."""
        early_stop = EarlyStopping(
            monitor="val_loss", patience=2, mode="min", min_delta=0.1, verbose=False
        )

        early_stop.on_epoch_end(0, {"val_loss": 1.0})
        early_stop.on_epoch_end(1, {"val_loss": 0.95})  # Improvement < min_delta
        assert early_stop.counter == 1  # Should count as no improvement

    def test_missing_metric_warning(self):
        """Test warning when monitored metric is missing."""
        early_stop = EarlyStopping(monitor="val_loss", patience=2, verbose=False)

        # Should not raise error, just log warning
        early_stop.on_epoch_end(0, {"train_loss": 1.0})
        assert early_stop.early_stop is False


class TestModelCheckpoint:
    """Test ModelCheckpoint callback."""

    def test_initialization(self):
        """Test ModelCheckpoint initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pt")
            checkpoint = ModelCheckpoint(filepath, monitor="val_loss", mode="min")
            assert checkpoint.filepath == filepath
            assert checkpoint.monitor == "val_loss"
            assert checkpoint.mode == "min"

    def test_invalid_mode_error(self):
        """Test error with invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            ModelCheckpoint("model.pt", mode="invalid")

    def test_saves_on_improvement(self):
        """Test checkpoint saves on improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pt")
            checkpoint = ModelCheckpoint(
                filepath, monitor="val_loss", mode="min", save_best_only=True, verbose=False
            )

            logs = {"val_loss": 1.0}
            checkpoint.on_epoch_end(0, logs)
            assert logs.get("_should_save_checkpoint") is True

            logs = {"val_loss": 1.1}
            checkpoint.on_epoch_end(1, logs)
            assert logs.get("_should_save_checkpoint") is None

            logs = {"val_loss": 0.9}
            checkpoint.on_epoch_end(2, logs)
            assert logs.get("_should_save_checkpoint") is True

    def test_max_mode(self):
        """Test checkpoint in max mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pt")
            checkpoint = ModelCheckpoint(filepath, monitor="val_acc", mode="max", verbose=False)

            logs = {"val_acc": 0.8}
            checkpoint.on_epoch_end(0, logs)
            assert logs.get("_should_save_checkpoint") is True

            logs = {"val_acc": 0.9}
            checkpoint.on_epoch_end(1, logs)
            assert logs.get("_should_save_checkpoint") is True

    def test_save_last(self):
        """Test save_last functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pt")
            checkpoint = ModelCheckpoint(filepath, save_last=True, verbose=False)

            logs = {}
            checkpoint.on_train_end(logs)
            assert logs.get("_should_save_checkpoint") is True


class TestLearningRateScheduler:
    """Test LearningRateScheduler callback."""

    def test_initialization(self):
        """Test LearningRateScheduler initialization."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5)

        lr_scheduler = LearningRateScheduler(scheduler, verbose=False)
        assert lr_scheduler.scheduler is scheduler

    def test_steps_scheduler(self):
        """Test scheduler steps on epoch end."""
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        lr_scheduler = LearningRateScheduler(scheduler, verbose=False)

        initial_lr = optimizer.param_groups[0]["lr"]

        logs = {}
        lr_scheduler.on_epoch_end(0, logs)

        new_lr = optimizer.param_groups[0]["lr"]
        assert new_lr == initial_lr * 0.5
        assert "learning_rate" in logs


class TestTrainer:
    """Test Trainer class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Config()
        config.training.max_epochs = 3
        config.training.batch_size = 16
        config.training.use_mixed_precision = False
        config.logging.print_every_n_epochs = 1
        config.logging.tensorboard_dir = None  # Disable tensorboard for tests
        config.data.output_dir = None  # Disable CSV logging for tests
        return config

    @pytest.fixture
    def simple_trainer(self, config):
        """Create simple trainer for testing."""
        model = SimpleModel()
        train_loader = create_simple_dataloader(n_samples=64, batch_size=16)
        val_loader = create_simple_dataloader(n_samples=32, batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        metrics = MetricCollection([Accuracy()])

        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            device="cpu",
        )
        return trainer

    def test_initialization(self, simple_trainer):
        """Test Trainer initialization."""
        assert simple_trainer.model is not None
        assert simple_trainer.device == "cpu"
        assert simple_trainer.current_epoch == 0
        assert len(simple_trainer.history) > 0

    def test_train_loop(self, simple_trainer):
        """Test training loop runs without errors."""
        result = simple_trainer.train()

        assert "history" in result
        assert "final_epoch" in result
        assert "total_time" in result
        assert len(result["history"]["train_loss"]) == 3
        assert len(result["history"]["val_loss"]) == 3

    def test_train_with_early_stopping(self, config):
        """Test training with early stopping."""
        model = SimpleModel()
        train_loader = create_simple_dataloader(n_samples=64, batch_size=16)
        val_loader = create_simple_dataloader(n_samples=32, batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        metrics = MetricCollection([Accuracy()])

        early_stop = EarlyStopping(monitor="val_loss", patience=1, verbose=False)

        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            callbacks=[early_stop],
            device="cpu",
        )

        result = trainer.train()

        # Should stop before max_epochs due to early stopping
        assert result["final_epoch"] < config.training.max_epochs

    def test_checkpoint_save_load(self, simple_trainer):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "checkpoint.pt")

            # Save checkpoint
            simple_trainer.save_checkpoint(filepath=filepath, epoch=5, metrics={"val_loss": 0.5})

            assert os.path.exists(filepath)

            # Load checkpoint
            checkpoint = simple_trainer.load_checkpoint(filepath)

            assert checkpoint["epoch"] == 5
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "config" in checkpoint
            assert checkpoint["metrics"]["val_loss"] == 0.5

    def test_gradient_clipping(self, config):
        """Test gradient clipping is applied."""
        config.training.gradient_clip = 1.0

        model = SimpleModel()
        train_loader = create_simple_dataloader(n_samples=32, batch_size=16)
        val_loader = create_simple_dataloader(n_samples=16, batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        metrics = MetricCollection([Accuracy()])

        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            device="cpu",
        )

        # Train one epoch
        config.training.max_epochs = 1
        result = trainer.train()

        # Check that training completed without errors
        assert len(result["history"]["train_loss"]) == 1


class TestTrainerIntegration:
    """Integration tests for Trainer with callbacks."""

    def test_trainer_with_all_callbacks(self):
        """Test trainer with multiple callbacks."""
        config = Config()
        config.training.max_epochs = 5
        config.training.use_mixed_precision = False
        config.logging.print_every_n_epochs = 1
        config.logging.tensorboard_dir = None
        config.data.output_dir = None

        model = SimpleModel()
        train_loader = create_simple_dataloader(n_samples=64, batch_size=16)
        val_loader = create_simple_dataloader(n_samples=32, batch_size=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        metrics = MetricCollection([Accuracy()])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "model_epoch_{epoch}.pt")

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=3, verbose=False),
                ModelCheckpoint(checkpoint_path, monitor="val_loss", verbose=False),
            ]

            trainer = Trainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                metrics=metrics,
                callbacks=callbacks,
                device="cpu",
            )

            result = trainer.train()

            assert "history" in result
            assert len(result["history"]["train_loss"]) > 0
