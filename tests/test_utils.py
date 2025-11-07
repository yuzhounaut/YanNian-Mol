"""Tests for utility modules."""

import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests

import numpy as np
import torch
import torch.nn as nn

from lifespan_predictor.utils.logging import setup_logger, get_logger
from lifespan_predictor.utils.io import (
    save_results,
    load_results,
    save_checkpoint,
    load_checkpoint,
    save_pickle,
    load_pickle,
    save_json,
    load_json,
)
from lifespan_predictor.utils.visualization import (
    plot_training_curves,
    plot_predictions,
    plot_roc_curve,
)


class TestLogging:
    """Tests for logging utilities."""

    def test_setup_logger_console_only(self):
        """Test logger setup with console handler only."""
        logger = setup_logger("test_logger", level=logging.INFO)

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_setup_logger_with_file(self):
        """Test logger setup with file handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger("test_logger_file", log_file=str(log_file))

            assert len(logger.handlers) == 2  # Console + File

            # Test logging to file
            logger.info("Test message")
            assert log_file.exists()

            # Close file handler before reading (Windows requirement)
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()

            content = log_file.read_text()
            assert "Test message" in content

    def test_get_logger(self):
        """Test getting existing logger."""
        logger1 = setup_logger("test_get_logger")
        logger2 = get_logger("test_get_logger")

        assert logger1 is logger2


class TestIO:
    """Tests for I/O utilities."""

    def test_save_and_load_results(self):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = {"accuracy": 0.95, "loss": 0.1}

            # Save results
            filepath = save_results(results, tmpdir, prefix="test", include_timestamp=False)
            assert Path(filepath).exists()

            # Load results
            loaded = load_results(filepath)
            assert "results" in loaded
            assert "metadata" in loaded
            assert loaded["results"]["accuracy"] == 0.95
            assert loaded["results"]["loss"] == 0.1

    def test_save_and_load_checkpoint(self):
        """Test saving and loading model checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple model and optimizer
            model = nn.Linear(10, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Save checkpoint
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            save_checkpoint(
                model,
                optimizer,
                epoch=5,
                filepath=str(checkpoint_path),
                metrics={"train_loss": 0.1},
            )

            assert checkpoint_path.exists()

            # Create new model and load checkpoint
            new_model = nn.Linear(10, 1)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            checkpoint_data = load_checkpoint(str(checkpoint_path), new_model, new_optimizer)

            assert checkpoint_data["epoch"] == 5
            assert checkpoint_data["metrics"]["train_loss"] == 0.1

            # Verify model weights were loaded
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)

    def test_save_and_load_pickle(self):
        """Test pickle serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"array": np.array([1, 2, 3]), "value": 42}
            filepath = Path(tmpdir) / "data.pkl"

            save_pickle(data, str(filepath))
            assert filepath.exists()

            loaded = load_pickle(str(filepath))
            assert loaded["value"] == 42
            assert np.array_equal(loaded["array"], np.array([1, 2, 3]))

    def test_save_and_load_json(self):
        """Test JSON serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"learning_rate": 0.001, "batch_size": 32}
            filepath = Path(tmpdir) / "config.json"

            save_json(data, str(filepath))
            assert filepath.exists()

            loaded = load_json(str(filepath))
            assert loaded["learning_rate"] == 0.001
            assert loaded["batch_size"] == 32


class TestVisualization:
    """Tests for visualization utilities."""

    def test_plot_training_curves(self):
        """Test plotting training curves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                "train_loss": [0.5, 0.4, 0.3, 0.2],
                "val_loss": [0.6, 0.5, 0.45, 0.4],
                "train_auc": [0.7, 0.75, 0.8, 0.85],
                "val_auc": [0.65, 0.7, 0.75, 0.8],
            }

            save_path = Path(tmpdir) / "training_curves.png"
            plot_training_curves(history, str(save_path))

            assert save_path.exists()

    def test_plot_predictions_classification(self):
        """Test plotting predictions for classification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            np.random.seed(42)
            y_true = np.array([0, 0, 0, 1, 1, 1])
            y_pred = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])

            save_path = Path(tmpdir) / "predictions_class.png"
            plot_predictions(y_true, y_pred, str(save_path), task="classification")

            assert save_path.exists()

    def test_plot_predictions_regression(self):
        """Test plotting predictions for regression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])

            save_path = Path(tmpdir) / "predictions_reg.png"
            plot_predictions(y_true, y_pred, str(save_path), task="regression")

            assert save_path.exists()

    def test_plot_roc_curve(self):
        """Test plotting ROC curve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
            y_pred_proba = np.array([0.1, 0.3, 0.6, 0.7, 0.8, 0.2, 0.9, 0.4])

            save_path = Path(tmpdir) / "roc_curve.png"
            auc_score = plot_roc_curve(y_true, y_pred_proba, str(save_path))

            assert save_path.exists()
            assert 0 <= auc_score <= 1
