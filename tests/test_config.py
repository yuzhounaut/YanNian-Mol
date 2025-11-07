"""Tests for configuration module."""

import os
import tempfile

import pytest
import yaml

from lifespan_predictor.config import Config


class TestConfig:
    """Test configuration loading and validation."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = Config()

        assert config.data.train_csv == "train.csv"
        assert config.featurization.max_atoms == 200
        assert config.model.enable_gnn is True
        assert config.training.task == "classification"
        assert config.random_seed == 42

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "data": {"train_csv": "custom_train.csv"},
            "training": {"batch_size": 64, "learning_rate": 0.001},
            "random_seed": 123,
        }

        config = Config.from_dict(config_dict)

        assert config.data.train_csv == "custom_train.csv"
        assert config.training.batch_size == 64
        assert config.training.learning_rate == 0.001
        assert config.random_seed == 123
        # Check defaults are still applied
        assert config.featurization.max_atoms == 200

    def test_from_yaml(self):
        """Test loading config from YAML file."""
        config_dict = {
            "data": {"train_csv": "test_train.csv"},
            "training": {"max_epochs": 50},
            "random_seed": 999,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = Config.from_yaml(temp_path)

            assert config.data.train_csv == "test_train.csv"
            assert config.training.max_epochs == 50
            assert config.random_seed == 999
        finally:
            os.unlink(temp_path)

    def test_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent_config.yaml")

    def test_save_config(self):
        """Test saving config to YAML file."""
        config = Config()
        config.data.train_csv = "saved_train.csv"
        config.random_seed = 777

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_config.yaml")
            config.save(save_path)

            # Load it back
            loaded_config = Config.from_yaml(save_path)

            assert loaded_config.data.train_csv == "saved_train.csv"
            assert loaded_config.random_seed == 777

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "data" in config_dict
        assert "featurization" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
        assert "device" in config_dict
        assert "logging" in config_dict
        assert "random_seed" in config_dict

    def test_invalid_task(self):
        """Test validation error for invalid task type."""
        with pytest.raises(ValueError, match="Task must be"):
            Config.from_dict({"training": {"task": "invalid_task"}})

    def test_invalid_logging_level(self):
        """Test validation error for invalid logging level."""
        with pytest.raises(ValueError, match="Logging level must be"):
            Config.from_dict({"logging": {"level": "INVALID"}})

    def test_invalid_maccs_nbits(self):
        """Test validation error for invalid MACCS bits."""
        with pytest.raises(ValueError, match="MACCS keys must have exactly 166 bits"):
            Config.from_dict({"featurization": {"maccs_nbits": 128}})

    def test_no_branches_enabled(self):
        """Test validation error when no model branches are enabled."""
        with pytest.raises(ValueError, match="At least one model branch must be enabled"):
            Config.from_dict(
                {"model": {"enable_gnn": False, "enable_fp_dnn": False, "enable_fp_cnn": False}}
            )

    def test_invalid_metric_for_task(self):
        """Test validation error for mismatched metric and task."""
        # Regression metric for classification task
        with pytest.raises(ValueError, match="For classification, main_metric must be"):
            Config.from_dict({"training": {"task": "classification", "main_metric": "RMSE"}})

        # Classification metric for regression task
        with pytest.raises(ValueError, match="For regression, main_metric must be"):
            Config.from_dict({"training": {"task": "regression", "main_metric": "AUC"}})

    def test_get_total_fp_dim(self):
        """Test calculating total fingerprint dimension."""
        config = Config()
        total_dim = config.get_total_fp_dim()

        expected = (
            config.featurization.morgan_nbits
            + config.featurization.rdkit_fp_nbits
            + config.featurization.maccs_nbits
        )

        assert total_dim == expected
        assert total_dim == 2048 + 2048 + 166

    def test_get_device_cpu(self):
        """Test getting CPU device string."""
        config = Config.from_dict({"device": {"use_cuda": False}})
        device = config.get_device()

        assert device == "cpu"

    def test_path_expansion(self):
        """Test environment variable expansion in paths."""
        # Set a test environment variable
        os.environ["TEST_DATA_DIR"] = "/test/data"

        config = Config.from_dict({"data": {"train_csv": "$TEST_DATA_DIR/train.csv"}})

        assert config.data.train_csv == "/test/data/train.csv"

        # Clean up
        del os.environ["TEST_DATA_DIR"]

    def test_validation_ranges(self):
        """Test validation of numeric ranges."""
        # Test negative batch size
        with pytest.raises(ValueError):
            Config.from_dict({"training": {"batch_size": 0}})

        # Test invalid dropout
        with pytest.raises(ValueError):
            Config.from_dict({"model": {"gnn_dropout": 1.5}})

        # Test invalid val_split
        with pytest.raises(ValueError):
            Config.from_dict({"training": {"val_split": 1.5}})
