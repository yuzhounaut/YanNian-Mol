"""
Integration tests for the lifespan prediction pipeline.

These tests verify that the full pipeline works end-to-end:
- Data preprocessing
- Feature generation
- Model training
- Inference
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from lifespan_predictor.config import Config
from lifespan_predictor.data.preprocessing import load_and_clean_csv
from lifespan_predictor.data.featurizers import CachedGraphFeaturizer
from lifespan_predictor.data.fingerprints import FingerprintGenerator
from lifespan_predictor.data.dataset import LifespanDataset
from lifespan_predictor.models.predictor import LifespanPredictor
from lifespan_predictor.training.trainer import Trainer
from lifespan_predictor.training.metrics import MetricCollection, AUC, Accuracy
from lifespan_predictor.training.callbacks import EarlyStopping


class TestPreprocessingPipeline:
    """Test full preprocessing pipeline."""

    @pytest.fixture
    def test_data_path(self):
        """Path to test data."""
        return Path(__file__).parent / "data" / "test_molecules.csv"

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_load_and_clean_pipeline(self, test_data_path):
        """Test loading and cleaning CSV data."""
        # Load and clean data
        df = load_and_clean_csv(
            str(test_data_path),
            smiles_column="SMILES",
            label_column="Life_extended",
            clean=True,
            drop_invalid=True,
        )

        # Verify data was loaded
        assert len(df) > 0
        assert "SMILES" in df.columns
        assert "Life_extended" in df.columns

        # Verify invalid SMILES were dropped
        assert len(df) < 15  # Original has 15 rows, 2 invalid

        # Verify all remaining SMILES are valid
        from rdkit import Chem

        for smiles in df["SMILES"]:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None

    def test_featurization_pipeline(self, test_data_path, temp_dir):
        """Test full featurization pipeline."""
        # Load data
        df = load_and_clean_csv(
            str(test_data_path),
            smiles_column="SMILES",
            label_column="Life_extended",
            clean=True,
            drop_invalid=True,
        )

        smiles_list = df["SMILES"].tolist()
        labels = df["Life_extended"].values

        # Create featurizer
        graph_cache_dir = Path(temp_dir) / "graph_features"
        featurizer = CachedGraphFeaturizer(
            cache_dir=str(graph_cache_dir), max_atoms=200, atom_feature_dim=75
        )

        # Featurize molecules
        adj_matrices, node_features, valid_labels = featurizer.featurize(
            smiles_list, labels=labels, force_recompute=False
        )

        # Verify outputs
        assert adj_matrices.shape[0] == len(smiles_list)
        assert node_features.shape[0] == len(smiles_list)
        assert valid_labels.shape[0] == len(smiles_list)
        assert adj_matrices.shape[1] == 200  # max_atoms
        assert node_features.shape[2] == 75  # atom_feature_dim

        # Test cache hit
        adj_matrices2, node_features2, valid_labels2 = featurizer.featurize(
            smiles_list, labels=labels, force_recompute=False
        )

        # Verify cached results match
        np.testing.assert_array_equal(adj_matrices, adj_matrices2)
        np.testing.assert_array_equal(node_features, node_features2)

    def test_fingerprint_pipeline(self, test_data_path, temp_dir):
        """Test fingerprint generation pipeline."""
        # Load data
        df = load_and_clean_csv(
            str(test_data_path),
            smiles_column="SMILES",
            label_column="Life_extended",
            clean=True,
            drop_invalid=True,
        )

        smiles_list = df["SMILES"].tolist()

        # Create fingerprint generator
        fp_cache_dir = Path(temp_dir) / "fingerprints"
        fp_generator = FingerprintGenerator(
            morgan_radius=2, morgan_nbits=2048, rdkit_fp_nbits=2048, n_jobs=1
        )

        # Generate fingerprints
        hashed_fps, non_hashed_fps = fp_generator.generate_fingerprints(
            smiles_list, cache_dir=str(fp_cache_dir)
        )

        # Verify outputs
        assert hashed_fps.shape[0] == len(smiles_list)
        assert non_hashed_fps.shape[0] == len(smiles_list)
        assert hashed_fps.shape[1] == 2048 + 2048  # morgan + rdkit
        assert non_hashed_fps.shape[1] == 166  # MACCS

        # Test cache hit
        hashed_fps2, non_hashed_fps2 = fp_generator.generate_fingerprints(
            smiles_list, cache_dir=str(fp_cache_dir)
        )

        # Verify cached results match
        np.testing.assert_array_equal(hashed_fps, hashed_fps2)
        np.testing.assert_array_equal(non_hashed_fps, non_hashed_fps2)


class TestTrainingPipeline:
    """Test full training pipeline."""

    @pytest.fixture
    def test_data_path(self):
        """Path to test data."""
        return Path(__file__).parent / "data" / "test_molecules.csv"

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        config = Config()
        config.data.output_dir = str(Path(temp_dir) / "output")
        config.data.graph_features_dir = str(Path(temp_dir) / "graph_features")
        config.data.fingerprints_dir = str(Path(temp_dir) / "fingerprints")
        config.training.max_epochs = 3
        config.training.batch_size = 4
        config.training.patience = 2
        config.training.use_mixed_precision = False
        config.device.use_cuda = False
        config.logging.tensorboard_dir = str(Path(temp_dir) / "runs")
        config.logging.log_file = None
        return config

    def test_dataset_creation(self, test_data_path, temp_dir, config):
        """Test dataset creation."""
        # Load and preprocess data
        df = load_and_clean_csv(
            str(test_data_path),
            smiles_column="SMILES",
            label_column="Life_extended",
            clean=True,
            drop_invalid=True,
        )

        smiles_list = df["SMILES"].tolist()
        labels = df["Life_extended"].values

        # Generate features
        featurizer = CachedGraphFeaturizer(
            cache_dir=config.data.graph_features_dir, max_atoms=200, atom_feature_dim=75
        )
        adj_matrices, node_features, valid_labels = featurizer.featurize(smiles_list, labels=labels)

        fp_generator = FingerprintGenerator(n_jobs=1)
        hashed_fps, non_hashed_fps = fp_generator.generate_fingerprints(
            smiles_list, cache_dir=config.data.fingerprints_dir
        )

        # Create dataset
        dataset = LifespanDataset(
            smiles_list=smiles_list,
            graph_features=(adj_matrices, node_features),
            fingerprints=(hashed_fps, non_hashed_fps),
            labels=valid_labels,
        )

        # Verify dataset
        assert len(dataset) == len(smiles_list)

        # Test data loader
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        assert hasattr(batch, "y")
        assert hasattr(batch, "fp_hashed")
        assert hasattr(batch, "fp_non_hashed")
        assert batch.y.shape[0] == 4

    def test_training_loop(self, test_data_path, temp_dir, config):
        """Test full training loop with small dataset."""
        # Load and preprocess data
        df = load_and_clean_csv(
            str(test_data_path),
            smiles_column="SMILES",
            label_column="Life_extended",
            clean=True,
            drop_invalid=True,
        )

        smiles_list = df["SMILES"].tolist()
        labels = df["Life_extended"].values.astype(np.float32)

        # Generate features
        featurizer = CachedGraphFeaturizer(
            cache_dir=config.data.graph_features_dir, max_atoms=200, atom_feature_dim=75
        )
        adj_matrices, node_features, valid_labels = featurizer.featurize(smiles_list, labels=labels)

        fp_generator = FingerprintGenerator(n_jobs=1)
        hashed_fps, non_hashed_fps = fp_generator.generate_fingerprints(
            smiles_list, cache_dir=config.data.fingerprints_dir
        )

        # Create dataset
        dataset = LifespanDataset(
            smiles_list=smiles_list,
            graph_features=(adj_matrices, node_features),
            fingerprints=(hashed_fps, non_hashed_fps),
            labels=valid_labels,
        )

        # Split into train/val
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        # Create model
        model = LifespanPredictor(config)

        # Create optimizer and criterion
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        # Create metrics
        metrics = MetricCollection([AUC(), Accuracy()])

        # Create trainer
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            metrics=metrics,
            callbacks=[EarlyStopping(patience=2, metric="val_loss")],
        )

        # Train
        history = trainer.train()

        # Verify training completed
        assert "history" in history
        assert "train_loss" in history["history"]
        assert "val_loss" in history["history"]
        assert len(history["history"]["train_loss"]) > 0
        assert len(history["history"]["val_loss"]) > 0

        # Verify loss decreased
        initial_loss = history["history"]["train_loss"][0]
        final_loss = history["history"]["train_loss"][-1]
        assert final_loss <= initial_loss  # Loss should not increase

    def test_checkpoint_save_load(self, test_data_path, temp_dir, config):
        """Test checkpoint saving and loading."""
        # Create simple model
        model = LifespanPredictor(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create dummy data loaders
        df = load_and_clean_csv(
            str(test_data_path),
            smiles_column="SMILES",
            label_column="Life_extended",
            clean=True,
            drop_invalid=True,
        )

        smiles_list = df["SMILES"].tolist()[:5]  # Use only 5 molecules
        labels = df["Life_extended"].values[:5].astype(np.float32)

        featurizer = CachedGraphFeaturizer(
            cache_dir=config.data.graph_features_dir, max_atoms=200, atom_feature_dim=75
        )
        adj_matrices, node_features, valid_labels = featurizer.featurize(smiles_list, labels=labels)

        fp_generator = FingerprintGenerator(n_jobs=1)
        hashed_fps, non_hashed_fps = fp_generator.generate_fingerprints(
            smiles_list, cache_dir=config.data.fingerprints_dir
        )

        dataset = LifespanDataset(
            smiles_list=smiles_list,
            graph_features=(adj_matrices, node_features),
            fingerprints=(hashed_fps, non_hashed_fps),
            labels=valid_labels,
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Create trainer
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            criterion=torch.nn.BCEWithLogitsLoss(),
            metrics=MetricCollection([AUC()]),
        )

        # Save checkpoint
        checkpoint_path = Path(temp_dir) / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=5, metrics={"val_loss": 0.5})

        # Verify checkpoint exists
        assert checkpoint_path.exists()

        # Load checkpoint
        checkpoint = trainer.load_checkpoint(str(checkpoint_path))

        # Verify checkpoint contents
        assert checkpoint["epoch"] == 5
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "config" in checkpoint
        assert checkpoint["metrics"]["val_loss"] == 0.5


class TestInferencePipeline:
    """Test inference pipeline."""

    @pytest.fixture
    def test_data_path(self):
        """Path to test data."""
        return Path(__file__).parent / "data" / "test_molecules.csv"

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create test configuration."""
        config = Config()
        config.data.graph_features_dir = str(Path(temp_dir) / "graph_features")
        config.data.fingerprints_dir = str(Path(temp_dir) / "fingerprints")
        config.device.use_cuda = False
        return config

    def test_inference_on_new_molecules(self, test_data_path, temp_dir, config):
        """Test inference on new molecules."""
        # Load data
        df = load_and_clean_csv(
            str(test_data_path),
            smiles_column="SMILES",
            label_column="Life_extended",
            clean=True,
            drop_invalid=True,
        )

        smiles_list = df["SMILES"].tolist()[:5]

        # Generate features
        featurizer = CachedGraphFeaturizer(
            cache_dir=config.data.graph_features_dir, max_atoms=200, atom_feature_dim=75
        )
        adj_matrices, node_features, _ = featurizer.featurize(smiles_list)

        fp_generator = FingerprintGenerator(n_jobs=1)
        hashed_fps, non_hashed_fps = fp_generator.generate_fingerprints(
            smiles_list, cache_dir=config.data.fingerprints_dir
        )

        # Create dataset (no labels for inference)
        dataset = LifespanDataset(
            smiles_list=smiles_list,
            graph_features=(adj_matrices, node_features),
            fingerprints=(hashed_fps, non_hashed_fps),
            labels=None,
        )

        # Create data loader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Create model
        model = LifespanPredictor(config)
        model.eval()

        # Run inference
        predictions = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(config.get_device())
                pred = model(batch)
                predictions.append(pred.cpu())

        predictions = torch.cat(predictions, dim=0).numpy()

        # Verify predictions
        assert predictions.shape[0] == len(smiles_list)
        assert predictions.shape[1] == 1

        # Verify predictions are in reasonable range
        assert np.all(np.isfinite(predictions))


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def edge_cases_path(self):
        """Path to edge cases data."""
        return Path(__file__).parent / "data" / "test_edge_cases.csv"

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_invalid_smiles_handling(self, edge_cases_path, temp_dir):
        """Test handling of invalid SMILES."""
        # Load data with invalid SMILES
        df = load_and_clean_csv(
            str(edge_cases_path),
            smiles_column="SMILES",
            label_column="Life_extended",
            clean=True,
            drop_invalid=True,
        )

        # Verify invalid SMILES were dropped
        assert len(df) < 10  # Original has 10 rows, some invalid

        # Verify all remaining SMILES are valid
        from rdkit import Chem

        for smiles in df["SMILES"]:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None

    def test_empty_dataset(self, temp_dir):
        """Test handling of empty dataset."""
        # Create empty CSV
        empty_csv = Path(temp_dir) / "empty.csv"
        pd.DataFrame(columns=["SMILES", "Life_extended"]).to_csv(empty_csv, index=False)

        # Load empty data
        df = load_and_clean_csv(
            str(empty_csv), smiles_column="SMILES", label_column="Life_extended"
        )

        # Verify empty dataframe
        assert len(df) == 0
