"""
Unit tests for dataset classes.

Tests the LifespanDataset, GraphDataBuilder, and collate functions.
"""

import os
import shutil
import tempfile

import dgl
import numpy as np
import pytest
from torch_geometric.data import Data

from lifespan_predictor.data import (
    CachedGraphFeaturizer,
    FingerprintGenerator,
    LifespanDataset,
    GraphDataBuilder,
    collate_lifespan_data,
    create_dataloader,
)


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return ["CCO", "CC", "CCC", "c1ccccc1"]


@pytest.fixture
def sample_features(sample_smiles):
    """Generate sample features for testing."""
    # Create temporary cache directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Generate graph features
        featurizer = CachedGraphFeaturizer(
            cache_dir=os.path.join(temp_dir, "features"), max_atoms=200, atom_feature_dim=75
        )
        adj, feat, sim, _ = featurizer.featurize(sample_smiles)

        # Generate fingerprints
        fp_gen = FingerprintGenerator(morgan_radius=2, morgan_nbits=2048, rdkit_fp_nbits=2048)
        hashed_fps, non_hashed_fps = fp_gen.generate_fingerprints(sample_smiles)

        # Create labels
        labels = np.array([0, 1, 0, 1], dtype=np.float32)

        yield {
            "smiles": sample_smiles,
            "adj": adj,
            "feat": feat,
            "sim": sim,
            "hashed_fps": hashed_fps,
            "non_hashed_fps": non_hashed_fps,
            "labels": labels,
            "temp_dir": temp_dir,
        }
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


class TestLifespanDataset:
    """Tests for LifespanDataset class."""

    def test_dataset_creation(self, sample_features):
        """Test basic dataset creation."""
        temp_root = os.path.join(sample_features["temp_dir"], "dataset")

        dataset = LifespanDataset(
            root=temp_root,
            smiles_list=sample_features["smiles"],
            graph_features=(
                sample_features["adj"],
                sample_features["feat"],
                sample_features["sim"],
            ),
            fingerprints=(sample_features["hashed_fps"], sample_features["non_hashed_fps"]),
            labels=sample_features["labels"],
        )

        assert len(dataset) == len(sample_features["smiles"])
        assert dataset.labels_array is not None

    def test_dataset_without_labels(self, sample_features):
        """Test dataset creation without labels."""
        temp_root = os.path.join(sample_features["temp_dir"], "dataset_no_labels")

        dataset = LifespanDataset(
            root=temp_root,
            smiles_list=sample_features["smiles"],
            graph_features=(
                sample_features["adj"],
                sample_features["feat"],
                sample_features["sim"],
            ),
            fingerprints=(sample_features["hashed_fps"], sample_features["non_hashed_fps"]),
            labels=None,
        )

        assert len(dataset) == len(sample_features["smiles"])
        assert dataset.labels_array is None

    def test_dataset_getitem(self, sample_features):
        """Test getting individual items from dataset."""
        temp_root = os.path.join(sample_features["temp_dir"], "dataset_getitem")

        dataset = LifespanDataset(
            root=temp_root,
            smiles_list=sample_features["smiles"],
            graph_features=(
                sample_features["adj"],
                sample_features["feat"],
                sample_features["sim"],
            ),
            fingerprints=(sample_features["hashed_fps"], sample_features["non_hashed_fps"]),
            labels=sample_features["labels"],
        )

        # Get first item
        data = dataset[0]

        # Check data structure
        assert isinstance(data, Data)
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "y")
        assert "_smiles" in data.__dict__
        assert hasattr(data, "hashed_fp")
        assert hasattr(data, "non_hashed_fp")

        # Check shapes
        assert data.x.shape[1] == 75  # atom_feature_dim
        assert data.edge_index.shape[0] == 2
        assert data.y.shape == (1,)
        assert data.hashed_fp.shape[0] == 4096  # morgan + rdkit
        assert data.non_hashed_fp.shape[0] == 166  # maccs

    def test_dataset_dimension_validation(self, sample_features):
        """Test dimension validation."""
        temp_root = os.path.join(sample_features["temp_dir"], "dataset_validation")

        # Test with mismatched dimensions
        with pytest.raises(ValueError, match="Dimension mismatch"):
            LifespanDataset(
                root=temp_root,
                smiles_list=sample_features["smiles"][:2],  # Only 2 SMILES
                graph_features=(
                    sample_features["adj"],  # 4 molecules
                    sample_features["feat"],
                    sample_features["sim"],
                ),
                fingerprints=(sample_features["hashed_fps"], sample_features["non_hashed_fps"]),
                labels=sample_features["labels"],
            )

    def test_dataset_label_dimension_validation(self, sample_features):
        """Test label dimension validation."""
        temp_root = os.path.join(sample_features["temp_dir"], "dataset_label_validation")

        # Test with mismatched label dimensions
        with pytest.raises(ValueError, match="Labels dimension mismatch"):
            LifespanDataset(
                root=temp_root,
                smiles_list=sample_features["smiles"],
                graph_features=(
                    sample_features["adj"],
                    sample_features["feat"],
                    sample_features["sim"],
                ),
                fingerprints=(sample_features["hashed_fps"], sample_features["non_hashed_fps"]),
                labels=np.array([0, 1]),  # Only 2 labels for 4 molecules
            )


class TestGraphDataBuilder:
    """Tests for GraphDataBuilder class."""

    def test_builder_initialization(self):
        """Test GraphDataBuilder initialization."""
        builder = GraphDataBuilder(use_edge_features=True)
        assert builder.use_edge_features is True
        assert builder.node_featurizer is not None
        assert builder.edge_featurizer is not None

    def test_build_dgl_graph(self):
        """Test building DGL graph from SMILES."""
        builder = GraphDataBuilder(use_edge_features=True)

        smiles = "CCO"
        graph = builder.build_dgl_graph(smiles, add_self_loop=True)

        # Check graph structure
        assert isinstance(graph, dgl.DGLGraph)
        assert "x" in graph.ndata
        assert "edge_attr" in graph.edata

        # Check dimensions
        assert graph.ndata["x"].shape[1] == 78  # ConvMolFeaturizer output
        assert graph.edata["edge_attr"].shape[1] == 11  # MolGraphConvFeaturizer output

    def test_build_dgl_graph_without_edge_features(self):
        """Test building DGL graph without edge features."""
        builder = GraphDataBuilder(use_edge_features=False)

        smiles = "CCO"
        graph = builder.build_dgl_graph(smiles, add_self_loop=True)

        assert isinstance(graph, dgl.DGLGraph)
        assert "x" in graph.ndata
        # Edge features should not be present
        assert "edge_attr" not in graph.edata or graph.edata["edge_attr"].shape[0] == 0

    def test_build_dgl_graph_invalid_smiles(self):
        """Test building DGL graph with invalid SMILES."""
        builder = GraphDataBuilder(use_edge_features=True)

        with pytest.raises(ValueError, match="Failed to parse SMILES"):
            builder.build_dgl_graph("INVALID_SMILES")

    def test_build_dgl_graph_from_data(self, sample_features):
        """Test building DGL graph from PyG Data object."""
        temp_root = os.path.join(sample_features["temp_dir"], "dataset_for_builder")

        # Create dataset
        dataset = LifespanDataset(
            root=temp_root,
            smiles_list=sample_features["smiles"],
            graph_features=(
                sample_features["adj"],
                sample_features["feat"],
                sample_features["sim"],
            ),
            fingerprints=(sample_features["hashed_fps"], sample_features["non_hashed_fps"]),
            labels=sample_features["labels"],
        )

        # Get a data object
        data = dataset[0]

        # Build DGL graph from data
        builder = GraphDataBuilder(use_edge_features=True)
        graph = builder.build_dgl_graph_from_data(data, add_self_loop=True)

        assert isinstance(graph, dgl.DGLGraph)
        assert "x" in graph.ndata


class TestCollateFunction:
    """Tests for collate_lifespan_data function."""

    def test_collate_function(self, sample_features):
        """Test collating a batch of Data objects."""
        temp_root = os.path.join(sample_features["temp_dir"], "dataset_for_collate")

        # Create dataset
        dataset = LifespanDataset(
            root=temp_root,
            smiles_list=sample_features["smiles"],
            graph_features=(
                sample_features["adj"],
                sample_features["feat"],
                sample_features["sim"],
            ),
            fingerprints=(sample_features["hashed_fps"], sample_features["non_hashed_fps"]),
            labels=sample_features["labels"],
        )

        # Get a batch
        batch = [dataset[i] for i in range(2)]

        # Collate
        batched_data = collate_lifespan_data(batch)

        # Check batched data
        assert hasattr(batched_data, "x")
        assert hasattr(batched_data, "edge_index")
        assert hasattr(batched_data, "y")
        assert hasattr(batched_data, "dgl_graph")

        # Check DGL graph
        assert isinstance(batched_data.dgl_graph, dgl.DGLGraph)
        assert batched_data.dgl_graph.batch_size == 2

    def test_create_dataloader(self, sample_features):
        """Test creating a DataLoader."""
        temp_root = os.path.join(sample_features["temp_dir"], "dataset_for_loader")

        # Create dataset
        dataset = LifespanDataset(
            root=temp_root,
            smiles_list=sample_features["smiles"],
            graph_features=(
                sample_features["adj"],
                sample_features["feat"],
                sample_features["sim"],
            ),
            fingerprints=(sample_features["hashed_fps"], sample_features["non_hashed_fps"]),
            labels=sample_features["labels"],
        )

        # Create dataloader
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)

        # Get a batch
        batch = next(iter(loader))

        # Check batch
        assert hasattr(batch, "x")
        assert hasattr(batch, "y")
        assert hasattr(batch, "dgl_graph")
        assert isinstance(batch.dgl_graph, dgl.DGLGraph)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
