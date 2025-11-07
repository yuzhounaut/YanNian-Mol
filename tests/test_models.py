"""
Unit tests for model architecture.

Tests the AttentiveFPModule, LifespanPredictor, and weight initialization utilities.
"""

import os
import shutil
import tempfile

import dgl
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from lifespan_predictor.config import Config
from lifespan_predictor.data import (
    CachedGraphFeaturizer,
    FingerprintGenerator,
    LifespanDataset,
    create_dataloader,
)
from lifespan_predictor.models import (
    AttentiveFPModule,
    LifespanPredictor,
    init_weights_xavier,
    init_weights_kaiming,
    initialize_model_weights,
    count_parameters,
    get_parameter_stats,
)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = Config()
    # Use smaller dimensions for faster testing
    config.model.gnn_graph_embed_dim = 64
    config.model.fp_cnn_output_dim = 32
    config.model.fp_dnn_layers = [128, 64]
    config.model.fp_dnn_output_dim = 32
    return config


@pytest.fixture
def sample_dgl_graph():
    """Create a sample DGL graph for testing."""
    # Create a simple graph with 3 nodes
    g = dgl.graph(([0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2]))
    g.ndata["x"] = torch.randn(3, 78)
    g.edata["edge_attr"] = torch.randn(6, 11)
    return g


@pytest.fixture
def sample_batch(sample_config):
    """Create a sample batch for testing."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Generate sample data
        smiles = ["CCO", "CC", "CCC"]

        # Generate graph features
        featurizer = CachedGraphFeaturizer(
            cache_dir=os.path.join(temp_dir, "features"), max_atoms=200, atom_feature_dim=75
        )
        adj, feat, sim, _ = featurizer.featurize(smiles)

        # Generate fingerprints
        fp_gen = FingerprintGenerator(morgan_radius=2, morgan_nbits=2048, rdkit_fp_nbits=2048)
        hashed_fps, non_hashed_fps = fp_gen.generate_fingerprints(smiles)

        # Create labels
        labels = np.array([0, 1, 0], dtype=np.float32)

        # Create dataset
        dataset = LifespanDataset(
            root=os.path.join(temp_dir, "dataset"),
            smiles_list=smiles,
            graph_features=(adj, feat, sim),
            fingerprints=(hashed_fps, non_hashed_fps),
            labels=labels,
        )

        # Create dataloader
        loader = create_dataloader(dataset, batch_size=3, shuffle=False)
        batch = next(iter(loader))

        yield batch
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


class TestAttentiveFPModule:
    """Tests for AttentiveFPModule class."""

    def test_module_initialization(self):
        """Test AttentiveFPModule initialization."""
        module = AttentiveFPModule(
            node_feat_size=78,
            edge_feat_size=11,
            num_layers=2,
            num_timesteps=2,
            graph_feat_size=128,
            dropout=0.5,
        )

        assert module.node_feat_size == 78
        assert module.edge_feat_size == 11
        assert module.num_layers == 2
        assert module.num_timesteps == 2
        assert module.graph_feat_size == 128
        assert module.dropout_rate == 0.5
        assert module.gnn is not None
        assert module.readout is not None

    def test_forward_pass_single_graph(self, sample_dgl_graph):
        """Test forward pass with a single graph."""
        module = AttentiveFPModule(node_feat_size=78, edge_feat_size=11, graph_feat_size=128)

        output = module([sample_dgl_graph])

        assert output.shape == (1, 128)
        assert not torch.isnan(output).any()

    def test_forward_pass_batched_graph(self, sample_dgl_graph):
        """Test forward pass with batched graphs."""
        module = AttentiveFPModule(node_feat_size=78, edge_feat_size=11, graph_feat_size=128)

        # Create another graph
        g2 = dgl.graph(([0, 1], [1, 0]))
        g2.ndata["x"] = torch.randn(2, 78)
        g2.edata["edge_attr"] = torch.randn(2, 11)

        # Batch graphs
        batched_g = dgl.batch([sample_dgl_graph, g2])

        output = module([batched_g])

        assert output.shape == (2, 128)
        assert not torch.isnan(output).any()

    def test_forward_pass_missing_node_features(self):
        """Test forward pass with missing node features."""
        module = AttentiveFPModule(node_feat_size=78, edge_feat_size=11, graph_feat_size=128)

        # Create graph without node features
        g = dgl.graph(([0, 1], [1, 0]))
        g.edata["edge_attr"] = torch.randn(2, 11)

        with pytest.raises(ValueError, match="must have node features"):
            module([g])

    def test_forward_pass_missing_edge_features(self):
        """Test forward pass with missing edge features."""
        module = AttentiveFPModule(node_feat_size=78, edge_feat_size=11, graph_feat_size=128)

        # Create graph without edge features
        g = dgl.graph(([0, 1], [1, 0]))
        g.ndata["x"] = torch.randn(2, 78)

        with pytest.raises(ValueError, match="must have edge features"):
            module([g])

    def test_forward_pass_wrong_feature_dimensions(self, sample_dgl_graph):
        """Test forward pass with wrong feature dimensions."""
        module = AttentiveFPModule(
            node_feat_size=100, edge_feat_size=11, graph_feat_size=128  # Wrong dimension
        )

        with pytest.raises(ValueError, match="Expected node features of size"):
            module([sample_dgl_graph])


class TestLifespanPredictor:
    """Tests for LifespanPredictor class."""

    def test_predictor_initialization_all_branches(self, sample_config):
        """Test LifespanPredictor initialization with all branches enabled."""
        model = LifespanPredictor(sample_config)

        assert model.enable_gnn is True
        assert model.enable_fp_cnn is True
        assert model.enable_fp_dnn is True
        assert "gnn" in model.branches
        assert "fp_cnn" in model.branches
        assert "fp_dnn" in model.branches
        assert model.prediction_head is not None

    def test_predictor_initialization_gnn_only(self, sample_config):
        """Test LifespanPredictor initialization with GNN branch only."""
        sample_config.model.enable_fp_cnn = False
        sample_config.model.enable_fp_dnn = False

        model = LifespanPredictor(sample_config)

        assert model.enable_gnn is True
        assert model.enable_fp_cnn is False
        assert model.enable_fp_dnn is False
        assert "gnn" in model.branches
        assert "fp_cnn" not in model.branches
        assert "fp_dnn" not in model.branches

    def test_predictor_initialization_no_branches(self, sample_config):
        """Test LifespanPredictor initialization with no branches enabled."""
        sample_config.model.enable_gnn = False
        sample_config.model.enable_fp_cnn = False
        sample_config.model.enable_fp_dnn = False

        with pytest.raises(ValueError, match="At least one model branch must be enabled"):
            LifespanPredictor(sample_config)

    def test_forward_pass_all_branches(self, sample_config, sample_batch):
        """Test forward pass with all branches enabled."""
        model = LifespanPredictor(sample_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        assert output.shape == (3, 1)  # batch_size=3, n_output_tasks=1
        assert not torch.isnan(output).any()

    def test_forward_pass_gnn_only(self, sample_config, sample_batch):
        """Test forward pass with GNN branch only."""
        sample_config.model.enable_fp_cnn = False
        sample_config.model.enable_fp_dnn = False

        model = LifespanPredictor(sample_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        assert output.shape == (3, 1)
        assert not torch.isnan(output).any()

    def test_forward_pass_fp_branches_only(self, sample_config, sample_batch):
        """Test forward pass with fingerprint branches only."""
        sample_config.model.enable_gnn = False

        model = LifespanPredictor(sample_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        assert output.shape == (3, 1)
        assert not torch.isnan(output).any()

    def test_get_enabled_branches(self, sample_config):
        """Test get_enabled_branches method."""
        model = LifespanPredictor(sample_config)

        enabled = model.get_enabled_branches()

        assert enabled["gnn"] is True
        assert enabled["fp_cnn"] is True
        assert enabled["fp_dnn"] is True

    def test_forward_gnn_missing_dgl_graph(self, sample_config):
        """Test _forward_gnn with missing dgl_graph."""
        model = LifespanPredictor(sample_config)

        # Create batch without dgl_graph
        batch = Data(
            x=torch.randn(3, 78),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            hashed_fp=torch.randn(1, 4096),
            non_hashed_fp=torch.randn(1, 166),
        )

        with pytest.raises(ValueError, match="must have 'dgl_graph' attribute"):
            model._forward_gnn(batch)

    def test_forward_fp_cnn_missing_fingerprints(self, sample_config):
        """Test _forward_fp_cnn with missing fingerprints."""
        model = LifespanPredictor(sample_config)

        # Create batch without non_hashed_fp
        batch = Data(
            x=torch.randn(3, 78),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            hashed_fp=torch.randn(1, 4096),
        )

        with pytest.raises(ValueError, match="must have 'non_hashed_fp' attribute"):
            model._forward_fp_cnn(batch)

    def test_forward_fp_dnn_missing_fingerprints(self, sample_config):
        """Test _forward_fp_dnn with missing fingerprints."""
        model = LifespanPredictor(sample_config)

        # Create batch without hashed_fp
        batch = Data(
            x=torch.randn(3, 78),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            non_hashed_fp=torch.randn(1, 166),
        )

        with pytest.raises(ValueError, match="must have 'hashed_fp' attribute"):
            model._forward_fp_dnn(batch)


class TestWeightInitialization:
    """Tests for weight initialization utilities."""

    def test_init_weights_xavier(self):
        """Test Xavier initialization."""
        model = nn.Linear(10, 5)
        init_weights_xavier(model)

        # Check that weights are initialized (not all zeros)
        assert not torch.all(model.weight == 0)
        # Check that bias is zero
        assert torch.all(model.bias == 0)

    def test_init_weights_kaiming(self):
        """Test Kaiming initialization."""
        model = nn.Linear(10, 5)
        init_weights_kaiming(model)

        # Check that weights are initialized (not all zeros)
        assert not torch.all(model.weight == 0)
        # Check that bias is initialized
        assert not torch.all(model.bias == 0)

    def test_initialize_model_weights(self, sample_config):
        """Test full model weight initialization."""
        model = LifespanPredictor(sample_config)

        # Initialize weights
        initialize_model_weights(model)

        # Check that weights are initialized
        for name, param in model.named_parameters():
            if "weight" in name:
                assert not torch.all(param == 0), f"Weights in {name} are all zeros"

    def test_initialize_model_weights_invalid_method(self, sample_config):
        """Test initialize_model_weights with invalid method."""
        model = LifespanPredictor(sample_config)

        with pytest.raises(ValueError, match="must be one of"):
            initialize_model_weights(model, gnn_init="invalid")

    def test_count_parameters(self, sample_config):
        """Test parameter counting."""
        model = LifespanPredictor(sample_config)

        n_params = count_parameters(model, trainable_only=True)

        assert n_params > 0
        assert isinstance(n_params, int)

    def test_get_parameter_stats(self, sample_config):
        """Test parameter statistics."""
        model = LifespanPredictor(sample_config)

        stats = get_parameter_stats(model)

        assert "total_params" in stats
        assert "trainable_params" in stats
        assert "non_trainable_params" in stats
        assert "param_size_mb" in stats
        assert stats["total_params"] > 0
        assert stats["trainable_params"] > 0
        assert stats["param_size_mb"] > 0


class TestModelOutputShapes:
    """Tests for model output shapes with different configurations."""

    def test_output_shape_single_task(self, sample_config, sample_batch):
        """Test output shape for single task."""
        sample_config.model.n_output_tasks = 1
        model = LifespanPredictor(sample_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        assert output.shape == (3, 1)

    def test_output_shape_multi_task(self, sample_config, sample_batch):
        """Test output shape for multiple tasks."""
        sample_config.model.n_output_tasks = 3
        model = LifespanPredictor(sample_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_batch)

        assert output.shape == (3, 3)

    def test_gradient_flow(self, sample_config, sample_batch):
        """Test that gradients flow through the model."""
        model = LifespanPredictor(sample_config)
        model.train()

        # Forward pass
        output = model(sample_batch)

        # Compute loss
        target = torch.zeros_like(output)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
