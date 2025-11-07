"""
Unit tests for molecular featurization module.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from lifespan_predictor.data.featurizers import CachedGraphFeaturizer


class TestCachedGraphFeaturizer:
    """Tests for CachedGraphFeaturizer class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def featurizer(self, temp_cache_dir):
        """Create a featurizer instance."""
        return CachedGraphFeaturizer(
            cache_dir=temp_cache_dir,
            max_atoms=50,
            atom_feature_dim=75,
            n_jobs=1,  # Use single process for testing
        )

    def test_initialization(self, temp_cache_dir):
        """Test featurizer initialization."""
        featurizer = CachedGraphFeaturizer(
            cache_dir=temp_cache_dir, max_atoms=100, atom_feature_dim=75
        )

        assert featurizer.max_atoms == 100
        assert featurizer.atom_feature_dim == 75
        assert Path(temp_cache_dir).exists()

    def test_featurize_single_molecule(self, featurizer):
        """Test featurization of a single molecule."""
        smiles_list = ["CCO"]  # Ethanol

        adj, feat, sim, labels = featurizer.featurize(smiles_list)

        # Check shapes
        assert adj.shape == (1, 50, 50)
        assert feat.shape == (1, 50, 75)
        assert sim.shape == (1, 50, 50)
        assert labels is None

    def test_featurize_multiple_molecules(self, featurizer):
        """Test featurization of multiple molecules."""
        smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]

        adj, feat, sim, labels = featurizer.featurize(smiles_list)

        # Check shapes
        assert adj.shape == (3, 50, 50)
        assert feat.shape == (3, 50, 75)
        assert sim.shape == (3, 50, 50)

    def test_featurize_with_labels(self, featurizer):
        """Test featurization with labels."""
        smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]
        labels_in = np.array([0, 1, 0])

        adj, feat, sim, labels_out = featurizer.featurize(smiles_list, labels=labels_in)

        # Check that labels are returned
        assert labels_out is not None
        assert len(labels_out) == 3
        np.testing.assert_array_equal(labels_out, labels_in)

    def test_featurize_invalid_molecules(self, featurizer):
        """Test handling of invalid molecules."""
        smiles_list = ["CCO", "invalid", "c1ccccc1"]

        adj, feat, sim, labels = featurizer.featurize(smiles_list)

        # Should skip invalid molecule
        assert adj.shape[0] == 2
        assert feat.shape[0] == 2
        assert sim.shape[0] == 2

    def test_featurize_molecules_exceeding_max_atoms(self, featurizer):
        """Test handling of molecules exceeding max_atoms."""
        # Create a large molecule (more than 50 atoms)
        large_smiles = "C" * 60  # Long chain
        smiles_list = ["CCO", large_smiles, "c1ccccc1"]

        adj, feat, sim, labels = featurizer.featurize(smiles_list)

        # Should skip molecule exceeding max_atoms
        assert adj.shape[0] == 2

    def test_cache_hit(self, featurizer):
        """Test that cached features are loaded correctly."""
        smiles_list = ["CCO", "c1ccccc1"]

        # First call - compute features
        adj1, feat1, sim1, _ = featurizer.featurize(smiles_list)

        # Second call - should load from cache
        adj2, feat2, sim2, _ = featurizer.featurize(smiles_list)

        # Results should be identical
        np.testing.assert_array_equal(adj1, adj2)
        np.testing.assert_array_equal(feat1, feat2)
        np.testing.assert_array_equal(sim1, sim2)

    def test_cache_miss_different_smiles(self, featurizer):
        """Test cache miss with different SMILES list."""
        smiles_list1 = ["CCO", "c1ccccc1"]
        smiles_list2 = ["CCO", "CC(C)C"]

        adj1, _, _, _ = featurizer.featurize(smiles_list1)
        adj2, _, _, _ = featurizer.featurize(smiles_list2)

        # Results should be different
        assert not np.array_equal(adj1, adj2)

    def test_force_recompute(self, featurizer):
        """Test force recompute ignores cache."""
        smiles_list = ["CCO", "c1ccccc1"]

        # First call
        adj1, _, _, _ = featurizer.featurize(smiles_list)

        # Second call with force_recompute
        adj2, _, _, _ = featurizer.featurize(smiles_list, force_recompute=True)

        # Results should be equal (same computation)
        np.testing.assert_array_equal(adj1, adj2)

    def test_labels_alignment_with_invalid_molecules(self, featurizer):
        """Test that labels are correctly aligned when some molecules are invalid."""
        smiles_list = ["CCO", "invalid", "c1ccccc1", "bad"]
        labels_in = np.array([0, 1, 2, 3])

        adj, feat, sim, labels_out = featurizer.featurize(smiles_list, labels=labels_in)

        # Should have 2 valid molecules
        assert adj.shape[0] == 2
        assert len(labels_out) == 2
        # Labels should correspond to valid molecules (indices 0 and 2)
        np.testing.assert_array_equal(labels_out, np.array([0, 2]))

    def test_output_dimensions(self, featurizer):
        """Test that output dimensions match configuration."""
        smiles_list = ["CCO"]

        adj, feat, sim, _ = featurizer.featurize(smiles_list)

        # Check dimensions
        assert adj.shape[1] == featurizer.max_atoms
        assert adj.shape[2] == featurizer.max_atoms
        assert feat.shape[1] == featurizer.max_atoms
        assert feat.shape[2] == featurizer.atom_feature_dim
        assert sim.shape[1] == featurizer.max_atoms
        assert sim.shape[2] == featurizer.max_atoms

    def test_adjacency_matrix_properties(self, featurizer):
        """Test properties of adjacency matrices."""
        smiles_list = ["c1ccccc1"]  # Benzene

        adj, _, _, _ = featurizer.featurize(smiles_list)

        # Adjacency matrix should be symmetric (after normalization)
        adj_matrix = adj[0]
        # Check symmetry for non-zero part
        non_zero_size = 6  # Benzene has 6 atoms
        adj_sub = adj_matrix[:non_zero_size, :non_zero_size]
        np.testing.assert_allclose(adj_sub, adj_sub.T, rtol=1e-5)

    def test_node_features_non_zero(self, featurizer):
        """Test that node features are non-zero for valid molecules."""
        smiles_list = ["CCO"]

        _, feat, _, _ = featurizer.featurize(smiles_list)

        # Node features should have non-zero values for actual atoms
        # Ethanol has 3 atoms (C, C, O)
        assert np.any(feat[0, :3, :] != 0)

    def test_similarity_graph_diagonal(self, featurizer):
        """Test that similarity graph has atomic numbers on diagonal."""
        smiles_list = ["CCO"]  # Ethanol: C(6), C(6), O(8)

        _, _, sim, _ = featurizer.featurize(smiles_list)

        # Check diagonal values (atomic numbers)
        # Carbon has atomic number 6, Oxygen has 8
        diagonal = np.diag(sim[0])
        # First two atoms should be carbon (6), third should be oxygen (8)
        assert diagonal[0] == 6
        assert diagonal[1] == 6
        assert diagonal[2] == 8

    def test_empty_smiles_list(self, featurizer):
        """Test handling of empty SMILES list."""
        smiles_list = []

        adj, feat, sim, labels = featurizer.featurize(smiles_list)

        # Should return empty arrays with correct shapes
        assert adj.shape == (0, 50, 50)
        assert feat.shape == (0, 50, 75)
        assert sim.shape == (0, 50, 50)

    def test_all_invalid_molecules(self, featurizer):
        """Test handling when all molecules are invalid."""
        smiles_list = ["invalid1", "invalid2", "bad"]

        adj, feat, sim, labels = featurizer.featurize(smiles_list)

        # Should return empty arrays
        assert adj.shape[0] == 0
        assert feat.shape[0] == 0
        assert sim.shape[0] == 0

    def test_labels_length_mismatch(self, featurizer):
        """Test error when labels length doesn't match SMILES list."""
        smiles_list = ["CCO", "c1ccccc1"]
        labels = np.array([0, 1, 2])  # Wrong length

        with pytest.raises(ValueError, match="Length of labels.*must match"):
            featurizer.featurize(smiles_list, labels=labels)

    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directory is created if it doesn't exist."""
        cache_path = Path(temp_cache_dir) / "subdir" / "cache"

        _ = CachedGraphFeaturizer(cache_dir=str(cache_path), max_atoms=50)

        assert cache_path.exists()

    def test_different_configurations_different_cache(self, temp_cache_dir):
        """Test that different configurations use different cache keys."""
        smiles_list = ["CCO"]

        # Featurizer with max_atoms=50
        featurizer1 = CachedGraphFeaturizer(cache_dir=temp_cache_dir, max_atoms=50, n_jobs=1)
        adj1, _, _, _ = featurizer1.featurize(smiles_list)

        # Featurizer with max_atoms=100
        featurizer2 = CachedGraphFeaturizer(cache_dir=temp_cache_dir, max_atoms=100, n_jobs=1)
        adj2, _, _, _ = featurizer2.featurize(smiles_list)

        # Shapes should be different
        assert adj1.shape[1] != adj2.shape[1]

    def test_parallel_processing(self, temp_cache_dir):
        """Test parallel processing produces same results as sequential."""
        smiles_list = ["CCO", "c1ccccc1", "CC(C)C", "CCC", "CCCC"]

        # Sequential processing
        featurizer_seq = CachedGraphFeaturizer(
            cache_dir=temp_cache_dir + "_seq", max_atoms=50, n_jobs=1
        )
        adj_seq, feat_seq, sim_seq, _ = featurizer_seq.featurize(smiles_list)

        # Parallel processing
        featurizer_par = CachedGraphFeaturizer(
            cache_dir=temp_cache_dir + "_par", max_atoms=50, n_jobs=2
        )
        adj_par, feat_par, sim_par, _ = featurizer_par.featurize(smiles_list)

        # Results should be identical
        np.testing.assert_array_equal(adj_seq, adj_par)
        np.testing.assert_array_equal(feat_seq, feat_par)
        np.testing.assert_array_equal(sim_seq, sim_par)
