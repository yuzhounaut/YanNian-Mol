"""
Unit tests for molecular fingerprint generation module.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from lifespan_predictor.data.fingerprints import FingerprintGenerator


class TestFingerprintGenerator:
    """Tests for FingerprintGenerator class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def generator(self):
        """Create a fingerprint generator instance."""
        return FingerprintGenerator(
            morgan_radius=2,
            morgan_nbits=2048,
            rdkit_fp_nbits=2048,
            maccs_nbits=166,
            n_jobs=1,  # Use single process for testing
        )

    def test_initialization(self):
        """Test generator initialization."""
        generator = FingerprintGenerator(
            morgan_radius=3, morgan_nbits=1024, rdkit_fp_nbits=1024, maccs_nbits=166
        )

        assert generator.morgan_radius == 3
        assert generator.morgan_nbits == 1024
        assert generator.rdkit_fp_nbits == 1024
        assert generator.maccs_nbits == 166

    def test_initialization_invalid_maccs_nbits(self):
        """Test that invalid MACCS nbits raises error."""
        with pytest.raises(ValueError, match="MACCS keys must have exactly 166 bits"):
            FingerprintGenerator(maccs_nbits=200)

    def test_generate_fingerprints_single_molecule(self, generator):
        """Test fingerprint generation for a single molecule."""
        smiles_list = ["CCO"]  # Ethanol

        hashed_fps, maccs_fps = generator.generate_fingerprints(smiles_list)

        # Check shapes
        assert hashed_fps.shape == (1, 4096)  # 2048 + 2048
        assert maccs_fps.shape == (1, 166)

        # Check data types
        assert hashed_fps.dtype == np.float32
        assert maccs_fps.dtype == np.float32

    def test_generate_fingerprints_multiple_molecules(self, generator):
        """Test fingerprint generation for multiple molecules."""
        smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]

        hashed_fps, maccs_fps = generator.generate_fingerprints(smiles_list)

        # Check shapes
        assert hashed_fps.shape == (3, 4096)
        assert maccs_fps.shape == (3, 166)

    def test_generate_fingerprints_empty_list(self, generator):
        """Test handling of empty SMILES list."""
        smiles_list = []

        hashed_fps, maccs_fps = generator.generate_fingerprints(smiles_list)

        # Should return empty arrays with correct shapes
        assert hashed_fps.shape == (0, 4096)
        assert maccs_fps.shape == (0, 166)

    def test_generate_fingerprints_invalid_molecules(self, generator):
        """Test handling of invalid molecules."""
        smiles_list = ["CCO", "invalid", "c1ccccc1"]

        hashed_fps, maccs_fps = generator.generate_fingerprints(smiles_list)

        # Should skip invalid molecule
        assert hashed_fps.shape[0] == 2
        assert maccs_fps.shape[0] == 2

    def test_generate_fingerprints_all_invalid(self, generator):
        """Test handling when all molecules are invalid."""
        smiles_list = ["invalid1", "invalid2", "bad"]

        hashed_fps, maccs_fps = generator.generate_fingerprints(smiles_list)

        # Should return empty arrays
        assert hashed_fps.shape[0] == 0
        assert maccs_fps.shape[0] == 0

    def test_morgan_fingerprints_non_zero(self, generator):
        """Test that Morgan fingerprints have non-zero values."""
        smiles_list = ["CCO"]

        hashed_fps, _ = generator.generate_fingerprints(smiles_list)

        # Morgan fingerprints should have some non-zero bits
        morgan_fp = hashed_fps[0, :2048]
        assert np.any(morgan_fp != 0)

    def test_rdkit_fingerprints_non_zero(self, generator):
        """Test that RDKit fingerprints have non-zero values."""
        smiles_list = ["CCO"]

        hashed_fps, _ = generator.generate_fingerprints(smiles_list)

        # RDKit fingerprints should have some non-zero bits
        rdkit_fp = hashed_fps[0, 2048:]
        assert np.any(rdkit_fp != 0)

    def test_maccs_fingerprints_non_zero(self, generator):
        """Test that MACCS keys have non-zero values."""
        smiles_list = ["CCO"]

        _, maccs_fps = generator.generate_fingerprints(smiles_list)

        # MACCS keys should have some non-zero bits
        assert np.any(maccs_fps[0] != 0)

    def test_fingerprints_binary(self, generator):
        """Test that fingerprints are binary (0 or 1)."""
        smiles_list = ["CCO", "c1ccccc1"]

        hashed_fps, maccs_fps = generator.generate_fingerprints(smiles_list)

        # All values should be 0 or 1
        assert np.all((hashed_fps == 0) | (hashed_fps == 1))
        assert np.all((maccs_fps == 0) | (maccs_fps == 1))

    def test_fingerprints_deterministic(self, generator):
        """Test that fingerprints are deterministic."""
        smiles_list = ["CCO", "c1ccccc1"]

        hashed_fps1, maccs_fps1 = generator.generate_fingerprints(smiles_list)
        hashed_fps2, maccs_fps2 = generator.generate_fingerprints(smiles_list)

        # Results should be identical
        np.testing.assert_array_equal(hashed_fps1, hashed_fps2)
        np.testing.assert_array_equal(maccs_fps1, maccs_fps2)

    def test_different_molecules_different_fingerprints(self, generator):
        """Test that different molecules have different fingerprints."""
        smiles1 = ["CCO"]
        smiles2 = ["c1ccccc1"]

        hashed_fps1, maccs_fps1 = generator.generate_fingerprints(smiles1)
        hashed_fps2, maccs_fps2 = generator.generate_fingerprints(smiles2)

        # Fingerprints should be different
        assert not np.array_equal(hashed_fps1, hashed_fps2)
        assert not np.array_equal(maccs_fps1, maccs_fps2)

    def test_cache_save_and_load(self, generator, temp_cache_dir):
        """Test that fingerprints are cached correctly."""
        smiles_list = ["CCO", "c1ccccc1"]

        # First call - compute and cache
        hashed_fps1, maccs_fps1 = generator.generate_fingerprints(
            smiles_list, cache_dir=temp_cache_dir
        )

        # Second call - load from cache
        hashed_fps2, maccs_fps2 = generator.generate_fingerprints(
            smiles_list, cache_dir=temp_cache_dir
        )

        # Results should be identical
        np.testing.assert_array_equal(hashed_fps1, hashed_fps2)
        np.testing.assert_array_equal(maccs_fps1, maccs_fps2)

    def test_cache_miss_different_smiles(self, generator, temp_cache_dir):
        """Test cache miss with different SMILES list."""
        smiles_list1 = ["CCO", "c1ccccc1"]
        smiles_list2 = ["CCO", "CC(C)C"]

        hashed_fps1, _ = generator.generate_fingerprints(smiles_list1, cache_dir=temp_cache_dir)
        hashed_fps2, _ = generator.generate_fingerprints(smiles_list2, cache_dir=temp_cache_dir)

        # Results should be different
        assert not np.array_equal(hashed_fps1, hashed_fps2)

    def test_force_recompute(self, generator, temp_cache_dir):
        """Test force recompute ignores cache."""
        smiles_list = ["CCO", "c1ccccc1"]

        # First call
        hashed_fps1, maccs_fps1 = generator.generate_fingerprints(
            smiles_list, cache_dir=temp_cache_dir
        )

        # Second call with force_recompute
        hashed_fps2, maccs_fps2 = generator.generate_fingerprints(
            smiles_list, cache_dir=temp_cache_dir, force_recompute=True
        )

        # Results should be equal (same computation)
        np.testing.assert_array_equal(hashed_fps1, hashed_fps2)
        np.testing.assert_array_equal(maccs_fps1, maccs_fps2)

    def test_different_configurations_different_cache(self, temp_cache_dir):
        """Test that different configurations use different cache keys."""
        smiles_list = ["CCO"]

        # Generator with radius=2
        generator1 = FingerprintGenerator(morgan_radius=2, morgan_nbits=2048, n_jobs=1)
        hashed_fps1, _ = generator1.generate_fingerprints(smiles_list, cache_dir=temp_cache_dir)

        # Generator with radius=3
        generator2 = FingerprintGenerator(morgan_radius=3, morgan_nbits=2048, n_jobs=1)
        hashed_fps2, _ = generator2.generate_fingerprints(smiles_list, cache_dir=temp_cache_dir)

        # Fingerprints should be different (different radius)
        assert not np.array_equal(hashed_fps1, hashed_fps2)

    def test_morgan_radius_effect(self):
        """Test that Morgan radius affects fingerprints."""
        smiles_list = ["CCO"]

        generator_r2 = FingerprintGenerator(morgan_radius=2, n_jobs=1)
        generator_r3 = FingerprintGenerator(morgan_radius=3, n_jobs=1)

        hashed_fps_r2, _ = generator_r2.generate_fingerprints(smiles_list)
        hashed_fps_r3, _ = generator_r3.generate_fingerprints(smiles_list)

        # Different radius should produce different fingerprints
        assert not np.array_equal(hashed_fps_r2, hashed_fps_r3)

    def test_fingerprint_dimensions_match_config(self):
        """Test that fingerprint dimensions match configuration."""
        generator = FingerprintGenerator(
            morgan_radius=2, morgan_nbits=1024, rdkit_fp_nbits=512, maccs_nbits=166, n_jobs=1
        )

        smiles_list = ["CCO"]
        hashed_fps, maccs_fps = generator.generate_fingerprints(smiles_list)

        # Check dimensions
        assert hashed_fps.shape[1] == 1024 + 512
        assert maccs_fps.shape[1] == 166

    def test_batch_compute_morgan(self, generator):
        """Test Morgan fingerprint batch computation."""
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(C)C"]]
        fps = generator._batch_compute_morgan(mols)

        assert fps.shape == (3, 2048)
        assert fps.dtype == np.float32

    def test_batch_compute_rdkit(self, generator):
        """Test RDKit fingerprint batch computation."""
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(C)C"]]
        fps = generator._batch_compute_rdkit(mols)

        assert fps.shape == (3, 2048)
        assert fps.dtype == np.float32

    def test_batch_compute_maccs(self, generator):
        """Test MACCS keys batch computation."""
        from rdkit import Chem

        mols = [Chem.MolFromSmiles(s) for s in ["CCO", "c1ccccc1", "CC(C)C"]]
        fps = generator._batch_compute_maccs(mols)

        assert fps.shape == (3, 166)
        assert fps.dtype == np.float32

    def test_validate_dimensions_success(self, generator):
        """Test dimension validation with correct dimensions."""
        morgan_fps = np.zeros((3, 2048), dtype=np.float32)
        rdkit_fps = np.zeros((3, 2048), dtype=np.float32)
        maccs_fps = np.zeros((3, 166), dtype=np.float32)

        # Should not raise
        generator._validate_dimensions(morgan_fps, rdkit_fps, maccs_fps)

    def test_validate_dimensions_morgan_mismatch(self, generator):
        """Test dimension validation with Morgan dimension mismatch."""
        morgan_fps = np.zeros((3, 1024), dtype=np.float32)  # Wrong size
        rdkit_fps = np.zeros((3, 2048), dtype=np.float32)
        maccs_fps = np.zeros((3, 166), dtype=np.float32)

        with pytest.raises(ValueError, match="Morgan fingerprint dimension mismatch"):
            generator._validate_dimensions(morgan_fps, rdkit_fps, maccs_fps)

    def test_validate_dimensions_rdkit_mismatch(self, generator):
        """Test dimension validation with RDKit dimension mismatch."""
        morgan_fps = np.zeros((3, 2048), dtype=np.float32)
        rdkit_fps = np.zeros((3, 1024), dtype=np.float32)  # Wrong size
        maccs_fps = np.zeros((3, 166), dtype=np.float32)

        with pytest.raises(ValueError, match="RDKit fingerprint dimension mismatch"):
            generator._validate_dimensions(morgan_fps, rdkit_fps, maccs_fps)

    def test_validate_dimensions_maccs_mismatch(self, generator):
        """Test dimension validation with MACCS dimension mismatch."""
        morgan_fps = np.zeros((3, 2048), dtype=np.float32)
        rdkit_fps = np.zeros((3, 2048), dtype=np.float32)
        maccs_fps = np.zeros((3, 200), dtype=np.float32)  # Wrong size

        with pytest.raises(ValueError, match="MACCS keys dimension mismatch"):
            generator._validate_dimensions(morgan_fps, rdkit_fps, maccs_fps)

    def test_validate_dimensions_molecule_count_mismatch(self, generator):
        """Test dimension validation with molecule count mismatch."""
        morgan_fps = np.zeros((3, 2048), dtype=np.float32)
        rdkit_fps = np.zeros((2, 2048), dtype=np.float32)  # Different count
        maccs_fps = np.zeros((3, 166), dtype=np.float32)

        with pytest.raises(ValueError, match="Number of molecules mismatch"):
            generator._validate_dimensions(morgan_fps, rdkit_fps, maccs_fps)

    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directory is created if it doesn't exist."""
        cache_path = Path(temp_cache_dir) / "subdir" / "cache"

        generator = FingerprintGenerator(n_jobs=1)
        smiles_list = ["CCO"]

        generator.generate_fingerprints(smiles_list, cache_dir=str(cache_path))

        assert cache_path.exists()

    def test_parallel_processing(self, temp_cache_dir):
        """Test parallel processing produces same results as sequential."""
        smiles_list = ["CCO", "c1ccccc1", "CC(C)C", "CCC", "CCCC"]

        # Sequential processing
        generator_seq = FingerprintGenerator(n_jobs=1)
        hashed_seq, maccs_seq = generator_seq.generate_fingerprints(
            smiles_list, cache_dir=temp_cache_dir + "_seq"
        )

        # Parallel processing
        generator_par = FingerprintGenerator(n_jobs=2)
        hashed_par, maccs_par = generator_par.generate_fingerprints(
            smiles_list, cache_dir=temp_cache_dir + "_par"
        )

        # Results should be identical
        np.testing.assert_array_equal(hashed_seq, hashed_par)
        np.testing.assert_array_equal(maccs_seq, maccs_par)

    def test_reference_molecules(self, generator):
        """Test fingerprints with reference molecules."""
        # Test with well-known molecules
        smiles_list = [
            "CCO",  # Ethanol
            "c1ccccc1",  # Benzene
            "CC(=O)O",  # Acetic acid
            "CC(C)C",  # Isobutane
        ]

        hashed_fps, maccs_fps = generator.generate_fingerprints(smiles_list)

        # All molecules should be processed
        assert hashed_fps.shape[0] == 4
        assert maccs_fps.shape[0] == 4

        # Each molecule should have unique fingerprints
        for i in range(4):
            for j in range(i + 1, 4):
                # At least some bits should be different
                assert not np.array_equal(hashed_fps[i], hashed_fps[j])
