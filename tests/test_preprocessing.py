"""
Unit tests for data preprocessing module.
"""

import pytest
import tempfile
import os

from lifespan_predictor.data.preprocessing import (
    clean_smiles,
    validate_smiles_list,
    load_and_clean_csv,
)


class TestCleanSmiles:
    """Tests for clean_smiles function."""

    def test_valid_smiles(self):
        """Test cleaning valid SMILES strings."""
        # Simple molecules
        assert clean_smiles("CCO") == "CCO"
        assert clean_smiles("c1ccccc1") == "c1ccccc1"
        assert clean_smiles("CC(C)C") == "CC(C)C"

    def test_canonicalization(self):
        """Test SMILES canonicalization."""
        # Different representations of the same molecule should canonicalize to same SMILES
        benzene1 = clean_smiles("c1ccccc1")
        benzene2 = clean_smiles("C1=CC=CC=C1")
        assert benzene1 == benzene2

        # Ethanol
        ethanol1 = clean_smiles("CCO")
        ethanol2 = clean_smiles("OCC")
        assert ethanol1 == ethanol2

    def test_salt_removal(self):
        """Test removal of salts and counterions."""
        # Sodium salt should be removed
        result = clean_smiles("CCO.[Na+]")
        assert result == "CCO"

        # Chloride salt
        result = clean_smiles("CC(=O)O.[Na+]")
        assert "[Na+]" not in result

    def test_invalid_smiles(self):
        """Test handling of invalid SMILES strings."""
        assert clean_smiles("invalid") is None
        assert clean_smiles("C(C") is None  # Unmatched parenthesis
        assert clean_smiles("C=C=C=C") is None  # Invalid bonding

    def test_empty_strings(self):
        """Test handling of empty strings."""
        assert clean_smiles("") is None
        assert clean_smiles("   ") is None
        assert clean_smiles(None) is None

    def test_whitespace_handling(self):
        """Test trimming of whitespace."""
        assert clean_smiles("  CCO  ") == "CCO"
        assert clean_smiles("\tCCO\n") == "CCO"

    def test_special_characters(self):
        """Test handling of special characters in SMILES."""
        # Valid SMILES with special characters
        assert clean_smiles("C[C@H](O)C") is not None  # Chiral center
        assert clean_smiles("C#N") is not None  # Triple bond
        assert clean_smiles("C=O") is not None  # Double bond

    def test_non_string_input(self):
        """Test handling of non-string inputs."""
        assert clean_smiles(123) is None
        assert clean_smiles([]) is None
        assert clean_smiles({}) is None


class TestValidateSmilesList:
    """Tests for validate_smiles_list function."""

    def test_all_valid_smiles(self):
        """Test validation with all valid SMILES."""
        smiles_list = ["CCO", "c1ccccc1", "CC(C)C"]
        valid, failed = validate_smiles_list(smiles_list)

        assert len(valid) == 3
        assert len(failed) == 0

    def test_mixed_valid_invalid(self):
        """Test validation with mix of valid and invalid SMILES."""
        smiles_list = ["CCO", "invalid", "c1ccccc1", "C(C", "CC"]
        valid, failed = validate_smiles_list(smiles_list)

        assert len(valid) == 3
        assert len(failed) == 2
        assert 1 in failed  # "invalid"
        assert 3 in failed  # "C(C"

    def test_all_invalid_smiles(self):
        """Test validation with all invalid SMILES."""
        smiles_list = ["invalid", "bad", "C(C"]
        valid, failed = validate_smiles_list(smiles_list)

        assert len(valid) == 0
        assert len(failed) == 3

    def test_empty_list(self):
        """Test validation with empty list."""
        valid, failed = validate_smiles_list([])

        assert len(valid) == 0
        assert len(failed) == 0

    def test_canonicalization_in_validation(self):
        """Test that validation canonicalizes SMILES when clean=True."""
        smiles_list = ["CCO", "OCC"]  # Two representations of ethanol
        valid, failed = validate_smiles_list(smiles_list, clean=True)

        assert len(valid) == 2
        assert len(failed) == 0
        # Both should be canonicalized to same form
        assert valid[0] == valid[1]

    def test_no_cleaning(self):
        """Test validation without cleaning."""
        smiles_list = ["CCO", "OCC"]
        valid, failed = validate_smiles_list(smiles_list, clean=False)

        assert len(valid) == 2
        assert len(failed) == 0
        # Should preserve original SMILES
        assert valid[0] == "CCO"
        assert valid[1] == "OCC"

    def test_failed_indices_correct(self):
        """Test that failed indices are correctly tracked."""
        smiles_list = ["CCO", "invalid1", "c1ccccc1", "invalid2", "CC"]
        valid, failed = validate_smiles_list(smiles_list)

        assert failed == [1, 3]


class TestLoadAndCleanCsv:
    """Tests for load_and_clean_csv function."""

    def test_load_valid_csv(self):
        """Test loading CSV with valid SMILES."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("SMILES,label\n")
            f.write("CCO,1\n")
            f.write("c1ccccc1,0\n")
            f.write("CC(C)C,1\n")
            temp_path = f.name

        try:
            df = load_and_clean_csv(temp_path, smiles_column="SMILES", label_column="label")

            assert len(df) == 3
            assert "SMILES" in df.columns
            assert "label" in df.columns
            assert list(df["label"]) == [1, 0, 1]
        finally:
            os.unlink(temp_path)

    def test_drop_invalid_smiles(self):
        """Test dropping rows with invalid SMILES."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("SMILES,label\n")
            f.write("CCO,1\n")
            f.write("invalid,0\n")
            f.write("c1ccccc1,1\n")
            temp_path = f.name

        try:
            df = load_and_clean_csv(temp_path, drop_invalid=True)

            assert len(df) == 2
            assert "invalid" not in df["SMILES"].values
        finally:
            os.unlink(temp_path)

    def test_keep_invalid_smiles(self):
        """Test keeping rows with invalid SMILES."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("SMILES,label\n")
            f.write("CCO,1\n")
            f.write("invalid,0\n")
            f.write("c1ccccc1,1\n")
            temp_path = f.name

        try:
            df = load_and_clean_csv(temp_path, drop_invalid=False)

            # Should keep all rows
            assert len(df) == 3
        finally:
            os.unlink(temp_path)

    def test_canonicalization_in_csv(self):
        """Test that SMILES are canonicalized when loading CSV."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("SMILES,label\n")
            f.write("CCO,1\n")
            f.write("OCC,0\n")  # Different representation of ethanol
            temp_path = f.name

        try:
            df = load_and_clean_csv(temp_path, clean=True)

            # Both should be canonicalized to same SMILES
            assert df.loc[0, "SMILES"] == df.loc[1, "SMILES"]
        finally:
            os.unlink(temp_path)

    def test_missing_smiles_column(self):
        """Test error handling when SMILES column is missing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("molecule,label\n")
            f.write("CCO,1\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="SMILES column.*not found"):
                load_and_clean_csv(temp_path, smiles_column="SMILES")
        finally:
            os.unlink(temp_path)

    def test_missing_label_column(self):
        """Test error handling when label column is missing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("SMILES,value\n")
            f.write("CCO,1\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Label column.*not found"):
                load_and_clean_csv(temp_path, label_column="label")
        finally:
            os.unlink(temp_path)

    def test_file_not_found(self):
        """Test error handling when CSV file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_and_clean_csv("nonexistent_file.csv")

    def test_custom_column_names(self):
        """Test loading CSV with custom column names."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("molecule,activity\n")
            f.write("CCO,1\n")
            f.write("c1ccccc1,0\n")
            temp_path = f.name

        try:
            df = load_and_clean_csv(temp_path, smiles_column="molecule", label_column="activity")

            assert len(df) == 2
            assert "molecule" in df.columns
            assert "activity" in df.columns
        finally:
            os.unlink(temp_path)

    def test_no_cleaning(self):
        """Test loading CSV without cleaning SMILES."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write("SMILES,label\n")
            f.write("CCO,1\n")
            f.write("OCC,0\n")
            temp_path = f.name

        try:
            df = load_and_clean_csv(temp_path, clean=False)

            # Should preserve original SMILES
            assert df.loc[0, "SMILES"] == "CCO"
            assert df.loc[1, "SMILES"] == "OCC"
        finally:
            os.unlink(temp_path)
