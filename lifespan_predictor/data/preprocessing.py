"""
Data preprocessing module for SMILES cleaning and validation.

This module provides functions to clean, validate, and canonicalize SMILES strings
before featurization, ensuring data quality and consistency.
"""

import logging
from typing import List, Optional, Tuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover

logger = logging.getLogger(__name__)


def clean_smiles(smiles: str) -> Optional[str]:
    """
    Clean and canonicalize a SMILES string.

    This function performs the following operations:
    1. Removes salts and counterions
    2. Canonicalizes the SMILES representation
    3. Validates the resulting molecule

    Parameters
    ----------
    smiles : str
        Input SMILES string to clean

    Returns
    -------
    Optional[str]
        Canonicalized SMILES string if valid, None if invalid

    Examples
    --------
    >>> clean_smiles("CCO")
    'CCO'
    >>> clean_smiles("CCO.[Na+]")
    'CCO'
    >>> clean_smiles("invalid")
    None
    """
    if not smiles or not isinstance(smiles, str):
        return None

    # Strip whitespace
    smiles = smiles.strip()

    if not smiles:
        return None

    try:
        # Parse SMILES to molecule
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            logger.debug(f"Failed to parse SMILES: {smiles}")
            return None

        # Remove salts and counterions
        remover = SaltRemover.SaltRemover()
        mol = remover.StripMol(mol)

        # Canonicalize SMILES
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # Validate the canonical SMILES
        if canonical_smiles and Chem.MolFromSmiles(canonical_smiles) is not None:
            return canonical_smiles
        else:
            logger.debug(f"Failed to canonicalize SMILES: {smiles}")
            return None

    except Exception as e:
        logger.debug(f"Error processing SMILES '{smiles}': {str(e)}")
        return None


def validate_smiles_list(smiles_list: List[str], clean: bool = True) -> Tuple[List[str], List[int]]:
    """
    Validate a list of SMILES strings and track failures.

    Parameters
    ----------
    smiles_list : List[str]
        List of SMILES strings to validate
    clean : bool, optional
        Whether to clean and canonicalize SMILES (default: True)

    Returns
    -------
    Tuple[List[str], List[int]]
        - List of valid (optionally cleaned) SMILES strings
        - List of indices of failed SMILES in the original list

    Examples
    --------
    >>> valid, failed = validate_smiles_list(["CCO", "invalid", "CC"])
    >>> len(valid)
    2
    >>> failed
    [1]
    """
    valid_smiles = []
    failed_indices = []

    for idx, smiles in enumerate(smiles_list):
        if clean:
            cleaned = clean_smiles(smiles)
            if cleaned is not None:
                valid_smiles.append(cleaned)
            else:
                failed_indices.append(idx)
                logger.warning(f"Invalid SMILES at index {idx}: {smiles}")
        else:
            # Just validate without cleaning
            if smiles and isinstance(smiles, str):
                mol = Chem.MolFromSmiles(smiles.strip())
                if mol is not None:
                    valid_smiles.append(smiles.strip())
                else:
                    failed_indices.append(idx)
                    logger.warning(f"Invalid SMILES at index {idx}: {smiles}")
            else:
                failed_indices.append(idx)
                logger.warning(f"Invalid SMILES at index {idx}: {smiles}")

    logger.info(
        f"Validated {len(smiles_list)} SMILES: "
        f"{len(valid_smiles)} valid, {len(failed_indices)} failed"
    )

    return valid_smiles, failed_indices


def load_and_clean_csv(
    csv_path: str,
    smiles_column: str = "SMILES",
    label_column: Optional[str] = None,
    clean: bool = True,
    drop_invalid: bool = True,
) -> pd.DataFrame:
    """
    Load CSV file and clean SMILES strings.

    This function loads a CSV file, validates and optionally cleans the SMILES
    strings, and returns a DataFrame with valid molecules.

    Parameters
    ----------
    csv_path : str
        Path to CSV file
    smiles_column : str, optional
        Name of column containing SMILES strings (default: 'SMILES')
    label_column : Optional[str], optional
        Name of column containing labels (default: None)
    clean : bool, optional
        Whether to clean and canonicalize SMILES (default: True)
    drop_invalid : bool, optional
        Whether to drop rows with invalid SMILES (default: True)

    Returns
    -------
    pd.DataFrame
        DataFrame with validated SMILES and labels

    Raises
    ------
    FileNotFoundError
        If CSV file does not exist
    ValueError
        If required columns are missing

    Examples
    --------
    >>> df = load_and_clean_csv("data.csv", smiles_column="SMILES")
    >>> "SMILES" in df.columns
    True
    """
    # Load CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {str(e)}")
        raise

    # Validate required columns
    if smiles_column not in df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    if label_column is not None and label_column not in df.columns:
        raise ValueError(
            f"Label column '{label_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Process SMILES
    original_count = len(df)

    if clean:
        # Clean and canonicalize SMILES
        df["cleaned_smiles"] = df[smiles_column].apply(clean_smiles)

        if drop_invalid:
            # Drop rows with invalid SMILES
            invalid_mask = df["cleaned_smiles"].isna()
            invalid_count = invalid_mask.sum()

            if invalid_count > 0:
                logger.warning(
                    f"Dropping {invalid_count} rows with invalid SMILES "
                    f"({invalid_count/original_count*100:.1f}%)"
                )
                df = df[~invalid_mask].copy()

            # Replace original SMILES with cleaned version
            df[smiles_column] = df["cleaned_smiles"]
            df = df.drop(columns=["cleaned_smiles"])
        else:
            # Keep original SMILES but add cleaned version
            df[smiles_column] = df["cleaned_smiles"].fillna(df[smiles_column])
            df = df.drop(columns=["cleaned_smiles"])
    else:
        # Just validate without cleaning
        valid_mask = df[smiles_column].apply(
            lambda x: x and isinstance(x, str) and Chem.MolFromSmiles(x.strip()) is not None
        )

        if drop_invalid:
            invalid_count = (~valid_mask).sum()
            if invalid_count > 0:
                logger.warning(
                    f"Dropping {invalid_count} rows with invalid SMILES "
                    f"({invalid_count/original_count*100:.1f}%)"
                )
                df = df[valid_mask].copy()

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    logger.info(f"Final dataset: {len(df)} valid molecules")

    return df
