"""
Molecular fingerprint generation module with caching support.

This module provides classes for generating various types of molecular fingerprints
(Morgan, RDKit, MACCS) with efficient batch processing and disk caching.
"""

import hashlib
import json
import logging
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FingerprintGenerator:
    """
    Unified fingerprint generator with batching and caching.

    This class generates multiple types of molecular fingerprints efficiently
    using batch processing and optional disk caching.

    Attributes
    ----------
    morgan_radius : int
        Radius for Morgan fingerprints
    morgan_nbits : int
        Number of bits for Morgan fingerprints
    rdkit_fp_nbits : int
        Number of bits for RDKit topological fingerprints
    maccs_nbits : int
        Number of bits for MACCS keys (always 166)
    n_jobs : int
        Number of parallel jobs for processing
    """

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_nbits: int = 2048,
        rdkit_fp_nbits: int = 2048,
        maccs_nbits: int = 166,
        n_jobs: int = -1,
    ):
        """
        Initialize the FingerprintGenerator.

        Parameters
        ----------
        morgan_radius : int, optional
            Radius for Morgan fingerprints (default: 2)
        morgan_nbits : int, optional
            Number of bits for Morgan fingerprints (default: 2048)
        rdkit_fp_nbits : int, optional
            Number of bits for RDKit fingerprints (default: 2048)
        maccs_nbits : int, optional
            Number of bits for MACCS keys (default: 166, fixed)
        n_jobs : int, optional
            Number of parallel jobs. -1 uses all CPU cores (default: -1)

        Raises
        ------
        ValueError
            If maccs_nbits is not 166 (MACCS keys have fixed size)

        Examples
        --------
        >>> generator = FingerprintGenerator(morgan_radius=2, morgan_nbits=2048)
        >>> hashed_fps, non_hashed_fps = generator.generate_fingerprints(["CCO", "CC"])
        """
        if maccs_nbits != 166:
            raise ValueError("MACCS keys must have exactly 166 bits")

        self.morgan_radius = morgan_radius
        self.morgan_nbits = morgan_nbits
        self.rdkit_fp_nbits = rdkit_fp_nbits
        self.maccs_nbits = maccs_nbits
        # RDKit MACCS keys actually return 167 bits (first bit is always 0)
        self._maccs_actual_nbits = 167
        self.n_jobs = cpu_count() if n_jobs == -1 else max(1, n_jobs)

        logger.info(
            f"Initialized FingerprintGenerator with morgan_radius={morgan_radius}, "
            f"morgan_nbits={morgan_nbits}, rdkit_fp_nbits={rdkit_fp_nbits}, "
            f"maccs_nbits={maccs_nbits}, n_jobs={self.n_jobs}"
        )

    def generate_fingerprints(
        self, smiles_list: List[str], cache_dir: Optional[str] = None, force_recompute: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate hashed and non-hashed fingerprints for a list of SMILES.

        This method generates three types of fingerprints:
        - Morgan (hashed): Circular fingerprints with hashing
        - RDKit (hashed): Topological fingerprints with hashing
        - MACCS (non-hashed): MACCS structural keys

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to process
        cache_dir : Optional[str], optional
            Directory for caching fingerprints. If None, no caching (default: None)
        force_recompute : bool, optional
            If True, ignore cache and recompute (default: False)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Hashed fingerprints: shape (n_molecules, morgan_nbits + rdkit_fp_nbits)
              Concatenation of Morgan and RDKit fingerprints
            - Non-hashed fingerprints: shape (n_molecules, maccs_nbits)
              MACCS keys

        Examples
        --------
        >>> generator = FingerprintGenerator()
        >>> smiles = ["CCO", "c1ccccc1", "CC(C)C"]
        >>> hashed, non_hashed = generator.generate_fingerprints(smiles)
        >>> hashed.shape
        (3, 4096)
        >>> non_hashed.shape
        (3, 166)
        """
        if not smiles_list:
            logger.warning("Empty SMILES list provided")
            hashed_dim = self.morgan_nbits + self.rdkit_fp_nbits
            return (
                np.zeros((0, hashed_dim), dtype=np.float32),
                np.zeros((0, self.maccs_nbits), dtype=np.float32),
            )

        # Check cache if directory provided
        if cache_dir is not None and not force_recompute:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            cache_key = self._generate_cache_key(smiles_list)
            cache_file = cache_path / f"{cache_key}_fingerprints.pkl"
            metadata_file = cache_path / f"{cache_key}_fingerprints_metadata.json"

            if cache_file.exists() and self._validate_cache(cache_file, metadata_file, smiles_list):
                logger.info(f"Loading fingerprints from cache: {cache_file}")
                return self._load_from_cache(cache_file)

        # Compute fingerprints
        logger.info(f"Computing fingerprints for {len(smiles_list)} molecules")

        # Convert SMILES to molecules
        mols = []
        valid_indices = []
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(idx)
            else:
                logger.warning(f"Failed to parse SMILES at index {idx}: {smiles}")

        if not mols:
            logger.error("No valid molecules to process")
            hashed_dim = self.morgan_nbits + self.rdkit_fp_nbits
            return (
                np.zeros((0, hashed_dim), dtype=np.float32),
                np.zeros((0, self.maccs_nbits), dtype=np.float32),
            )

        logger.info(f"Successfully parsed {len(mols)}/{len(smiles_list)} molecules")

        # Generate fingerprints in batches
        morgan_fps = self._batch_compute_morgan(mols)
        rdkit_fps = self._batch_compute_rdkit(mols)
        maccs_fps = self._batch_compute_maccs(mols)

        # Validate dimensions
        self._validate_dimensions(morgan_fps, rdkit_fps, maccs_fps)

        # Concatenate hashed fingerprints
        hashed_fps = np.concatenate([morgan_fps, rdkit_fps], axis=1)

        # Save to cache if directory provided
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            cache_key = self._generate_cache_key(smiles_list)
            cache_file = cache_path / f"{cache_key}_fingerprints.pkl"
            metadata_file = cache_path / f"{cache_key}_fingerprints_metadata.json"

            self._save_to_cache(
                cache_file, metadata_file, hashed_fps, maccs_fps, smiles_list, valid_indices
            )

        return hashed_fps, maccs_fps

    def _batch_compute_morgan(self, mols: List[Chem.Mol]) -> np.ndarray:
        """
        Compute Morgan fingerprints in batch with parallel processing.

        Parameters
        ----------
        mols : List[Chem.Mol]
            List of RDKit molecule objects

        Returns
        -------
        np.ndarray
            Morgan fingerprints, shape (n_molecules, morgan_nbits)
        """
        logger.debug(f"Computing Morgan fingerprints for {len(mols)} molecules")

        if self.n_jobs > 1 and len(mols) > 10:
            # Parallel processing
            with Pool(processes=self.n_jobs) as pool:
                fps = list(
                    tqdm(
                        pool.imap(self._compute_morgan_single, mols),
                        total=len(mols),
                        desc="Morgan fingerprints",
                    )
                )
        else:
            # Sequential processing
            fps = [
                self._compute_morgan_single(mol) for mol in tqdm(mols, desc="Morgan fingerprints")
            ]

        return np.array(fps, dtype=np.float32)

    def _compute_morgan_single(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute Morgan fingerprint for a single molecule.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object

        Returns
        -------
        np.ndarray
            Morgan fingerprint bit vector
        """
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=self.morgan_radius, nBits=self.morgan_nbits
            )
            return np.array(fp, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Error computing Morgan fingerprint: {str(e)}")
            return np.zeros(self.morgan_nbits, dtype=np.float32)

    def _batch_compute_rdkit(self, mols: List[Chem.Mol]) -> np.ndarray:
        """
        Compute RDKit topological fingerprints in batch with parallel processing.

        Parameters
        ----------
        mols : List[Chem.Mol]
            List of RDKit molecule objects

        Returns
        -------
        np.ndarray
            RDKit fingerprints, shape (n_molecules, rdkit_fp_nbits)
        """
        logger.debug(f"Computing RDKit fingerprints for {len(mols)} molecules")

        if self.n_jobs > 1 and len(mols) > 10:
            # Parallel processing
            with Pool(processes=self.n_jobs) as pool:
                fps = list(
                    tqdm(
                        pool.imap(self._compute_rdkit_single, mols),
                        total=len(mols),
                        desc="RDKit fingerprints",
                    )
                )
        else:
            # Sequential processing
            fps = [self._compute_rdkit_single(mol) for mol in tqdm(mols, desc="RDKit fingerprints")]

        return np.array(fps, dtype=np.float32)

    def _compute_rdkit_single(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute RDKit fingerprint for a single molecule.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object

        Returns
        -------
        np.ndarray
            RDKit fingerprint bit vector
        """
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=self.rdkit_fp_nbits)
            return np.array(fp, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Error computing RDKit fingerprint: {str(e)}")
            return np.zeros(self.rdkit_fp_nbits, dtype=np.float32)

    def _batch_compute_maccs(self, mols: List[Chem.Mol]) -> np.ndarray:
        """
        Compute MACCS keys in batch with parallel processing.

        Parameters
        ----------
        mols : List[Chem.Mol]
            List of RDKit molecule objects

        Returns
        -------
        np.ndarray
            MACCS keys, shape (n_molecules, 166)
        """
        logger.debug(f"Computing MACCS keys for {len(mols)} molecules")

        if self.n_jobs > 1 and len(mols) > 10:
            # Parallel processing
            with Pool(processes=self.n_jobs) as pool:
                fps = list(
                    tqdm(
                        pool.imap(self._compute_maccs_single, mols),
                        total=len(mols),
                        desc="MACCS keys",
                    )
                )
        else:
            # Sequential processing
            fps = [self._compute_maccs_single(mol) for mol in tqdm(mols, desc="MACCS keys")]

        return np.array(fps, dtype=np.float32)

    def _compute_maccs_single(self, mol: Chem.Mol) -> np.ndarray:
        """
        Compute MACCS keys for a single molecule.

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object

        Returns
        -------
        np.ndarray
            MACCS keys bit vector (166 bits, excluding first bit)
        """
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            # RDKit returns 167 bits, but first bit is always 0
            # We skip it to get the standard 166 MACCS keys
            fp_array = np.array(fp, dtype=np.float32)
            return fp_array[1:]  # Skip first bit
        except Exception as e:
            logger.warning(f"Error computing MACCS keys: {str(e)}")
            return np.zeros(self.maccs_nbits, dtype=np.float32)

    def _validate_dimensions(
        self, morgan_fps: np.ndarray, rdkit_fps: np.ndarray, maccs_fps: np.ndarray
    ) -> None:
        """
        Validate fingerprint dimensions match configuration.

        Parameters
        ----------
        morgan_fps : np.ndarray
            Morgan fingerprints
        rdkit_fps : np.ndarray
            RDKit fingerprints
        maccs_fps : np.ndarray
            MACCS keys

        Raises
        ------
        ValueError
            If dimensions don't match configuration
        """
        if morgan_fps.shape[1] != self.morgan_nbits:
            raise ValueError(
                f"Morgan fingerprint dimension mismatch: "
                f"expected {self.morgan_nbits}, got {morgan_fps.shape[1]}"
            )

        if rdkit_fps.shape[1] != self.rdkit_fp_nbits:
            raise ValueError(
                f"RDKit fingerprint dimension mismatch: "
                f"expected {self.rdkit_fp_nbits}, got {rdkit_fps.shape[1]}"
            )

        if maccs_fps.shape[1] != self.maccs_nbits:
            raise ValueError(
                f"MACCS keys dimension mismatch: "
                f"expected {self.maccs_nbits}, got {maccs_fps.shape[1]}"
            )

        if not (morgan_fps.shape[0] == rdkit_fps.shape[0] == maccs_fps.shape[0]):
            raise ValueError(
                f"Number of molecules mismatch: "
                f"Morgan={morgan_fps.shape[0]}, "
                f"RDKit={rdkit_fps.shape[0]}, "
                f"MACCS={maccs_fps.shape[0]}"
            )

        logger.debug("Fingerprint dimensions validated successfully")

    def _generate_cache_key(self, smiles_list: List[str]) -> str:
        """
        Generate a unique cache key from SMILES list.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        str
            MD5 hash of the SMILES list and configuration
        """
        # Create deterministic string from SMILES list
        smiles_str = "|".join(sorted(smiles_list))

        # Add configuration to cache key
        config_str = (
            f"{self.morgan_radius}_{self.morgan_nbits}_" f"{self.rdkit_fp_nbits}_{self.maccs_nbits}"
        )

        # Generate MD5 hash
        cache_str = f"{smiles_str}_{config_str}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _validate_cache(
        self, cache_file: Path, metadata_file: Path, smiles_list: List[str]
    ) -> bool:
        """
        Validate cached fingerprints.

        Parameters
        ----------
        cache_file : Path
            Path to cache file
        metadata_file : Path
            Path to metadata file
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        bool
            True if cache is valid, False otherwise
        """
        try:
            # Check if both files exist
            if not cache_file.exists() or not metadata_file.exists():
                return False

            # Load metadata
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Validate configuration
            if metadata.get("morgan_radius") != self.morgan_radius:
                logger.debug("Cache invalid: morgan_radius mismatch")
                return False

            if metadata.get("morgan_nbits") != self.morgan_nbits:
                logger.debug("Cache invalid: morgan_nbits mismatch")
                return False

            if metadata.get("rdkit_fp_nbits") != self.rdkit_fp_nbits:
                logger.debug("Cache invalid: rdkit_fp_nbits mismatch")
                return False

            if metadata.get("maccs_nbits") != self.maccs_nbits:
                logger.debug("Cache invalid: maccs_nbits mismatch")
                return False

            # Validate SMILES list
            cached_smiles = metadata.get("smiles_list", [])
            if sorted(cached_smiles) != sorted(smiles_list):
                logger.debug("Cache invalid: SMILES list mismatch")
                return False

            # Check file integrity
            if cache_file.stat().st_size == 0:
                logger.debug("Cache invalid: empty cache file")
                return False

            return True

        except Exception as e:
            logger.debug(f"Cache validation error: {str(e)}")
            return False

    def _save_to_cache(
        self,
        cache_file: Path,
        metadata_file: Path,
        hashed_fps: np.ndarray,
        maccs_fps: np.ndarray,
        smiles_list: List[str],
        valid_indices: List[int],
    ) -> None:
        """
        Save fingerprints to cache.

        Parameters
        ----------
        cache_file : Path
            Path to cache file
        metadata_file : Path
            Path to metadata file
        hashed_fps : np.ndarray
            Hashed fingerprints (Morgan + RDKit)
        maccs_fps : np.ndarray
            MACCS keys
        smiles_list : List[str]
            Original SMILES list
        valid_indices : List[int]
            Indices of successfully processed molecules
        """
        try:
            # Save fingerprints
            cache_data = {
                "hashed_fps": hashed_fps,
                "maccs_fps": maccs_fps,
                "valid_indices": valid_indices,
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            metadata = {
                "morgan_radius": self.morgan_radius,
                "morgan_nbits": self.morgan_nbits,
                "rdkit_fp_nbits": self.rdkit_fp_nbits,
                "maccs_nbits": self.maccs_nbits,
                "smiles_list": smiles_list,
                "valid_indices": valid_indices,
                "n_molecules": len(smiles_list),
                "n_valid": len(valid_indices),
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved fingerprints to cache: {cache_file}")

        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
            # Clean up partial files
            if cache_file.exists():
                cache_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()

    def _load_from_cache(self, cache_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load fingerprints from cache.

        Parameters
        ----------
        cache_file : Path
            Path to cache file

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Hashed fingerprints (Morgan + RDKit)
            - MACCS keys
        """
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            hashed_fps = cache_data["hashed_fps"]
            maccs_fps = cache_data["maccs_fps"]

            return hashed_fps, maccs_fps

        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            raise
