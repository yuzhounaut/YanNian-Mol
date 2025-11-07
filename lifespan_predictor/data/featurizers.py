"""
Molecular featurization module with caching support.

This module provides classes for converting SMILES strings to molecular graph features
with disk caching to avoid redundant computations.
"""

import hashlib
import json
import logging
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import deepchem as dc
import numpy as np
import scipy.sparse as sp
from rdkit import Chem
from tqdm import tqdm

from lifespan_predictor.utils.memory import cleanup_temp_files

logger = logging.getLogger(__name__)


class CachedGraphFeaturizer:
    """
    Graph featurizer with disk caching for molecular features.

    This class converts SMILES strings to graph representations (adjacency matrices
    and node features) and caches the results to disk to avoid recomputation.

    Attributes
    ----------
    cache_dir : str
        Directory for storing cached features
    max_atoms : int
        Maximum number of atoms allowed in a molecule
    atom_feature_dim : int
        Dimension of atom feature vectors
    """

    def __init__(
        self,
        cache_dir: str,
        max_atoms: int = 200,
        atom_feature_dim: int = 75,
        n_jobs: int = -1,
        use_memory_mapping: bool = True,
    ):
        """
        Initialize the CachedGraphFeaturizer.

        Parameters
        ----------
        cache_dir : str
            Directory path for storing cached features
        max_atoms : int, optional
            Maximum number of atoms in molecule (default: 200)
        atom_feature_dim : int, optional
            Dimension of atom features from ConvMolFeaturizer (default: 75)
        n_jobs : int, optional
            Number of parallel jobs for featurization. -1 uses all CPU cores (default: -1)
        use_memory_mapping : bool, optional
            Use memory-mapped arrays for large feature matrices (default: True)

        Examples
        --------
        >>> featurizer = CachedGraphFeaturizer("cache/features", max_atoms=200)
        >>> adj, feat, labels = featurizer.featurize(["CCO", "CC"], labels=np.array([0, 1]))
        """
        self.cache_dir = Path(cache_dir)
        self.max_atoms = max_atoms
        self.atom_feature_dim = atom_feature_dim
        self.n_jobs = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        self.use_memory_mapping = use_memory_mapping

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Clean up any leftover temporary files
        cleanup_temp_files(str(self.cache_dir), "*.mmap")

        logger.info(
            f"Initialized CachedGraphFeaturizer with cache_dir={cache_dir}, "
            f"max_atoms={max_atoms}, atom_feature_dim={atom_feature_dim}, "
            f"n_jobs={self.n_jobs}, use_memory_mapping={use_memory_mapping}"
        )

    def featurize(
        self,
        smiles_list: List[str],
        labels: Optional[np.ndarray] = None,
        force_recompute: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Featurize a list of SMILES strings with caching.

        This method computes adjacency matrices, node features, and similarity graphs
        for molecules. Results are cached based on a hash of the SMILES list.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings to featurize
        labels : Optional[np.ndarray], optional
            Labels corresponding to molecules (default: None)
        force_recompute : bool, optional
            If True, ignore cache and recompute features (default: False)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
            - Adjacency matrices: shape (n_molecules, max_atoms, max_atoms)
            - Node features: shape (n_molecules, max_atoms, atom_feature_dim)
            - Similarity graphs: shape (n_molecules, max_atoms, max_atoms)
            - Labels: shape (n_molecules,) or None if not provided

        Raises
        ------
        ValueError
            If labels are provided but length doesn't match smiles_list

        Examples
        --------
        >>> featurizer = CachedGraphFeaturizer("cache")
        >>> smiles = ["CCO", "CC", "CCC"]
        >>> adj, feat, sim, labels = featurizer.featurize(smiles)
        >>> adj.shape
        (3, 200, 200)
        """
        if labels is not None and len(labels) != len(smiles_list):
            raise ValueError(
                f"Length of labels ({len(labels)}) must match "
                f"length of smiles_list ({len(smiles_list)})"
            )

        # Generate cache key from SMILES list
        cache_key = self._generate_cache_key(smiles_list)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.json"

        # Check if cached features exist and are valid
        if not force_recompute and cache_file.exists():
            if self._validate_cache(cache_file, metadata_file, smiles_list):
                logger.info(f"Loading features from cache: {cache_file}")
                return self._load_from_cache(cache_file, labels)
            else:
                logger.warning("Cache validation failed, recomputing features")

        # Compute features
        logger.info(f"Computing features for {len(smiles_list)} molecules")
        adj_matrices, node_features, sim_graphs, valid_indices = self._compute_features_batch(
            smiles_list
        )

        # Align labels with valid molecules
        aligned_labels = None
        if labels is not None:
            aligned_labels = labels[valid_indices]

        # Save to cache
        self._save_to_cache(
            cache_file,
            metadata_file,
            adj_matrices,
            node_features,
            sim_graphs,
            smiles_list,
            valid_indices,
        )

        return adj_matrices, node_features, sim_graphs, aligned_labels

    def _compute_features_batch(
        self, smiles_list: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Compute features for a batch of SMILES strings with parallel processing.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]
            - Padded adjacency matrices
            - Padded node features
            - Padded similarity graphs
            - List of valid indices from original smiles_list
        """
        all_adj_matrices = []
        all_node_features = []
        all_sim_graphs = []
        valid_indices = []

        # Use parallel processing if n_jobs > 1 and we have enough molecules
        if self.n_jobs > 1 and len(smiles_list) > 10:
            logger.info(f"Using parallel processing with {self.n_jobs} workers")
            results = self._parallel_featurize(smiles_list)
        else:
            logger.info("Using sequential processing")
            results = self._sequential_featurize(smiles_list)

        # Process results
        for idx, (adj, feat, sim, error) in enumerate(results):
            if error is not None:
                logger.warning(
                    f"Error featurizing molecule at index {idx} ({smiles_list[idx]}): {error}"
                )
                continue

            if adj is not None and feat is not None and sim is not None:
                all_adj_matrices.append(adj)
                all_node_features.append(feat)
                all_sim_graphs.append(sim)
                valid_indices.append(idx)
            else:
                logger.debug(f"Skipping invalid molecule at index {idx}: {smiles_list[idx]}")

        if not valid_indices:
            logger.error("No molecules were successfully featurized")
            # Return empty arrays with correct shapes
            empty_adj = np.zeros((0, self.max_atoms, self.max_atoms), dtype=np.float32)
            empty_feat = np.zeros((0, self.max_atoms, self.atom_feature_dim), dtype=np.float32)
            empty_sim = np.zeros((0, self.max_atoms, self.max_atoms), dtype=np.float32)
            return empty_adj, empty_feat, empty_sim, []

        # Pad and stack all features
        padded_adj = np.array(
            [self._pad_array(m, (self.max_atoms, self.max_atoms)) for m in all_adj_matrices],
            dtype=np.float32,
        )

        padded_feat = np.array(
            [
                self._pad_array(f, (self.max_atoms, self.atom_feature_dim))
                for f in all_node_features
            ],
            dtype=np.float32,
        )

        padded_sim = np.array(
            [self._pad_array(s, (self.max_atoms, self.max_atoms)) for s in all_sim_graphs],
            dtype=np.float32,
        )

        logger.info(f"Successfully featurized {len(valid_indices)}/{len(smiles_list)} molecules")

        return padded_adj, padded_feat, padded_sim, valid_indices

    def _sequential_featurize(
        self, smiles_list: List[str]
    ) -> List[
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]
    ]:
        """
        Featurize molecules sequentially with progress bar.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]]
            List of (adj, feat, sim, error) tuples
        """
        results = []
        for smiles in tqdm(smiles_list, desc="Featurizing molecules"):
            try:
                adj, feat, sim = self._compute_features(smiles)
                results.append((adj, feat, sim, None))
            except Exception as e:
                results.append((None, None, None, str(e)))
        return results

    def _parallel_featurize(
        self, smiles_list: List[str]
    ) -> List[
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]
    ]:
        """
        Featurize molecules in parallel with progress bar.

        Parameters
        ----------
        smiles_list : List[str]
            List of SMILES strings

        Returns
        -------
        List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]]
            List of (adj, feat, sim, error) tuples
        """
        try:
            with Pool(processes=self.n_jobs) as pool:
                # Use imap for progress tracking
                results = list(
                    tqdm(
                        pool.imap(self._compute_features_wrapper, smiles_list),
                        total=len(smiles_list),
                        desc="Featurizing molecules (parallel)",
                    )
                )
            return results
        except Exception as e:
            logger.warning(f"Parallel processing failed: {str(e)}. Falling back to sequential.")
            return self._sequential_featurize(smiles_list)

    def _compute_features_wrapper(
        self, smiles: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
        """
        Wrapper for _compute_features to handle exceptions in parallel processing.

        Parameters
        ----------
        smiles : str
            SMILES string

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]
            (adj, feat, sim, error) tuple
        """
        try:
            adj, feat, sim = self._compute_features(smiles)
            return (adj, feat, sim, None)
        except Exception as e:
            return (None, None, None, str(e))

    def _compute_features(
        self, smiles: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute features for a single molecule.

        Parameters
        ----------
        smiles : str
            SMILES string

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]
            - Normalized adjacency matrix
            - Node features
            - Similarity graph
            Returns (None, None, None) if featurization fails
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug(f"Failed to parse SMILES: {smiles}")
            return None, None, None

        num_atoms = mol.GetNumAtoms()

        # Check atom count
        if num_atoms == 0:
            logger.debug(f"Empty molecule: {smiles}")
            return None, None, None

        if num_atoms > self.max_atoms:
            logger.debug(f"Molecule exceeds max_atoms ({num_atoms} > {self.max_atoms}): {smiles}")
            return None, None, None

        try:
            # Compute adjacency matrix
            adj = Chem.GetAdjacencyMatrix(mol).astype(np.float32)

            # Normalize adjacency matrix
            norm_adj = self._preprocess_graph(adj)
            if norm_adj is None:
                return None, None, None

            # Compute node features using DeepChem
            featurizer = dc.feat.ConvMolFeaturizer()
            features_obj = featurizer.featurize([mol])[0]

            if features_obj is None or isinstance(features_obj, bool):
                logger.debug(f"DeepChem featurization failed for: {smiles}")
                return None, None, None

            node_features = features_obj.get_atom_features()

            # Compute similarity graph (adjacency with bond orders + atomic numbers on diagonal)
            Chem.Kekulize(mol)
            atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            sim_graph = Chem.GetAdjacencyMatrix(mol, useBO=True).astype(np.float32)

            # Add atomic numbers to diagonal
            for i, atom_num in enumerate(atoms):
                sim_graph[i, i] = atom_num

            return norm_adj, node_features, sim_graph

        except Exception as e:
            logger.debug(f"Error computing features for {smiles}: {str(e)}")
            return None, None, None

    def _preprocess_graph(self, adj: np.ndarray) -> Optional[np.ndarray]:
        """
        Normalize adjacency matrix using symmetric normalization.

        Applies the transformation: D^(-1/2) * (A + I) * D^(-1/2)
        where A is the adjacency matrix, I is identity, and D is the degree matrix.

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix

        Returns
        -------
        Optional[np.ndarray]
            Normalized adjacency matrix, or None if normalization fails
        """
        if adj is None or not isinstance(adj, np.ndarray):
            return None

        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            return None

        if adj.shape[0] == 0:
            return adj

        # Add self-loops
        adj_with_self_loops = adj + sp.eye(adj.shape[0])

        # Compute degree matrix
        rowsum = np.array(adj_with_self_loops.sum(1)).flatten()

        # Avoid division by zero
        rowsum[rowsum == 0] = 1e-9

        # Symmetric normalization
        degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5))
        adj_normalized = (
            adj_with_self_loops.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
        )

        return np.array(adj_normalized, dtype=np.float32)

    def _pad_array(
        self, arr: np.ndarray, target_shape: Tuple[int, ...], constant_value: float = 0.0
    ) -> np.ndarray:
        """
        Pad array to target shape.

        Parameters
        ----------
        arr : np.ndarray
            Array to pad
        target_shape : Tuple[int, ...]
            Target shape
        constant_value : float, optional
            Value to use for padding (default: 0.0)

        Returns
        -------
        np.ndarray
            Padded array
        """
        if not isinstance(arr, np.ndarray):
            return np.full(target_shape, constant_value, dtype=np.float32)

        pad_width = []
        for i in range(len(target_shape)):
            if i < arr.ndim:
                pad_width.append((0, max(0, target_shape[i] - arr.shape[i])))
            else:
                pad_width.append((0, target_shape[i]))

        return np.pad(arr, pad_width, "constant", constant_values=constant_value)

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
            MD5 hash of the SMILES list
        """
        # Create a deterministic string from SMILES list
        smiles_str = "|".join(sorted(smiles_list))

        # Add configuration to cache key
        config_str = f"{self.max_atoms}_{self.atom_feature_dim}"

        # Generate MD5 hash
        cache_str = f"{smiles_str}_{config_str}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _validate_cache(
        self, cache_file: Path, metadata_file: Path, smiles_list: List[str]
    ) -> bool:
        """
        Validate cached features.

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
            if metadata.get("max_atoms") != self.max_atoms:
                logger.debug("Cache invalid: max_atoms mismatch")
                return False

            if metadata.get("atom_feature_dim") != self.atom_feature_dim:
                logger.debug("Cache invalid: atom_feature_dim mismatch")
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
        adj_matrices: np.ndarray,
        node_features: np.ndarray,
        sim_graphs: np.ndarray,
        smiles_list: List[str],
        valid_indices: List[int],
    ) -> None:
        """
        Save features to cache.

        Parameters
        ----------
        cache_file : Path
            Path to cache file
        metadata_file : Path
            Path to metadata file
        adj_matrices : np.ndarray
            Adjacency matrices
        node_features : np.ndarray
            Node features
        sim_graphs : np.ndarray
            Similarity graphs
        smiles_list : List[str]
            Original SMILES list
        valid_indices : List[int]
            Indices of successfully featurized molecules
        """
        try:
            # Save features
            cache_data = {
                "adj_matrices": adj_matrices,
                "node_features": node_features,
                "sim_graphs": sim_graphs,
                "valid_indices": valid_indices,
            }

            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            metadata = {
                "max_atoms": self.max_atoms,
                "atom_feature_dim": self.atom_feature_dim,
                "smiles_list": smiles_list,
                "valid_indices": valid_indices,
                "n_molecules": len(smiles_list),
                "n_valid": len(valid_indices),
            }

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Saved features to cache: {cache_file}")

        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
            # Clean up partial files
            if cache_file.exists():
                cache_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()

    def _load_from_cache(
        self, cache_file: Path, labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load features from cache.

        Parameters
        ----------
        cache_file : Path
            Path to cache file
        labels : Optional[np.ndarray], optional
            Labels to align with cached features

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
            - Adjacency matrices
            - Node features
            - Similarity graphs
            - Aligned labels (if provided)
        """
        try:
            with open(cache_file, "rb") as f:
                cache_data = pickle.load(f)

            adj_matrices = cache_data["adj_matrices"]
            node_features = cache_data["node_features"]
            sim_graphs = cache_data["sim_graphs"]
            valid_indices = cache_data["valid_indices"]

            # Align labels if provided
            aligned_labels = None
            if labels is not None:
                aligned_labels = labels[valid_indices]

            return adj_matrices, node_features, sim_graphs, aligned_labels

        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            raise
