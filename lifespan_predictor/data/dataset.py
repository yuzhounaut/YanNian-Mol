"""
Dataset classes for lifespan prediction.

This module provides PyTorch Geometric dataset classes and utilities for
loading and batching molecular data with graph features and fingerprints.
"""

import logging
from typing import Callable, List, Optional, Tuple

import dgl
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


class LifespanDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for lifespan prediction.

    This dataset stores molecular data including graph features, fingerprints,
    and labels. It creates PyG Data objects that can be efficiently batched
    for training.

    Attributes
    ----------
    smiles_list : List[str]
        List of SMILES strings
    graph_features : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of (adjacency_matrices, node_features, similarity_graphs)
    fingerprints : Tuple[np.ndarray, np.ndarray]
        Tuple of (hashed_fingerprints, non_hashed_fingerprints)
    labels : Optional[np.ndarray]
        Labels for supervised learning
    """

    def __init__(
        self,
        root: str,
        smiles_list: List[str],
        graph_features: Tuple[np.ndarray, np.ndarray, np.ndarray],
        fingerprints: Tuple[np.ndarray, np.ndarray],
        labels: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        """
        Initialize the LifespanDataset.

        Parameters
        ----------
        root : str
            Root directory for storing processed data
        smiles_list : List[str]
            List of SMILES strings
        graph_features : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Tuple of (adjacency_matrices, node_features, similarity_graphs)
            - adjacency_matrices: shape (n_molecules, max_atoms, max_atoms)
            - node_features: shape (n_molecules, max_atoms, atom_feature_dim)
            - similarity_graphs: shape (n_molecules, max_atoms, max_atoms)
        fingerprints : Tuple[np.ndarray, np.ndarray]
            Tuple of (hashed_fingerprints, non_hashed_fingerprints)
            - hashed_fingerprints: shape (n_molecules, hashed_fp_dim)
            - non_hashed_fingerprints: shape (n_molecules, non_hashed_fp_dim)
        labels : Optional[np.ndarray], optional
            Labels for supervised learning, shape (n_molecules,) (default: None)
        transform : Optional[Callable], optional
            Transform to apply to each Data object (default: None)
        pre_transform : Optional[Callable], optional
            Transform to apply before saving (default: None)

        Raises
        ------
        ValueError
            If input dimensions don't match

        Examples
        --------
        >>> from lifespan_predictor.data import CachedGraphFeaturizer, FingerprintGenerator
        >>> featurizer = CachedGraphFeaturizer("cache/features")
        >>> fp_gen = FingerprintGenerator()
        >>> smiles = ["CCO", "CC", "CCC"]
        >>> adj, feat, sim, labels = featurizer.featurize(smiles, labels=np.array([0, 1, 0]))
        >>> hashed_fps, non_hashed_fps = fp_gen.generate_fingerprints(smiles)
        >>> dataset = LifespanDataset(
        ...     root="data/processed",
        ...     smiles_list=smiles,
        ...     graph_features=(adj, feat, sim),
        ...     fingerprints=(hashed_fps, non_hashed_fps),
        ...     labels=labels
        ... )
        """
        self.smiles_list = smiles_list
        self.adj_matrices, self.node_features, self.sim_graphs = graph_features
        self.hashed_fps, self.non_hashed_fps = fingerprints
        self.labels_array = labels

        # Validate dimensions
        self._validate_dimensions()

        # Store data list for processing
        self._data_list = None

        super(LifespanDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

        logger.info(
            f"Loaded LifespanDataset with {len(self)} samples from {self.processed_paths[0]}"
        )

    def _validate_dimensions(self) -> None:
        """
        Validate that all input dimensions match.

        Raises
        ------
        ValueError
            If dimensions don't match
        """
        n_smiles = len(self.smiles_list)
        n_adj = len(self.adj_matrices)
        n_feat = len(self.node_features)
        n_sim = len(self.sim_graphs)
        n_hashed = len(self.hashed_fps)
        n_non_hashed = len(self.non_hashed_fps)

        if not (n_smiles == n_adj == n_feat == n_sim == n_hashed == n_non_hashed):
            raise ValueError(
                f"Dimension mismatch: SMILES={n_smiles}, Adj={n_adj}, "
                f"Feat={n_feat}, Sim={n_sim}, Hashed_FP={n_hashed}, "
                f"Non_Hashed_FP={n_non_hashed}"
            )

        if self.labels_array is not None and len(self.labels_array) != n_smiles:
            raise ValueError(
                f"Labels dimension mismatch: Labels={len(self.labels_array)}, " f"SMILES={n_smiles}"
            )

        logger.debug(f"Validated dimensions for {n_smiles} molecules")

    @property
    def raw_file_names(self) -> List[str]:
        """
        List of raw file names (not used in this implementation).

        Returns
        -------
        List[str]
            Empty list
        """
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """
        List of processed file names.

        Returns
        -------
        List[str]
            List containing the processed data filename
        """
        return ["lifespan_data.pt"]

    def download(self) -> None:
        """
        Download raw data (not needed in this implementation).
        """

    def process(self) -> None:
        """
        Process raw data and save to disk.

        This method creates PyG Data objects for each molecule and saves
        them in a collated format.
        """
        logger.info(f"Processing {len(self.smiles_list)} molecules...")

        data_list = []
        skipped_indices = []

        for idx in range(len(self.smiles_list)):
            try:
                data = self._create_pyg_data(idx)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            except Exception as e:
                logger.warning(
                    f"Failed to create Data object for molecule {idx} "
                    f"({self.smiles_list[idx]}): {str(e)}"
                )
                skipped_indices.append(idx)

        if skipped_indices:
            logger.warning(f"Skipped {len(skipped_indices)}/{len(self.smiles_list)} molecules")

        if not data_list:
            raise ValueError("No valid Data objects created")

        # Collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        logger.info(
            f"Processed {len(data_list)} molecules and saved to " f"{self.processed_paths[0]}"
        )

    def _create_pyg_data(self, idx: int) -> Data:
        """
        Create a PyG Data object for a single molecule.

        Parameters
        ----------
        idx : int
            Index of the molecule

        Returns
        -------
        Data
            PyG Data object containing all features

        Raises
        ------
        ValueError
            If SMILES parsing fails or features are invalid
        """
        smiles = self.smiles_list[idx]

        # Parse SMILES to get actual number of atoms
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            raise ValueError(f"Empty molecule: {smiles}")

        # Extract unpadded features for this molecule
        # The stored features are padded to max_atoms, we need to extract the actual atoms
        adj_matrix = self.adj_matrices[idx][:num_atoms, :num_atoms]
        node_feat = self.node_features[idx][:num_atoms, :]
        sim_graph = self.sim_graphs[idx][:num_atoms, :num_atoms]

        # Convert adjacency matrix to edge_index (PyG format)
        edge_index = self._adj_to_edge_index(adj_matrix)

        # Create PyG Data object with only the essential graph data
        data = Data(
            x=torch.from_numpy(node_feat).float(),
            edge_index=edge_index,
            hashed_fp=torch.from_numpy(self.hashed_fps[idx]).float(),
            non_hashed_fp=torch.from_numpy(self.non_hashed_fps[idx]).float(),
        )

        # Add label if available
        if self.labels_array is not None:
            data.y = torch.tensor([self.labels_array[idx]], dtype=torch.float32)

        # Store metadata as Python attributes (not tensor attributes)
        # This prevents PyG from trying to collate them
        data.__dict__["_smiles"] = smiles
        data.__dict__["_num_atoms"] = num_atoms
        data.__dict__["_adj_matrix"] = torch.from_numpy(adj_matrix).float()
        data.__dict__["_sim_graph"] = torch.from_numpy(sim_graph).float()

        return data

    def _adj_to_edge_index(self, adj_matrix: np.ndarray) -> torch.Tensor:
        """
        Convert adjacency matrix to edge_index format.

        Parameters
        ----------
        adj_matrix : np.ndarray
            Adjacency matrix, shape (num_atoms, num_atoms)

        Returns
        -------
        torch.Tensor
            Edge index in PyG format, shape (2, num_edges)
        """
        # Find non-zero entries (edges)
        # Use a threshold to handle floating point adjacency matrices
        edge_indices = np.where(adj_matrix > 0.01)

        if len(edge_indices[0]) == 0:
            # No edges found, return empty edge_index with correct shape
            return torch.zeros((2, 0), dtype=torch.long)

        edge_index = torch.tensor(np.array([edge_indices[0], edge_indices[1]]), dtype=torch.long)
        return edge_index

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples
        """
        return len(self.smiles_list)

    def __repr__(self) -> str:
        """
        String representation of the dataset.

        Returns
        -------
        str
            String representation
        """
        return f"{self.__class__.__name__}({len(self)} samples, " f"root={self.root})"


class GraphDataBuilder:
    """
    Builder for DGL graph objects from molecular data.

    This class creates DGL graphs that can be used with DGL-LifeSci models
    like AttentiveFP. It reuses featurizers for efficiency.

    Attributes
    ----------
    use_edge_features : bool
        Whether to include edge features in the graph
    node_featurizer : dc.feat.ConvMolFeaturizer
        DeepChem featurizer for node features
    edge_featurizer : dc.feat.MolGraphConvFeaturizer
        DeepChem featurizer for edge features
    """

    def __init__(self, use_edge_features: bool = True):
        """
        Initialize the GraphDataBuilder.

        Parameters
        ----------
        use_edge_features : bool, optional
            Whether to include edge features (default: True)

        Examples
        --------
        >>> builder = GraphDataBuilder(use_edge_features=True)
        >>> graph = builder.build_dgl_graph("CCO", node_features)
        """
        self.use_edge_features = use_edge_features

        # Initialize featurizers once for reuse
        try:
            import deepchem as dc

            self.node_featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
            self.edge_featurizer = dc.feat.MolGraphConvFeaturizer(
                use_edges=True, use_partial_charge=True
            )
        except ImportError:
            raise ImportError(
                "DeepChem is required for GraphDataBuilder. "
                "Install it with: pip install deepchem"
            )

        logger.info(f"Initialized GraphDataBuilder with use_edge_features={use_edge_features}")

    def build_dgl_graph(
        self, smiles: str, node_features: Optional[np.ndarray] = None, add_self_loop: bool = True
    ) -> dgl.DGLGraph:
        """
        Build a DGL graph from SMILES and optional node features.

        Parameters
        ----------
        smiles : str
            SMILES string
        node_features : Optional[np.ndarray], optional
            Pre-computed node features. If None, will compute from SMILES (default: None)
        add_self_loop : bool, optional
            Whether to add self-loops to the graph (default: True)

        Returns
        -------
        dgl.DGLGraph
            DGL graph with node and edge features

        Raises
        ------
        ValueError
            If SMILES parsing fails or featurization fails

        Examples
        --------
        >>> builder = GraphDataBuilder()
        >>> graph = builder.build_dgl_graph("CCO")
        >>> graph.ndata['x'].shape
        torch.Size([3, 78])
        """
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Failed to parse SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            raise ValueError(f"Empty molecule: {smiles}")

        # Get node features
        if node_features is None:
            node_feat_obj = self.node_featurizer.featurize([mol])[0]
            if node_feat_obj is None or isinstance(node_feat_obj, bool):
                raise ValueError(f"Node featurization failed for: {smiles}")
            node_features = node_feat_obj.get_atom_features()

        # Get edge information
        edge_feat_obj = self.edge_featurizer.featurize([mol])[0]
        if edge_feat_obj is None or isinstance(edge_feat_obj, bool):
            raise ValueError(f"Edge featurization failed for: {smiles}")

        edge_index = edge_feat_obj.edge_index  # Shape: (2, num_edges)
        edge_features = edge_feat_obj.edge_features if self.use_edge_features else None

        # Create DGL graph
        src = edge_index[0]
        dst = edge_index[1]
        g = dgl.graph(
            (torch.from_numpy(src).long(), torch.from_numpy(dst).long()), num_nodes=num_atoms
        )

        # Add node features
        g.ndata["x"] = torch.from_numpy(node_features).float()

        # Add edge features if available
        if edge_features is not None:
            g.edata["edge_attr"] = torch.from_numpy(edge_features).float()

        # Add self-loops
        if add_self_loop:
            g = dgl.add_self_loop(g)

        return g

    def build_dgl_graph_from_data(self, data: Data, add_self_loop: bool = True) -> dgl.DGLGraph:
        """
        Build a DGL graph from a PyG Data object.

        Parameters
        ----------
        data : Data
            PyG Data object containing SMILES and features
        add_self_loop : bool, optional
            Whether to add self-loops (default: True)

        Returns
        -------
        dgl.DGLGraph
            DGL graph

        Examples
        --------
        >>> builder = GraphDataBuilder()
        >>> # Assume data is a PyG Data object
        >>> graph = builder.build_dgl_graph_from_data(data)
        """
        # Get SMILES from __dict__ if available
        smiles = data.__dict__.get("_smiles", None)
        if smiles is None:
            raise ValueError("Data object does not contain SMILES information")

        return self.build_dgl_graph(
            smiles=smiles,
            node_features=data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x,
            add_self_loop=add_self_loop,
        )


def collate_lifespan_data(batch: List[Data]) -> Data:
    """
    Collate function for batching LifespanDataset samples.

    This function handles batching of PyG Data objects with DGL graphs.
    It creates DGL graphs on-the-fly and batches them properly.

    Parameters
    ----------
    batch : List[Data]
        List of PyG Data objects

    Returns
    -------
    Data
        Batched PyG Data object with additional dgl_graph attribute

    Examples
    --------
    >>> from torch_geometric.loader import DataLoader
    >>> dataset = LifespanDataset(...)
    >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_lifespan_data)
    >>> for batch in loader:
    ...     # batch.dgl_graph contains batched DGL graphs
    ...     pass
    """
    # Use PyG's default collate for most attributes
    from torch_geometric.data import Batch

    batched_data = Batch.from_data_list(batch)

    # Create DGL graphs for each sample and batch them
    builder = GraphDataBuilder(use_edge_features=True)
    dgl_graphs = []

    for data in batch:
        try:
            # Get SMILES from __dict__ if available
            smiles = data.__dict__.get("_smiles", None)
            if smiles is None:
                # Fallback: try to reconstruct from node features
                logger.warning("SMILES not found in data object, using fallback")
                # Create a simple graph from the data
                num_nodes = data.x.shape[0]
                g = dgl.graph(([], []), num_nodes=num_nodes)
                g.ndata["x"] = data.x
                dgl_graphs.append(g)
                continue

            # Build DGL graph from SMILES and node features
            dgl_graph = builder.build_dgl_graph(
                smiles=smiles,
                node_features=data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x,
                add_self_loop=True,
            )
            dgl_graphs.append(dgl_graph)
        except Exception as e:
            logger.warning(f"Failed to create DGL graph: {str(e)}")
            # Create a dummy single-node graph as fallback
            dummy_graph = dgl.graph(([], []), num_nodes=1)
            dummy_graph.ndata["x"] = torch.zeros(1, 78)  # Default node feature dim
            dgl_graphs.append(dummy_graph)

    # Batch DGL graphs
    if dgl_graphs:
        batched_dgl_graph = dgl.batch(dgl_graphs)
        # Store as a regular attribute in __dict__ to avoid PyG collation issues
        batched_data.__dict__["dgl_graph"] = batched_dgl_graph

    return batched_data


def create_dataloader(
    dataset: LifespanDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for LifespanDataset with custom collate function.

    Parameters
    ----------
    dataset : LifespanDataset
        Dataset to load
    batch_size : int, optional
        Batch size (default: 32)
    shuffle : bool, optional
        Whether to shuffle data (default: True)
    num_workers : int, optional
        Number of worker processes (default: 0)
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyG DataLoader with custom collate function

    Examples
    --------
    >>> dataset = LifespanDataset(...)
    >>> loader = create_dataloader(dataset, batch_size=32, shuffle=True)
    >>> for batch in loader:
    ...     # Process batch
    ...     pass
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_lifespan_data,
        **kwargs,
    )
