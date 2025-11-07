"""
AttentiveFP GNN module using DGL-LifeSci.

This module implements the AttentiveFP (Attentive Fingerprint) graph neural network
for molecular property prediction using the DGL-LifeSci library.
"""

import logging
from typing import List

import dgl
import torch
import torch.nn as nn
from dgllife.model import AttentiveFPGNN, AttentiveFPReadout

logger = logging.getLogger(__name__)


class AttentiveFPModule(nn.Module):
    """
    AttentiveFP GNN module for molecular graph embedding.

    This module uses the AttentiveFP architecture from DGL-LifeSci to process
    molecular graphs and generate fixed-size graph embeddings. It includes
    dropout and layer normalization for regularization.

    The AttentiveFP model uses graph attention mechanisms to learn molecular
    representations by aggregating information from neighboring atoms.

    Parameters
    ----------
    node_feat_size : int
        Dimension of input node features
    edge_feat_size : int
        Dimension of input edge features
    num_layers : int, optional
        Number of GNN layers (default: 2)
    num_timesteps : int, optional
        Number of timesteps for graph attention (default: 2)
    graph_feat_size : int, optional
        Dimension of output graph embedding (default: 128)
    dropout : float, optional
        Dropout rate (default: 0.5)

    Attributes
    ----------
    gnn : AttentiveFPGNN
        The GNN component for node embedding
    readout : AttentiveFPReadout
        The readout component for graph-level embedding
    dropout : nn.Dropout
        Dropout layer
    layer_norm : nn.LayerNorm
        Layer normalization

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> # Create a simple graph
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 0]))
    >>> g.ndata['x'] = torch.randn(3, 78)
    >>> g.edata['edge_attr'] = torch.randn(3, 11)
    >>> # Create model
    >>> model = AttentiveFPModule(
    ...     node_feat_size=78,
    ...     edge_feat_size=11,
    ...     num_layers=2,
    ...     num_timesteps=2,
    ...     graph_feat_size=128,
    ...     dropout=0.5
    ... )
    >>> # Forward pass
    >>> embedding = model([g])
    >>> embedding.shape
    torch.Size([1, 128])
    """

    def __init__(
        self,
        node_feat_size: int,
        edge_feat_size: int,
        num_layers: int = 2,
        num_timesteps: int = 2,
        graph_feat_size: int = 128,
        dropout: float = 0.5,
    ):
        """
        Initialize the AttentiveFPModule.

        Parameters
        ----------
        node_feat_size : int
            Dimension of input node features
        edge_feat_size : int
            Dimension of input edge features
        num_layers : int, optional
            Number of GNN layers (default: 2)
        num_timesteps : int, optional
            Number of timesteps for graph attention (default: 2)
        graph_feat_size : int, optional
            Dimension of output graph embedding (default: 128)
        dropout : float, optional
            Dropout rate (default: 0.5)
        """
        super(AttentiveFPModule, self).__init__()

        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.graph_feat_size = graph_feat_size
        self.dropout_rate = dropout

        # AttentiveFP GNN for node-level processing
        self.gnn = AttentiveFPGNN(
            node_feat_size=node_feat_size,
            edge_feat_size=edge_feat_size,
            num_layers=num_layers,
            graph_feat_size=graph_feat_size,
            dropout=dropout,
        )

        # AttentiveFP Readout for graph-level embedding
        self.readout = AttentiveFPReadout(
            feat_size=graph_feat_size, num_timesteps=num_timesteps, dropout=dropout
        )

        # Additional regularization layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(graph_feat_size)

        logger.info(
            f"Initialized AttentiveFPModule with node_feat_size={node_feat_size}, "
            f"edge_feat_size={edge_feat_size}, num_layers={num_layers}, "
            f"num_timesteps={num_timesteps}, graph_feat_size={graph_feat_size}, "
            f"dropout={dropout}"
        )

    def forward(self, dgl_graphs: List[dgl.DGLGraph]) -> torch.Tensor:
        """
        Forward pass to generate graph embeddings.

        Parameters
        ----------
        dgl_graphs : List[dgl.DGLGraph]
            List of DGL graphs or a batched DGL graph
            Each graph should have:
            - ndata['x']: node features, shape (num_nodes, node_feat_size)
            - edata['edge_attr']: edge features, shape (num_edges, edge_feat_size)

        Returns
        -------
        torch.Tensor
            Graph embeddings, shape (batch_size, graph_feat_size)

        Raises
        ------
        ValueError
            If graphs don't have required node or edge features

        Examples
        --------
        >>> model = AttentiveFPModule(78, 11, graph_feat_size=128)
        >>> # Create batched graph
        >>> g1 = dgl.graph(([0, 1], [1, 0]))
        >>> g1.ndata['x'] = torch.randn(2, 78)
        >>> g1.edata['edge_attr'] = torch.randn(2, 11)
        >>> g2 = dgl.graph(([0, 1, 2], [1, 2, 0]))
        >>> g2.ndata['x'] = torch.randn(3, 78)
        >>> g2.edata['edge_attr'] = torch.randn(3, 11)
        >>> batched_g = dgl.batch([g1, g2])
        >>> embeddings = model([batched_g])
        >>> embeddings.shape
        torch.Size([2, 128])
        """
        # Handle both single batched graph and list of graphs
        if isinstance(dgl_graphs, dgl.DGLGraph):
            g = dgl_graphs
        elif isinstance(dgl_graphs, list) and len(dgl_graphs) == 1:
            g = dgl_graphs[0]
        elif isinstance(dgl_graphs, list):
            g = dgl.batch(dgl_graphs)
        else:
            raise ValueError(f"Expected DGLGraph or list of DGLGraphs, got {type(dgl_graphs)}")

        # Validate graph has required features
        if "x" not in g.ndata:
            raise ValueError("Graph must have node features in ndata['x']")
        if "edge_attr" not in g.edata:
            raise ValueError("Graph must have edge features in edata['edge_attr']")

        # Extract features
        node_feats = g.ndata["x"]
        edge_feats = g.edata["edge_attr"]

        # Validate feature dimensions
        if node_feats.shape[-1] != self.node_feat_size:
            raise ValueError(
                f"Expected node features of size {self.node_feat_size}, "
                f"got {node_feats.shape[-1]}"
            )
        if edge_feats.shape[-1] != self.edge_feat_size:
            raise ValueError(
                f"Expected edge features of size {self.edge_feat_size}, "
                f"got {edge_feats.shape[-1]}"
            )

        # GNN forward pass to get node embeddings
        node_embeddings = self.gnn(g, node_feats, edge_feats)

        # Readout to get graph-level embeddings
        graph_embeddings = self.readout(g, node_embeddings, get_node_weight=False)

        # Apply layer normalization
        graph_embeddings = self.layer_norm(graph_embeddings)

        # Apply dropout
        graph_embeddings = self.dropout(graph_embeddings)

        return graph_embeddings

    def __repr__(self) -> str:
        """
        String representation of the module.

        Returns
        -------
        str
            String representation
        """
        return (
            f"{self.__class__.__name__}("
            f"node_feat_size={self.node_feat_size}, "
            f"edge_feat_size={self.edge_feat_size}, "
            f"num_layers={self.num_layers}, "
            f"num_timesteps={self.num_timesteps}, "
            f"graph_feat_size={self.graph_feat_size}, "
            f"dropout={self.dropout_rate})"
        )
