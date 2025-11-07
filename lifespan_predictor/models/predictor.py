"""
Main predictor model combining GNN, CNN, and DNN branches.

This module implements the LifespanPredictor model that combines multiple
branches (GNN, CNN, DNN) for molecular property prediction.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.data import Data

from lifespan_predictor.config import Config
from lifespan_predictor.models.attentive_fp import AttentiveFPModule

logger = logging.getLogger(__name__)


class LifespanPredictor(nn.Module):
    """
    Multi-branch predictor combining GNN, CNN, and DNN for lifespan prediction.

    This model combines three branches:
    1. GNN branch: Processes molecular graphs using AttentiveFP
    2. CNN branch: Processes MACCS fingerprints using 1D convolutions
    3. DNN branch: Processes Morgan and RDKit fingerprints using fully connected layers

    The outputs from enabled branches are concatenated and passed through
    a final prediction head.

    Parameters
    ----------
    config : Config
        Configuration object containing model hyperparameters

    Attributes
    ----------
    config : Config
        Configuration object
    enable_gnn : bool
        Whether GNN branch is enabled
    enable_fp_cnn : bool
        Whether fingerprint CNN branch is enabled
    enable_fp_dnn : bool
        Whether fingerprint DNN branch is enabled
    gnn_module : Optional[AttentiveFPModule]
        GNN module for graph processing
    fp_cnn : Optional[nn.Sequential]
        CNN module for MACCS fingerprints
    fp_dnn : Optional[nn.Sequential]
        DNN module for Morgan/RDKit fingerprints
    prediction_head : nn.Sequential
        Final prediction layers

    Examples
    --------
    >>> from lifespan_predictor.config import Config
    >>> config = Config()
    >>> model = LifespanPredictor(config)
    >>> # Forward pass with batch
    >>> output = model(batch)
    """

    def __init__(self, config: Config, use_gradient_checkpointing: bool = False):
        """
        Initialize the LifespanPredictor.

        Parameters
        ----------
        config : Config
            Configuration object containing model hyperparameters
        use_gradient_checkpointing : bool, optional
            Enable gradient checkpointing to reduce memory usage (default: False)
        """
        super(LifespanPredictor, self).__init__()

        self.config = config
        self.enable_gnn = config.model.enable_gnn
        self.enable_fp_cnn = config.model.enable_fp_cnn
        self.enable_fp_dnn = config.model.enable_fp_dnn
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Validate at least one branch is enabled
        if not (self.enable_gnn or self.enable_fp_cnn or self.enable_fp_dnn):
            raise ValueError("At least one model branch must be enabled")

        # Initialize branches using ModuleDict for dynamic selection
        self.branches = nn.ModuleDict()

        # GNN branch
        if self.enable_gnn:
            self.branches["gnn"] = AttentiveFPModule(
                node_feat_size=config.model.gnn_node_input_dim,
                edge_feat_size=config.model.gnn_edge_input_dim,
                num_layers=config.model.gnn_num_layers,
                num_timesteps=config.model.gnn_num_timesteps,
                graph_feat_size=config.model.gnn_graph_embed_dim,
                dropout=config.model.gnn_dropout,
            )
            logger.info("Initialized GNN branch")

        # CNN branch for MACCS fingerprints
        if self.enable_fp_cnn:
            self.branches["fp_cnn"] = self._build_fp_cnn_branch(
                input_dim=config.featurization.maccs_nbits,
                output_dim=config.model.fp_cnn_output_dim,
                dropout=config.model.fp_dropout,
            )
            logger.info("Initialized fingerprint CNN branch")

        # DNN branch for Morgan and RDKit fingerprints
        if self.enable_fp_dnn:
            fp_dnn_input_dim = (
                config.featurization.morgan_nbits + config.featurization.rdkit_fp_nbits
            )
            self.branches["fp_dnn"] = self._build_fp_dnn_branch(
                input_dim=fp_dnn_input_dim,
                hidden_dims=config.model.fp_dnn_layers,
                output_dim=config.model.fp_dnn_output_dim,
                dropout=config.model.fp_dropout,
            )
            logger.info("Initialized fingerprint DNN branch")

        # Calculate total feature dimension from all enabled branches
        total_feat_dim = 0
        if self.enable_gnn:
            total_feat_dim += config.model.gnn_graph_embed_dim
        if self.enable_fp_cnn:
            total_feat_dim += config.model.fp_cnn_output_dim
        if self.enable_fp_dnn:
            total_feat_dim += config.model.fp_dnn_output_dim

        # Prediction head
        self.prediction_head = self._build_prediction_head(
            input_dim=total_feat_dim, n_output_tasks=config.model.n_output_tasks
        )

        # Enable gradient checkpointing if requested
        if self.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")

        logger.info(
            f"Initialized LifespanPredictor with total_feat_dim={total_feat_dim}, "
            f"n_output_tasks={config.model.n_output_tasks}, "
            f"gradient_checkpointing={use_gradient_checkpointing}"
        )

    def _enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing for memory efficiency.

        This trades compute for memory by not storing intermediate activations
        during forward pass and recomputing them during backward pass.
        """
        # Enable checkpointing for CNN branch
        if self.enable_fp_cnn and "fp_cnn" in self.branches:
            # Wrap CNN in checkpoint
            original_cnn = self.branches["fp_cnn"]
            self.branches["fp_cnn"] = nn.Sequential(
                *[
                    (
                        nn.utils.checkpoint.checkpoint(layer, use_reentrant=False)
                        if isinstance(layer, (nn.Conv1d, nn.Linear))
                        else layer
                    )
                    for layer in original_cnn
                ]
            )

        # Enable checkpointing for DNN branch
        if self.enable_fp_dnn and "fp_dnn" in self.branches:
            # Wrap DNN in checkpoint
            original_dnn = self.branches["fp_dnn"]
            self.branches["fp_dnn"] = nn.Sequential(
                *[
                    (
                        nn.utils.checkpoint.checkpoint(layer, use_reentrant=False)
                        if isinstance(layer, nn.Linear)
                        else layer
                    )
                    for layer in original_dnn
                ]
            )

    def _build_fp_cnn_branch(
        self, input_dim: int, output_dim: int, dropout: float
    ) -> nn.Sequential:
        """
        Build CNN branch for MACCS fingerprints.

        Parameters
        ----------
        input_dim : int
            Input dimension (MACCS fingerprint size)
        output_dim : int
            Output dimension
        dropout : float
            Dropout rate

        Returns
        -------
        nn.Sequential
            CNN branch module
        """
        return nn.Sequential(
            # Reshape to (batch, 1, input_dim) for 1D convolution
            nn.Unflatten(1, (1, input_dim)),
            # First conv layer
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Second conv layer
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # Final linear layer
            nn.Linear(64, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _build_fp_dnn_branch(
        self, input_dim: int, hidden_dims: list, output_dim: int, dropout: float
    ) -> nn.Sequential:
        """
        Build DNN branch for Morgan and RDKit fingerprints.

        Parameters
        ----------
        input_dim : int
            Input dimension (Morgan + RDKit fingerprint size)
        hidden_dims : list
            List of hidden layer dimensions
        output_dim : int
            Output dimension
        dropout : float
            Dropout rate

        Returns
        -------
        nn.Sequential
            DNN branch module
        """
        layers = []

        # Input layer
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.extend([nn.Linear(prev_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)])

        return nn.Sequential(*layers)

    def _build_prediction_head(self, input_dim: int, n_output_tasks: int) -> nn.Sequential:
        """
        Build final prediction head.

        Parameters
        ----------
        input_dim : int
            Input dimension (concatenated features from all branches)
        n_output_tasks : int
            Number of output tasks

        Returns
        -------
        nn.Sequential
            Prediction head module
        """
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, n_output_tasks),
        )

    def forward(self, batch: Data) -> torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        batch : Data
            Batched PyG Data object containing:
            - dgl_graph: Batched DGL graph (if GNN enabled)
            - hashed_fp: Hashed fingerprints (Morgan + RDKit) (if DNN enabled)
            - non_hashed_fp: Non-hashed fingerprints (MACCS) (if CNN enabled)

        Returns
        -------
        torch.Tensor
            Predictions, shape (batch_size, n_output_tasks)

        Examples
        --------
        >>> model = LifespanPredictor(config)
        >>> output = model(batch)
        >>> output.shape
        torch.Size([32, 1])
        """
        branch_outputs = []

        # GNN branch
        if self.enable_gnn:
            gnn_output = self._forward_gnn(batch)
            branch_outputs.append(gnn_output)

        # CNN branch for MACCS fingerprints
        if self.enable_fp_cnn:
            cnn_output = self._forward_fp_cnn(batch)
            branch_outputs.append(cnn_output)

        # DNN branch for Morgan and RDKit fingerprints
        if self.enable_fp_dnn:
            dnn_output = self._forward_fp_dnn(batch)
            branch_outputs.append(dnn_output)

        # Concatenate all branch outputs
        if len(branch_outputs) == 1:
            combined_features = branch_outputs[0]
        else:
            combined_features = torch.cat(branch_outputs, dim=1)

        # Final prediction
        predictions = self.prediction_head(combined_features)

        return predictions

    def _forward_gnn(self, batch: Data) -> torch.Tensor:
        """
        Forward pass through GNN branch.

        Parameters
        ----------
        batch : Data
            Batched PyG Data object with dgl_graph attribute

        Returns
        -------
        torch.Tensor
            GNN branch output, shape (batch_size, gnn_graph_embed_dim)

        Raises
        ------
        ValueError
            If dgl_graph is not found in batch
        """
        if not hasattr(batch, "dgl_graph") and "dgl_graph" not in batch.__dict__:
            raise ValueError(
                "Batch must have 'dgl_graph' attribute for GNN branch. "
                "Make sure to use collate_lifespan_data as collate function."
            )

        dgl_graph = batch.__dict__.get("dgl_graph", getattr(batch, "dgl_graph", None))

        if dgl_graph is None:
            raise ValueError("dgl_graph is None")

        return self.branches["gnn"]([dgl_graph])

    def _forward_fp_cnn(self, batch: Data) -> torch.Tensor:
        """
        Forward pass through fingerprint CNN branch.

        Parameters
        ----------
        batch : Data
            Batched PyG Data object with non_hashed_fp attribute (MACCS)

        Returns
        -------
        torch.Tensor
            CNN branch output, shape (batch_size, fp_cnn_output_dim)

        Raises
        ------
        ValueError
            If non_hashed_fp is not found in batch
        """
        if not hasattr(batch, "non_hashed_fp"):
            raise ValueError("Batch must have 'non_hashed_fp' attribute for CNN branch")

        maccs_fp = batch.non_hashed_fp
        return self.branches["fp_cnn"](maccs_fp)

    def _forward_fp_dnn(self, batch: Data) -> torch.Tensor:
        """
        Forward pass through fingerprint DNN branch.

        Parameters
        ----------
        batch : Data
            Batched PyG Data object with hashed_fp attribute (Morgan + RDKit)

        Returns
        -------
        torch.Tensor
            DNN branch output, shape (batch_size, fp_dnn_output_dim)

        Raises
        ------
        ValueError
            If hashed_fp is not found in batch
        """
        if not hasattr(batch, "hashed_fp"):
            raise ValueError("Batch must have 'hashed_fp' attribute for DNN branch")

        hashed_fp = batch.hashed_fp
        return self.branches["fp_dnn"](hashed_fp)

    def get_enabled_branches(self) -> Dict[str, bool]:
        """
        Get dictionary of enabled branches.

        Returns
        -------
        Dict[str, bool]
            Dictionary mapping branch names to enabled status
        """
        return {"gnn": self.enable_gnn, "fp_cnn": self.enable_fp_cnn, "fp_dnn": self.enable_fp_dnn}

    def __repr__(self) -> str:
        """
        String representation of the model.

        Returns
        -------
        str
            String representation
        """
        enabled_branches = [k for k, v in self.get_enabled_branches().items() if v]
        return (
            f"{self.__class__.__name__}("
            f"enabled_branches={enabled_branches}, "
            f"n_output_tasks={self.config.model.n_output_tasks})"
        )
