"""
Custom neural network layers and initialization utilities.

This module provides utility functions for weight initialization
and custom layer implementations.
"""

import logging
import math

import torch.nn as nn

logger = logging.getLogger(__name__)


def init_weights_xavier(module: nn.Module) -> None:
    """
    Initialize weights using Xavier (Glorot) initialization.

    This initialization is suitable for layers with tanh or sigmoid activations,
    and is commonly used for GNN layers.

    Parameters
    ----------
    module : nn.Module
        Module to initialize

    Examples
    --------
    >>> model = nn.Linear(10, 5)
    >>> init_weights_xavier(model)
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_weights_kaiming(module: nn.Module, nonlinearity: str = "relu") -> None:
    """
    Initialize weights using Kaiming (He) initialization.

    This initialization is suitable for layers with ReLU activations,
    and is commonly used for CNN and DNN layers.

    Parameters
    ----------
    module : nn.Module
        Module to initialize
    nonlinearity : str, optional
        Type of nonlinearity ('relu' or 'leaky_relu') (default: 'relu')

    Examples
    --------
    >>> model = nn.Linear(10, 5)
    >>> init_weights_kaiming(model)
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
        if module.bias is not None:
            # Calculate fan_in for bias initialization
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity=nonlinearity)
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(module.bias, -bound, bound)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def init_weights_normal(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """
    Initialize weights using normal distribution.

    Parameters
    ----------
    module : nn.Module
        Module to initialize
    mean : float, optional
        Mean of the normal distribution (default: 0.0)
    std : float, optional
        Standard deviation of the normal distribution (default: 0.01)

    Examples
    --------
    >>> model = nn.Linear(10, 5)
    >>> init_weights_normal(model, mean=0.0, std=0.01)
    """
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def initialize_model_weights(
    model: nn.Module, gnn_init: str = "xavier", cnn_init: str = "kaiming", dnn_init: str = "kaiming"
) -> None:
    """
    Initialize weights for the entire model with branch-specific strategies.

    This function applies different initialization strategies to different
    branches of the model:
    - GNN branch: Xavier initialization (better for graph attention)
    - CNN branch: Kaiming initialization (better for ReLU activations)
    - DNN branch: Kaiming initialization (better for ReLU activations)

    Parameters
    ----------
    model : nn.Module
        Model to initialize
    gnn_init : str, optional
        Initialization method for GNN branch ('xavier', 'kaiming', or 'normal')
        (default: 'xavier')
    cnn_init : str, optional
        Initialization method for CNN branch ('xavier', 'kaiming', or 'normal')
        (default: 'kaiming')
    dnn_init : str, optional
        Initialization method for DNN branch ('xavier', 'kaiming', or 'normal')
        (default: 'kaiming')

    Raises
    ------
    ValueError
        If initialization method is not recognized

    Examples
    --------
    >>> from lifespan_predictor.models import LifespanPredictor
    >>> from lifespan_predictor.config import Config
    >>> config = Config()
    >>> model = LifespanPredictor(config)
    >>> initialize_model_weights(model)
    """
    init_methods = {
        "xavier": init_weights_xavier,
        "kaiming": init_weights_kaiming,
        "normal": init_weights_normal,
    }

    # Validate initialization methods
    for init_name, init_method in [
        ("gnn_init", gnn_init),
        ("cnn_init", cnn_init),
        ("dnn_init", dnn_init),
    ]:
        if init_method not in init_methods:
            raise ValueError(
                f"{init_name} must be one of {list(init_methods.keys())}, " f"got '{init_method}'"
            )

    # Initialize branches if they exist
    if hasattr(model, "branches"):
        # GNN branch
        if "gnn" in model.branches:
            logger.info(f"Initializing GNN branch with {gnn_init} initialization")
            model.branches["gnn"].apply(init_methods[gnn_init])

        # CNN branch
        if "fp_cnn" in model.branches:
            logger.info(f"Initializing CNN branch with {cnn_init} initialization")
            model.branches["fp_cnn"].apply(init_methods[cnn_init])

        # DNN branch
        if "fp_dnn" in model.branches:
            logger.info(f"Initializing DNN branch with {dnn_init} initialization")
            model.branches["fp_dnn"].apply(init_methods[dnn_init])

    # Initialize prediction head
    if hasattr(model, "prediction_head"):
        logger.info("Initializing prediction head with kaiming initialization")
        model.prediction_head.apply(init_weights_kaiming)

    logger.info("Model weight initialization complete")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Parameters
    ----------
    model : nn.Module
        Model to count parameters for
    trainable_only : bool, optional
        If True, count only trainable parameters (default: True)

    Returns
    -------
    int
        Number of parameters

    Examples
    --------
    >>> from lifespan_predictor.models import LifespanPredictor
    >>> from lifespan_predictor.config import Config
    >>> config = Config()
    >>> model = LifespanPredictor(config)
    >>> n_params = count_parameters(model)
    >>> print(f"Model has {n_params:,} trainable parameters")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_parameter_stats(model: nn.Module) -> dict:
    """
    Get statistics about model parameters.

    Parameters
    ----------
    model : nn.Module
        Model to analyze

    Returns
    -------
    dict
        Dictionary containing parameter statistics:
        - total_params: Total number of parameters
        - trainable_params: Number of trainable parameters
        - non_trainable_params: Number of non-trainable parameters
        - param_size_mb: Approximate size in megabytes

    Examples
    --------
    >>> from lifespan_predictor.models import LifespanPredictor
    >>> from lifespan_predictor.config import Config
    >>> config = Config()
    >>> model = LifespanPredictor(config)
    >>> stats = get_parameter_stats(model)
    >>> print(f"Model size: {stats['param_size_mb']:.2f} MB")
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params

    # Estimate size in MB (assuming float32, 4 bytes per parameter)
    param_size_mb = (total_params * 4) / (1024**2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "param_size_mb": param_size_mb,
    }


def print_model_summary(model: nn.Module) -> None:
    """
    Print a summary of the model architecture and parameters.

    Parameters
    ----------
    model : nn.Module
        Model to summarize

    Examples
    --------
    >>> from lifespan_predictor.models import LifespanPredictor
    >>> from lifespan_predictor.config import Config
    >>> config = Config()
    >>> model = LifespanPredictor(config)
    >>> print_model_summary(model)
    """
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(model)
    print("=" * 80)

    stats = get_parameter_stats(model)
    print(f"Total parameters: {stats['total_params']:,}")
    print(f"Trainable parameters: {stats['trainable_params']:,}")
    print(f"Non-trainable parameters: {stats['non_trainable_params']:,}")
    print(f"Estimated size: {stats['param_size_mb']:.2f} MB")
    print("=" * 80)
