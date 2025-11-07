"""
Model architecture components for lifespan prediction.

This module provides neural network models and utilities for molecular
property prediction, including GNN, CNN, and DNN branches.
"""

from lifespan_predictor.models.attentive_fp import AttentiveFPModule
from lifespan_predictor.models.predictor import LifespanPredictor
from lifespan_predictor.models.layers import (
    init_weights_xavier,
    init_weights_kaiming,
    init_weights_normal,
    initialize_model_weights,
    count_parameters,
    get_parameter_stats,
    print_model_summary,
)

__all__ = [
    "AttentiveFPModule",
    "LifespanPredictor",
    "init_weights_xavier",
    "init_weights_kaiming",
    "init_weights_normal",
    "initialize_model_weights",
    "count_parameters",
    "get_parameter_stats",
    "print_model_summary",
]
