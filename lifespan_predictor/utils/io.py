"""I/O utilities for saving and loading data."""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_results(
    results: Dict[str, Any],
    output_dir: str,
    prefix: str = "results",
    include_timestamp: bool = True,
) -> str:
    """
    Save results dictionary with metadata.

    Args:
        results: Dictionary containing results to save
        output_dir: Directory to save results
        prefix: Prefix for output filename (default: "results")
        include_timestamp: Whether to include timestamp in filename (default: True)

    Returns:
        Path to saved file

    Example:
        >>> results = {"accuracy": 0.95, "loss": 0.1}
        >>> save_results(results, "outputs", prefix="validation")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Add metadata
    results_with_metadata = {
        "results": results,
        "metadata": {"timestamp": datetime.now().isoformat(), "saved_by": "lifespan_predictor"},
    }

    # Generate filename
    if include_timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp_str}.json"
    else:
        filename = f"{prefix}.json"

    filepath = output_path / filename

    # Save as JSON
    with open(filepath, "w") as f:
        json.dump(results_with_metadata, f, indent=2, default=str)

    return str(filepath)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.

    Args:
        filepath: Path to results file

    Returns:
        Dictionary containing results and metadata

    Example:
        >>> data = load_results("outputs/results_20240101_120000.json")
        >>> print(data["results"]["accuracy"])
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    scheduler: Optional[Any] = None,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> None:
    """
    Save model checkpoint with full training state.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch number
        filepath: Path to save checkpoint
        scheduler: Optional learning rate scheduler
        metrics: Optional dictionary of training metrics
        config: Optional configuration dictionary
        **kwargs: Additional data to save in checkpoint

    Example:
        >>> save_checkpoint(
        ...     model, optimizer, epoch=10,
        ...     filepath="checkpoints/model_epoch10.pt",
        ...     metrics={"train_loss": 0.1, "val_loss": 0.15}
        ... )
    """
    # Create directory if needed
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    # Add optional components
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if metrics is not None:
        checkpoint["metrics"] = metrics

    if config is not None:
        checkpoint["config"] = config

    # Add any additional kwargs
    checkpoint.update(kwargs)

    # Save checkpoint
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint and restore training state.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint to (default: None, uses checkpoint device)

    Returns:
        Dictionary containing checkpoint data (epoch, metrics, etc.)

    Example:
        >>> checkpoint_data = load_checkpoint(
        ...     "checkpoints/best_model.pt",
        ...     model=model,
        ...     optimizer=optimizer
        ... )
        >>> start_epoch = checkpoint_data['epoch'] + 1
    """
    # Load checkpoint
    if device is not None:
        checkpoint = torch.load(filepath, map_location=device)
    else:
        checkpoint = torch.load(filepath)

    # Restore model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore scheduler state if provided
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save object using pickle.

    Args:
        obj: Object to save
        filepath: Path to save file

    Example:
        >>> data = {"features": np.array([1, 2, 3])}
        >>> save_pickle(data, "data/features.pkl")
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath: str) -> Any:
    """
    Load object from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object

    Example:
        >>> data = load_pickle("data/features.pkl")
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(obj: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """
    Save dictionary as JSON file.

    Args:
        obj: Dictionary to save
        filepath: Path to save file
        indent: JSON indentation level (default: 2)

    Example:
        >>> config = {"learning_rate": 0.001, "batch_size": 32}
        >>> save_json(config, "config/params.json")
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary

    Example:
        >>> config = load_json("config/params.json")
    """
    with open(filepath, "r") as f:
        return json.load(f)
