"""Visualization utilities for plotting results."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# Set seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (12, 5),
) -> None:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with metric names as keys and lists of values as values.
                 Expected format: {"train_loss": [...], "val_loss": [...], ...}
        save_path: Path to save the plot
        metrics: List of metric names to plot. If None, plots all metrics found.
        figsize: Figure size (width, height) in inches

    Example:
        >>> history = {
        ...     "train_loss": [0.5, 0.4, 0.3],
        ...     "val_loss": [0.6, 0.5, 0.45],
        ...     "train_auc": [0.7, 0.8, 0.85],
        ...     "val_auc": [0.65, 0.75, 0.8]
        ... }
        >>> plot_training_curves(history, "plots/training_curves.png")
    """
    # Create output directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Determine which metrics to plot
    if metrics is None:
        # Extract unique metric names (remove train_/val_ prefix)
        metric_names = set()
        for key in history.keys():
            if key.startswith("train_"):
                metric_names.add(key[6:])  # Remove 'train_'
            elif key.startswith("val_"):
                metric_names.add(key[4:])  # Remove 'val_'
        metrics = sorted(list(metric_names))

    # Create subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        train_key = f"train_{metric}"
        val_key = f"val_{metric}"

        # Plot training curve
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], "b-", label="Training", linewidth=2)

        # Plot validation curve
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], "r-", label="Validation", linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} over Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    task: str = "classification",
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot predictions vs ground truth.

    Args:
        y_true: True labels/values
        y_pred: Predicted labels/values (probabilities for classification)
        save_path: Path to save the plot
        task: Task type - "classification" or "regression"
        figsize: Figure size (width, height) in inches

    Example:
        >>> # Classification
        >>> plot_predictions(y_true, y_pred_proba, "plots/predictions.png", task="classification")
        >>>
        >>> # Regression
        >>> plot_predictions(y_true, y_pred, "plots/predictions.png", task="regression")
    """
    # Create output directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    if task == "classification":
        # For classification, plot probability distribution by class
        # Assuming binary classification
        class_0_probs = y_pred[y_true == 0]
        class_1_probs = y_pred[y_true == 1]

        bins = np.linspace(0, 1, 30)
        ax.hist(
            class_0_probs, bins=bins, alpha=0.6, label="Class 0", color="blue", edgecolor="black"
        )
        ax.hist(
            class_1_probs, bins=bins, alpha=0.6, label="Class 1", color="red", edgecolor="black"
        )

        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Frequency")
        ax.set_title("Prediction Distribution by True Class")
        ax.legend()
        ax.grid(True, alpha=0.3)

    elif task == "regression":
        # For regression, plot scatter of true vs predicted
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, edgecolors="k", linewidth=0.5)

        # Add diagonal line (perfect prediction)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction"
        )

        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("True vs Predicted Values")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    else:
        raise ValueError(f"Unknown task type: {task}. Must be 'classification' or 'regression'")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str, figsize: tuple = (8, 6)
) -> float:
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        save_path: Path to save the plot
        figsize: Figure size (width, height) in inches

    Returns:
        AUC score

    Example:
        >>> auc_score = plot_roc_curve(y_true, y_pred_proba, "plots/roc_curve.png")
        >>> print(f"AUC: {auc_score:.3f}")
    """
    # Create output directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color="darkorange", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--", label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return roc_auc


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    normalize: bool = False,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
        class_names: Names of classes for labels
        figsize: Figure size (width, height) in inches
        normalize: Whether to normalize the confusion matrix

    Example:
        >>> plot_confusion_matrix(y_true, y_pred, "plots/confusion_matrix.png",
        ...                       class_names=["Negative", "Positive"])
    """
    from sklearn.metrics import confusion_matrix

    # Create output directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        square=True,
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: str,
    top_n: int = 20,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot feature importance.

    Args:
        feature_names: Names of features
        importances: Importance scores for each feature
        save_path: Path to save the plot
        top_n: Number of top features to display
        figsize: Figure size (width, height) in inches

    Example:
        >>> plot_feature_importance(feature_names, importances, "plots/feature_importance.png")
    """
    # Create output directory
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_importances, align="center", color="steelblue", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
