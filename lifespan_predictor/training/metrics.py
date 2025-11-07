"""
Metrics module for model evaluation.

This module provides a collection of metrics for both classification and regression tasks,
with support for numpy arrays and PyTorch tensors.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, List, Optional
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import pearsonr


class Metric(ABC):
    """
    Abstract base class for all metrics.

    All metrics should inherit from this class and implement the compute() method.
    Metrics can handle both numpy arrays and PyTorch tensors.
    """

    def __init__(self):
        """Initialize the metric."""

    @abstractmethod
    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute the metric value.

        Args:
            y_true: Ground truth labels/values
            y_pred: Predicted labels/values or probabilities

        Returns:
            Computed metric value as a float

        Raises:
            ValueError: If inputs have incompatible shapes or invalid values
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the metric name.

        Returns:
            Name of the metric
        """

    @property
    def higher_is_better(self) -> bool:
        """
        Indicate whether higher values are better for this metric.

        Returns:
            True if higher is better, False otherwise
        """
        return True

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert input to numpy array.

        Args:
            data: Input data as numpy array or torch tensor

        Returns:
            Data as numpy array
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Validate input arrays.

        Args:
            y_true: Ground truth array
            y_pred: Prediction array

        Raises:
            ValueError: If inputs have incompatible shapes or contain invalid values
        """
        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Shape mismatch: y_true has {y_true.shape[0]} samples, "
                f"y_pred has {y_pred.shape[0]} samples"
            )

        if np.any(np.isnan(y_true)):
            raise ValueError("y_true contains NaN values")

        if np.any(np.isnan(y_pred)):
            raise ValueError("y_pred contains NaN values")

        if np.any(np.isinf(y_true)):
            raise ValueError("y_true contains infinite values")

        if np.any(np.isinf(y_pred)):
            raise ValueError("y_pred contains infinite values")


# ============================================================================
# Classification Metrics
# ============================================================================


class AUC(Metric):
    """
    Area Under the ROC Curve (AUC) metric for binary classification.

    This metric measures the area under the Receiver Operating Characteristic curve,
    which plots the true positive rate against the false positive rate.

    Examples:
        >>> auc = AUC()
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred = np.array([0.1, 0.4, 0.35, 0.8])
        >>> score = auc.compute(y_true, y_pred)
    """

    @property
    def name(self) -> str:
        return "AUC"

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute AUC score.

        Args:
            y_true: Ground truth binary labels (0 or 1)
            y_pred: Predicted probabilities for the positive class

        Returns:
            AUC score between 0 and 1

        Raises:
            ValueError: If y_true contains only one class or inputs are invalid
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Check if we have both classes
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            raise ValueError(
                f"AUC requires at least 2 classes, but y_true contains only {unique_classes}"
            )

        try:
            return float(roc_auc_score(y_true, y_pred))
        except ValueError as e:
            raise ValueError(f"Error computing AUC: {str(e)}")


class Accuracy(Metric):
    """
    Accuracy metric for classification.

    Computes the fraction of predictions that match the ground truth labels.

    Examples:
        >>> acc = Accuracy()
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> score = acc.compute(y_true, y_pred)
    """

    @property
    def name(self) -> str:
        return "Accuracy"

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute accuracy score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (not probabilities)

        Returns:
            Accuracy score between 0 and 1
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        return float(accuracy_score(y_true, y_pred))


class F1Score(Metric):
    """
    F1 Score metric for classification.

    The F1 score is the harmonic mean of precision and recall.

    Args:
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')

    Examples:
        >>> f1 = F1Score(average='binary')
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> score = f1.compute(y_true, y_pred)
    """

    def __init__(self, average: str = "binary"):
        """
        Initialize F1Score metric.

        Args:
            average: Averaging strategy for multi-class classification
        """
        super().__init__()
        self.average = average

    @property
    def name(self) -> str:
        return f"F1Score_{self.average}"

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute F1 score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (not probabilities)

        Returns:
            F1 score between 0 and 1
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Handle single class case
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return 0.0

        return float(f1_score(y_true, y_pred, average=self.average, zero_division=0))


class Precision(Metric):
    """
    Precision metric for classification.

    Precision is the ratio of true positives to all positive predictions.

    Args:
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')

    Examples:
        >>> prec = Precision(average='binary')
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> score = prec.compute(y_true, y_pred)
    """

    def __init__(self, average: str = "binary"):
        """
        Initialize Precision metric.

        Args:
            average: Averaging strategy for multi-class classification
        """
        super().__init__()
        self.average = average

    @property
    def name(self) -> str:
        return f"Precision_{self.average}"

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute precision score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (not probabilities)

        Returns:
            Precision score between 0 and 1
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Handle single class case
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return 0.0

        return float(precision_score(y_true, y_pred, average=self.average, zero_division=0))


class Recall(Metric):
    """
    Recall metric for classification.

    Recall is the ratio of true positives to all actual positive samples.

    Args:
        average: Averaging strategy ('binary', 'micro', 'macro', 'weighted')

    Examples:
        >>> rec = Recall(average='binary')
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 0, 0])
        >>> score = rec.compute(y_true, y_pred)
    """

    def __init__(self, average: str = "binary"):
        """
        Initialize Recall metric.

        Args:
            average: Averaging strategy for multi-class classification
        """
        super().__init__()
        self.average = average

    @property
    def name(self) -> str:
        return f"Recall_{self.average}"

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute recall score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (not probabilities)

        Returns:
            Recall score between 0 and 1
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Handle single class case
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return 0.0

        return float(recall_score(y_true, y_pred, average=self.average, zero_division=0))


# ============================================================================
# Regression Metrics
# ============================================================================


class RMSE(Metric):
    """
    Root Mean Squared Error (RMSE) metric for regression.

    RMSE measures the square root of the average squared differences between
    predicted and actual values.

    Examples:
        >>> rmse = RMSE()
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> score = rmse.compute(y_true, y_pred)
    """

    @property
    def name(self) -> str:
        return "RMSE"

    @property
    def higher_is_better(self) -> bool:
        return False

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute RMSE score.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            RMSE score (lower is better)
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))


class MAE(Metric):
    """
    Mean Absolute Error (MAE) metric for regression.

    MAE measures the average absolute differences between predicted and actual values.

    Examples:
        >>> mae = MAE()
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> score = mae.compute(y_true, y_pred)
    """

    @property
    def name(self) -> str:
        return "MAE"

    @property
    def higher_is_better(self) -> bool:
        return False

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute MAE score.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            MAE score (lower is better)
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        return float(mean_absolute_error(y_true, y_pred))


class R2Score(Metric):
    """
    R-squared (coefficient of determination) metric for regression.

    R² measures the proportion of variance in the dependent variable that is
    predictable from the independent variable(s).

    Examples:
        >>> r2 = R2Score()
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> score = r2.compute(y_true, y_pred)
    """

    @property
    def name(self) -> str:
        return "R2"

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute R² score.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            R² score (higher is better, max 1.0)
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        return float(r2_score(y_true, y_pred))


class PearsonCorrelation(Metric):
    """
    Pearson Correlation Coefficient metric for regression.

    Measures the linear correlation between predicted and actual values.

    Examples:
        >>> pearson = PearsonCorrelation()
        >>> y_true = np.array([3.0, -0.5, 2.0, 7.0])
        >>> y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        >>> score = pearson.compute(y_true, y_pred)
    """

    @property
    def name(self) -> str:
        return "PearsonR"

    def compute(
        self, y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """
        Compute Pearson correlation coefficient.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Pearson correlation coefficient between -1 and 1

        Raises:
            ValueError: If inputs have insufficient variance
        """
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)

        self._validate_inputs(y_true, y_pred)

        # Flatten arrays if needed
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # Check for sufficient variance
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            raise ValueError(
                "Cannot compute Pearson correlation: one or both arrays have zero variance"
            )

        correlation, _ = pearsonr(y_true, y_pred)
        return float(correlation)


# ============================================================================
# Metric Collection
# ============================================================================


class MetricCollection:
    """
    Collection of metrics for batch evaluation.

    This class manages multiple metrics and computes them all at once,
    handling edge cases like single-class batches and NaN values.

    Args:
        metrics: List of Metric instances to compute

    Examples:
        >>> metrics = MetricCollection([AUC(), Accuracy(), F1Score()])
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_pred_proba = np.array([0.1, 0.4, 0.35, 0.8])
        >>> y_pred_labels = (y_pred_proba > 0.5).astype(int)
        >>> results = metrics.compute_all(y_true, y_pred_proba, y_pred_labels)
    """

    def __init__(self, metrics: List[Metric]):
        """
        Initialize MetricCollection.

        Args:
            metrics: List of Metric instances

        Raises:
            ValueError: If metrics list is empty
        """
        if not metrics:
            raise ValueError("MetricCollection requires at least one metric")

        self.metrics = metrics
        self._metric_dict = {metric.name: metric for metric in metrics}

    def compute_all(
        self,
        y_true: Union[np.ndarray, torch.Tensor],
        y_pred: Union[np.ndarray, torch.Tensor],
        y_pred_labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Compute all metrics in the collection.

        This method handles edge cases gracefully:
        - Single class in y_true: Returns 0.0 for classification metrics
        - NaN values: Raises ValueError with clear message
        - Insufficient samples: Raises ValueError

        Args:
            y_true: Ground truth labels/values
            y_pred: Predicted probabilities (for classification) or values (for regression)
            y_pred_labels: Predicted labels for classification metrics (optional)
                          If None, will threshold y_pred at 0.5 for binary classification

        Returns:
            Dictionary mapping metric names to computed values

        Raises:
            ValueError: If inputs are invalid or incompatible
        """
        results = {}

        # Convert to numpy for validation
        y_true_np = self._to_numpy(y_true)
        y_pred_np = self._to_numpy(y_pred)

        # Basic validation
        if y_true_np.shape[0] == 0:
            raise ValueError("Cannot compute metrics on empty arrays")

        if y_true_np.shape[0] != y_pred_np.shape[0]:
            raise ValueError(
                f"Shape mismatch: y_true has {y_true_np.shape[0]} samples, "
                f"y_pred has {y_pred_np.shape[0]} samples"
            )

        # Check for NaN or inf values
        if np.any(np.isnan(y_true_np)) or np.any(np.isinf(y_true_np)):
            raise ValueError("y_true contains NaN or infinite values")

        if np.any(np.isnan(y_pred_np)) or np.any(np.isinf(y_pred_np)):
            raise ValueError("y_pred contains NaN or infinite values")

        # Generate labels if not provided (for binary classification)
        if y_pred_labels is None:
            # Check if this looks like classification (probabilities between 0 and 1)
            if np.all((y_pred_np >= 0) & (y_pred_np <= 1)):
                y_pred_labels = (y_pred_np > 0.5).astype(int)
            else:
                # For regression, use predictions as-is
                y_pred_labels = y_pred_np
        else:
            y_pred_labels = self._to_numpy(y_pred_labels)

        # Check for single class (affects classification metrics)
        unique_classes = np.unique(y_true_np.flatten())
        has_single_class = len(unique_classes) < 2

        # Compute each metric
        for metric in self.metrics:
            try:
                # Determine which predictions to use
                if isinstance(metric, (AUC,)):
                    # AUC needs probabilities
                    pred_to_use = y_pred
                elif isinstance(metric, (Accuracy, F1Score, Precision, Recall)):
                    # Classification metrics need labels
                    pred_to_use = y_pred_labels
                else:
                    # Regression metrics use predictions as-is
                    pred_to_use = y_pred

                # Handle single class case for classification metrics
                if has_single_class and isinstance(metric, (AUC, F1Score, Precision, Recall)):
                    results[metric.name] = 0.0
                else:
                    results[metric.name] = metric.compute(y_true, pred_to_use)

            except ValueError as e:
                # Log the error but continue with other metrics
                results[metric.name] = float("nan")
                print(f"Warning: Could not compute {metric.name}: {str(e)}")
            except Exception as e:
                # Catch any other unexpected errors
                results[metric.name] = float("nan")
                print(f"Error computing {metric.name}: {str(e)}")

        return results

    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert input to numpy array.

        Args:
            data: Input data as numpy array or torch tensor

        Returns:
            Data as numpy array
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def get_metric(self, name: str) -> Optional[Metric]:
        """
        Get a metric by name.

        Args:
            name: Name of the metric

        Returns:
            Metric instance or None if not found
        """
        return self._metric_dict.get(name)

    def __len__(self) -> int:
        """Return the number of metrics in the collection."""
        return len(self.metrics)

    def __repr__(self) -> str:
        """Return string representation of the collection."""
        metric_names = [m.name for m in self.metrics]
        return f"MetricCollection({metric_names})"
