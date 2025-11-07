"""
Unit tests for metrics module.

Tests cover:
- Base Metric class functionality
- Classification metrics (AUC, Accuracy, F1, Precision, Recall)
- Regression metrics (RMSE, MAE, R2, Pearson)
- MetricCollection class
- Edge cases and error handling
"""

import pytest
import numpy as np
import torch
from lifespan_predictor.training.metrics import (
    AUC,
    Accuracy,
    F1Score,
    Precision,
    Recall,
    RMSE,
    MAE,
    R2Score,
    PearsonCorrelation,
    MetricCollection,
)


class TestBaseMetric:
    """Test base Metric class functionality."""

    def test_to_numpy_with_numpy_array(self):
        """Test conversion of numpy array."""
        metric = AUC()  # Use concrete class
        data = np.array([1, 2, 3])
        result = metric._to_numpy(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_to_numpy_with_torch_tensor(self):
        """Test conversion of torch tensor."""
        metric = AUC()
        data = torch.tensor([1, 2, 3])
        result = metric._to_numpy(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_validate_inputs_shape_mismatch(self):
        """Test validation catches shape mismatch."""
        metric = AUC()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])
        with pytest.raises(ValueError, match="Shape mismatch"):
            metric._validate_inputs(y_true, y_pred)

    def test_validate_inputs_nan_in_y_true(self):
        """Test validation catches NaN in y_true."""
        metric = AUC()
        y_true = np.array([1, np.nan, 3])
        y_pred = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="NaN"):
            metric._validate_inputs(y_true, y_pred)

    def test_validate_inputs_inf_in_y_pred(self):
        """Test validation catches infinite values."""
        metric = AUC()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, np.inf, 3])
        with pytest.raises(ValueError, match="infinite"):
            metric._validate_inputs(y_true, y_pred)


class TestClassificationMetrics:
    """Test classification metrics."""

    def test_auc_perfect_prediction(self):
        """Test AUC with perfect predictions."""
        auc = AUC()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.0, 0.1, 0.9, 1.0])
        score = auc.compute(y_true, y_pred)
        assert score == 1.0

    def test_auc_random_prediction(self):
        """Test AUC with random predictions."""
        auc = AUC()
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        score = auc.compute(y_true, y_pred)
        assert score == 0.5

    def test_auc_single_class_error(self):
        """Test AUC raises error with single class."""
        auc = AUC()
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.6, 0.9])
        with pytest.raises(ValueError, match="at least 2 classes"):
            auc.compute(y_true, y_pred)

    def test_auc_with_torch_tensors(self):
        """Test AUC works with torch tensors."""
        auc = AUC()
        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0.0, 0.1, 0.9, 1.0])
        score = auc.compute(y_true, y_pred)
        assert score == 1.0

    def test_accuracy_perfect_prediction(self):
        """Test accuracy with perfect predictions."""
        acc = Accuracy()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        score = acc.compute(y_true, y_pred)
        assert score == 1.0

    def test_accuracy_half_correct(self):
        """Test accuracy with 50% correct predictions."""
        acc = Accuracy()
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0])
        score = acc.compute(y_true, y_pred)
        assert score == 0.5

    def test_f1score_perfect_prediction(self):
        """Test F1 score with perfect predictions."""
        f1 = F1Score(average="binary")
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        score = f1.compute(y_true, y_pred)
        assert score == 1.0

    def test_f1score_single_class(self):
        """Test F1 score handles single class gracefully."""
        f1 = F1Score(average="binary")
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        score = f1.compute(y_true, y_pred)
        assert score == 0.0

    def test_precision_perfect_prediction(self):
        """Test precision with perfect predictions."""
        prec = Precision(average="binary")
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        score = prec.compute(y_true, y_pred)
        assert score == 1.0

    def test_recall_perfect_prediction(self):
        """Test recall with perfect predictions."""
        rec = Recall(average="binary")
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        score = rec.compute(y_true, y_pred)
        assert score == 1.0

    def test_metric_names(self):
        """Test metric names are correct."""
        assert AUC().name == "AUC"
        assert Accuracy().name == "Accuracy"
        assert F1Score().name == "F1Score_binary"
        assert Precision().name == "Precision_binary"
        assert Recall().name == "Recall_binary"

    def test_higher_is_better(self):
        """Test higher_is_better property for classification metrics."""
        assert AUC().higher_is_better is True
        assert Accuracy().higher_is_better is True
        assert F1Score().higher_is_better is True


class TestRegressionMetrics:
    """Test regression metrics."""

    def test_rmse_perfect_prediction(self):
        """Test RMSE with perfect predictions."""
        rmse = RMSE()
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([3.0, -0.5, 2.0, 7.0])
        score = rmse.compute(y_true, y_pred)
        assert score == 0.0

    def test_rmse_known_value(self):
        """Test RMSE with known expected value."""
        rmse = RMSE()
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        score = rmse.compute(y_true, y_pred)
        # Expected: sqrt((0.5^2 + 0.5^2 + 0^2 + 1^2) / 4) = sqrt(1.5/4) = sqrt(0.375) ≈ 0.612
        assert abs(score - 0.612) < 0.01

    def test_mae_perfect_prediction(self):
        """Test MAE with perfect predictions."""
        mae = MAE()
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([3.0, -0.5, 2.0, 7.0])
        score = mae.compute(y_true, y_pred)
        assert score == 0.0

    def test_mae_known_value(self):
        """Test MAE with known expected value."""
        mae = MAE()
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        score = mae.compute(y_true, y_pred)
        # Expected: (0.5 + 0.5 + 0 + 1) / 4 = 0.5
        assert score == 0.5

    def test_r2_perfect_prediction(self):
        """Test R2 with perfect predictions."""
        r2 = R2Score()
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([3.0, -0.5, 2.0, 7.0])
        score = r2.compute(y_true, y_pred)
        assert score == 1.0

    def test_r2_mean_prediction(self):
        """Test R2 with mean predictions (should be 0)."""
        r2 = R2Score()
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([2.5, 2.5, 2.5, 2.5])  # Mean of y_true
        score = r2.compute(y_true, y_pred)
        assert abs(score - 0.0) < 0.01

    def test_pearson_perfect_correlation(self):
        """Test Pearson with perfect correlation."""
        pearson = PearsonCorrelation()
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        score = pearson.compute(y_true, y_pred)
        assert abs(score - 1.0) < 0.01

    def test_pearson_negative_correlation(self):
        """Test Pearson with negative correlation."""
        pearson = PearsonCorrelation()
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([4.0, 3.0, 2.0, 1.0])
        score = pearson.compute(y_true, y_pred)
        assert abs(score - (-1.0)) < 0.01

    def test_pearson_zero_variance_error(self):
        """Test Pearson raises error with zero variance."""
        pearson = PearsonCorrelation()
        y_true = np.array([1.0, 1.0, 1.0, 1.0])
        y_pred = np.array([2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="zero variance"):
            pearson.compute(y_true, y_pred)

    def test_regression_metric_names(self):
        """Test metric names are correct."""
        assert RMSE().name == "RMSE"
        assert MAE().name == "MAE"
        assert R2Score().name == "R2"
        assert PearsonCorrelation().name == "PearsonR"

    def test_higher_is_better_regression(self):
        """Test higher_is_better property for regression metrics."""
        assert RMSE().higher_is_better is False
        assert MAE().higher_is_better is False
        assert R2Score().higher_is_better is True
        assert PearsonCorrelation().higher_is_better is True


class TestMetricCollection:
    """Test MetricCollection class."""

    def test_initialization(self):
        """Test MetricCollection initialization."""
        metrics = [AUC(), Accuracy()]
        collection = MetricCollection(metrics)
        assert len(collection) == 2

    def test_empty_metrics_error(self):
        """Test error when initializing with empty list."""
        with pytest.raises(ValueError, match="at least one metric"):
            MetricCollection([])

    def test_compute_all_classification(self):
        """Test computing all classification metrics."""
        metrics = [AUC(), Accuracy(), F1Score()]
        collection = MetricCollection(metrics)

        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.4, 0.6, 0.9])
        y_pred_labels = (y_pred_proba > 0.5).astype(int)

        results = collection.compute_all(y_true, y_pred_proba, y_pred_labels)

        assert "AUC" in results
        assert "Accuracy" in results
        assert "F1Score_binary" in results
        assert all(0 <= v <= 1 for v in results.values())

    def test_compute_all_regression(self):
        """Test computing all regression metrics."""
        metrics = [RMSE(), MAE(), R2Score()]
        collection = MetricCollection(metrics)

        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])

        results = collection.compute_all(y_true, y_pred)

        assert "RMSE" in results
        assert "MAE" in results
        assert "R2" in results
        assert not np.isnan(results["RMSE"])

    def test_compute_all_single_class_handling(self):
        """Test handling of single class in classification."""
        metrics = [AUC(), F1Score()]
        collection = MetricCollection(metrics)

        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.5, 0.6, 0.7, 0.8])

        results = collection.compute_all(y_true, y_pred)

        # Should return 0.0 for metrics that can't be computed
        assert results["AUC"] == 0.0
        assert results["F1Score_binary"] == 0.0

    def test_compute_all_empty_array_error(self):
        """Test error with empty arrays."""
        metrics = [Accuracy()]
        collection = MetricCollection(metrics)

        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises(ValueError, match="empty arrays"):
            collection.compute_all(y_true, y_pred)

    def test_compute_all_shape_mismatch_error(self):
        """Test error with shape mismatch."""
        metrics = [Accuracy()]
        collection = MetricCollection(metrics)

        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])

        with pytest.raises(ValueError, match="Shape mismatch"):
            collection.compute_all(y_true, y_pred)

    def test_compute_all_nan_handling(self):
        """Test error with NaN values."""
        metrics = [Accuracy()]
        collection = MetricCollection(metrics)

        y_true = np.array([0, 1, np.nan])
        y_pred = np.array([0, 1, 1])

        with pytest.raises(ValueError, match="NaN"):
            collection.compute_all(y_true, y_pred)

    def test_compute_all_auto_threshold(self):
        """Test automatic thresholding for binary classification."""
        metrics = [Accuracy()]
        collection = MetricCollection(metrics)

        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.4, 0.6, 0.9])

        # Should auto-threshold at 0.5
        results = collection.compute_all(y_true, y_pred_proba)
        assert "Accuracy" in results
        assert results["Accuracy"] == 1.0

    def test_get_metric(self):
        """Test getting metric by name."""
        metrics = [AUC(), Accuracy()]
        collection = MetricCollection(metrics)

        auc_metric = collection.get_metric("AUC")
        assert auc_metric is not None
        assert isinstance(auc_metric, AUC)

        none_metric = collection.get_metric("NonExistent")
        assert none_metric is None

    def test_repr(self):
        """Test string representation."""
        metrics = [AUC(), Accuracy()]
        collection = MetricCollection(metrics)

        repr_str = repr(collection)
        assert "MetricCollection" in repr_str
        assert "AUC" in repr_str
        assert "Accuracy" in repr_str

    def test_with_torch_tensors(self):
        """Test MetricCollection works with torch tensors."""
        metrics = [Accuracy(), F1Score()]
        collection = MetricCollection(metrics)

        y_true = torch.tensor([0, 0, 1, 1])
        y_pred = torch.tensor([0, 0, 1, 1])

        results = collection.compute_all(y_true, y_pred)
        assert results["Accuracy"] == 1.0
        assert results["F1Score_binary"] == 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_2d_arrays_flattened(self):
        """Test that 2D arrays are properly flattened."""
        acc = Accuracy()
        y_true = np.array([[0], [1], [1], [0]])
        y_pred = np.array([[0], [1], [1], [0]])
        score = acc.compute(y_true, y_pred)
        assert score == 1.0

    def test_mixed_types(self):
        """Test mixing numpy and torch inputs."""
        acc = Accuracy()
        y_true = np.array([0, 1, 1, 0])
        y_pred = torch.tensor([0, 1, 1, 0])
        score = acc.compute(y_true, y_pred)
        assert score == 1.0

    def test_large_arrays(self):
        """Test with larger arrays."""
        acc = Accuracy()
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=1000)
        y_pred = y_true.copy()  # Perfect prediction
        score = acc.compute(y_true, y_pred)
        assert score == 1.0

    def test_float_labels(self):
        """Test that float labels work for classification."""
        acc = Accuracy()
        y_true = np.array([0.0, 1.0, 1.0, 0.0])
        y_pred = np.array([0.0, 1.0, 1.0, 0.0])
        score = acc.compute(y_true, y_pred)
        assert score == 1.0
