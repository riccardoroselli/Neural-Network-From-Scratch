# nn/metrics.py
import numpy as np


class Metric:
    """Base class for all evaluation metrics."""

    def compute(self, y_pred, y_true):
        """
        Compute the metric value.
        
        Args:
            y_pred: predicted values from the model
            y_true: ground truth values
        
        Returns:
            scalar metric value
        """
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.compute(y_pred, y_true)

    def __repr__(self):
        return self.__class__.__name__


# ==================== Classification Metrics ====================

class Accuracy(Metric):
    """Accuracy metric for classification tasks."""

    def compute(self, y_pred, y_true):
        # Binary classification: threshold at 0.5
        if y_pred.shape[-1] == 1:
            preds = (y_pred >= 0.5).astype(int)
            targets = y_true.astype(int)
        # Multi-class: argmax over classes
        else:
            preds = np.argmax(y_pred, axis=1)
            targets = np.argmax(y_true, axis=1)
        
        return float(np.mean(preds == targets))


class Precision(Metric):
    """Precision metric for binary classification."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def compute(self, y_pred, y_true):
        preds = (y_pred >= self.threshold).astype(int)
        
        true_positives = np.sum((preds == 1) & (y_true == 1))
        false_positives = np.sum((preds == 1) & (y_true == 0))
        
        denominator = true_positives + false_positives
        if denominator == 0:
            return 0.0
        
        return float(true_positives / denominator)


class Recall(Metric):
    """Recall metric for binary classification."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def compute(self, y_pred, y_true):
        preds = (y_pred >= self.threshold).astype(int)
        
        true_positives = np.sum((preds == 1) & (y_true == 1))
        false_negatives = np.sum((preds == 0) & (y_true == 1))
        
        denominator = true_positives + false_negatives
        if denominator == 0:
            return 0.0
        
        return float(true_positives / denominator)


class F1Score(Metric):
    """F1 Score for binary classification."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def compute(self, y_pred, y_true):
        preds = (y_pred >= self.threshold).astype(int)
        
        # Calculate confusion matrix components
        true_positives = np.sum((preds == 1) & (y_true == 1))
        false_positives = np.sum((preds == 1) & (y_true == 0))
        false_negatives = np.sum((preds == 0) & (y_true == 1))
        
        # Calculate precision
        precision_denom = true_positives + false_positives
        precision = true_positives / precision_denom if precision_denom > 0 else 0.0
        
        # Calculate recall
        recall_denom = true_positives + false_negatives
        recall = true_positives / recall_denom if recall_denom > 0 else 0.0
        
        # Calculate F1
        f1_denom = precision + recall
        if f1_denom == 0:
            return 0.0
        
        return float(2 * (precision * recall) / f1_denom)


# ==================== Regression Metrics ====================

class MEE(Metric):
    """
    Mean Euclidean Error.
    Formula: (1/N) * sum(||y_pred - y_true||_2)
    """

    def compute(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        
        diff = y_pred - y_true
        euclidean_distances = np.linalg.norm(diff, axis=1)
        
        return float(np.mean(euclidean_distances))


class MSE(Metric):
    """
    Mean Squared Error.
    Formula: (1/N) * sum(||y_pred - y_true||^2)
    """

    def compute(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        
        diff = y_pred - y_true
        squared_errors = diff ** 2
        mse_per_sample = np.sum(squared_errors, axis=1)
        
        return float(np.mean(mse_per_sample))
