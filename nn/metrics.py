# nn/metrics.py
import numpy as np


class Metric:
    """
    Base class for all evaluation metrics.

    """

    def compute(self, y_pred, y_true):
        """
        Compute the metric value.

        Args:
            y_pred: predicted values from the model.
            y_true: ground truth values.
        
        Returns:
            scalar metric value.
        """
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.compute(y_pred, y_true)

    def __repr__(self):
        return self.__class__.__name__


class Accuracy(Metric):
    """
    Accuracy metric for classification tasks.

    """

    def compute(self, y_pred, y_true):
        """
        Determine accuracy by comparing predicted classes with targets.
        
        """
        # For binary classification (MONK), apply 0.5 threshold
        if y_pred.shape[-1] == 1:
            preds = (y_pred >= 0.5).astype(int)
            targets = y_true.astype(int)
        # For multi-class classification (Softmax), use the index of the max value
        else:
            preds = np.argmax(y_pred, axis=1)
            targets = np.argmax(y_true, axis=1)
            
        return float(np.mean(preds == targets))


class MEE(Metric):
    """
    Mean Euclidean Error (MEE).

    """

    def compute(self, y_pred, y_true):
        """
        Formula: (1/l) * sum(||y_pred_p - y_true_p||_2)
        """
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        
        # Calculate Euclidean distance for each pattern along the feature axis
        dist = np.linalg.norm(y_pred - y_true, axis=1)
        return float(np.mean(dist))


class MSE(Metric):
    """
    Mean Squared Error (MSE).

    """

    def compute(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        # Average sum of squared errors per sample
        return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))


class Precision(Metric):
    """
    Precision metric for binary classification.

    """

    def compute(self, y_pred, y_true):
        preds = (y_pred >= 0.5).astype(int)
        true_positives = np.sum((preds == 1) & (y_true == 1))
        false_positives = np.sum((preds == 1) & (y_true == 0))
        
        denominator = true_positives + false_positives
        if denominator == 0:
            return 0.0
        return float(true_positives / denominator)


class Recall(Metric):
    """
    Recall metric for binary classification.

    """

    def compute(self, y_pred, y_true):
        preds = (y_pred >= 0.5).astype(int)
        true_positives = np.sum((preds == 1) & (y_true == 1))
        false_negatives = np.sum((preds == 0) & (y_true == 1))
        
        denominator = true_positives + false_negatives
        if denominator == 0:
            return 0.0
        return float(true_positives / denominator)


class F1Score(Metric):
    """
    F1 Score for binary classification.

    """

    def compute(self, y_pred, y_true):
        precision_val = Precision().compute(y_pred, y_true)
        recall_val = Recall().compute(y_pred, y_true)
        
        denominator = precision_val + recall_val
        if denominator == 0:
            return 0.0
        return float(2 * (precision_val * recall_val) / denominator)