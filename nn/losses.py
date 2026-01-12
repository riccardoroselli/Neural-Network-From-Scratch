# nn/losses.py
import numpy as np


class Loss:
    """Base class for loss functions"""

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        """Gradient w.r.t. predictions"""
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def __repr__(self):
        return self.__class__.__name__


class MSE(Loss):
    """Mean Squared Error - regression loss"""

    def forward(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))

    def backward(self, y_pred, y_true):
        # NOTE: no /N here (Dense already averages)
        return y_pred - y_true


class MEE(Loss):
    """Mean Euclidean Error - regression loss (L2 norm per sample)"""

    def forward(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        return float(np.mean(np.linalg.norm(y_pred - y_true, axis=1)))

    def backward(self, y_pred, y_true):
        """
        Gradient: (y_pred - y_true) / ||y_pred - y_true||_2
        NOTE: no /N here (Dense already averages)
        """
        diff = y_pred - y_true
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return diff / norms


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy - binary classification loss (use with Sigmoid)"""

    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self, y_pred, y_true):
        """
        Gradient: -(y/p) + (1-y)/(1-p)
        NOTE: no /N here (Dense already averages)
        """
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return -(y_true / y_pred) + (1.0 - y_true) / (1.0 - y_pred)


class CrossEntropy(Loss):
    """
    Categorical Cross-Entropy - multi-class classification loss.
    Use with Softmax activation (gradient is simplified as y_pred - y_true).
    """

    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Expects:
            y_pred: softmax probabilities, shape (N, num_classes)
            y_true: one-hot encoded labels, shape (N, num_classes)
        """
        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return float(loss)

    def backward(self, y_pred, y_true):
        """
        Combined Softmax+CrossEntropy gradient: (y_pred - y_true)
        
        This is the simplified gradient when Softmax is the last activation.
        The Softmax.backward() should just pass this gradient through.
        """
        # Simplified gradient (no clipping needed here)
        return y_pred - y_true
