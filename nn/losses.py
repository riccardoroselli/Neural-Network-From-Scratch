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

    def _fix_shapes(self, y_pred, y_true):
        """
        Promote 1D arrays to 2D column vectors so every loss can assume a
        (n_samples, n_outputs) layout. Each array is reshaped independently.
        """
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        
        return y_pred, y_true


# ==================== Regression Losses ====================

class MSE(Loss):
    """
    Mean Squared Error.
    Suitable for single or multi-target regression.
    """

    def forward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)
        
        diff = y_pred - y_true
        squared_errors = diff ** 2
        loss_per_sample = np.sum(squared_errors, axis=1)
        
        return float(np.mean(loss_per_sample))

    def backward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)
        
        N = y_true.shape[0]
        return (2.0 / N) * (y_pred - y_true)


class MEE(Loss):
    """
    Mean Euclidean Error.
    Ideal for multi-target regression (e.g., ML-CUP).
    """

    def forward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)
        
        diff = y_pred - y_true
        euclidean_distances = np.linalg.norm(diff, axis=1)
        
        return float(np.mean(euclidean_distances))

    def backward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)
        
        eps = 1e-12
        diff = y_pred - y_true
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.clip(norms, eps, None)
        
        N = y_true.shape[0]
        return (diff / norms) / N


# ==================== Classification Losses ====================

class BinaryCrossEntropy(Loss):
    """
    Binary Cross Entropy for binary classification.
    Use with Sigmoid activation.
    
    Formula:
        loss = (1/N) * sum[ -y*log(p) - (1-y)*log(1-p) ]
        gradient = (1/N) * (p - y) / (p * (1-p))
    """

    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)
        y_pred = self._clip_probs(y_pred)
        
        # Binary cross-entropy formula
        log_loss = y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)
        loss_per_sample = -np.sum(log_loss, axis=1)
        
        return float(np.mean(loss_per_sample))

    def backward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)
        y_pred = self._clip_probs(y_pred)
        
        N = y_true.shape[0]
        return (1.0 / N) * ((y_pred - y_true) / (y_pred * (1.0 - y_pred)))

    def _clip_probs(self, probs):
        """Clip probabilities to avoid log(0)"""
        return np.clip(probs, self.eps, 1.0 - self.eps)


class CrossEntropy(Loss):
    """
    Categorical Cross-Entropy for multi-class classification.
    Use with Softmax activation.
    
    Formula:
        loss = (1/N) * sum[ -sum_c y[c] * log(p[c]) ]
        gradient = (p - y) / N  (combined Softmax+CE derivative)
    
    Note: Assumes y_true is one-hot encoded.
    """

    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        y_pred = self._clip_probs(y_pred)
        
        # Cross-entropy formula
        log_probs = np.log(y_pred)
        loss_per_sample = -np.sum(y_true * log_probs, axis=1)
        
        return float(np.mean(loss_per_sample))

    def backward(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        
        N = y_true.shape[0]
        return (y_pred - y_true) / N

    def _clip_probs(self, probs):
        """Clip probabilities to avoid log(0)"""
        return np.clip(probs, self.eps, 1.0 - self.eps)
