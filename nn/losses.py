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
        Internal helper to resolve 1D vector ambiguity.
        Uses y_pred (which is network-external and reliable) to understand how
        to interpret y_true (which comes from the user).
        """
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        if y_true.ndim == 1:
            N, D = y_pred.shape
            if N == 1 and D > 1 and y_true.shape[0] == D:
                 y_true = y_true.reshape(1, -1)
            else:
                 y_true = y_true.reshape(-1, 1)
        
        return y_pred, y_true


class MSE(Loss):
    """
    Mean Squared Error.
    Gestisce automaticamente regressione Batch e Multi-Target singolo.
    """

    def forward(self, y_pred, y_true):
        # Auto-correzione delle forme
        y_pred, y_true = self._fix_shapes(y_pred, y_true)

        err2 = (y_pred - y_true) ** 2          # (N, D)
        per_sample = np.sum(err2, axis=1)      # (N,)
        return float(np.mean(per_sample))      # scalar

    def backward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)

        N = y_true.shape[0]
        return (2.0 / N) * (y_pred - y_true)


class MEE(Loss):
    """
    Mean Euclidean Error.
    Ideale per Regressione Multi-Target.
    """

    def forward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)
        return float(np.mean(np.linalg.norm(y_pred - y_true, axis=1)))

    def backward(self, y_pred, y_true):
        y_pred, y_true = self._fix_shapes(y_pred, y_true)

        eps = 1e-12
        diff = y_pred - y_true                          
        norms = np.linalg.norm(diff, axis=1, keepdims=True)  
        norms = np.clip(norms, eps, None)

        N = y_true.shape[0]
        return (diff / norms) / N


class BinaryCrossEntropy(Loss):
    """
    The gradient (p - y) / (p*(1-p)) appears numerically unstable,
    but when used with Sigmoid activation, the terms p*(1-p) cancel out
    perfectly, yielding the stable gradient (p - y) / N.
    The clipping to [eps, 1-eps] provides additional numerical safety.

    USE WITH SIGMOID
    """

    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)

        per_elem = y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)  # (N, D)
        per_sample = -np.sum(per_elem, axis=1)                                       # (N,)
        loss = np.mean(per_sample)                                                   # scalar
        return float(loss)

    def backward(self, y_pred, y_true):
        """
        Gradient w.r.t. probabilities y_pred (Sigmoid output).
        Matches the batch-mean convention in forward() by dividing by N (batch size).
        """
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)

        N = y_true.shape[0]
        return (1.0 / N) * ((y_pred - y_true) / (y_pred * (1.0 - y_pred)))


class CrossEntropy(Loss):
    """
    Categorical Cross-Entropy (batch-mean).
    USE WITH SOFTMAX.

    loss = (1/N) * sum_i [ - sum_c y_true[i,c] * log(y_pred[i,c]) ]
    backward returns the standard combined Softmax+CE gradient:
        dloss/dz = (y_pred - y_true) / N

    WARNING: This loss assumes y_true is One-Hot encoded and strictly coupled with Softmax.
    """

    def __init__(self, eps=1e-12):
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return float(loss)

    def backward(self, y_pred, y_true):
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)

        N = y_true.shape[0]
        return (y_pred - y_true) / N