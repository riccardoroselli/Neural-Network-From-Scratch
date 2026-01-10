# nn/losses.py
import numpy as np

Array = np.ndarray


class Loss:
    """Base class for loss functions"""

    def forward(self, y_pred: Array, y_true: Array) -> float:
        raise NotImplementedError

    def backward(self, y_pred: Array, y_true: Array) -> Array:
        """Gradient w.r.t. predictions"""
        raise NotImplementedError

    def __call__(self, y_pred: Array, y_true: Array) -> float:
        return self.forward(y_pred, y_true)

    def __repr__(self):
        return self.__class__.__name__


class MSE(Loss):
    """Mean Squared Error - regression loss"""

    def forward(self, y_pred: Array, y_true: Array) -> float:
        y_pred = np.atleast_2d(y_pred)
        y_true = np.atleast_2d(y_true)
        return float(np.mean(np.sum((y_pred - y_true) ** 2, axis=1)))

    def backward(self, y_pred: Array, y_true: Array) -> Array:
        # NOTE: no /N here (Dense already averages)
        return (y_pred - y_true)


class BinaryCrossEntropy(Loss):
    """Binary Cross Entropy - binary classification loss (expects probabilities)"""

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def forward(self, y_pred: Array, y_true: Array) -> float:
        # Ensure column vectors
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def backward(self, y_pred: Array, y_true: Array) -> Array:
        """
        Returns dL/dy_pred (same shape as y_pred).
        IMPORTANT: no /N here (Dense.backward already divides by N).
        """
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        y_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)

        # This is exactly the derivative of BCE wrt probability y_pred:
        # dL/dp = -(y/p) + (1-y)/(1-p)
        # (equivalently (p - y)/(p(1-p)))
        dY = (-(y_true / y_pred) + (1.0 - y_true) / (1.0 - y_pred))
        return dY
