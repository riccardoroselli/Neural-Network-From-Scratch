import numpy as np
from typing import Optional
from .core import Module, Array


class Tanh(Module):
    def __init__(self):
        self._A_cache: Optional[Array] = None

    def forward(self, X: Array, training: bool = True) -> Array:
        A = np.tanh(X)
        self._A_cache = A
        return A

    def backward(self, dA: Array) -> Array:
        assert self._A_cache is not None, "forward must be called before backward"
        A = self._A_cache
        return dA * (1.0 - A ** 2)


class Sigmoid(Module):
    def __init__(self):
        self._A_cache: Optional[Array] = None

    def forward(self, X: Array, training: bool = True) -> Array:
        # numerically stable sigmoid
        # sigmoid(x) = 1/(1+exp(-x))
        # stable split for x>=0 and x<0
        A = np.empty_like(X, dtype=float)
        pos = (X >= 0)
        neg = ~pos
        A[pos] = 1.0 / (1.0 + np.exp(-X[pos]))
        exp_x = np.exp(X[neg])
        A[neg] = exp_x / (1.0 + exp_x)

        self._A_cache = A
        return A

    def backward(self, dA: Array) -> Array:
        assert self._A_cache is not None, "forward must be called before backward"
        A = self._A_cache
        return dA * (A * (1.0 - A))


class ReLU(Module):
    def __init__(self):
        self._X_cache: Optional[Array] = None

    def forward(self, X: Array, training: bool = True) -> Array:
        self._X_cache = X
        return np.maximum(0.0, X)

    def backward(self, dA: Array) -> Array:
        assert self._X_cache is not None, "forward must be called before backward"
        X = self._X_cache
        return dA * (X > 0.0)
