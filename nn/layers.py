import numpy as np
from typing import Optional, Callable
from .core import Module, Array

def xavier_uniform(in_dim: int, out_dim: int, rng: np.random.Generator) -> Array:
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    return rng.uniform(-limit, limit, size=(in_dim, out_dim))

class Dense(Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        initializer: Callable[[int, int, np.random.Generator], Array] = xavier_uniform,
        seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(seed)
        self.W = initializer(in_dim, out_dim, rng)
        self.b = np.zeros((1, out_dim))

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        self._X_cache = None

    def forward(self, X: Array, training: bool = True) -> Array:
        self._X_cache = X
        return X @ self.W + self.b

    def backward(self, dZ: Array) -> Array:
        X = self._X_cache
        assert X is not None

        N = X.shape[0]
        self.dW = (X.T @ dZ) / N
        self.db = np.sum(dZ, axis=0, keepdims=True) / N
        dX = dZ @ self.W.T
        return dX

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db
