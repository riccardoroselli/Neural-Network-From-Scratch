import numpy as np
from typing import Optional
from .core import Module, Array


class Dropout(Module):
    """
    Inverted Dropout:
      - training=True: randomly zeros activations with prob p,
        scales survivors by 1/(1-p)
      - training=False: identity
    """
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        assert 0.0 <= p < 1.0, "Dropout p must be in [0,1)"
        self.p = float(p)
        self.keep_prob = 1.0 - self.p
        self.rng = np.random.default_rng(seed)
        self._mask: Optional[Array] = None

    def forward(self, X: Array, training: bool = True) -> Array:
        if (not training) or self.p == 0.0:
            self._mask = None
            return X

        self._mask = (self.rng.random(X.shape) < self.keep_prob)
        return (X * self._mask) / self.keep_prob

    def backward(self, dY: Array) -> Array:
        if self._mask is None:
            return dY
        return (dY * self._mask) / self.keep_prob
