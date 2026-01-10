import numpy as np
from typing import List
from .core import Module
from .layers import Dense


class Regularizer:
    def penalty(self, modules: List[Module]) -> float:
        return 0.0

    def add_gradients(self, modules: List[Module]) -> None:
        pass


class L2(Regularizer):
    """
    L2 regularization on Dense weights (NOT biases):
      penalty = 0.5 * lam * sum(||W||^2)
      adds gradients: dW += lam * W
    """
    def __init__(self, lam: float = 0.0):
        self.lam = float(lam)

    def penalty(self, modules: List[Module]) -> float:
        if self.lam == 0.0:
            return 0.0

        s = 0.0
        for m in modules:
            if isinstance(m, Dense):
                s += float(np.sum(m.W * m.W))
        return 0.5 * self.lam * s

    def add_gradients(self, modules: List[Module]) -> None:
        if self.lam == 0.0:
            return

        for m in modules:
            if isinstance(m, Dense):
                m.dW += self.lam * m.W
