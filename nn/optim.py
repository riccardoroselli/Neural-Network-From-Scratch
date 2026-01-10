from typing import List
from .core import Module


class Optimizer:
    def step(self, modules: List[Module]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.1):
        self.lr = float(lr)

    def step(self, modules: List[Module]) -> None:
        for m in modules:
            for p, g in m.params_and_grads():
                p -= self.lr * g
