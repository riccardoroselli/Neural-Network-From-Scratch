import numpy as np
from typing import List, Optional
from .core import Module, Array


class Model:
    """
    Sequential model:
      - modules define the network graph in order
      - forward: left->right
      - backward: right->left
    """
    def __init__(
        self,
        modules: List[Module],
        loss=None,
        optimizer=None,
        regularizer=None,
        metrics: Optional[list] = None,
        callbacks: Optional[list] = None,
    ):
        self.modules = modules

        # These are placeholders for later steps (training)
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.metrics = metrics or []
        self.callbacks = callbacks or []

    def forward(self, X: Array, training: bool = True) -> Array:
        for m in self.modules:
            X = m.forward(X, training=training)
        return X

    def backward(self, dY: Array) -> None:
        for m in reversed(self.modules):
            dY = m.backward(dY)

    def params_and_grads(self):
        for m in self.modules:
            for p_g in m.params_and_grads():
                yield p_g

    # --- below will be used later for training/eval ---

    def compute_loss(self, y_true, y_pred) -> float:
        assert self.loss is not None
        base = self.loss.forward(y_pred, y_true)   # <-- swapped order

        reg = 0.0
        if self.regularizer is not None:
            reg = self.regularizer.penalty(self.modules)
        return float(base + reg)

    def step(self) -> None:
        assert self.optimizer is not None, "Model.optimizer is not set"

        # regularizer may add gradients, e.g. L2: dW += lam * W
        if self.regularizer is not None:
            self.regularizer.add_gradients(self.modules)

        self.optimizer.step(self.modules)

    def predict_proba(self, X: Array) -> Array:
        return self.forward(X, training=False)

    def predict(self, X: Array, threshold: float = 0.5) -> Array:
        y = self.predict_proba(X)
        return (y >= threshold).astype(int)
