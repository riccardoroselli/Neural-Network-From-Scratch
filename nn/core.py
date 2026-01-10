import numpy as np
from typing import Iterable, Tuple

Array = np.ndarray

class Module:
    def forward(self, X: Array, training: bool = True) -> Array:
        raise NotImplementedError

    def backward(self, dY: Array) -> Array:
        raise NotImplementedError

    def params_and_grads(self) -> Iterable[Tuple[Array, Array]]:
        return []
