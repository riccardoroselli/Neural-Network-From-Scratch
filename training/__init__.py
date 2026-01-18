"""
Training utilities for the NumPy-only NN framework.

Keeps training / experimentation separate from the pure NN engine in `nn/`.

Main entry points:
- BatchIterator (mini-batching)
- History (epoch logs)
- Trainer (fit/evaluate)
"""

from .dataloader import BatchIterator
from .history import History
from .trainer import Trainer

__all__ = ["BatchIterator", "History", "Trainer"]
