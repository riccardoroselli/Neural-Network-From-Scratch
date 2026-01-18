import numpy as np


class BatchIterator:
    """
    Simple mini-batch iterator over (X, y).

    Parameters
    ----------
    X : np.ndarray
        Input array, shape (N, D)
    y : np.ndarray | None
        Target array, shape (N, ...) or None
    batch_size : int
    shuffle : bool
    drop_last : bool
        If True, drop the last incomplete batch.
    seed : int | None
        If provided and shuffle=True, shuffling will be deterministic.
    """

    def __init__(self, X, y=None, batch_size=32, shuffle=True, drop_last=False, seed=None):
        if X is None:
            raise ValueError("X cannot be None")
        self.X = X
        self.y = y

        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed

        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X and y must have the same first dimension")

    def __len__(self):
        n = len(self.X)
        bs = self.batch_size
        if self.drop_last:
            return n // bs
        return (n + bs - 1) // bs

    def __iter__(self):
        n = len(self.X)
        idx = np.arange(n)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(idx)

        bs = self.batch_size
        end = (n // bs) * bs if self.drop_last else n

        for start in range(0, end, bs):
            batch_idx = idx[start:start + bs]
            Xb = self.X[batch_idx]
            if self.y is None:
                yield Xb
            else:
                yield Xb, self.y[batch_idx]
