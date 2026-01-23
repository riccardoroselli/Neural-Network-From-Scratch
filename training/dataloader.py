# nn/dataloader.py
import numpy as np


class BatchIterator:
    """
    Mini-batch iterator for training data.
    
    Iterates over dataset (X, y) in batches, with optional shuffling
    and support for incomplete final batches.
    
    Example:
        >>> loader = BatchIterator(X_train, y_train, batch_size=32, shuffle=True)
        >>> for X_batch, y_batch in loader:
        ...     # train on batch
    
    Args:
        X: input array, shape (N, D)
        y: target array, shape (N, ...) or None
        batch_size: number of samples per batch
        shuffle: whether to shuffle data each epoch
        drop_last: if True, drop incomplete final batch
        seed: random seed for shuffling (for reproducibility)
    """

    def __init__(self, X, y=None, batch_size=32, shuffle=True, drop_last=False, seed=None):
        if X is None:
            raise ValueError("X cannot be None")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        
        if y is not None and len(X) != len(y):
            raise ValueError(
                f"X and y must have same length: len(X)={len(X)}, len(y)={len(y)}"
            )
        
        self.X = X
        self.y = y
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = seed

    def __len__(self):
        """Return number of batches per epoch"""
        num_samples = len(self.X)
        
        if self.drop_last:
            return num_samples // self.batch_size
        
        # Ceiling division: number of batches including incomplete final batch
        return (num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches"""
        num_samples = len(self.X)
        indices = np.arange(num_samples)
        
        # Shuffle if requested
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)
        
        # Determine iteration range
        if self.drop_last:
            max_start = (num_samples // self.batch_size) * self.batch_size
        else:
            max_start = num_samples
        
        # Yield batches
        for start in range(0, max_start, self.batch_size):
            end = start + self.batch_size
            batch_indices = indices[start:end]
            
            X_batch = self.X[batch_indices]
            
            if self.y is None:
                yield X_batch
            else:
                y_batch = self.y[batch_indices]
                yield X_batch, y_batch
