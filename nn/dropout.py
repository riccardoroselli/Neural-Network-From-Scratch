# nn/dropout.py
import numpy as np
from .core import Module


class Dropout(Module):
    """
    Inverted Dropout:
    - training=True: randomly zeros activations with prob p,
                     scales survivors by 1/(1-p)
    - training=False: identity (no dropout)
    """
    
    def __init__(self, p=0.5, seed=None):
        assert 0.0 <= p < 1.0, "Dropout p must be in [0, 1)"
        self.p = p
        self.keep_prob = 1.0 - self.p
        self.rng = np.random.default_rng(seed)
        self.mask = None
    
    def forward(self, X, training=True):
        if not training or self.p == 0.0:
            self.mask = None
            return X
        
        self.mask = (self.rng.random(X.shape) < self.keep_prob)
        return (X * self.mask) / self.keep_prob
    
    def backward(self, dY):
        # Protezione contro divisione per zero e overflow
        if self.keep_prob < 1e-10:
            return np.zeros_like(dY)
            
        # Calcolo standard (Inverted Dropout)
        return (dY * self.mask) / self.keep_prob
