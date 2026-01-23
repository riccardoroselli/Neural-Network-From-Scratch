# nn/dropout.py
import numpy as np
from .core import Module


class Dropout(Module):
    """
    Inverted Dropout implementation.
    
    During training:
        - Randomly zeros activations with probability p
        - Scales surviving activations by 1/(1-p) to maintain expected value
    
    During inference:
        - Acts as identity (no dropout applied)
    """
    
    def __init__(self, p=0.5, seed=None):
        """
        Args:
            p: dropout probability (fraction of neurons to drop)
            seed: random seed for reproducibility
        """
        assert 0.0 <= p < 1.0, f"Dropout probability must be in [0, 1), got {p}"
        
        self.p = p
        self.keep_prob = 1.0 - p
        self.scale = 1.0 / self.keep_prob if self.keep_prob > 0 else 1.0
        self.rng = np.random.default_rng(seed)
        self.mask = None
    
    def forward(self, X, training=True):
        """
        Forward pass with dropout.
        
        Args:
            X: input activations
            training: if True, apply dropout; if False, return X unchanged
        
        Returns:
            output after dropout (scaled if training=True)
        """
        # Inference mode or no dropout
        if not training or self.p == 0.0:
            return X
        
        # Generate binary mask and scale
        self.mask = self.rng.random(X.shape) < self.keep_prob
        return X * self.mask * self.scale
    
    def backward(self, dY):
        """
        Backward pass: apply same mask and scaling as forward pass.
        
        Args:
            dY: gradient from next layer
        
        Returns:
            gradient with dropout mask applied
        """
        if self.mask is None:
            return dY
        
        return dY * self.mask * self.scale
