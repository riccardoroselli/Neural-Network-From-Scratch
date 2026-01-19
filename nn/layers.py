# nn/layers.py
import numpy as np
from .core import Module
from .initializers import xavier_uniform, he_uniform, zeros


class Dense(Module):
    """Fully connected (dense) layer"""
    
    def __init__(self, in_dim, out_dim, initializer=xavier_uniform, seed=None):
        """
        Args:
            in_dim: number of input features
            out_dim: number of output features
            initializer: weight initialization function
            seed: random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        
        # Initialize parameters
        self.W = initializer(in_dim, out_dim, rng)
        self.b = zeros(1, out_dim, rng)
        
        # Initialize gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Cache for backward pass
        self._X_cache = None

        self._initializer = initializer
        self._seed = seed
    
    def forward(self, X, training=True):
        """Linear transformation: Y = X @ W + b"""
        self._X_cache = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, dZ):
        """
        Backpropagate gradients through the layer.
        
        Args:
            dZ: gradient from next layer, shape (N, out_dim)
        
        Returns:
            dX: gradient to previous layer, shape (N, in_dim)
        """
        X = self._X_cache
        assert X is not None, "forward must be called before backward"
        
        # Compute gradients
        self.dW = np.dot(X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        
        # Gradient for previous layer
        dX = np.dot(dZ, self.W.T)
        return dX
    
    def params_and_grads(self):
        """Yield (param, grad) pairs for optimizer"""
        yield self.W, self.dW
        yield self.b, self.db

    def reset(self):
        """Reinitialize weights and biases to random values."""
        in_dim, out_dim = self.W.shape
        rng = np.random.default_rng(self._seed)
        self.W = self._initializer(in_dim, out_dim, rng)
        self.b = zeros(1, out_dim, rng)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._X_cache = None