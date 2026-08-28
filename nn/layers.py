# nn/layers.py
import numpy as np
from .core import Module
from .initializers import xavier_uniform, zeros


class Dense(Module):
    """
    Fully connected (dense) layer.
    
    Performs linear transformation: Y = X @ W + b
    """
    
    def __init__(self, in_dim, out_dim, initializer=xavier_uniform, seed=None):
        """
        Args:
            in_dim: number of input features
            out_dim: number of output features
            initializer: weight initialization function (default: Xavier)
            seed: random seed for reproducibility
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._initializer = initializer
        self._seed = seed
        
        self._X_cache = None
        
        # Initialize weights and biases
        self._initialize_parameters()
    
    def forward(self, X, training=True):
        """
        Forward pass: Y = X @ W + b
        
        Args:
            X: input tensor, shape (N, in_dim)
            training: unused (a Dense layer behaves identically in training
                      and inference; only modules such as Dropout differ)
        
        Returns:
            output tensor, shape (N, out_dim)
        """
        self._X_cache = X
        return np.dot(X, self.W) + self.b
    
    def backward(self, dZ):
        """
        Backward pass: compute gradients w.r.t. weights, biases, and inputs.
        
        Args:
            dZ: gradient from next layer, shape (N, out_dim)
        
        Returns:
            dX: gradient to pass to previous layer, shape (N, in_dim)
        """
        assert self._X_cache is not None, "forward() must be called before backward()"
        
        X = self._X_cache
        
        # Gradients for parameters
        self.dW = np.dot(X.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        
        # Gradient for previous layer
        dX = np.dot(dZ, self.W.T)
        
        return dX
    
    def params_and_grads(self):
        """Yield (parameter, gradient) pairs for optimizer"""
        yield self.W, self.dW
        yield self.b, self.db
    
    def reset(self):
        """Reinitialize weights and biases to random values"""
        self._initialize_parameters()
        self._X_cache = None
    
    def _initialize_parameters(self):
        """Initialize or reinitialize weights, biases, and gradients"""
        rng = np.random.default_rng(self._seed)
        
        self.W = self._initializer(self.in_dim, self.out_dim, rng)
        self.b = zeros(1, self.out_dim, rng)
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
