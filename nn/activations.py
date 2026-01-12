# nn/activations.py
import numpy as np
from .core import Module


class ReLU(Module):
    def __init__(self):
        self.X = None
    
    def forward(self, X, training=True):
        self.X = X
        return np.maximum(0.0, X)
    
    def backward(self, dA):
        return dA * (self.X > 0.0)


class Sigmoid(Module):
    def __init__(self):
        self.X = None
    
    def forward(self, X, training=True):
        self.X = np.clip(X, -500, 500)
        return 1.0 / (1.0 + np.exp(-self.X))
    
    def backward(self, dA):
        A = 1.0 / (1.0 + np.exp(-self.X))
        return dA * (A * (1.0 - A))


class Tanh(Module):
    def __init__(self):
        self.X = None
    
    def forward(self, X, training=True):
        self.X = X
        return np.tanh(X)
    
    def backward(self, dA):
        return dA * (1.0 - np.tanh(self.X) ** 2)


class Identity(Module):
    def forward(self, X, training=True):
        return X
    
    def backward(self, dA):
        return dA

class Softmax(Module):
    """
    Softmax activation (use with CrossEntropy loss).
    The combined gradient is handled by the loss function.
    """
    
    def forward(self, X, training=True):
        # Numerical stability
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def backward(self, dA):
        # If used with CrossEntropy, the gradient is already simplified
        # and passed directly from the loss
        return dA
