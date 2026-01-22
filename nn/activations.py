# nn/activations.py
import numpy as np
from .core import Module

class ReLU(Module):
    def __init__(self):
        self.mask = None
    
    def forward(self, X, training=True):
        self.mask = (X > 0)
        return X * self.mask
    
    def backward(self, dA):
        return dA * self.mask

class Sigmoid(Module):
    def __init__(self):
        self.A = None
    
    def forward(self, X, training=True):
        X_clipped = np.clip(X, -20, 20) 
        self.A = 1.0 / (1.0 + np.exp(-X_clipped))
        return self.A
    
    def backward(self, dA):
        return dA * (self.A * (1.0 - self.A))

class Tanh(Module):
    def __init__(self):
        self.A = None 
    
    def forward(self, X, training=True):
        self.A = np.tanh(X)
        return self.A
    
    def backward(self, dA):
        return dA * (1.0 - self.A ** 2)

class Identity(Module):
    def forward(self, X, training=True):
        return X
    
    def backward(self, dA):
        return dA

class Softmax(Module):
    """
    IMPORTANT: This implementation assumes the gradient is already the combined Softmax+CrossEntropy
    gradient computed by the loss function. 
    The backward pass is a simple pass-through: dL/dz = (p - y) / N. DO NOT use this Softmax with 
    other losses or in standalone fashion. For correct behavior, always pair with CrossEntropy loss. 
    """
    
    def forward(self, X, training=True):
        # Numerical stability
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        self.A = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.A
    
    def backward(self, dA):
        # If used with CrossEntropy, the gradient is already simplified
        # and passed directly from the loss
        return dA
