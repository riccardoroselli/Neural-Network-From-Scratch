# nn/activations.py
import numpy as np
from .core import Module


class ReLU(Module):
    """
    Rectified Linear Unit activation.
    
    Formula:
        f(x) = max(0, x)
        f'(x) = 1 if x > 0, else 0
    """
    
    def __init__(self):
        self.mask = None
    
    def forward(self, X, training=True):
        """Apply ReLU activation"""
        self.mask = (X > 0)
        return X * self.mask
    
    def backward(self, dA):
        """Gradient: zero out negative inputs"""
        return dA * self.mask


class Sigmoid(Module):
    """
    Sigmoid activation.
    
    Formula:
        f(x) = 1 / (1 + exp(-x))
        f'(x) = f(x) * (1 - f(x))
    
    Use with BinaryCrossEntropy loss for binary classification.
    """
    
    def __init__(self):
        self.output = None
    
    def forward(self, X, training=True):
        """
        Apply sigmoid activation with numerical stability.
        
        Clips input to [-20, 20] to prevent overflow in exp().
        """
        X_clipped = np.clip(X, -20, 20)
        self.output = 1.0 / (1.0 + np.exp(-X_clipped))
        return self.output
    
    def backward(self, dA):
        """Gradient using cached output"""
        return dA * self.output * (1.0 - self.output)


class Tanh(Module):
    """
    Hyperbolic tangent activation.
    
    Formula:
        f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        f'(x) = 1 - f(x)^2
    
    Popular for hidden layers (outputs in range [-1, 1]).
    """
    
    def __init__(self):
        self.output = None
    
    def forward(self, X, training=True):
        """Apply tanh activation"""
        self.output = np.tanh(X)
        return self.output
    
    def backward(self, dA):
        """Gradient using cached output"""
        return dA * (1.0 - self.output ** 2)


class Identity(Module):
    """
    Identity activation (no transformation).
    
    Formula:
        f(x) = x
        f'(x) = 1
    
    Use for regression tasks as output activation.
    """
    
    def forward(self, X, training=True):
        """Return input unchanged"""
        return X
    
    def backward(self, dA):
        """Gradient is identity"""
        return dA


class Softmax(Module):
    """
    Softmax activation for multi-class classification.
    
    Formula:
        f(x_i) = exp(x_i) / sum(exp(x_j))
    
    Note: Gradient is typically fused with CrossEntropy loss for
    numerical stability. The backward() method here is a pass-through.
    
    Use with CrossEntropy loss.
    """
    
    def __init__(self):
        self.output = None
    
    def forward(self, X, training=True):
        """
        Apply softmax with numerical stability.
        
        Subtracts max(X) before exp to prevent overflow.
        """
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dA):
        """
        Pass-through gradient.
        
        The actual Softmax gradient is computed by CrossEntropy loss
        for numerical stability (combined Softmax+CE derivative).
        """
        return dA
