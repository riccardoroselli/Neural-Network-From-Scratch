# nn/regularizers.py
import numpy as np
from .layers import Dense


class Regularizer:
    """Base class for regularization techniques"""
    
    def penalty(self, modules):
        """
        Compute regularization penalty term to add to loss.
        
        Args:
            modules: list of Module instances
        
        Returns:
            scalar penalty value
        """
        return 0.0
    
    def add_gradients(self, modules):
        """
        Add regularization gradients to parameter gradients.
        
        Args:
            modules: list of Module instances
        """
        pass

    def _get_dense_layers(self, modules):
        """Helper to extract Dense layers from modules list"""
        return [m for m in modules if isinstance(m, Dense)]


class L2(Regularizer):
    """
    L2 regularization (weight decay) on Dense layer weights.
    
    Penalty: 0.5 * lambda * sum(||W||^2)
    Gradient: dW += lambda * W
    
    Note: Applied only to weights, NOT biases.
    """
    
    def __init__(self, lam=0.0):
        """
        Args:
            lam: regularization strength (lambda)
        """
        self.lam = lam
    
    def penalty(self, modules):
        """Compute L2 penalty on all Dense layer weights"""
        if self.lam == 0.0:
            return 0.0
        
        total_penalty = 0.0
        for layer in self._get_dense_layers(modules):
            total_penalty += np.sum(layer.W ** 2)
        
        return 0.5 * self.lam * total_penalty
    
    def add_gradients(self, modules):
        """Add L2 gradient (lambda * W) to weight gradients"""
        if self.lam == 0.0:
            return
        
        for layer in self._get_dense_layers(modules):
            layer.dW += self.lam * layer.W


class L1(Regularizer):
    """
    L1 regularization (Lasso) on Dense layer weights.
    
    Penalty: lambda * sum(|W|)
    Gradient: dW += lambda * sign(W)
    
    Promotes sparsity (many weights become exactly zero).
    """
    
    def __init__(self, lam=0.0):
        """
        Args:
            lam: regularization strength (lambda)
        """
        self.lam = lam
    
    def penalty(self, modules):
        """Compute L1 penalty on all Dense layer weights"""
        if self.lam == 0.0:
            return 0.0
        
        total_penalty = 0.0
        for layer in self._get_dense_layers(modules):
            total_penalty += np.sum(np.abs(layer.W))
        
        return self.lam * total_penalty
    
    def add_gradients(self, modules):
        """Add L1 gradient (lambda * sign(W)) to weight gradients"""
        if self.lam == 0.0:
            return
        
        for layer in self._get_dense_layers(modules):
            layer.dW += self.lam * np.sign(layer.W)
