# nn/regularizers.py
import numpy as np
from .core import Module
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


class L2(Regularizer):
    """
    L2 regularization (weight decay) on Dense layer weights.
    
    Penalty: 0.5 * lam * sum(||W||^2)
    Gradient: dW += lam * W
    
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
        
        penalty = 0.0
        for m in modules:
            if isinstance(m, Dense):
                penalty += np.sum(m.W * m.W)
        
        return 0.5 * self.lam * penalty
    
    def add_gradients(self, modules):
        """Add L2 gradient (lam * W) to weight gradients"""
        if self.lam == 0.0:
            return
        
        for m in modules:
            if isinstance(m, Dense):
                m.dW += self.lam * m.W


class L1(Regularizer):
    """
    L1 regularization (Lasso) on Dense layer weights.
    
    Penalty: lam * sum(|W|)
    Gradient: dW += lam * sign(W)
    
    Promotes sparsity (many weights become exactly zero).
    """
    
    def __init__(self, lam=0.0):
        self.lam = lam
    
    def penalty(self, modules):
        """Compute L1 penalty on all Dense layer weights"""
        if self.lam == 0.0:
            return 0.0
        
        penalty = 0.0
        for m in modules:
            if isinstance(m, Dense):
                penalty += np.sum(np.abs(m.W))
        
        return self.lam * penalty
    
    def add_gradients(self, modules):
        """Add L1 gradient (lam * sign(W)) to weight gradients"""
        if self.lam == 0.0:
            return
        
        for m in modules:
            if isinstance(m, Dense):
                m.dW += self.lam * np.sign(m.W)
