# nn/optim.py
from .core import Module
import numpy as np


class Optimizer:
    """Base class for optimizers"""
    
    def step(self, modules):
        """
        Update parameters of all modules.
        
        Args:
            modules: list of Module instances with learnable parameters
        """
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer"""
    
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def step(self, modules):
        """Apply gradient descent: param -= lr * grad"""
        for m in modules:
            for p, g in m.params_and_grads():
                p -= self.lr * g

class SGDMomentum(Optimizer):
    """SGD with momentum"""
    
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
    
    def step(self, modules):
        for m in modules:
            for p, g in m.params_and_grads():
                # Get or initialize velocity for this parameter
                p_id = id(p)
                if p_id not in self.velocities:
                    self.velocities[p_id] = np.zeros_like(p)
                
                # Update velocity and parameter
                v = self.velocities[p_id]
                v[:] = self.momentum * v - self.lr * g
                p += v


class Adam(Optimizer):
    """Adam optimizer"""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0   # timestep
    
    def step(self, modules):
        self.t += 1
        
        for m in modules:
            for p, g in m.params_and_grads():
                p_id = id(p)
                
                # Initialize moments if needed
                if p_id not in self.m:
                    self.m[p_id] = np.zeros_like(p)
                    self.v[p_id] = np.zeros_like(p)
                
                # Update biased moments
                self.m[p_id] = self.beta1 * self.m[p_id] + (1 - self.beta1) * g
                self.v[p_id] = self.beta2 * self.v[p_id] + (1 - self.beta2) * (g ** 2)
                
                # Bias correction
                m_hat = self.m[p_id] / (1 - self.beta1 ** self.t)
                v_hat = self.v[p_id] / (1 - self.beta2 ** self.t)
                
                # Update parameters
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
