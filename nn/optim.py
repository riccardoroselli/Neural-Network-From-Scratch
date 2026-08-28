# nn/optim.py
import numpy as np


class Optimizer:
    """Base class for all optimizers"""

    def step(self, modules):
        """
        Update parameters of all modules.

        Args:
            modules: list of Module instances with learnable parameters
        """
        raise NotImplementedError

    @staticmethod
    def _iter_keyed_params(modules):
        """
        Yield (state_key, param, grad) for every learnable parameter.

        The key is the parameter's position in the network - (module index,
        parameter index within that module) - rather than id(param).

        id() is only unique among *live* objects: reinitializing a layer frees
        its arrays, and a later allocation can land on the same address and
        inherit the previous parameter's optimizer state. A positional key is
        stable across reinitialization and cannot collide, so momentum and
        Adam moments always stay matched to the parameter they belong to.
        """
        for module_index, module in enumerate(modules):
            for param_index, (param, grad) in enumerate(module.params_and_grads()):
                yield (module_index, param_index), param, grad


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule:
        theta = theta - lr * grad
    
    Args:
        lr: learning rate (step size)
    """
    
    def __init__(self, lr=0.1):
        self.lr = lr
    
    def step(self, modules):
        """Apply gradient descent to all parameters"""
        for module in modules:
            for param, grad in module.params_and_grads():
                param -= self.lr * grad


class SGDMomentum(Optimizer):
    """
    SGD with Momentum optimizer.
    
    Update rule:
        v = momentum * v - lr * grad
        theta = theta + v
    
    Momentum accumulates gradients across iterations, smoothing
    the optimization trajectory and accelerating convergence.
    
    Args:
        lr: learning rate
        momentum: momentum factor (typically 0.9)
    """
    
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
    
    def step(self, modules):
        """Apply momentum-based gradient descent"""
        for key, param, grad in self._iter_keyed_params(modules):
            # Initialize velocity if needed
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(param)

            # Update velocity
            velocity = self.velocities[key]
            velocity *= self.momentum
            velocity -= self.lr * grad

            # Update parameter
            param += velocity


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines momentum (first moment) and RMSProp (second moment)
    with bias correction for early training steps.
    
    Update rule:
        m = beta1 * m + (1 - beta1) * grad          (first moment)
        v = beta2 * v + (1 - beta2) * grad^2        (second moment)
        m_hat = m / (1 - beta1^t)                   (bias correction)
        v_hat = v / (1 - beta2^t)                   (bias correction)
        theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    
    Args:
        lr: learning rate (typically 0.001)
        beta1: exponential decay rate for first moment (typically 0.9)
        beta2: exponential decay rate for second moment (typically 0.999)
        eps: small constant for numerical stability (typically 1e-8)
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Timestep counter
    
    def step(self, modules):
        """Apply Adam optimization step"""
        self.t += 1

        for key, param, grad in self._iter_keyed_params(modules):
            # Initialize moments if needed
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

            # Compute bias-corrected moments
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameter
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
