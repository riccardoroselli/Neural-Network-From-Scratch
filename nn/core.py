# nn/core.py
import numpy as np

class Module:
    """
    Base class for all network components (layers, activations, dropout, etc.).
    
    Components may have learnable parameters (Dense) or be stateless (ReLU).
    All components must implement forward/backward for gradient flow.
    """
    
    def forward(self, X, training=True):
        """
        Forward pass through the module.
        
        Args:
            X: input data
            training: whether in training mode (affects Dropout, BatchNorm, etc.)
        
        Returns:
            output after applying the module transformation
        """
        raise NotImplementedError
    
    def backward(self, dY):
        """
        Backward pass through the module.
        
        Args:
            dY: gradient flowing back from the next layer
        
        Returns:
            gradient to pass to the previous layer
        """
        raise NotImplementedError
    
    def params_and_grads(self):
        """
        Yields (param, grad) pairs for learnable parameters.
        Default: empty (for stateless modules like activations).
        
        Yields:
            tuples of (parameter_array, gradient_array)
        """
        return []
