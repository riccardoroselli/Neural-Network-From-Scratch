# nn/core.py

class Module:
    """
    Base class for all neural network components.
    
    Components can be:
        - Parametric (Dense layers with weights/biases)
        - Non-parametric (ReLU, Tanh, etc.)
    
    All components must implement forward() and backward() for gradient flow.
    Parametric components must also implement params_and_grads() to expose
    their learnable parameters for optimization.
    """

    def forward(self, X, training=True):
        """
        Forward pass through the module.
        
        Args:
            X: input data
            training: whether in training mode (affects Dropout, etc.)
        
        Returns:
            output after applying the module transformation
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def backward(self, dY):
        """
        Backward pass through the module.
        
        Args:
            dY: gradient flowing back from the next layer
        
        Returns:
            gradient to pass to the previous layer
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement backward()")

    def params_and_grads(self):
        """
        Generator yielding (parameter, gradient) pairs.
        
        For parametric modules (Dense), yields weight/bias arrays and gradients.
        For non-parametric modules (activations), yields nothing.
        
        Yields:
            tuple: (parameter_array, gradient_array)
        """
        # Default: no parameters (for activations, dropout, etc.)
        return
        yield  # Make it a generator (unreachable but syntactically correct)

    def __repr__(self):
        """String representation of the module"""
        return f"{self.__class__.__name__}()"
