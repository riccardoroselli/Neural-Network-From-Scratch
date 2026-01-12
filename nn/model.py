# nn/model.py
import numpy as np
from .core import Module


class Model:
    """
    Sequential neural network model.
    
    Modules are executed in order during forward pass (left->right)
    and in reverse order during backward pass (right->left).
    """
    
    def __init__(self, modules, loss=None, optimizer=None, regularizer=None, 
                 metrics=None, callbacks=None):
        """
        Args:
            modules: list of Module instances (layers, activations, dropout)
            loss: Loss instance for training
            optimizer: Optimizer instance for parameter updates
            regularizer: Regularizer instance (L1, L2, etc.)
            metrics: list of Metric instances for evaluation
            callbacks: list of Callback instances for training hooks
        """
        self.modules = modules
        
        # Training components
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.metrics = metrics or []
        self.callbacks = callbacks or []
    
    def forward(self, X, training=True):
        """
        Forward pass through all modules.
        
        Args:
            X: input data, shape (N, input_dim)
            training: if True, enables dropout and other training-specific behavior
        
        Returns:
            output after passing through all modules
        """
        for m in self.modules:
            X = m.forward(X, training=training)
        return X
    
    def backward(self, dY):
        """
        Backward pass through all modules in reverse order.
        
        Args:
            dY: gradient from loss function
        """
        for m in reversed(self.modules):
            dY = m.backward(dY)
    
    def params_and_grads(self):
        """
        Generator yielding all (parameter, gradient) pairs from all modules.
        Used by optimizer to update parameters.
        """
        for m in self.modules:
            for p_g in m.params_and_grads():
                yield p_g
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute total loss (base loss + regularization penalty).
        
        Args:
            y_true: ground truth labels
            y_pred: model predictions
        
        Returns:
            scalar loss value
        """
        assert self.loss is not None, "Model.loss is not set"
        
        # Base loss
        base_loss = self.loss.forward(y_pred, y_true)
        
        # Regularization penalty
        reg_penalty = 0.0
        if self.regularizer is not None:
            reg_penalty = self.regularizer.penalty(self.modules)
        
        return float(base_loss + reg_penalty)
    
    def step(self):
        """
        Perform one optimization step:
        1. Add regularization gradients to parameter gradients
        2. Update parameters using optimizer
        """
        assert self.optimizer is not None, "Model.optimizer is not set"
        
        # Add regularization gradients (e.g., L2: dW += lam * W)
        if self.regularizer is not None:
            self.regularizer.add_gradients(self.modules)
        
        # Update parameters
        self.optimizer.step(self.modules)
    
    def predict_proba(self, X):
        """
        Predict probabilities (inference mode, no dropout).
        
        Args:
            X: input data
        
        Returns:
            predicted probabilities/values
        """
        return self.forward(X, training=False)
    
    def predict(self, X, threshold=0.5):
        """
        Predict binary classes using threshold.
        
        Args:
            X: input data
            threshold: decision threshold (default 0.5)
        
        Returns:
            binary predictions (0 or 1)
        """
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)
    
    def __repr__(self):
        """String representation of the model"""
        lines = ["Model("]
        for i, m in enumerate(self.modules):
            lines.append(f"  ({i}) {m.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)
