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
        self.loss = loss
        self.optimizer = optimizer
        self.regularizer = regularizer
        self.metrics = metrics or []
        self.callbacks = callbacks or []

    # ==================== Forward/Backward Pass ====================

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

    # ==================== Parameter Management ====================

    def params_and_grads(self):
        """Generator yielding all (parameter, gradient) pairs from all modules."""
        for m in self.modules:
            for p_g in m.params_and_grads():
                yield p_g

    def parameters(self):
        """Generator yielding all parameter arrays in stable order."""
        for m in self.modules:
            if hasattr(m, "params_and_grads"):
                for p, _ in m.params_and_grads():
                    yield p

    # ==================== State Management ====================

    def get_state(self):
        """
        Snapshot model parameters and optimizer state.
        
        Returns:
            dict: {'params': list of arrays, 'optimizer': dict or None}
        """
        # Save parameters
        param_state = [p.copy() for p in self.parameters()]
        
        # Save optimizer state
        opt_state = None
        if self.optimizer is not None:
            opt_state = {}
            
            # SGDMomentum state
            if hasattr(self.optimizer, 'velocities'):
                opt_state['velocities'] = {
                    k: v.copy() for k, v in self.optimizer.velocities.items()
                }
            
            # Adam state
            if hasattr(self.optimizer, 'm'):
                opt_state['m'] = {k: v.copy() for k, v in self.optimizer.m.items()}
                opt_state['v'] = {k: v.copy() for k, v in self.optimizer.v.items()}
                opt_state['t'] = self.optimizer.t
        
        return {'params': param_state, 'optimizer': opt_state}

    def set_state(self, state):
        """
        Restore model parameters and optimizer state.
        
        Args:
            state: dict from get_state()
        """
        param_state = state['params']
        opt_state = state.get('optimizer')
        
        # Restore parameters (in-place to keep same memory references)
        params = list(self.parameters())
        
        if len(param_state) != len(params):
            raise ValueError(
                f"State mismatch: expected {len(params)} parameters, got {len(param_state)}"
            )
        
        for p, saved in zip(params, param_state):
            if p.shape != saved.shape:
                raise ValueError(
                    f"Shape mismatch: parameter {p.shape} vs saved {saved.shape}"
                )
            # Copy values in-place to preserve memory references
            np.copyto(p, saved)
        
        # Restore optimizer state
        if opt_state is not None and self.optimizer is not None:
            # SGDMomentum
            if 'velocities' in opt_state:
                self.optimizer.velocities = {
                    k: v.copy() for k, v in opt_state['velocities'].items()
                }
            
            # Adam
            if 'm' in opt_state:
                self.optimizer.m = {k: v.copy() for k, v in opt_state['m'].items()}
                self.optimizer.v = {k: v.copy() for k, v in opt_state['v'].items()}
                self.optimizer.t = opt_state['t']

    def reset(self):
        """
        Reset model to initial state:
        - Reinitialize all parameters (weights, biases)
        - Clear optimizer state (momentum, Adam stats)
        - Reset callbacks
        """
        # Reset all modules
        for module in self.modules:
            if hasattr(module, 'reset'):
                module.reset()
        
        # Reset optimizer state
        if self.optimizer is not None:
            # SGDMomentum
            if hasattr(self.optimizer, 'velocities'):
                self.optimizer.velocities = {}
            
            # Adam
            if hasattr(self.optimizer, 'm'):
                self.optimizer.m = {}
                self.optimizer.v = {}
                self.optimizer.t = 0
        
        # Reset callbacks
        for cb in self.callbacks:
            if hasattr(cb, 'reset'):
                cb.reset()

    # ==================== Training ====================

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
        1. Add regularization gradients
        2. Update parameters
        """
        assert self.optimizer is not None, "Model.optimizer is not set"
        
        # Add regularization gradients (e.g., L2: dW += lam * W)
        if self.regularizer is not None:
            self.regularizer.add_gradients(self.modules)
        
        # Update parameters
        self.optimizer.step(self.modules)

    # ==================== Inference ====================

    def predict_proba(self, X):
        """Predict probabilities (inference mode, no dropout)."""
        return self.forward(X, training=False)

    def predict(self, X, threshold=0.5):
        """Predict binary classes using threshold."""
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)

    # ==================== Representation ====================

    def __repr__(self):
        """String representation of model architecture."""
        lines = ["Model("]
        for i, m in enumerate(self.modules):
            lines.append(f"  ({i}) {m.__class__.__name__}")
        lines.append(")")
        return "\n".join(lines)
