# nn/callbacks.py
import numpy as np


class Callback:
    """
    Base callback class with no-op hooks.
    
    Callbacks receive references to the model and logs during training.
    Any hook can return True to request training to stop.
    
    Integration points:
        - Model.fit(...)
        - Trainer.fit(...)
        - Custom training loops
    
    Callback lifecycle:
        1. set_model(model) - called once at setup
        2. on_train_begin(logs) - called at training start
        3. For each epoch:
            - on_epoch_begin(epoch, logs)
            - For each batch:
                - on_batch_begin(batch, logs)
                - on_batch_end(batch, logs)
            - on_epoch_end(epoch, logs)
        4. on_train_end(logs) - called at training end
    """

    def set_model(self, model):
        """Store reference to the model being trained"""
        self.model = model

    def on_train_begin(self, logs=None):
        """Called at the start of training"""
        return False

    def on_train_end(self, logs=None):
        """Called at the end of training"""
        return False

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of each epoch"""
        return False

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        return False

    def on_batch_begin(self, batch, logs=None):
        """Called at the start of each batch"""
        return False

    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch"""
        return False


class EarlyStopping(Callback):
    """
    Early stopping callback to halt training when metric stops improving.
    
    Monitors a metric (e.g., validation loss) and stops training after
    a specified number of epochs with no improvement. Optionally restores
    model weights from the best epoch.
    
    Args:
        monitor: metric name to track (e.g., 'val_loss', 'val_Accuracy')
        patience: number of epochs with no improvement before stopping
        min_delta: minimum change to qualify as improvement
        mode: 'min', 'max', or 'auto'
            - 'min': lower is better (for loss)
            - 'max': higher is better (for accuracy)
            - 'auto': infer from monitor name
        restore_best_weights: if True, restore model to best epoch when stopping
        verbose: 0 = silent, 1 = print improvements and stopping
    
    Example:
        >>> callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        >>> model = Model(..., callbacks=[callback])
        >>> trainer.fit(X, y, X_val=X_val, y_val=y_val, epochs=100)
        # Training stops early if val_loss doesn't improve for 10 epochs
    """

    def __init__(
        self,
        monitor="val_loss",
        patience=10,
        min_delta=0.0,
        mode="auto",
        restore_best_weights=True,
        verbose=0,
    ):
        self.monitor = str(monitor)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = str(mode)
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = int(verbose)
        
        self.model = None
        
        # Training state (reset at train begin)
        self.best_value = None
        self.best_epoch = None
        self.wait = 0
        self.stopped_epoch = None
        self.best_state = None
        self._is_improvement = None  # Function pointer set at train begin

    def should_stop(self):
        """Check if training should stop"""
        return self.stopped_epoch is not None

    def on_train_begin(self, logs=None):
        """Initialize state at training start"""
        self.wait = 0
        self.best_epoch = None
        self.best_state = None
        self.stopped_epoch = None
        
        # Determine comparison mode
        mode = self.mode
        if mode == "auto":
            metric_name = self.monitor.lower()
            mode = "max" if ("acc" in metric_name or "accuracy" in metric_name) else "min"
        
        if mode not in ("min", "max"):
            raise ValueError(
                f"EarlyStopping mode must be 'min', 'max', or 'auto', got {self.mode!r}"
            )
        
        # Set initial best value and comparison function
        if mode == "min":
            self.best_value = np.inf
            self._is_improvement = self._is_improvement_min
        else:
            self.best_value = -np.inf
            self._is_improvement = self._is_improvement_max
        
        return False

    def on_epoch_end(self, epoch, logs=None):
        """Check for improvement and potentially stop training"""
        logs = logs or {}
        
        # Skip if metric not available
        if self.monitor not in logs:
            return False
        
        current_value = float(logs[self.monitor])
        
        # Check for improvement
        if self._is_improvement(current_value, self.best_value):
            self.best_value = current_value
            self.best_epoch = int(epoch)
            self.wait = 0
            
            # Save best state if requested
            if self.restore_best_weights:
                self.best_state = self._snapshot_state()
            
            if self.verbose:
                print(
                    f"[EarlyStopping] Epoch {epoch}: {self.monitor} improved to {current_value:.6g}"
                )
            
            return False
        
        # No improvement - increment wait counter
        self.wait += 1
        
        # Check if patience exhausted
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            
            if self.verbose:
                print(
                    f"[EarlyStopping] Stopping at epoch {epoch} "
                    f"(best epoch={self.best_epoch}, best {self.monitor}={self.best_value:.6g})"
                )
            
            # Restore best weights if requested
            if self.restore_best_weights and self.best_state is not None:
                self._restore_state(self.best_state)
                if self.verbose:
                    print("[EarlyStopping] Restored weights from best epoch")
            
            return True  # Signal to stop training
        
        return False

    # ==================== Improvement Checks ====================

    def _is_improvement_min(self, current, best):
        """Check improvement for metrics where lower is better"""
        return current < (best - self.min_delta)

    def _is_improvement_max(self, current, best):
        """Check improvement for metrics where higher is better"""
        return current > (best + self.min_delta)

    # ==================== State Management ====================

    def _snapshot_state(self):
        """
        Save current model state.
        
        Prefers Model.get_state() if available (includes optimizer state).
        Falls back to copying parameters directly for older implementations.
        """
        if self.model is None:
            raise RuntimeError(
                "EarlyStopping: model not set. Call set_model(model) first."
            )
        
        # Preferred: use Model.get_state() (includes optimizer state)
        if hasattr(self.model, "get_state") and callable(self.model.get_state):
            return self.model.get_state()
        
        # Fallback: copy parameters only (for backward compatibility)
        return [param.copy() for param in self._iter_params_fallback()]

    def _restore_state(self, state):
        """
        Restore model to saved state.
        
        Prefers Model.set_state() if available (restores optimizer too).
        Falls back to in-place parameter restoration for older implementations.
        """
        if self.model is None:
            raise RuntimeError(
                "EarlyStopping: model not set. Call set_model(model) first."
            )
        
        # Preferred: use Model.set_state() (restores optimizer too)
        if hasattr(self.model, "set_state") and callable(self.model.set_state):
            self.model.set_state(state)
            return
        
        # Fallback: restore parameters in-place
        for param, saved_param in zip(self._iter_params_fallback(), state):
            param[...] = saved_param

    def _iter_params_fallback(self):
        """
        Iterate over model parameters (fallback for older implementations).
        
        Used when Model.get_state() / set_state() are not available.
        """
        modules = getattr(self.model, "modules", None)
        if modules is None:
            raise RuntimeError(
                "EarlyStopping: model has no 'modules' attribute."
            )
        
        for module in modules:
            if hasattr(module, "params_and_grads"):
                for param, _ in module.params_and_grads():
                    yield param
