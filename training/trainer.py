# training/trainer.py
import sys

from training.dataloader import BatchIterator
from training.history import History


class Trainer:
    """
    Training orchestrator for neural network models.
    
    Handles mini-batch training loop with validation, metrics tracking,
    callbacks (early stopping, etc.), and progress visualization.
    
    Example:
        >>> model = Model(...)
        >>> trainer = Trainer(model, verbose=1)
        >>> history = trainer.fit(X_train, y_train, 
        ...                       X_val=X_val, y_val=y_val,
        ...                       epochs=100, batch_size=32)
    """

    def __init__(self, model, verbose=1):
        """
        Args:
            model: Model instance to train
            verbose: 0 = silent, >= 1 = banner and per-epoch progress bar
        """
        self.model = model
        self.verbose = int(verbose)
        self._banner_printed = False

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs=100,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        seed=None,
        include_reg_in_val=False,
    ):
        """
        Train the model using mini-batch gradient descent.
        
        Args:
            X_train: training input data, shape (N, D)
            y_train: training targets, shape (N, ...) 
            X_val: optional validation input data
            y_val: optional validation targets
            epochs: number of training epochs
            batch_size: mini-batch size
            shuffle: whether to shuffle training data each epoch
            drop_last: whether to drop incomplete final batch
            seed: random seed for reproducibility
            include_reg_in_val: whether to include regularization in validation loss
        
        Returns:
            History object containing per-epoch metrics
        """
        # Validate model configuration
        if self.model.loss is None:
            raise ValueError("Model.loss is not set. Cannot train without loss function.")
        if self.model.optimizer is None:
            raise ValueError("Model.optimizer is not set. Cannot train without optimizer.")
        
        # Setup callbacks
        callbacks = getattr(self.model, "callbacks", []) or []
        for callback in callbacks:
            if hasattr(callback, "set_model"):
                callback.set_model(self.model)
        
        # Initialize history
        history = History()
        
        # Trigger on_train_begin
        if self._trigger_callbacks(callbacks, "on_train_begin"):
            return history  # Early stop requested
        
        # Print banner once
        if self.verbose >= 1 and not self._banner_printed:
            self._print_banner()
            self._banner_printed = True
        
        # Training loop
        for epoch in range(epochs):
            # Trigger on_epoch_begin
            self._trigger_callbacks(callbacks, "on_epoch_begin", epoch=epoch)
            
            # Train for one epoch
            train_metrics = self._train_epoch(
                X_train,
                y_train,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=None if seed is None else seed + epoch + 1,
            )
            
            # Collect metrics
            epoch_logs = dict(train_metrics)
            
            # Validation (if provided)
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(
                    X_val,
                    y_val,
                    batch_size=batch_size,
                    include_regularization=include_reg_in_val,
                )
                # Prefix with 'val_'
                for key, value in val_metrics.items():
                    epoch_logs[f"val_{key}"] = value
            
            # Log metrics
            history.log(**epoch_logs)
            
            # Print progress
            if self.verbose >= 1:
                self._print_progress(epoch + 1, epochs, epoch_logs)
            
            # Trigger on_epoch_end (check for early stopping)
            if self._trigger_callbacks(callbacks, "on_epoch_end", epoch=epoch, logs=epoch_logs):
                break  # Stop training
        
        # Final newline after progress bar
        if self.verbose >= 1:
            print()
        
        # Trigger on_train_end
        self._trigger_callbacks(callbacks, "on_train_end", logs=history.to_dict())
        
        return history

    def evaluate(self, X, y, batch_size=256, include_regularization=False):
        """
        Evaluate model on a dataset without training.
        
        Args:
            X: input data
            y: target data
            batch_size: batch size for evaluation
            include_regularization: whether to include regularization penalty in loss
        
        Returns:
            dict of metrics: {'loss': ..., 'Accuracy': ..., etc.}
        """
        iterator = BatchIterator(X, y, batch_size=batch_size, shuffle=False)
        
        metrics = self.model.metrics or []
        
        # Accumulators
        total_samples = 0
        loss_sum = 0.0
        metric_sums = {metric.__class__.__name__: 0.0 for metric in metrics}
        
        # Iterate over batches
        for X_batch, y_batch in iterator:
            # Forward pass (inference mode)
            y_pred = self.model.forward(X_batch, training=False)
            
            # Compute loss
            base_loss = float(self.model.loss.forward(y_pred, y_batch))
            reg_penalty = 0.0
            
            if include_regularization and self.model.regularizer is not None:
                reg_penalty = float(self.model.regularizer.penalty(self.model.modules))
            
            batch_loss = base_loss + reg_penalty
            
            # Accumulate
            batch_size_actual = len(X_batch)
            total_samples += batch_size_actual
            loss_sum += batch_loss * batch_size_actual
            
            # Compute metrics
            for metric in metrics:
                metric_name = metric.__class__.__name__
                metric_value = float(metric(y_pred, y_batch))
                metric_sums[metric_name] += metric_value * batch_size_actual
        
        # Average over all samples
        results = {"loss": loss_sum / max(total_samples, 1)}
        for metric_name, metric_sum in metric_sums.items():
            results[metric_name] = metric_sum / max(total_samples, 1)
        
        return results

    def _train_epoch(self, X, y, batch_size, shuffle, drop_last, seed):
        """
        Train for a single epoch.
        
        Returns:
            dict of training metrics
        """
        callbacks = getattr(self.model, "callbacks", []) or []
        metrics = self.model.metrics or []
        
        iterator = BatchIterator(
            X, y,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
        )
        
        # Accumulators
        total_samples = 0
        loss_sum = 0.0
        metric_sums = {metric.__class__.__name__: 0.0 for metric in metrics}
        
        # Train over batches
        for batch_idx, (X_batch, y_batch) in enumerate(iterator):
            # Trigger on_batch_begin
            self._trigger_callbacks(callbacks, "on_batch_begin", batch=batch_idx)
            
            # Forward pass
            y_pred = self.model.forward(X_batch, training=True)
            
            # Compute loss (with regularization)
            loss = float(self.model.compute_loss(y_true=y_batch, y_pred=y_pred))
            
            # Backward pass
            dY = self.model.loss.backward(y_pred, y_batch)
            self.model.backward(dY)
            
            # Update parameters
            self.model.step()
            
            # Accumulate metrics
            batch_size_actual = len(X_batch)
            total_samples += batch_size_actual
            loss_sum += loss * batch_size_actual
            
            for metric in metrics:
                metric_name = metric.__class__.__name__
                metric_value = float(metric(y_pred, y_batch))
                metric_sums[metric_name] += metric_value * batch_size_actual
            
            # Trigger on_batch_end
            self._trigger_callbacks(callbacks, "on_batch_end", batch=batch_idx, logs={"loss": loss})
        
        # Average over all samples
        results = {"loss": loss_sum / max(total_samples, 1)}
        for metric_name, metric_sum in metric_sums.items():
            results[metric_name] = metric_sum / max(total_samples, 1)
        
        return results

    def _trigger_callbacks(self, callbacks, hook_name, **kwargs):
        """
        Trigger a callback hook on all callbacks.
        
        Args:
            callbacks: list of callback objects
            hook_name: name of hook method (e.g., 'on_epoch_end')
            **kwargs: arguments to pass to hook
        
        Returns:
            bool: True if any callback requests stopping
        """
        stop_requested = False
        
        for callback in callbacks:
            if hasattr(callback, hook_name):
                hook = getattr(callback, hook_name)
                result = hook(**kwargs)
                if result:
                    stop_requested = True
        
        return stop_requested

    def _print_banner(self):
        """Print ASCII art banner for I Cavalli team"""
        banner = r"""
      _____     ___                 _ _ _ 
      \_   \   / __\__ ___   ____ _| | (_)
       / /\/  / /  / _` \ \ / / _` | | | |
    /\/ /_   / /__| (_| |\ V / (_| | | | |
    \____/   \____/\__,_| \_/ \__,_|_|_|_|
        """
        print(banner)

    def _print_progress(self, current_epoch, total_epochs, logs):
        """
        Print progress bar with metrics (overwrites previous line).
        
        Args:
            current_epoch: current epoch number (1-indexed for display)
            total_epochs: total number of epochs
            logs: dict of metrics to display
        """
        progress = current_epoch / total_epochs
        bar_length = 30
        filled = int(bar_length * progress)
        
        # Build progress bar
        bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
        
        # Format metrics
        metrics_str = ' '.join([f'{key}: {value:.4f}' for key, value in logs.items()])
        
        # Print with carriage return to overwrite
        sys.stdout.write(f'\rEpoch {current_epoch}/{total_epochs} [{bar}] {metrics_str}')
        sys.stdout.flush()
