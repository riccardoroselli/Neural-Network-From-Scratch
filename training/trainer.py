import numpy as np
from .dataloader import BatchIterator
from .history import History
import sys


class Trainer:
    """Mini-batch training loop for nn.model.Model with metrics, callbacks, and history logging."""

    def __init__(self, model, verbose=1):
        self.model = model
        self.verbose = int(verbose)
        self._ascii_banner_printed = False

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
        """Train the model."""
        model = self.model
        if model.loss is None:
            raise ValueError("Model.loss is not set")
        if model.optimizer is None:
            raise ValueError("Model.optimizer is not set")

        callbacks = getattr(model, "callbacks", []) or []
        for cb in callbacks:
            if hasattr(cb, "set_model"):
                cb.set_model(model)

        history = History()

        stop = False
        for cb in callbacks:
            if hasattr(cb, "on_train_begin"):
                stop = bool(cb.on_train_begin(logs=None)) or stop
        if stop:
            return history

        if self.verbose >= 1 and not self._ascii_banner_printed:
            self._print_banner()
            self._ascii_banner_printed = True

        epochs = int(epochs)

        for epoch in range(1, epochs + 1):
            for cb in callbacks:
                if hasattr(cb, "on_epoch_begin"):
                    cb.on_epoch_begin(epoch, logs=None)

            train_logs = self._run_epoch_train(
                X_train,
                y_train,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=None if seed is None else int(seed) + epoch,
            )

            logs = dict(train_logs)

            if X_val is not None and y_val is not None:
                val_logs = self.evaluate(
                    X_val,
                    y_val,
                    batch_size=max(int(batch_size), 1),
                    include_regularization=include_reg_in_val,
                )
                for k, v in val_logs.items():
                    logs[f"val_{k}"] = v

            history.log(**logs)

            if self.verbose:
                self._print_progress(epoch, epochs, logs)

            stop = False
            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    stop = bool(cb.on_epoch_end(epoch, logs=logs)) or stop
                if hasattr(cb, "should_stop") and callable(cb.should_stop):
                    stop = bool(cb.should_stop()) or stop

            if stop:
                break

        if self.verbose >= 1:
            print()

        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(logs=history.to_dict())

        return history

    def evaluate(self, X, y, batch_size=256, include_regularization=False):
        """Evaluate loss + metrics on a dataset (no gradients, training=False)."""
        model = self.model
        iterator = BatchIterator(X, y, batch_size=batch_size, shuffle=False)

        n_total = 0
        loss_sum = 0.0

        metrics = model.metrics or []
        metric_sums = {m.__class__.__name__: 0.0 for m in metrics}

        for Xb, yb in iterator:
            y_pred = model.forward(Xb, training=False)

            base_loss = float(model.loss.forward(y_pred, yb))
            reg = 0.0
            if include_regularization and model.regularizer is not None:
                reg = float(model.regularizer.penalty(model.modules))
            batch_loss = base_loss + reg

            bs = len(Xb)
            n_total += bs
            loss_sum += batch_loss * bs

            for m in metrics:
                name = m.__class__.__name__
                metric_sums[name] += float(m(y_pred, yb)) * bs

        out = {"loss": loss_sum / max(n_total, 1)}
        for name, s in metric_sums.items():
            out[name] = s / max(n_total, 1)
        return out

    def _run_epoch_train(self, X, y, batch_size, shuffle, drop_last, seed):
        model = self.model
        callbacks = getattr(model, "callbacks", []) or []
        metrics = model.metrics or []

        iterator = BatchIterator(
            X, y,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed
        )

        n_total = 0
        loss_sum = 0.0
        metric_sums = {m.__class__.__name__: 0.0 for m in metrics}

        for batch_idx, (Xb, yb) in enumerate(iterator):
            for cb in callbacks:
                if hasattr(cb, "on_batch_begin"):
                    cb.on_batch_begin(batch_idx, logs=None)

            y_pred = model.forward(Xb, training=True)
            loss = float(model.compute_loss(y_true=yb, y_pred=y_pred))

            dY = model.loss.backward(y_pred, yb)
            model.backward(dY)
            model.step()

            bs = len(Xb)
            n_total += bs
            loss_sum += loss * bs

            for m in metrics:
                name = m.__class__.__name__
                metric_sums[name] += float(m(y_pred, yb)) * bs

            for cb in callbacks:
                if hasattr(cb, "on_batch_end"):
                    cb.on_batch_end(batch_idx, logs={"loss": loss})

        out = {"loss": loss_sum / max(n_total, 1)}
        for name, s in metric_sums.items():
            out[name] = s / max(n_total, 1)
        return out

    def _print_banner(self):
        """Print ASCII art banner"""
        banner = r"""
      _____     ___                 _ _ _ 
      \_   \   / __\__ ___   ____ _| | (_)
       / /\/  / /  / _` \ \ / / _` | | | |
    /\/ /_   / /__| (_| |\ V / (_| | | | |
    \____/   \____/\__,_| \_/ \__,_|_|_|_|
    """
        print(banner)

    def _print_progress(self, current_epoch, total_epochs, logs):
        """Print progress bar with metrics"""
        progress = current_epoch / total_epochs
        bar_length = 30
        filled_length = int(bar_length * progress)

        bar = '=' * filled_length + '>' + '.' * (bar_length - filled_length - 1)
        metrics_str = ' '.join([f'{k}: {v:.4f}' for k, v in logs.items()])

        sys.stdout.write(f'Epoch {current_epoch}/{total_epochs} [{bar}] {metrics_str}')
        sys.stdout.flush()