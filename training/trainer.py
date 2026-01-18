import numpy as np
from .dataloader import BatchIterator
from .history import History


class Trainer:
    """
    Mini-batch training loop for your `nn.model.Model`.

    Supports:
    - shuffling, mini-batches
    - train/eval forward modes via Model.forward(training=...)
    - metrics (Metric.__call__)
    - callbacks (including EarlyStopping)
    - history logging

    Notes for your codebase:
    - Loss.backward requires (y_pred, y_true) arguments. :contentReference[oaicite:5]{index=5}
    - Metrics implement compute() and __call__(). :contentReference[oaicite:6]{index=6}
    """

    def __init__(self, model, verbose=1):
        self.model = model
        self.verbose = int(verbose)

    # ----------------------------- public API -----------------------------

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
        Train the model.

        Parameters
        ----------
        X_train, y_train : np.ndarray
        X_val, y_val : np.ndarray | None
        epochs : int
        batch_size : int
        shuffle : bool
        drop_last : bool
        seed : int | None
            Base seed for shuffling each epoch deterministically.
        include_reg_in_val : bool
            If True, validation loss includes regularization penalty.

        Returns
        -------
        History
        """
        model = self.model
        if model.loss is None:
            raise ValueError("Model.loss is not set")
        if model.optimizer is None:
            raise ValueError("Model.optimizer is not set")

        # Bind callbacks to model
        callbacks = getattr(model, "callbacks", []) or []
        for cb in callbacks:
            if hasattr(cb, "set_model"):
                cb.set_model(model)

        history = History()

        # on_train_begin
        stop = False
        for cb in callbacks:
            if hasattr(cb, "on_train_begin"):
                stop = bool(cb.on_train_begin(logs=None)) or stop
        if stop:
            return history

        epochs = int(epochs)

        for epoch in range(1, epochs + 1):
            # on_epoch_begin
            for cb in callbacks:
                if hasattr(cb, "on_epoch_begin"):
                    cb.on_epoch_begin(epoch, logs=None)

            # Train epoch
            train_logs = self._run_epoch_train(
                X_train,
                y_train,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                seed=None if seed is None else int(seed) + epoch,
            )

            logs = dict(train_logs)

            # Validation epoch
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
                self._print_epoch(epoch, logs)

            # on_epoch_end + early stop
            stop = False
            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    stop = bool(cb.on_epoch_end(epoch, logs=logs)) or stop
                if hasattr(cb, "should_stop") and callable(cb.should_stop):
                    stop = bool(cb.should_stop()) or stop

            if stop:
                break

        # on_train_end
        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(logs=history.to_dict())

        return history

    def evaluate(self, X, y, batch_size=256, include_regularization=False):
        """
        Evaluate loss + metrics on a dataset (no gradients, training=False).

        Returns
        -------
        dict: {"loss": ..., "Accuracy": ..., ...}
        """
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

    # ----------------------------- internal helpers -----------------------------

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
            # on_batch_begin
            for cb in callbacks:
                if hasattr(cb, "on_batch_begin"):
                    cb.on_batch_begin(batch_idx, logs=None)

            # forward (training=True enables Dropout behavior) :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}
            y_pred = model.forward(Xb, training=True)

            # loss (+ regularization penalty via Model.compute_loss) :contentReference[oaicite:9]{index=9}
            loss = float(model.compute_loss(y_true=yb, y_pred=y_pred))

            # backward
            dY = model.loss.backward(y_pred, yb)  # IMPORTANT: your losses require args :contentReference[oaicite:10]{index=10}
            model.backward(dY)

            # step (regularizer grads + optimizer update inside Model.step) :contentReference[oaicite:11]{index=11}
            model.step()

            bs = len(Xb)
            n_total += bs
            loss_sum += loss * bs

            for m in metrics:
                name = m.__class__.__name__
                metric_sums[name] += float(m(y_pred, yb)) * bs

            # on_batch_end
            for cb in callbacks:
                if hasattr(cb, "on_batch_end"):
                    cb.on_batch_end(batch_idx, logs={"loss": loss})

        out = {"loss": loss_sum / max(n_total, 1)}
        for name, s in metric_sums.items():
            out[name] = s / max(n_total, 1)
        return out

    def _print_epoch(self, epoch, logs):
        parts = [f"epoch {epoch:03d}"]
        if "loss" in logs:
            parts.append(f"loss={float(logs['loss']):.6g}")

        # print other keys (metrics + val_*)
        for k, v in logs.items():
            if k == "loss":
                continue
            if isinstance(v, (float, int, np.floating, np.integer)):
                parts.append(f"{k}={float(v):.6g}")

        print(" - ".join(parts))
