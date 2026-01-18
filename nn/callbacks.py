# nn/callbacks.py
import numpy as np


class Callback:
    """
    Base Callback (no-ops by default).

    This file is designed to work with the API:
        Model(..., callbacks=[...])

    You can integrate it later either in:
    - Model.fit(...)
    - a separate trainer.py
    - or a manual training loop

    The contract is:
    - callbacks get a reference to the model via set_model(model)
    - callbacks can receive logs dicts (e.g., {"loss": ..., "val_loss": ...})
    - any callback hook may return True to request stopping training
    """

    def set_model(self, model):
        self.model = model

    def on_train_begin(self, logs=None):
        return False

    def on_train_end(self, logs=None):
        return False

    def on_epoch_begin(self, epoch, logs=None):
        return False

    def on_epoch_end(self, epoch, logs=None):
        return False

    def on_batch_begin(self, batch, logs=None):
        return False

    def on_batch_end(self, batch, logs=None):
        return False


class EarlyStopping(Callback):
    """
    EarlyStopping callback.

    Parameters
    ----------
    monitor : str
        Key in logs to monitor. Common: "val_loss", "loss", "val_accuracy".
    patience : int
        Stop after this many epochs with no improvement.
    min_delta : float
        Minimum change to count as an improvement.
    mode : str
        "min", "max", or "auto".
        - "min": lower is better (loss)
        - "max": higher is better (accuracy)
        - "auto": infer from monitor name ("acc"/"accuracy" -> "max", else "min")
    restore_best_weights : bool
        If True, restore model parameters from the best epoch when stopping.
    verbose : int
        0 = silent, 1 = prints improvements / stopping info.
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

        # state
        self._best = None
        self._best_epoch = None
        self._wait = 0
        self._stop = False
        self._best_weights = None
        self._is_improvement = None  # set at train begin

    def should_stop(self):
        return self._stop

    def on_train_begin(self, logs=None):
        self._stop = False
        self._wait = 0
        self._best_epoch = None
        self._best_weights = None

        mode = self.mode
        if mode == "auto":
            name = self.monitor.lower()
            mode = "max" if ("acc" in name or "accuracy" in name) else "min"

        if mode not in ("min", "max"):
            raise ValueError(f"EarlyStopping: mode must be 'min', 'max', or 'auto', got {self.mode!r}")

        if mode == "min":
            self._best = np.inf
            self._is_improvement = self._is_improvement_min
        else:
            self._best = -np.inf
            self._is_improvement = self._is_improvement_max

        return False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # If the metric isn't present, do nothing (allows flexible loops)
        if self.monitor not in logs:
            return False

        current = float(logs[self.monitor])

        if self._is_improvement(current, self._best):
            self._best = current
            self._best_epoch = int(epoch)
            self._wait = 0

            if self.restore_best_weights:
                self._best_weights = self._snapshot_weights()

            if self.verbose:
                print(f"[EarlyStopping] epoch={epoch} improved {self.monitor} -> {current:.6g}")

            return False

        self._wait += 1
        if self._wait > self.patience:
            self._stop = True

            if self.verbose:
                print(
                    f"[EarlyStopping] stopping at epoch={epoch} "
                    f"(best epoch={self._best_epoch}, best {self.monitor}={self._best:.6g})"
                )

            if self.restore_best_weights and self._best_weights is not None:
                self._restore_weights(self._best_weights)
                if self.verbose:
                    print("[EarlyStopping] restored best weights")

            return True

        return False

    # ----------------- improvement rules -----------------

    def _is_improvement_min(self, current, best):
        return current < (best - self.min_delta)

    def _is_improvement_max(self, current, best):
        return current > (best + self.min_delta)

    # ----------------- snapshot / restore -----------------

    def _iter_params(self):
        """
        Iterate parameter arrays in a stable order.

        We rely on your convention that learnable modules implement:
            params_and_grads() -> yields (param_array, grad_array)
        """
        if self.model is None:
            raise RuntimeError("EarlyStopping: model is not set. Make sure Model sets callbacks via cb.set_model(self).")

        modules = getattr(self.model, "modules", None)
        if modules is None:
            raise RuntimeError("EarlyStopping: model has no attribute 'modules'.")

        for m in modules:
            if hasattr(m, "params_and_grads"):
                for p, _ in m.params_and_grads():
                    yield p

    def _snapshot_weights(self):
        # copy arrays to keep the "best" state immutable
        return [p.copy() for p in self._iter_params()]

    def _restore_weights(self, weights):
        # restore in-place to preserve references (important for momentum/Adam state)
        for p, w in zip(self._iter_params(), weights):
            p[...] = w
