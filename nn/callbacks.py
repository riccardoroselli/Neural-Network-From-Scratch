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
        self._best_state = None
        self._is_improvement = None  # set at train begin

    def should_stop(self):
        return self._stop

    def on_train_begin(self, logs=None):
        self._stop = False
        self._wait = 0
        self._best_epoch = None
        self._best_state = None

        mode = self.mode
        if mode == "auto":
            name = self.monitor.lower()
            mode = "max" if ("acc" in name or "accuracy" in name) else "min"

        if mode not in ("min", "max"):
            raise ValueError(
                f"EarlyStopping: mode must be 'min', 'max', or 'auto', got {self.mode!r}"
            )

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
                self._best_state = self._snapshot_state()

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

            if self.restore_best_weights and self._best_state is not None:
                self._restore_state(self._best_state)
                if self.verbose:
                    print("[EarlyStopping] restored best weights")

            return True

        return False

    # ----------------- improvement rules -----------------

    def _is_improvement_min(self, current, best):
        return current < (best - self.min_delta)

    def _is_improvement_max(self, current, best):
        return current > (best + self.min_delta)

    # ----------------- state snapshot / restore -----------------

    def _snapshot_state(self):
        """
        Snapshot model state.

        Prefer Model.get_state() if available (cleaner, more robust).
        Fallback: snapshot via params_and_grads iteration.
        """
        if self.model is None:
            raise RuntimeError("EarlyStopping: model is not set. Did you call cb.set_model(model)?")

        if hasattr(self.model, "get_state") and callable(self.model.get_state):
            return self.model.get_state()

        # fallback (older approach)
        return [p.copy() for p in self._iter_params_fallback()]

    def _restore_state(self, state):
        """
        Restore model state.

        Prefer Model.set_state() if available.
        Fallback: restore via params iteration (in-place).
        """
        if self.model is None:
            raise RuntimeError("EarlyStopping: model is not set. Did you call cb.set_model(model)?")

        if hasattr(self.model, "set_state") and callable(self.model.set_state):
            self.model.set_state(state)
            return

        for p, saved in zip(self._iter_params_fallback(), state):
            p[...] = saved

    def _iter_params_fallback(self):
        """
        Iterate parameter arrays in a stable order using params_and_grads().
        """
        modules = getattr(self.model, "modules", None)
        if modules is None:
            raise RuntimeError("EarlyStopping: model has no attribute 'modules'.")

        for m in modules:
            if hasattr(m, "params_and_grads"):
                for p, _ in m.params_and_grads():
                    yield p
