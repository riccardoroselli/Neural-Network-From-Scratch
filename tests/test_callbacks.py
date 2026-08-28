"""
EarlyStopping behaviour: improvement tracking, stopping, and weight restoration.

    python -m unittest tests.test_callbacks
"""
import unittest

import numpy as np

from nn.activations import Sigmoid, Tanh
from nn.callbacks import Callback, EarlyStopping
from nn.layers import Dense
from nn.losses import BinaryCrossEntropy
from nn.model import Model
from nn.optim import Adam
from training.trainer import Trainer


def overfitting_problem():
    """Small noisy train set against a clean val set: val loss dips then climbs."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 8))
    y = (X[:, 0] + 0.4 * rng.standard_normal(40) > 0).astype(float).reshape(-1, 1)
    X_val = rng.standard_normal((200, 8))
    y_val = (X_val[:, 0] > 0).astype(float).reshape(-1, 1)
    return X, y, X_val, y_val


def build(callback, hidden=(32, 32), lr=0.02):
    modules, prev = [], 8
    for h in hidden:
        modules += [Dense(prev, h, seed=len(modules) + 1), Tanh()]
        prev = h
    modules += [Dense(prev, 1, seed=99), Sigmoid()]
    return Model(modules, loss=BinaryCrossEntropy(), optimizer=Adam(lr=lr),
                 callbacks=[callback])


class TestEarlyStoppingRestoration(unittest.TestCase):

    def test_restores_best_weights_when_the_epoch_limit_is_reached(self):
        """
        The regression this guards: patience never runs out, so on_epoch_end
        never restores, and the model used to be left on its final (much worse)
        weights even though the best state had been captured.
        """
        X, y, X_val, y_val = overfitting_problem()
        es = EarlyStopping(monitor="val_loss", patience=10_000,
                           restore_best_weights=True)
        trainer = Trainer(build(es), verbose=0)
        history = trainer.fit(X, y, X_val=X_val, y_val=y_val,
                              epochs=300, batch_size=8, seed=0)

        val_loss = np.array(history.get("val_loss"))
        best = float(val_loss.min())

        self.assertFalse(es.should_stop())              # patience never fired
        self.assertGreater(val_loss[-1], best * 1.5)    # the run really did degrade
        self.assertAlmostEqual(trainer.evaluate(X_val, y_val)["loss"], best, places=10)

    def test_restores_best_weights_when_patience_is_exhausted(self):
        X, y, X_val, y_val = overfitting_problem()
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        trainer = Trainer(build(es, hidden=(8,), lr=0.05), verbose=0)
        history = trainer.fit(X, y, X_val=X_val, y_val=y_val,
                              epochs=400, batch_size=8, seed=0)

        best = float(np.min(history.get("val_loss")))
        self.assertTrue(es.should_stop())
        self.assertAlmostEqual(trainer.evaluate(X_val, y_val)["loss"], best, places=10)

    def test_restores_exactly_once(self):
        X, y, X_val, y_val = overfitting_problem()
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        trainer = Trainer(build(es, hidden=(8,), lr=0.05), verbose=0)
        trainer.fit(X, y, X_val=X_val, y_val=y_val, epochs=400, batch_size=8, seed=0)

        restored_at_stop = es._restored
        weights = [p.copy() for p in trainer.model.parameters()]
        es.on_train_end()  # a second call must be a no-op
        for before, after in zip(weights, trainer.model.parameters()):
            np.testing.assert_array_equal(before, after)
        self.assertTrue(restored_at_stop)

    def test_does_not_restore_when_disabled(self):
        X, y, X_val, y_val = overfitting_problem()
        es = EarlyStopping(monitor="val_loss", patience=10_000,
                           restore_best_weights=False)
        trainer = Trainer(build(es, hidden=(32,), lr=0.05), verbose=0)
        history = trainer.fit(X, y, X_val=X_val, y_val=y_val,
                              epochs=200, batch_size=8, seed=0)

        final = float(history.get("val_loss")[-1])
        self.assertAlmostEqual(trainer.evaluate(X_val, y_val)["loss"], final, places=10)

    def test_no_validation_set_is_a_no_op(self):
        X, y, _, _ = overfitting_problem()
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = Trainer(build(es, hidden=(4,), lr=0.05), verbose=0).fit(
            X, y, epochs=20, batch_size=8, seed=0)

        self.assertEqual(len(history.get("loss")), 20)
        self.assertIsNone(es.best_state)
        self.assertFalse(es.should_stop())

    def test_state_is_reset_between_consecutive_fits(self):
        X, y, X_val, y_val = overfitting_problem()
        es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        trainer = Trainer(build(es, hidden=(8,), lr=0.05), verbose=0)
        trainer.fit(X, y, X_val=X_val, y_val=y_val, epochs=100, batch_size=8, seed=0)
        self.assertTrue(es._restored)

        trainer.model.reset()
        trainer.fit(X, y, X_val=X_val, y_val=y_val, epochs=5, batch_size=8, seed=0)
        self.assertIsNone(es.stopped_epoch)


class TestEarlyStoppingMonitoring(unittest.TestCase):

    def test_mode_auto_infers_direction_from_the_metric_name(self):
        es = EarlyStopping(monitor="val_loss", mode="auto")
        es.on_train_begin()
        self.assertEqual(es.best_value, np.inf)

        es = EarlyStopping(monitor="val_Accuracy", mode="auto")
        es.on_train_begin()
        self.assertEqual(es.best_value, -np.inf)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            EarlyStopping(monitor="val_loss", mode="sideways").on_train_begin()

    def test_min_delta_requires_a_real_improvement(self):
        es = EarlyStopping(monitor="val_loss", patience=2, min_delta=0.1,
                           restore_best_weights=False)
        es.set_model(None)
        es.on_train_begin()

        self.assertFalse(es.on_epoch_end(0, {"val_loss": 1.0}))
        self.assertFalse(es.on_epoch_end(1, {"val_loss": 0.95}))  # below min_delta
        self.assertTrue(es.on_epoch_end(2, {"val_loss": 0.94}))   # patience spent
        self.assertEqual(es.best_epoch, 0)

    def test_missing_monitor_key_is_ignored(self):
        es = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=False)
        es.set_model(None)
        es.on_train_begin()
        for epoch in range(10):
            self.assertFalse(es.on_epoch_end(epoch, {"loss": 0.5}))
        self.assertIsNone(es.stopped_epoch)


class TestCallbackProtocol(unittest.TestCase):

    def test_every_hook_is_triggered_in_order(self):
        class Recorder(Callback):
            def __init__(self):
                self.events = []

            def on_train_begin(self, logs=None):
                self.events.append("train_begin"); return False

            def on_epoch_begin(self, epoch, logs=None):
                self.events.append(f"epoch_begin:{epoch}"); return False

            def on_batch_begin(self, batch, logs=None):
                self.events.append("batch_begin"); return False

            def on_batch_end(self, batch, logs=None):
                self.events.append("batch_end"); return False

            def on_epoch_end(self, epoch, logs=None):
                self.events.append(f"epoch_end:{epoch}"); return False

            def on_train_end(self, logs=None):
                self.events.append("train_end"); return False

        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 4))
        y = (X[:, 0] > 0).astype(float).reshape(-1, 1)

        rec = Recorder()
        model = Model([Dense(4, 2, seed=1), Tanh(), Dense(2, 1, seed=2), Sigmoid()],
                      loss=BinaryCrossEntropy(), optimizer=Adam(lr=0.01),
                      callbacks=[rec])
        Trainer(model, verbose=0).fit(X, y, epochs=2, batch_size=8, seed=0)

        self.assertEqual(rec.events[0], "train_begin")
        self.assertEqual(rec.events[-1], "train_end")
        self.assertEqual(rec.events[1], "epoch_begin:0")
        self.assertIn("epoch_end:1", rec.events)

    def test_any_callback_can_stop_training(self):
        class StopAfterThree(Callback):
            def on_epoch_end(self, epoch, logs=None):
                return epoch >= 2

        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 4))
        y = (X[:, 0] > 0).astype(float).reshape(-1, 1)

        model = Model([Dense(4, 2, seed=1), Tanh(), Dense(2, 1, seed=2), Sigmoid()],
                      loss=BinaryCrossEntropy(), optimizer=Adam(lr=0.01),
                      callbacks=[StopAfterThree()])
        history = Trainer(model, verbose=0).fit(X, y, epochs=50, batch_size=8, seed=0)
        self.assertEqual(len(history.get("loss")), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
