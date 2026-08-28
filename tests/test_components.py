"""
Coverage for the parts of the library the MONK and CUP experiments never
exercise: the classification metrics, L1, He initialization, the multi-class
path, state save/restore, and the batching and config utilities.

These components are part of the framework's claimed surface, so they are
verified here rather than left untested.

    python -m unittest tests.test_components
"""
import json
import os
import tempfile
import unittest

import numpy as np

from data.data_handler.data_loader import denormalize, normalize
from nn.activations import Identity, Softmax, Tanh
from nn.initializers import he_uniform, xavier_uniform, zeros
from nn.layers import Dense
from nn.losses import MEE, MSE, CrossEntropy
from nn.metrics import Accuracy, F1Score, MEE as MEEMetric, MSE as MSEMetric, Precision, Recall
from nn.model import Model
from nn.optim import SGD, Adam
from nn.regularizers import L1, L2
from training.config import load_config
from training.dataloader import BatchIterator
from training.history import History
from training.model_factory import build_model_from_cfg
from training.trainer import Trainer


class TestClassificationMetrics(unittest.TestCase):
    """
    Hand-worked confusion matrix:
        y_true =  [1, 1, 1, 0, 0, 0, 0]
        y_prob =  [.9,.8,.3,.7,.2,.1,.4]  -> preds [1,1,0,1,0,0,0]
        TP=2, FP=1, FN=1, TN=3
        precision = recall = f1 = 2/3
    """

    def setUp(self):
        self.y_true = np.array([[1], [1], [1], [0], [0], [0], [0]])
        self.y_prob = np.array([[.9], [.8], [.3], [.7], [.2], [.1], [.4]])

    def test_precision(self):
        self.assertAlmostEqual(Precision()(self.y_prob, self.y_true), 2 / 3)

    def test_recall(self):
        self.assertAlmostEqual(Recall()(self.y_prob, self.y_true), 2 / 3)

    def test_f1(self):
        self.assertAlmostEqual(F1Score()(self.y_prob, self.y_true), 2 / 3)

    def test_threshold_is_honoured(self):
        # At 0.85 only the first sample is predicted positive: TP=1, FP=0.
        self.assertAlmostEqual(Precision(threshold=0.85)(self.y_prob, self.y_true), 1.0)
        self.assertAlmostEqual(Recall(threshold=0.85)(self.y_prob, self.y_true), 1 / 3)

    def test_no_positive_prediction_does_not_divide_by_zero(self):
        y_prob = np.zeros((4, 1))
        y_true = np.array([[1], [0], [1], [0]])
        self.assertEqual(Precision()(y_prob, y_true), 0.0)
        self.assertEqual(Recall()(y_prob, y_true), 0.0)
        self.assertEqual(F1Score()(y_prob, y_true), 0.0)

    def test_perfect_prediction(self):
        y_true = np.array([[1], [0], [1], [0]])
        y_prob = np.array([[.99], [.01], [.98], [.02]])
        for metric in (Precision(), Recall(), F1Score(), Accuracy()):
            self.assertAlmostEqual(metric(y_prob, y_true), 1.0)

    def test_accuracy_binary_and_multiclass(self):
        self.assertAlmostEqual(Accuracy()(self.y_prob, self.y_true), 5 / 7)
        probs = np.array([[.7, .2, .1], [.1, .8, .1], [.2, .2, .6], [.5, .3, .2]])
        onehot = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        self.assertAlmostEqual(Accuracy()(probs, onehot), 0.75)


class TestRegressionMetrics(unittest.TestCase):

    def test_mee_is_the_mean_euclidean_distance(self):
        y_pred = np.array([[3.0, 4.0], [0.0, 0.0]])
        y_true = np.array([[0.0, 0.0], [6.0, 8.0]])
        self.assertAlmostEqual(MEEMetric()(y_pred, y_true), (5.0 + 10.0) / 2)

    def test_mse_metric_matches_the_mse_loss(self):
        rng = np.random.default_rng(0)
        y_pred, y_true = rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
        self.assertAlmostEqual(MSEMetric()(y_pred, y_true), MSE().forward(y_pred, y_true))

    def test_mee_metric_matches_the_mee_loss(self):
        rng = np.random.default_rng(1)
        y_pred, y_true = rng.standard_normal((6, 3)), rng.standard_normal((6, 3))
        self.assertAlmostEqual(MEEMetric()(y_pred, y_true), MEE().forward(y_pred, y_true))


class TestInitializers(unittest.TestCase):

    def test_xavier_bounds_and_shape(self):
        W = xavier_uniform(10, 5, np.random.default_rng(0))
        self.assertEqual(W.shape, (10, 5))
        self.assertLessEqual(np.abs(W).max(), np.sqrt(6.0 / 15))

    def test_he_bounds_and_shape(self):
        W = he_uniform(10, 5, np.random.default_rng(0))
        self.assertEqual(W.shape, (10, 5))
        self.assertLessEqual(np.abs(W).max(), np.sqrt(6.0 / 10))

    def test_he_is_wider_than_xavier_for_the_same_fan_in(self):
        """He compensates for ReLU zeroing half the activations."""
        self.assertGreater(np.sqrt(6.0 / 64), np.sqrt(6.0 / (64 + 64)))

    def test_zeros(self):
        np.testing.assert_array_equal(zeros(3, 4), np.zeros((3, 4)))

    def test_dense_accepts_a_custom_initializer(self):
        layer = Dense(8, 4, initializer=he_uniform, seed=0)
        self.assertLessEqual(np.abs(layer.W).max(), np.sqrt(6.0 / 8))
        np.testing.assert_array_equal(layer.b, np.zeros((1, 4)))

    def test_initialization_is_seed_reproducible(self):
        a = Dense(5, 3, seed=7)
        b = Dense(5, 3, seed=7)
        np.testing.assert_array_equal(a.W, b.W)
        self.assertFalse(np.array_equal(Dense(5, 3, seed=7).W, Dense(5, 3, seed=8).W))


class TestL1Regularizer(unittest.TestCase):

    def test_penalty_value(self):
        layer = Dense(2, 2, seed=0)
        layer.W[...] = np.array([[1.0, -2.0], [3.0, -4.0]])
        self.assertAlmostEqual(L1(lam=0.5).penalty([layer]), 0.5 * 10.0)

    def test_gradient_is_lambda_times_sign(self):
        layer = Dense(2, 2, seed=0)
        layer.W[...] = np.array([[1.0, -2.0], [3.0, -4.0]])
        layer.dW[...] = 0.0
        L1(lam=0.5).add_gradients([layer])
        np.testing.assert_allclose(layer.dW, 0.5 * np.sign(layer.W))

    def test_l1_drives_weights_toward_zero(self):
        """The property that distinguishes L1 from L2: sparsity."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 6))
        y = X @ rng.standard_normal((6, 1))

        def train(regularizer):
            model = Model([Dense(6, 1, seed=1), Identity()], loss=MSE(),
                          optimizer=SGD(lr=0.01), regularizer=regularizer)
            Trainer(model, verbose=0).fit(X, y, epochs=60, batch_size=16, seed=0)
            return np.abs(model.modules[0].W).sum()

        self.assertLess(train(L1(lam=0.5)), train(L1(lam=0.0)))

    def test_l2_shrinks_weights_relative_to_no_regularization(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 6))
        y = X @ rng.standard_normal((6, 1))

        def train(regularizer):
            model = Model([Dense(6, 1, seed=1), Identity()], loss=MSE(),
                          optimizer=SGD(lr=0.01), regularizer=regularizer)
            Trainer(model, verbose=0).fit(X, y, epochs=60, batch_size=16, seed=0)
            return float((model.modules[0].W ** 2).sum())

        self.assertLess(train(L2(lam=0.5)), train(L2(lam=0.0)))


class TestMulticlassPath(unittest.TestCase):
    """The task='multiclass' branch of the factory: Softmax + CrossEntropy."""

    def setUp(self):
        rng = np.random.default_rng(0)
        centres = np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 3.0]])
        self.labels = rng.integers(0, 3, size=150)
        self.X = centres[self.labels] + 0.5 * rng.standard_normal((150, 2))
        self.y = np.zeros((150, 3))
        self.y[np.arange(150), self.labels] = 1.0
        self.cfg = {"model": {"hidden_units": [12], "activation": "tanh", "dropout": 0.0},
                    "optim": {"name": "adam", "lr": 0.05},
                    "callbacks": {"early_stopping": False}}

    def test_factory_builds_a_softmax_head(self):
        model = build_model_from_cfg(self.cfg, seed=0, in_dim=2, out_dim=3,
                                     task="multiclass")
        self.assertIsInstance(model.modules[-1], Softmax)
        self.assertIsInstance(model.loss, CrossEntropy)

    def test_outputs_are_a_probability_distribution(self):
        model = build_model_from_cfg(self.cfg, seed=0, in_dim=2, out_dim=3,
                                     task="multiclass")
        probs = model.predict_proba(self.X)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(150), atol=1e-12)
        self.assertTrue(np.all(probs >= 0))

    def test_it_learns_three_separable_classes(self):
        model = build_model_from_cfg(self.cfg, seed=0, in_dim=2, out_dim=3,
                                     task="multiclass")
        trainer = Trainer(model, verbose=0)
        trainer.fit(self.X, self.y, epochs=60, batch_size=16, seed=0)
        self.assertGreater(trainer.evaluate(self.X, self.y)["Accuracy"], 0.95)

    def test_unknown_task_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model_from_cfg(self.cfg, seed=0, in_dim=2, out_dim=3, task="ranking")

    def test_unknown_activation_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model_from_cfg({"model": {"activation": "swish"}}, seed=0,
                                 in_dim=2, out_dim=1)


class TestModelStateAndPrediction(unittest.TestCase):

    def _model(self):
        return Model([Dense(4, 3, seed=1), Tanh(), Dense(3, 1, seed=2), Identity()],
                     loss=MSE(), optimizer=Adam(lr=0.01))

    def test_get_state_set_state_round_trip(self):
        model = self._model()
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((20, 4)), rng.standard_normal((20, 1))

        state = model.get_state()
        before = model.predict_proba(X).copy()

        Trainer(model, verbose=0).fit(X, y, epochs=20, batch_size=8, seed=0)
        self.assertFalse(np.allclose(model.predict_proba(X), before))

        model.set_state(state)
        np.testing.assert_allclose(model.predict_proba(X), before)

    def test_set_state_preserves_array_identity(self):
        """Optimizer state is positional now, but in-place restore still matters."""
        model = self._model()
        params_before = [id(p) for p in model.parameters()]
        model.set_state(model.get_state())
        self.assertEqual([id(p) for p in model.parameters()], params_before)

    def test_set_state_rejects_a_mismatched_state(self):
        model = self._model()
        state = model.get_state()
        state["params"] = state["params"][:-1]
        with self.assertRaises(ValueError):
            model.set_state(state)

    def test_optimizer_state_is_captured_and_restored(self):
        model = self._model()
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((20, 4)), rng.standard_normal((20, 1))
        Trainer(model, verbose=0).fit(X, y, epochs=5, batch_size=8, seed=0)

        state = model.get_state()
        self.assertGreater(state["optimizer"]["t"], 0)
        model.optimizer.t = 0
        model.set_state(state)
        self.assertGreater(model.optimizer.t, 0)

    def test_predict_thresholds_probabilities(self):
        model = Model([Dense(3, 1, seed=1), Identity()], loss=MSE())
        X = np.zeros((4, 3))
        model.modules[0].b[...] = np.array([[0.6]])
        np.testing.assert_array_equal(model.predict(X), np.ones((4, 1), dtype=int))
        np.testing.assert_array_equal(model.predict(X, threshold=0.7),
                                      np.zeros((4, 1), dtype=int))

    def test_repr_lists_the_architecture(self):
        text = repr(self._model())
        self.assertIn("Dense", text)
        self.assertIn("Tanh", text)


class TestBatchIterator(unittest.TestCase):

    def setUp(self):
        self.X = np.arange(10).reshape(10, 1)
        self.y = np.arange(10).reshape(10, 1) * 2

    def test_batch_count_with_and_without_drop_last(self):
        self.assertEqual(len(BatchIterator(self.X, self.y, batch_size=3)), 4)
        self.assertEqual(len(BatchIterator(self.X, self.y, batch_size=3, drop_last=True)), 3)

    def test_every_sample_appears_exactly_once(self):
        seen = np.concatenate([xb.ravel() for xb, _ in
                               BatchIterator(self.X, self.y, batch_size=3, shuffle=True, seed=0)])
        np.testing.assert_array_equal(np.sort(seen), np.arange(10))

    def test_pairs_stay_aligned_after_shuffling(self):
        for xb, yb in BatchIterator(self.X, self.y, batch_size=4, shuffle=True, seed=1):
            np.testing.assert_array_equal(yb, xb * 2)

    def test_drop_last_discards_the_incomplete_batch(self):
        batches = list(BatchIterator(self.X, self.y, batch_size=3,
                                     shuffle=False, drop_last=True))
        self.assertEqual([len(xb) for xb, _ in batches], [3, 3, 3])

    def test_shuffle_is_seed_reproducible(self):
        def order(seed):
            return np.concatenate([xb.ravel() for xb, _ in
                                   BatchIterator(self.X, self.y, batch_size=3,
                                                 shuffle=True, seed=seed)])
        np.testing.assert_array_equal(order(5), order(5))
        self.assertFalse(np.array_equal(order(5), order(6)))

    def test_works_without_targets(self):
        batches = list(BatchIterator(self.X, None, batch_size=4, shuffle=False))
        self.assertEqual(len(batches), 3)
        self.assertEqual(batches[0].shape, (4, 1))

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            BatchIterator(None, self.y)
        with self.assertRaises(ValueError):
            BatchIterator(self.X, self.y, batch_size=0)
        with self.assertRaises(ValueError):
            BatchIterator(self.X, self.y[:5])


class TestHistory(unittest.TestCase):

    def test_logging_and_retrieval(self):
        history = History()
        history.log(loss=0.5, Accuracy=0.8)
        history.log(loss=0.3, Accuracy=0.9)
        self.assertEqual(history.get("loss"), [0.5, 0.3])
        self.assertEqual(history.last("loss"), 0.3)
        self.assertEqual(set(history.keys()), {"loss", "Accuracy"})
        self.assertEqual(history.to_dict()["Accuracy"], [0.8, 0.9])

    def test_missing_keys_return_the_default(self):
        history = History()
        self.assertIsNone(history.last("nope"))
        self.assertEqual(history.last("nope", 42), 42)
        self.assertIsNone(history.get("nope"))
        self.assertEqual(repr(history), "History(empty)")


class TestConfigLoading(unittest.TestCase):

    def test_json_and_yaml_agree(self):
        payload = {"base": {"optim": {"lr": 0.1}}, "grid": {"optim.lr": [0.1, 0.2]}}
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "c.json")
            with open(j, "w") as f:
                json.dump(payload, f)
            self.assertEqual(load_config(j), payload)

            y = os.path.join(d, "c.yaml")
            with open(y, "w") as f:
                f.write("base:\n  optim:\n    lr: 0.1\ngrid:\n  optim.lr: [0.1, 0.2]\n")
            self.assertEqual(load_config(y), payload)

    def test_non_mapping_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.json")
            with open(path, "w") as f:
                json.dump([1, 2, 3], f)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_real_project_configs_load(self):
        for name in ("MONK/configs/monk1", "MONK/configs/monk2",
                     "MONK/configs/monk3", "CUP/configs/cup_coarse"):
            cfg = load_config(f"experiments/{name}.yaml")
            self.assertIn("base", cfg)
            self.assertIn("grid", cfg)


class TestNormalization(unittest.TestCase):

    def test_round_trip_on_targets(self):
        y = np.random.default_rng(0).standard_normal((40, 4)) * 12 - 5
        yn, mean, std = normalize(y)
        np.testing.assert_allclose(denormalize(yn, mean, std), y, atol=1e-12)

    def test_statistics_are_per_column(self):
        X = np.array([[0.0, 100.0], [2.0, 300.0], [4.0, 200.0]])
        _, mean, std = normalize(X)
        np.testing.assert_allclose(mean, [2.0, 200.0])
        self.assertEqual(len(std), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
