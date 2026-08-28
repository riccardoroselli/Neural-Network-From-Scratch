"""
End-to-end and methodological checks for the training pipeline:
splitting, normalization hygiene, k-fold, grid search, and determinism.

    python -m unittest tests.test_pipeline
"""
import csv
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from data.data_handler.data_loader import load_monk, normalize, denormalize
from nn.metrics import Accuracy
from training.holdout_cv import holdout_validation
from training.kfold_cv import kfold_cross_validation, _is_regression_task
from training.model_factory import build_model_from_cfg
from training.trainer import Trainer


def toy_classification(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int).reshape(-1, 1)
    return X, y


def toy_regression(n=120, d=6, k=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, k))
    return X, X @ W + 0.1 * rng.standard_normal((n, k))


BINARY_CFG = {
    "model": {"hidden_units": [6], "activation": "tanh", "dropout": 0.0},
    "optim": {"name": "sgd", "lr": 0.1, "momentum": 0.9},
    "regularizer": {"l2": 0.0},
    "training": {"epochs": 12, "batch_size": 16},
    "callbacks": {"early_stopping": False},
}

REGRESSION_CFG = {
    "model": {"hidden_units": [8, 4], "activation": "tanh", "dropout": 0.0},
    "optim": {"name": "sgd", "lr": 0.01, "momentum": 0.9},
    "regularizer": {"l2": 1e-4},
    "training": {"epochs": 12, "batch_size": 16},
    "callbacks": {"early_stopping": False},
}


class TestNormalization(unittest.TestCase):

    def test_standardizes_to_zero_mean_unit_variance(self):
        X = np.random.default_rng(0).standard_normal((50, 4)) * 7 + 3
        Xn, mean, std = normalize(X)
        np.testing.assert_allclose(Xn.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(Xn.std(axis=0), 1, atol=1e-12)

    def test_transform_mode_reuses_supplied_statistics(self):
        rng = np.random.default_rng(1)
        Xtr, Xte = rng.standard_normal((50, 4)), rng.standard_normal((20, 4))
        _, mean, std = normalize(Xtr)
        Xte_n, m2, s2 = normalize(Xte, mean=mean, std=std)
        np.testing.assert_array_equal(m2, mean)
        np.testing.assert_array_equal(s2, std)
        np.testing.assert_allclose(Xte_n, (Xte - mean) / std)
        # The held-out set must NOT come out standardized on its own statistics.
        self.assertGreater(abs(float(Xte_n.mean())), 1e-6)

    def test_constant_column_does_not_divide_by_zero(self):
        X = np.ones((10, 3))
        Xn, _, std = normalize(X)
        self.assertTrue(np.all(np.isfinite(Xn)))
        np.testing.assert_array_equal(std, np.ones(3))

    def test_denormalize_inverts_normalize(self):
        y = np.random.default_rng(2).standard_normal((30, 4)) * 5 - 2
        yn, mean, std = normalize(y)
        np.testing.assert_allclose(denormalize(yn, mean, std), y, atol=1e-12)


class TestHoldout(unittest.TestCase):

    def test_split_sizes_and_disjointness(self):
        X, y = toy_classification(n=100)
        model = build_model_from_cfg(BINARY_CFG, seed=0, in_dim=X.shape[1], out_dim=1)
        tr, va, hist = holdout_validation(
            X, y, model, Trainer(model, verbose=0), val_split=0.25,
            stratified=True, epochs=5, batch_size=16, seed=0, verbose=0)
        self.assertEqual(len(hist.get("loss")), 5)
        self.assertIn("loss", va)
        self.assertIn("Accuracy", va)

    def test_normalization_statistics_come_from_the_training_split_only(self):
        """
        The validation split must be standardized with the training split's
        mean and std. If statistics were fitted on the full array first, the
        validation split would come out with mean ~0, which is the leak.
        """
        rng = np.random.default_rng(4)
        X = rng.standard_normal((200, 5)) * 3 + 10
        # Give the tail a very different location so a leak is unmistakable.
        X[150:] += 40
        y = (X[:, 0] > X[:, 0].mean()).astype(int).reshape(-1, 1)

        captured = {}
        import training.holdout_cv as holdout_module
        original = holdout_module.normalize

        def spy(data, mean=None, std=None):
            out = original(data, mean=mean, std=std)
            captured.setdefault("calls", []).append(
                (data.shape, mean is None))
            return out

        holdout_module.normalize = spy
        try:
            model = build_model_from_cfg(BINARY_CFG, seed=0, in_dim=5, out_dim=1)
            holdout_validation(X, y, model, Trainer(model, verbose=0),
                               val_split=0.25, stratified=False, epochs=2,
                               batch_size=32, seed=0, verbose=0,
                               normalize_data=True)
        finally:
            holdout_module.normalize = original

        calls = captured["calls"]
        # First call fits (mean is None) on the training split; second call
        # transforms the validation split with supplied statistics.
        self.assertTrue(calls[0][1], "first normalize() call must fit statistics")
        self.assertFalse(calls[1][1], "second call must reuse the fitted statistics")
        self.assertGreater(calls[0][0][0], calls[1][0][0],
                           "statistics must be fitted on the larger (training) split")


class TestKFold(unittest.TestCase):

    def test_task_detection(self):
        self.assertTrue(_is_regression_task(np.zeros((10, 4))))          # multi-output
        self.assertTrue(_is_regression_task(np.zeros((10, 1))))          # float 1-D
        self.assertFalse(_is_regression_task(np.zeros((10, 1), dtype=int)))
        self.assertFalse(_is_regression_task(np.array([0, 1, 1, 0])))

    def test_folds_partition_the_dataset(self):
        from sklearn.model_selection import KFold
        X, _ = toy_regression(n=100)
        seen = []
        for _, val_idx in KFold(n_splits=5, shuffle=True, random_state=0).split(X):
            seen.append(set(val_idx.tolist()))
        union = set().union(*seen)
        self.assertEqual(union, set(range(100)))
        self.assertEqual(sum(len(s) for s in seen), 100)  # disjoint

    def test_classification_runs_and_reports_per_fold_stats(self):
        X, y = toy_classification(n=100)
        model = build_model_from_cfg(BINARY_CFG, seed=0, in_dim=X.shape[1], out_dim=1)
        tr, va, hists, folds = kfold_cross_validation(
            X, y, model, Trainer(model, verbose=0), k=4, epochs=5,
            batch_size=16, seed=0, verbose=0)
        self.assertEqual(len(folds), 4)
        self.assertEqual(len(hists), 4)
        for key in ("loss", "Accuracy"):
            self.assertIn("mean", va[key])
            self.assertIn("std", va[key])

    def test_regression_runs_end_to_end(self):
        X, y = toy_regression(n=100)
        model = build_model_from_cfg(REGRESSION_CFG, seed=0, in_dim=X.shape[1],
                                     out_dim=y.shape[1], task="regression")
        tr, va, hists, folds = kfold_cross_validation(
            X, y, model, Trainer(model, verbose=0), k=4, epochs=5,
            batch_size=16, seed=0, verbose=0,
            normalize_data=True, normalize_target=True)
        self.assertEqual(len(folds), 4)
        self.assertIn("MEE", va)


class TestTrainingRuns(unittest.TestCase):

    def test_short_training_run_reduces_loss(self):
        X, y = toy_classification(n=200)
        model = build_model_from_cfg(BINARY_CFG, seed=0, in_dim=X.shape[1], out_dim=1)
        hist = Trainer(model, verbose=0).fit(X, y, epochs=40, batch_size=32, seed=0)
        losses = hist.get("loss")
        self.assertLess(losses[-1], losses[0])

    def test_evaluate_matches_a_manual_full_batch_computation(self):
        X, y = toy_classification(n=64)
        model = build_model_from_cfg(BINARY_CFG, seed=0, in_dim=X.shape[1], out_dim=1)
        trainer = Trainer(model, verbose=0)
        trainer.fit(X, y, epochs=5, batch_size=16, seed=0)

        batched = trainer.evaluate(X, y, batch_size=16)
        y_pred = model.forward(X, training=False)
        self.assertAlmostEqual(batched["loss"],
                               model.loss.forward(y_pred, y), places=10)
        self.assertAlmostEqual(batched["Accuracy"],
                               Accuracy()(y_pred, y), places=10)

    def test_batch_size_does_not_change_evaluation(self):
        X, y = toy_classification(n=97)  # deliberately not a multiple of any batch
        model = build_model_from_cfg(BINARY_CFG, seed=0, in_dim=X.shape[1], out_dim=1)
        trainer = Trainer(model, verbose=0)
        trainer.fit(X, y, epochs=3, batch_size=16, seed=0)
        a = trainer.evaluate(X, y, batch_size=8)
        b = trainer.evaluate(X, y, batch_size=64)
        self.assertAlmostEqual(a["loss"], b["loss"], places=10)
        self.assertAlmostEqual(a["Accuracy"], b["Accuracy"], places=10)


class TestDeterminism(unittest.TestCase):
    """Running the same configuration twice must give identical numbers."""

    def test_training_run_is_reproducible(self):
        X, y = toy_classification(n=120)

        def run():
            model = build_model_from_cfg(BINARY_CFG, seed=7, in_dim=X.shape[1], out_dim=1)
            hist = Trainer(model, verbose=0).fit(X, y, epochs=15, batch_size=16, seed=7)
            return np.array(hist.get("loss"))

        np.testing.assert_array_equal(run(), run())

    def test_holdout_run_is_reproducible(self):
        X, y = toy_classification(n=120)

        def run():
            model = build_model_from_cfg(BINARY_CFG, seed=7, in_dim=X.shape[1], out_dim=1)
            _, val, _ = holdout_validation(
                X, y, model, Trainer(model, verbose=0), val_split=0.2,
                stratified=True, epochs=10, batch_size=16, seed=7, verbose=0)
            return val

        self.assertEqual(run(), run())

    def test_kfold_run_is_reproducible(self):
        X, y = toy_classification(n=120)

        def run():
            model = build_model_from_cfg(BINARY_CFG, seed=7, in_dim=X.shape[1], out_dim=1)
            _, val, _, _ = kfold_cross_validation(
                X, y, model, Trainer(model, verbose=0), k=4, epochs=8,
                batch_size=16, seed=7, verbose=0)
            return val

        self.assertEqual(run(), run())

    def test_dropout_run_is_reproducible(self):
        X, y = toy_classification(n=120)
        cfg = dict(BINARY_CFG)
        cfg["model"] = dict(BINARY_CFG["model"], dropout=0.3)

        def run():
            model = build_model_from_cfg(cfg, seed=7, in_dim=X.shape[1], out_dim=1)
            hist = Trainer(model, verbose=0).fit(X, y, epochs=10, batch_size=16, seed=7)
            return np.array(hist.get("loss"))

        np.testing.assert_array_equal(run(), run())

    def test_unseeded_fit_is_not_reproducible(self):
        """
        Documents the hazard behind the blind-test submission: omitting seed=
        leaves BatchIterator on fresh OS entropy, so two runs differ even
        though the weights are seeded identically.
        """
        X, y = toy_classification(n=120)

        def run():
            model = build_model_from_cfg(BINARY_CFG, seed=7, in_dim=X.shape[1], out_dim=1)
            hist = Trainer(model, verbose=0).fit(X, y, epochs=10, batch_size=16)
            return np.array(hist.get("loss"))

        self.assertFalse(np.array_equal(run(), run()))


class TestGridSearchEndToEnd(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_small_two_phase_selection(self):
        """Coarse hold-out then fine k-fold, with all expected artifacts."""
        from experiments.MONK.common import build_model, load_full_data
        from training.model_selection import run_two_phase_selection

        cfg = {
            "selection": {"top_k": 3,
                          "refine": {"steps": 2, "sig_digits": 1,
                                     "rules": {"optim.lr": {"clip": [1e-4, 1.0]}}}},
            "base": {
                "seed": 0,
                "data": {"train_path": "data/MONK/MONK1/monks-1.train"},
                "split": {"val_size": 0.2, "stratified": True},
                "cv": {"k": 3, "shuffle": True, "seed": 0},
                "model": {"hidden_units": [4], "activation": "tanh", "dropout": 0.0},
                "optim": {"name": "sgd", "lr": 0.1, "momentum": 0.0},
                "regularizer": {"l2": 0.0},
                "callbacks": {"early_stopping": False},
                "training": {"epochs": 8, "batch_size": 32, "shuffle": True},
            },
            "grid": {"optim.lr": [0.05, 0.2], "model.hidden_units": [[4], [8]]},
        }
        cfg_path = os.path.join(self.tmp, "cfg.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        out = run_two_phase_selection(
            config_path=cfg_path, build_model_fn=build_model,
            load_full_data_fn=load_full_data, out_dir=self.tmp, seeds=[0],
            top_k=3, verbose=0, n_jobs=2)

        for name in ("coarse_runs.csv", "coarse_summary.csv", "fine_grid.json",
                     "refine_report.json", "fine_config.json", "fine_runs.csv",
                     "fine_summary.csv", "best_config.json"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp, name)), name)

        self.assertIsNotNone(out["best_config"])
        self.assertEqual(len(out["coarse_summary"]), 4)  # 2 lr x 2 architectures
        self.assertIsNotNone(out["best_score"])

        # Both phases write one row per configuration and seed, with no rows
        # carried over from a previous search into the same directory.
        for name, expected in [("coarse_runs.csv", 4), ("fine_runs.csv", None)]:
            with open(os.path.join(self.tmp, name)) as f:
                rows = list(csv.DictReader(f))
            if expected is not None:
                self.assertEqual(len(rows), expected, name)

    def test_reruns_do_not_accumulate_rows(self):
        """
        run_grid appends to its CSV, so a second search into the same output
        directory used to leave the first run's rows behind in coarse_runs.csv.
        Both phases must now start from a clean file.
        """
        from experiments.MONK.common import build_model, load_full_data
        from training.model_selection import run_two_phase_selection

        cfg = {
            "selection": {"top_k": 2,
                          "refine": {"steps": 2, "sig_digits": 1,
                                     "rules": {"optim.lr": {"clip": [1e-4, 1.0]}}}},
            "base": {
                "seed": 0,
                "data": {"train_path": "data/MONK/MONK1/monks-1.train"},
                "split": {"val_size": 0.2, "stratified": True},
                "cv": {"k": 2, "shuffle": True, "seed": 0},
                "model": {"hidden_units": [4], "activation": "tanh", "dropout": 0.0},
                "optim": {"name": "sgd", "lr": 0.1, "momentum": 0.0},
                "regularizer": {"l2": 0.0},
                "callbacks": {"early_stopping": False},
                "training": {"epochs": 3, "batch_size": 32, "shuffle": True},
            },
            "grid": {"optim.lr": [0.05, 0.2]},
        }
        cfg_path = os.path.join(self.tmp, "cfg.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        def run():
            run_two_phase_selection(
                config_path=cfg_path, build_model_fn=build_model,
                load_full_data_fn=load_full_data, out_dir=self.tmp, seeds=[0],
                top_k=2, verbose=0, n_jobs=2)
            counts = {}
            for name in ("coarse_runs.csv", "fine_runs.csv"):
                with open(os.path.join(self.tmp, name)) as f:
                    counts[name] = len(list(csv.DictReader(f)))
            return counts

        first = run()
        second = run()
        self.assertEqual(first["coarse_runs.csv"], 2)
        self.assertEqual(first, second,
                         "a repeated search must not accumulate rows")

    def test_ensemble_utilities(self):
        from evaluation.ensemble_utils import (accuracy, bce_loss, majority_vote,
                                            mean_std, predict_proba, stack_histories)
        X, y = toy_classification(n=80)
        models = []
        for s in range(5):
            m = build_model_from_cfg(BINARY_CFG, seed=s, in_dim=X.shape[1], out_dim=1)
            Trainer(m, verbose=0).fit(X, y, epochs=10, batch_size=16, seed=s)
            models.append(m)

        P = np.stack([predict_proba(m, X) for m in models], axis=0)
        self.assertEqual(P.shape, (5, 80))

        y_hat, p_mean, vote_frac = majority_vote(P)
        self.assertEqual(y_hat.shape, (80,))
        self.assertTrue(np.all((vote_frac >= 0) & (vote_frac <= 1)))
        self.assertGreaterEqual(accuracy(y.ravel(), y_hat), 0.0)
        self.assertGreater(bce_loss(y.ravel(), p_mean), 0.0)

        mean, std = mean_std([1.0, 2.0, 3.0])
        self.assertAlmostEqual(mean, 2.0)

        stacked = stack_histories([{"loss": [1, 2, 3]}, {"loss": [1, 2]}], "loss")
        self.assertEqual(stacked.shape, (2, 3))
        self.assertTrue(np.isnan(stacked[1, 2]))


class TestDataLoaders(unittest.TestCase):

    def test_monk_encoding(self):
        X, y = load_monk("data/MONK/MONK1/monks-1.train", encode=True)
        self.assertEqual(X.shape, (124, 17))
        self.assertEqual(y.shape, (124, 1))
        # One-hot: every sample sets exactly one bit per attribute group.
        for start, width in [(0, 3), (3, 3), (6, 2), (8, 3), (11, 4), (15, 2)]:
            np.testing.assert_array_equal(X[:, start:start + width].sum(axis=1),
                                          np.ones(124))
        self.assertTrue(set(np.unique(y)) <= {0, 1})

    def test_monk_raw_mode(self):
        X, y = load_monk("data/MONK/MONK1/monks-1.train", encode=False)
        self.assertEqual(X.shape, (124, 6))

    def test_cup_shapes(self):
        from data.data_handler.data_loader import load_cup
        X, y = load_cup("data/CUP/ML-CUP25-TR.csv", training=True)
        self.assertEqual(X.shape, (500, 12))
        self.assertEqual(y.shape, (500, 4))
        Xb = load_cup("data/CUP/ML-CUP25-TS.csv", training=False)
        self.assertEqual(Xb.shape, (1000, 12))
        self.assertTrue(np.all(np.isfinite(X)) and np.all(np.isfinite(y)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
