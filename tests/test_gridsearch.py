"""
Grid-search aggregation: metric coercion, ranking, and divergence handling.

    python -m unittest tests.test_gridsearch
"""
import unittest

from training.config import expand_grid, get_by_path, set_by_path
from training.gridsearch import (_compute_mean_std, _extract_objective,
                                 _infer_mode_from_monitor, _safe_float,
                                 summarize_rows)
from training.refine_grid import build_linear_values, round_sig


def row(config_id, score):
    return {"config_id": config_id, "score_mean": _safe_float(score),
            "val_loss_mean": _safe_float(score), "val_loss_std": None,
            "val_acc_mean": None, "val_acc_std": None,
            "best_epoch_mean": None, "best_epoch_std": None, "config_json": "{}"}


class TestSafeFloat(unittest.TestCase):

    def test_parses_finite_numbers(self):
        self.assertEqual(_safe_float("1.5"), 1.5)
        self.assertEqual(_safe_float(2), 2.0)

    def test_rejects_non_finite(self):
        """A diverged run is a missing measurement, not a very bad one."""
        for value in ("nan", "inf", "-inf", float("nan"), float("inf")):
            self.assertIsNone(_safe_float(value))

    def test_rejects_unparseable(self):
        for value in (None, "", "abc", object()):
            self.assertIsNone(_safe_float(value))


class TestMeanStd(unittest.TestCase):

    def test_single_value_has_undefined_std(self):
        mean, std = _compute_mean_std([2.0])
        self.assertEqual(mean, 2.0)
        self.assertIsNone(std, "std of one sample must not be reported as 0.0")

    def test_multiple_values(self):
        self.assertEqual(_compute_mean_std([2.0, 4.0]), (3.0, 1.0))

    def test_empty(self):
        self.assertEqual(_compute_mean_std([]), (None, None))


class TestRanking(unittest.TestCase):

    def test_diverged_configs_rank_last_when_minimizing(self):
        rows = [row("a", 1.0), row("b", "nan"), row("c", 0.5),
                row("d", "inf"), row("e", 2.0)]
        summary = summarize_rows(rows, objective_mode="min")
        scores = [r["score_mean"] for r in summary]
        finite = [s for s in scores if s is not None]
        self.assertEqual(finite, [0.5, 1.0, 2.0])
        self.assertTrue(all(s is None for s in scores[len(finite):]))

    def test_diverged_configs_rank_last_when_maximizing(self):
        rows = [row("a", 1.0), row("b", "nan"), row("c", 0.5), row("d", 2.0)]
        summary = summarize_rows(rows, objective_mode="max")
        scores = [r["score_mean"] for r in summary]
        finite = [s for s in scores if s is not None]
        self.assertEqual(finite, [2.0, 1.0, 0.5])
        self.assertTrue(all(s is None for s in scores[len(finite):]))

    def test_runs_are_grouped_by_config(self):
        rows = [row("a", 1.0), row("a", 3.0), row("b", 2.0)]
        summary = summarize_rows(rows, objective_mode="min")
        by_id = {r["config_id"]: r for r in summary}
        self.assertEqual(by_id["a"]["n_runs"], 2)
        self.assertEqual(by_id["a"]["score_mean"], 2.0)
        self.assertEqual(by_id["a"]["score_std"], 1.0)
        self.assertEqual(by_id["b"]["n_runs"], 1)
        self.assertIsNone(by_id["b"]["score_std"])


class TestObjective(unittest.TestCase):

    def test_objective_selection(self):
        self.assertEqual(_extract_objective("val_loss", 0.3, 0.9), 0.3)
        self.assertEqual(_extract_objective("val_acc", 0.3, 0.9), 0.9)
        self.assertEqual(_extract_objective("anything", 0.3, 0.9), 0.3)

    def test_mode_inference(self):
        self.assertEqual(_infer_mode_from_monitor("val_loss"), "min")
        self.assertEqual(_infer_mode_from_monitor("val_Accuracy"), "max")


class TestConfigExpansion(unittest.TestCase):

    def test_cartesian_product_size_and_content(self):
        cfg = {"base": {"optim": {"lr": 0.1, "momentum": 0.0}},
               "grid": {"optim.lr": [0.01, 0.1], "optim.momentum": [0.0, 0.9]}}
        runs = expand_grid(cfg)
        self.assertEqual(len(runs), 4)
        self.assertEqual({(r["optim"]["lr"], r["optim"]["momentum"]) for r in runs},
                         {(0.01, 0.0), (0.01, 0.9), (0.1, 0.0), (0.1, 0.9)})

    def test_base_is_not_mutated(self):
        cfg = {"base": {"optim": {"lr": 0.1}}, "grid": {"optim.lr": [0.5]}}
        expand_grid(cfg)
        self.assertEqual(cfg["base"]["optim"]["lr"], 0.1)

    def test_empty_grid_yields_the_base(self):
        cfg = {"base": {"optim": {"lr": 0.1}}}
        self.assertEqual(expand_grid(cfg), [{"optim": {"lr": 0.1}}])

    def test_missing_base_is_rejected(self):
        with self.assertRaises(ValueError):
            expand_grid({"grid": {"optim.lr": [0.1]}})

    def test_scalar_grid_value_is_rejected(self):
        with self.assertRaises(ValueError):
            expand_grid({"base": {}, "grid": {"optim.lr": 0.1}})

    def test_path_helpers_round_trip(self):
        cfg = {}
        set_by_path(cfg, "a.b.c", 7)
        self.assertEqual(cfg, {"a": {"b": {"c": 7}}})
        self.assertEqual(get_by_path(cfg, "a.b.c"), 7)
        self.assertEqual(get_by_path(cfg, "a.b.missing", "fallback"), "fallback")


class TestGridRefinement(unittest.TestCase):

    def test_round_sig(self):
        self.assertEqual(round_sig(0.0456, 1), 0.05)
        self.assertEqual(round_sig(123.0, 1), 100.0)
        self.assertEqual(round_sig(0.0, 1), 0.0)

    def test_linear_values_are_sorted_and_deduplicated(self):
        values = build_linear_values(0.01, 0.1, steps=5, sig_digits=1)
        self.assertEqual(values, sorted(values))
        self.assertEqual(len(values), len(set(values)))

    def test_degenerate_range_collapses_to_one_value(self):
        self.assertEqual(build_linear_values(0.5, 0.5, steps=5, sig_digits=1), [0.5])

    def test_clipping_is_applied(self):
        values = build_linear_values(0.0, 10.0, steps=5, sig_digits=1, clip=(0.0, 1.0))
        self.assertTrue(all(0.0 <= v <= 1.0 for v in values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
