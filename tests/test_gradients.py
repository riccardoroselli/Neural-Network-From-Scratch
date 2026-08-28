"""
Finite-difference gradient checks for every differentiable component.

Run as a test suite:
    python -m unittest tests.test_gradients

Run as a report (prints the max relative error per component):
    python -m tests.test_gradients
"""
import unittest

import numpy as np

from nn.activations import ReLU, Sigmoid, Tanh, Identity, Softmax
from nn.layers import Dense
from nn.dropout import Dropout
from nn.losses import MSE, MEE, BinaryCrossEntropy, CrossEntropy
from nn.model import Model
from nn.regularizers import L1, L2
from tests.gradcheck import (
    check_loss_grad,
    check_module_input_grad,
    numeric_gradient,
    relative_error,
)

TOL = 1e-6          # any correct analytical gradient clears this easily
RESULTS = {}        # component name -> max relative error, for the report


def record(name, err):
    RESULTS[name] = err
    return err


def one_hot(labels, n_classes):
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


class TestActivationGradients(unittest.TestCase):
    """d(activation)/dX against central differences."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_relu(self):
        # Kept away from the kink at 0, where the derivative is undefined.
        X = self.rng.uniform(0.5, 2.0, size=(6, 4)) * self.rng.choice([-1, 1], size=(6, 4))
        err, _, _ = check_module_input_grad(ReLU(), X)
        self.assertLess(record("ReLU", err), TOL)

    def test_sigmoid(self):
        # Inside the [-20, 20] window, where forward() does not clip.
        X = self.rng.uniform(-6, 6, size=(6, 4))
        err, _, _ = check_module_input_grad(Sigmoid(), X)
        self.assertLess(record("Sigmoid", err), TOL)

    def test_tanh(self):
        X = self.rng.uniform(-3, 3, size=(6, 4))
        err, _, _ = check_module_input_grad(Tanh(), X)
        self.assertLess(record("Tanh", err), TOL)

    def test_identity(self):
        X = self.rng.standard_normal((6, 4))
        err, _, _ = check_module_input_grad(Identity(), X)
        self.assertLess(record("Identity", err), TOL)

    def test_sigmoid_saturated_region_is_documented(self):
        """
        forward() clips its input to [-20, 20], so beyond that the function is
        constant and the true derivative is 0. backward() ignores the clip and
        returns s(1-s) ~ 2e-9 instead. The discrepancy is negligible in
        magnitude but it is a real deviation, so it is pinned here rather than
        left to be discovered by a future gradient check.
        """
        X = np.array([[25.0, -25.0]])
        sigmoid = Sigmoid()
        sigmoid.forward(X.copy())
        analytic = sigmoid.backward(np.ones_like(X))

        self.assertTrue(np.all(analytic > 0))       # code returns nonzero ...
        self.assertTrue(np.all(analytic < 1e-8))    # ... but negligibly so
        record("Sigmoid (|x|>20, clipped)", 1.0)    # relative error is 1.0 by construction


class TestLossGradients(unittest.TestCase):
    """d(loss)/d(y_pred) against central differences."""

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_mse_multi_output(self):
        y_pred = self.rng.standard_normal((8, 4))
        y_true = self.rng.standard_normal((8, 4))
        err, _, _ = check_loss_grad(MSE(), y_pred, y_true)
        self.assertLess(record("MSE loss", err), TOL)

    def test_mse_single_output(self):
        y_pred = self.rng.standard_normal((8, 1))
        y_true = self.rng.standard_normal((8, 1))
        err, _, _ = check_loss_grad(MSE(), y_pred, y_true)
        self.assertLess(err, TOL)

    def test_mee_multi_output(self):
        # Residuals held away from zero, where the Euclidean norm is not
        # differentiable and backward() falls back to its eps clip.
        y_true = self.rng.standard_normal((8, 4))
        y_pred = y_true + self.rng.uniform(0.5, 1.5, size=(8, 4))
        err, _, _ = check_loss_grad(MEE(), y_pred, y_true)
        self.assertLess(record("MEE loss", err), TOL)

    def test_binary_cross_entropy(self):
        y_pred = self.rng.uniform(0.05, 0.95, size=(8, 1))
        y_true = self.rng.integers(0, 2, size=(8, 1)).astype(float)
        err, _, _ = check_loss_grad(BinaryCrossEntropy(), y_pred, y_true)
        self.assertLess(record("BinaryCrossEntropy loss", err), TOL)

    def test_cross_entropy_alone_is_a_fused_gradient(self):
        """
        CrossEntropy.backward returns (p - y)/N, which is the gradient w.r.t.
        the *softmax logits*, not w.r.t. p. Checked directly against p it is
        wrong by construction. This documents that CrossEntropy is only valid
        immediately after Softmax, whose backward() is a pass-through.
        """
        y_pred = self.rng.uniform(0.1, 0.9, size=(6, 3))
        y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
        y_true = one_hot(self.rng.integers(0, 3, size=6), 3)

        err, _, _ = check_loss_grad(CrossEntropy(), y_pred, y_true)
        record("CrossEntropy loss (w.r.t. p, unfused)", err)
        self.assertGreater(err, 0.1)  # deliberately not the derivative of the loss

    def test_softmax_cross_entropy_pair(self):
        """The fused pair must be correct w.r.t. the logits."""
        logits = self.rng.standard_normal((6, 3))
        y_true = one_hot(self.rng.integers(0, 3, size=6), 3)

        softmax, loss = Softmax(), CrossEntropy()
        probs = softmax.forward(logits.copy())
        analytic = softmax.backward(loss.backward(probs, y_true))

        def scalar():
            return float(loss.forward(softmax.forward(logits), y_true))

        numeric = numeric_gradient(scalar, logits)
        err = relative_error(analytic, numeric)
        self.assertLess(record("Softmax + CrossEntropy (fused)", err), TOL)

    def test_sigmoid_binary_cross_entropy_pair(self):
        """
        The BCE gradient divides by p(1-p) and Sigmoid's backward multiplies it
        straight back out, so the pair must cancel to (p - y)/N exactly.
        """
        logits = self.rng.uniform(-4, 4, size=(8, 1))
        y_true = self.rng.integers(0, 2, size=(8, 1)).astype(float)

        sigmoid, loss = Sigmoid(), BinaryCrossEntropy()
        probs = sigmoid.forward(logits.copy())
        analytic = sigmoid.backward(loss.backward(probs, y_true))

        def scalar():
            return float(loss.forward(sigmoid.forward(logits), y_true))

        numeric = numeric_gradient(scalar, logits)
        err = relative_error(analytic, numeric)
        self.assertLess(record("Sigmoid + BinaryCrossEntropy (fused)", err), TOL)

        # And the closed form it should collapse to:
        expected = (probs - y_true) / len(y_true)
        self.assertLess(relative_error(analytic, expected), 1e-10)


class TestDenseGradients(unittest.TestCase):
    """dX, dW and db for a Dense layer."""

    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.layer = Dense(5, 3, seed=1)
        self.X = self.rng.standard_normal((7, 5))
        self.dY = self.rng.standard_normal((7, 3))

    def _analytic(self):
        self.layer.forward(self.X.copy())
        return self.layer.backward(self.dY)

    def _scalar(self):
        return float(np.sum(self.dY * self.layer.forward(self.X)))

    def test_input_gradient(self):
        dX = self._analytic()
        numeric = numeric_gradient(self._scalar, self.X)
        self.assertLess(record("Dense dX", relative_error(dX, numeric)), TOL)

    def test_weight_gradient(self):
        self._analytic()
        numeric = numeric_gradient(self._scalar, self.layer.W)
        self.assertLess(record("Dense dW", relative_error(self.layer.dW, numeric)), TOL)

    def test_bias_gradient(self):
        self._analytic()
        numeric = numeric_gradient(self._scalar, self.layer.b)
        self.assertLess(record("Dense db", relative_error(self.layer.db, numeric)), TOL)


class TestDropoutGradient(unittest.TestCase):
    """
    Dropout is stochastic, so the mask is pinned by re-seeding the layer's RNG
    before every forward call; the finite-difference probe then sees the same
    deterministic function the analytical gradient was computed for.
    """

    def test_input_gradient_with_frozen_mask(self):
        rng = np.random.default_rng(11)
        X = rng.standard_normal((6, 4))
        dropout = Dropout(p=0.3, seed=99)

        def forward_fixed_mask(x):
            dropout.rng = np.random.default_rng(99)
            return dropout.forward(x, training=True)

        Y = forward_fixed_mask(X.copy())
        dY = rng.standard_normal(Y.shape)
        analytic = dropout.backward(dY)

        numeric = numeric_gradient(lambda: float(np.sum(dY * forward_fixed_mask(X))), X)
        self.assertLess(record("Dropout dX (p=0.3)", relative_error(analytic, numeric)), TOL)

    def test_inference_mode_is_identity(self):
        X = np.random.default_rng(5).standard_normal((4, 3))
        dropout = Dropout(p=0.5, seed=1)
        np.testing.assert_array_equal(dropout.forward(X, training=False), X)

    def test_inference_forward_clears_the_cached_mask(self):
        """A later backward() must not reuse a mask from an earlier pass."""
        X = np.ones((4, 3))
        dropout = Dropout(p=0.5, seed=1)
        dropout.forward(X, training=True)
        dropout.forward(X, training=False)
        self.assertIsNone(dropout.mask)
        np.testing.assert_array_equal(dropout.backward(np.ones((4, 3))), np.ones((4, 3)))

    def test_reset_restarts_the_mask_stream(self):
        X = np.ones((4, 3))
        dropout = Dropout(p=0.4, seed=13)
        first = dropout.forward(X, training=True).copy()
        dropout.reset()
        self.assertIsNone(dropout.mask)
        np.testing.assert_array_equal(dropout.forward(X, training=True), first)

    def test_model_reset_restarts_dropout(self):
        """Model.reset() must reach Dropout like every other stateful module."""
        from nn.model import Model
        from nn.losses import MSE

        dropout = Dropout(p=0.4, seed=13)
        model = Model(modules=[dropout], loss=MSE())
        X = np.ones((4, 3))
        first = model.forward(X, training=True).copy()
        model.reset()
        np.testing.assert_array_equal(model.forward(X, training=True), first)

    def test_expected_value_is_preserved(self):
        """
        Inverted dropout must leave E[output] == input.

        A single draw of 16k elements has a standard error of ~6.5e-3, so the
        mean is averaged over 40 independent seeds to get a tolerance that
        reflects the estimator rather than one noisy sample.
        """
        X = np.ones((2000, 8))
        means = [Dropout(p=0.4, seed=s).forward(X, training=True).mean()
                 for s in range(40)]
        self.assertAlmostEqual(float(np.mean(means)), 1.0, places=2)


class TestRegularizerGradients(unittest.TestCase):
    """
    The penalty added to the loss and the gradient added to dW must be exactly
    consistent: d(penalty)/dW must equal what add_gradients() contributes.
    """

    def _layers(self, seed=2):
        rng = np.random.default_rng(seed)
        layers = [Dense(4, 3, seed=1), Dense(3, 2, seed=2)]
        for layer in layers:
            layer.W[...] = rng.standard_normal(layer.W.shape)
            layer.dW[...] = 0.0
            layer.db[...] = 0.0
        return layers

    def test_l2_penalty_matches_gradient(self):
        lam = 0.037
        layers = self._layers()
        reg = L2(lam=lam)
        reg.add_gradients(layers)

        worst = 0.0
        for layer in layers:
            analytic = layer.dW.copy()
            numeric = numeric_gradient(lambda: reg.penalty(layers), layer.W)
            worst = max(worst, relative_error(analytic, numeric))
        self.assertLess(record("L2 penalty vs gradient", worst), TOL)

    def test_l1_penalty_matches_gradient(self):
        lam = 0.021
        layers = self._layers(seed=4)
        reg = L1(lam=lam)
        reg.add_gradients(layers)

        worst = 0.0
        for layer in layers:
            analytic = layer.dW.copy()
            numeric = numeric_gradient(lambda: reg.penalty(layers), layer.W)
            worst = max(worst, relative_error(analytic, numeric))
        self.assertLess(record("L1 penalty vs gradient", worst), TOL)

    def test_biases_are_never_penalized(self):
        for reg in (L2(lam=0.5), L1(lam=0.5)):
            layers = self._layers()
            for layer in layers:
                layer.b[...] = 3.0
            before = reg.penalty(layers)
            for layer in layers:
                layer.b[...] = -7.0
            self.assertEqual(before, reg.penalty(layers))

            reg.add_gradients(layers)
            for layer in layers:
                np.testing.assert_array_equal(layer.db, np.zeros_like(layer.db))

    def test_zero_lambda_is_a_no_op(self):
        for reg in (L2(lam=0.0), L1(lam=0.0)):
            layers = self._layers()
            self.assertEqual(reg.penalty(layers), 0.0)
            reg.add_gradients(layers)
            for layer in layers:
                np.testing.assert_array_equal(layer.dW, np.zeros_like(layer.dW))


class TestEndToEndModelGradients(unittest.TestCase):
    """
    Whole-network check: gradients of the total objective (data loss plus
    regularization penalty) w.r.t. every parameter of a multi-layer model,
    exercising forward, backward, and the regularizer together.
    """

    def _check_model(self, modules, loss, y_true, X, regularizer=None, label=""):
        model = Model(modules=modules, loss=loss, regularizer=regularizer)

        def objective():
            y_pred = model.forward(X, training=False)
            return float(model.compute_loss(y_true=y_true, y_pred=y_pred))

        y_pred = model.forward(X, training=False)
        model.backward(model.loss.backward(y_pred, y_true))
        if regularizer is not None:
            regularizer.add_gradients(model.modules)

        worst = 0.0
        for module in model.modules:
            for param, grad in module.params_and_grads():
                numeric = numeric_gradient(objective, param)
                worst = max(worst, relative_error(grad, numeric))

        self.assertLess(record(label, worst), TOL)
        return worst

    def test_regression_network_mse(self):
        rng = np.random.default_rng(21)
        X = rng.standard_normal((9, 6))
        y = rng.standard_normal((9, 3))
        modules = [Dense(6, 5, seed=1), Tanh(), Dense(5, 4, seed=2), Tanh(),
                   Dense(4, 3, seed=3), Identity()]
        self._check_model(modules, MSE(), y, X, label="Model 6-5-4-3 tanh + MSE")

    def test_regression_network_mse_with_l2(self):
        rng = np.random.default_rng(22)
        X = rng.standard_normal((9, 6))
        y = rng.standard_normal((9, 3))
        modules = [Dense(6, 5, seed=1), Tanh(), Dense(5, 3, seed=2), Identity()]
        self._check_model(modules, MSE(), y, X, regularizer=L2(lam=0.05),
                          label="Model + L2 (loss and gradient)")

    def test_regression_network_mee(self):
        rng = np.random.default_rng(23)
        X = rng.standard_normal((9, 6))
        y = rng.standard_normal((9, 4)) * 5.0
        modules = [Dense(6, 8, seed=1), Tanh(), Dense(8, 4, seed=2), Identity()]
        self._check_model(modules, MEE(), y, X, label="Model + MEE loss")

    def test_binary_classifier_relu_sigmoid_bce(self):
        rng = np.random.default_rng(24)
        X = rng.standard_normal((10, 7))
        y = rng.integers(0, 2, size=(10, 1)).astype(float)
        modules = [Dense(7, 6, seed=1), ReLU(), Dense(6, 4, seed=2), ReLU(),
                   Dense(4, 1, seed=3), Sigmoid()]
        self._check_model(modules, BinaryCrossEntropy(), y, X,
                          label="Model 7-6-4-1 relu + sigmoid + BCE")

    def test_multiclass_softmax_cross_entropy(self):
        rng = np.random.default_rng(25)
        X = rng.standard_normal((10, 5))
        y = one_hot(rng.integers(0, 4, size=10), 4)
        modules = [Dense(5, 6, seed=1), Tanh(), Dense(6, 4, seed=2), Softmax()]
        self._check_model(modules, CrossEntropy(), y, X,
                          label="Model 5-6-4 tanh + softmax + CE")

    def test_network_with_dropout_frozen_mask(self):
        """Dropout in training mode, with its mask pinned by re-seeding."""
        rng = np.random.default_rng(26)
        X = rng.standard_normal((8, 5))
        y = rng.standard_normal((8, 2))

        dropout = Dropout(p=0.25, seed=77)
        modules = [Dense(5, 6, seed=1), Tanh(), dropout, Dense(6, 2, seed=2), Identity()]
        model = Model(modules=modules, loss=MSE())

        def forward_fixed_mask():
            dropout.rng = np.random.default_rng(77)
            return model.forward(X, training=True)

        def objective():
            return float(model.compute_loss(y_true=y, y_pred=forward_fixed_mask()))

        y_pred = forward_fixed_mask()
        model.backward(model.loss.backward(y_pred, y))

        worst = 0.0
        for module in model.modules:
            for param, grad in module.params_and_grads():
                numeric = numeric_gradient(objective, param)
                worst = max(worst, relative_error(grad, numeric))
        self.assertLess(record("Model with Dropout (frozen mask)", worst), TOL)


def _report():
    """Run every check and print the max relative error per component."""
    suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=0).run(suite)

    width = max(len(k) for k in RESULTS)
    print("\n" + "=" * (width + 22))
    print("GRADIENT CHECK — max relative error (central differences, h=1e-5)")
    print("=" * (width + 22))
    for name, err in RESULTS.items():
        verdict = "OK" if err < TOL else "expected mismatch"
        print(f"{name:<{width}}  {err:>10.3e}  {verdict}")
    print("=" * (width + 22))
    print(f"{'tests run':<{width}}  {result.testsRun:>10d}")
    print(f"{'failures':<{width}}  {len(result.failures):>10d}")
    print(f"{'errors':<{width}}  {len(result.errors):>10d}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_report())
