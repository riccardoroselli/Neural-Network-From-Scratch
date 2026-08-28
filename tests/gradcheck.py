"""
Finite-difference gradient checking utilities.

Central differences are used throughout:

    df/dx ~= (f(x + h) - f(x - h)) / (2h)

which has O(h^2) truncation error, against O(h) for forward differences.
With float64 and h ~ 1e-5 the total error floor sits around 1e-10, so any
correctly implemented analytical gradient should come in well below 1e-6.

Relative error follows the standard formulation:

    err = |analytic - numeric| / max(|analytic| + |numeric|, eps)
"""
import numpy as np


def numeric_gradient(f, x, h=1e-5):
    """
    Numerical gradient of a scalar function f w.r.t. array x.

    Args:
        f: callable taking no arguments and returning a scalar. It must read
           the *current* contents of x, which this function perturbs in place.
        x: array to differentiate with respect to (modified and restored).
        h: perturbation step.

    Returns:
        array of the same shape as x holding df/dx.
    """
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        original = x[idx]

        x[idx] = original + h
        f_plus = f()

        x[idx] = original - h
        f_minus = f()

        x[idx] = original  # restore before moving on
        grad[idx] = (f_plus - f_minus) / (2.0 * h)

        it.iternext()

    return grad


def relative_error(analytic, numeric, eps=1e-12):
    """
    Max relative error between an analytical and a numerical gradient.

    Returns 0.0 when both gradients are uniformly zero.
    """
    analytic = np.asarray(analytic, dtype=float)
    numeric = np.asarray(numeric, dtype=float)

    numerator = np.abs(analytic - numeric)
    denominator = np.abs(analytic) + np.abs(numeric)

    # Where both gradients vanish the relative error is defined as 0.
    both_zero = denominator < eps
    denominator = np.where(both_zero, 1.0, denominator)
    errors = np.where(both_zero, 0.0, numerator / denominator)

    return float(np.max(errors))


def check_module_input_grad(module, X, seed=0, h=1e-5):
    """
    Gradient check of a Module w.r.t. its input.

    A random-but-fixed upstream gradient dY is used so the check exercises the
    full chain rule rather than a sum-reduction special case: the scalar under
    test is s = sum(dY * module.forward(X)), whose derivative w.r.t. X is
    exactly what backward(dY) should return.
    """
    rng = np.random.default_rng(seed)
    Y = module.forward(X.copy(), training=True)
    dY = rng.standard_normal(np.shape(Y))

    analytic = module.backward(dY)

    def scalar():
        return float(np.sum(dY * module.forward(X, training=True)))

    numeric = numeric_gradient(scalar, X, h=h)
    return relative_error(analytic, numeric), analytic, numeric


def check_loss_grad(loss, y_pred, y_true, h=1e-5):
    """Gradient check of a Loss w.r.t. its predictions."""
    analytic = loss.backward(y_pred.copy(), y_true)

    def scalar():
        return float(loss.forward(y_pred, y_true))

    numeric = numeric_gradient(scalar, y_pred, h=h)
    return relative_error(analytic, numeric), analytic, numeric
