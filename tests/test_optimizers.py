"""
Optimizer update rules, checked step by step against reference formulas
computed independently in the test.

    python -m unittest tests.test_optimizers
"""
import unittest

import numpy as np

from nn.core import Module
from nn.layers import Dense
from nn.optim import SGD, SGDMomentum, Adam


class FakeParam(Module):
    """A minimal parametric module with a fixed, controllable gradient."""

    def __init__(self, value, grad):
        self.p = np.array(value, dtype=float)
        self.g = np.array(grad, dtype=float)

    def params_and_grads(self):
        yield self.p, self.g


class TestSGD(unittest.TestCase):

    def test_update_rule(self):
        module = FakeParam([1.0, -2.0], [0.5, 0.25])
        SGD(lr=0.1).step([module])
        np.testing.assert_allclose(module.p, [1.0 - 0.05, -2.0 - 0.025])

    def test_updates_in_place(self):
        """
        The optimizer must mutate the layer's own arrays. If it rebound them
        instead, Dense.W would silently stop tracking the updates.
        """
        layer = Dense(3, 2, seed=0)
        W_before = layer.W
        layer.dW[...] = 1.0
        layer.db[...] = 1.0
        SGD(lr=0.1).step([layer])
        self.assertIs(layer.W, W_before)
        self.assertTrue(np.allclose(layer.W, W_before))

    def test_multiple_steps_accumulate(self):
        module = FakeParam([0.0], [1.0])
        opt = SGD(lr=0.1)
        for _ in range(5):
            opt.step([module])
        np.testing.assert_allclose(module.p, [-0.5])


class TestSGDMomentum(unittest.TestCase):
    """
    Classical (heavy-ball) momentum:
        v <- mu * v - lr * g
        p <- p + v
    """

    def test_matches_reference_over_many_steps(self):
        lr, mu = 0.05, 0.9
        grads = [0.5, -0.2, 0.3, 0.3, -0.1, 0.4, 0.0, -0.6]

        module = FakeParam([1.0], [0.0])
        opt = SGDMomentum(lr=lr, momentum=mu)

        p_ref, v_ref = 1.0, 0.0
        for g in grads:
            module.g[...] = g
            opt.step([module])

            v_ref = mu * v_ref - lr * g
            p_ref = p_ref + v_ref
            self.assertAlmostEqual(float(module.p[0]), p_ref, places=12)

    def test_zero_momentum_equals_plain_sgd(self):
        a = FakeParam([2.0], [0.7])
        b = FakeParam([2.0], [0.7])
        SGDMomentum(lr=0.1, momentum=0.0).step([a])
        SGD(lr=0.1).step([b])
        np.testing.assert_allclose(a.p, b.p)

    def test_velocity_is_per_parameter(self):
        """Two parameters in one module must not share a velocity buffer."""
        layer = Dense(2, 2, seed=0)
        layer.dW[...] = 1.0
        layer.db[...] = 2.0
        opt = SGDMomentum(lr=0.1, momentum=0.9)
        opt.step([layer])
        self.assertEqual(len(opt.velocities), 2)
        velocities = list(opt.velocities.values())
        self.assertFalse(np.allclose(velocities[0].ravel()[0], velocities[1].ravel()[0]))


class TestAdam(unittest.TestCase):
    """
        m <- b1*m + (1-b1)*g          v <- b2*v + (1-b2)*g^2
        m_hat = m/(1-b1^t)            v_hat = v/(1-b2^t)
        p <- p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def test_matches_reference_over_many_steps(self):
        lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
        grads = [0.5, -0.2, 0.3, 0.1, -0.4, 0.25, 0.0, 0.8, -0.15, 0.05]

        module = FakeParam([1.0], [0.0])
        opt = Adam(lr=lr, beta1=b1, beta2=b2, eps=eps)

        p_ref, m_ref, v_ref = 1.0, 0.0, 0.0
        for t, g in enumerate(grads, start=1):
            module.g[...] = g
            opt.step([module])

            m_ref = b1 * m_ref + (1 - b1) * g
            v_ref = b2 * v_ref + (1 - b2) * g * g
            m_hat = m_ref / (1 - b1 ** t)
            v_hat = v_ref / (1 - b2 ** t)
            p_ref = p_ref - lr * m_hat / (np.sqrt(v_hat) + eps)

            self.assertEqual(opt.t, t)
            self.assertAlmostEqual(float(module.p[0]), p_ref, places=12)

    def test_bias_correction_makes_first_step_full_size(self):
        """
        At t=1 bias correction gives m_hat/sqrt(v_hat) == sign(g), so the first
        step must be almost exactly lr. Without correction it would be ~lr/10.
        """
        module = FakeParam([0.0], [0.5])
        Adam(lr=0.01).step([module])
        self.assertAlmostEqual(float(module.p[0]), -0.01, places=7)

    def test_timestep_is_shared_across_parameters(self):
        """t counts optimizer steps, not parameter visits."""
        layer = Dense(3, 2, seed=0)
        opt = Adam(lr=0.01)
        opt.step([layer])
        opt.step([layer])
        self.assertEqual(opt.t, 2)

    def test_scale_invariance(self):
        """
        Adam's step size is invariant to a constant rescaling of the gradient,
        up to the eps term in the denominator. With g=1e-4 that eps shifts the
        step by ~1e-4 relative, so the two agree to about 6 decimal places.
        """
        small = FakeParam([1.0], [1e-4])
        large = FakeParam([1.0], [1e4])
        Adam(lr=0.01).step([small])
        Adam(lr=0.01).step([large])
        self.assertAlmostEqual(float(small.p[0]), float(large.p[0]), places=5)


class TestOptimizerStateKeying(unittest.TestCase):
    """
    Optimizer state is keyed on each parameter's position in the network, not
    on id(param). These tests pin the properties that keying buys.
    """

    def test_state_is_keyed_by_position_not_identity(self):
        layer = Dense(3, 2, seed=0)
        layer.dW[...] = 1.0
        layer.db[...] = 1.0

        opt = SGDMomentum(lr=0.1, momentum=0.9)
        opt.step([layer])
        self.assertEqual(set(opt.velocities), {(0, 0), (0, 1)})

    def test_reinitializing_a_layer_does_not_corrupt_optimizer_state(self):
        """
        Dense.reset() allocates fresh W and b arrays. Under id() keying the new
        arrays could land on freed addresses and inherit another parameter's
        state - observed in practice as a new (1,2) bias picking up a (3,2)
        weight buffer. Positional keys stay matched, so the buffers keep the
        right shapes and the step stays well defined.
        """
        layer = Dense(3, 2, seed=0)
        layer.dW[...] = 1.0
        layer.db[...] = 1.0

        opt = SGDMomentum(lr=0.1, momentum=0.9)
        opt.step([layer])

        layer.reset()
        layer.dW[...] = 1.0
        layer.db[...] = 1.0
        opt.step([layer])  # must not raise

        self.assertEqual(set(opt.velocities), {(0, 0), (0, 1)})
        self.assertEqual(opt.velocities[(0, 0)].shape, layer.W.shape)
        self.assertEqual(opt.velocities[(0, 1)].shape, layer.b.shape)

    def test_state_does_not_grow_across_reinitializations(self):
        """Stale entries used to accumulate on every reset; they no longer do."""
        layer = Dense(4, 3, seed=0)
        opt = Adam(lr=0.01)
        for _ in range(10):
            layer.dW[...] = 1.0
            layer.db[...] = 1.0
            opt.step([layer])
            layer.reset()
        self.assertEqual(len(opt.m), 2)
        self.assertEqual(len(opt.v), 2)

    def test_distinct_modules_get_distinct_state(self):
        a, b = Dense(3, 2, seed=0), Dense(2, 2, seed=1)
        for layer in (a, b):
            layer.dW[...] = 1.0
            layer.db[...] = 1.0
        opt = SGDMomentum(lr=0.1, momentum=0.9)
        opt.step([a, b])
        self.assertEqual(set(opt.velocities), {(0, 0), (0, 1), (1, 0), (1, 1)})

    def test_model_reset_clears_optimizer_state(self):
        from nn.model import Model
        from nn.losses import MSE

        layer = Dense(3, 2, seed=0)
        model = Model(modules=[layer], loss=MSE(), optimizer=Adam(lr=0.01))
        layer.dW[...] = 1.0
        layer.db[...] = 1.0
        model.step()
        self.assertEqual(model.optimizer.t, 1)
        self.assertEqual(len(model.optimizer.m), 2)

        model.reset()
        self.assertEqual(model.optimizer.t, 0)
        self.assertEqual(len(model.optimizer.m), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
