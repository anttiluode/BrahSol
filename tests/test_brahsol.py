"""Run: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from brahsol import ACTIONS, NeuralMedium, ReturnModel, run, summarize


class TestMedium(unittest.TestCase):
    def test_seed_reproducible(self):
        a, b = NeuralMedium(10), NeuralMedium(10)
        for action in [0, 4, 1, 7, 3, 3]:
            np.testing.assert_allclose(a.step(action), b.step(action))
        np.testing.assert_allclose(a.W, b.W)

    def test_plasticity_switch(self):
        dynamic, frozen = NeuralMedium(4, True), NeuralMedium(4, False)
        for i in range(80):
            dynamic.step(i % ACTIONS)
            frozen.step(i % ACTIONS)
        self.assertGreater(dynamic.drift(), 0)
        self.assertEqual(frozen.drift(), 0)

    def test_ports_differ(self):
        a = NeuralMedium(5)
        b = a.clone()
        self.assertGreater(np.linalg.norm(a.step(0)-b.step(4)), 0.01)

    def test_sensor_modes(self):
        self.assertEqual(NeuralMedium(1, sensor_mode="sum").step(0).shape, (1,))
        self.assertEqual(NeuralMedium(1, sensor_mode="four").step(0).shape, (4,))


class TestModel(unittest.TestCase):
    def test_rls_learns_response(self):
        m = ReturnModel(1)
        for v in np.linspace(-1, 1, 120):
            m.update(np.array([v]), 2, np.array([0.3+0.4*v]))
        estimate, uncertainty = m.predict(np.array([0.25]), 2)
        self.assertAlmostEqual(float(estimate[0]), 0.4, delta=0.02)
        self.assertGreaterEqual(uncertainty, 0)
        self.assertEqual(m.counts[2], 120)

    def test_finite_run(self):
        result = run(seed=2, policy="active", train_steps=80, test_steps=30)
        for name in ("test_rmse", "persistence_rmse", "relative_skill", "train_homeo_mse"):
            self.assertTrue(np.isfinite(getattr(result, name)), name)
        self.assertEqual(result.action_coverage, 8)
        self.assertEqual(len(summarize([result])), 1)

    def test_breaking_return_pairing_hurts_prediction(self):
        intact = run(seed=2, policy="active", train_steps=140, test_steps=60)
        shuffled = run(seed=2, policy="active", train_steps=140, test_steps=60,
                       shuffled_returns=True)
        self.assertLess(intact.test_rmse, shuffled.test_rmse)

    def test_balanced_policy_cover(self):
        from brahsol import Controller
        c = Controller("balanced", 4, 19)
        chosen = [c.choose(np.zeros(4), i) for i in range(ACTIONS*3)]
        for i in range(0, len(chosen), ACTIONS):
            self.assertEqual(set(chosen[i:i+ACTIONS]), set(range(ACTIONS)))


if __name__ == "__main__":
    unittest.main()
