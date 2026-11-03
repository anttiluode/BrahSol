"""V2 checks for plasticity, input remapping, counterfactuals and forgetting."""
import unittest
import numpy as np
from brahsol_v2 import ChangingMedium, AdaptiveRLS, paired_run, POLICIES

class TestDynamicMedium(unittest.TestCase):
    def test_reproducible(self):
        a,b=ChangingMedium(12),ChangingMedium(12)
        for k in range(70):
            if k==30: a.change_connectivity();b.change_connectivity()
            np.testing.assert_allclose(a.step(k%8),b.step(k%8))
            np.testing.assert_allclose(a.W,b.W)

    def test_plasticity_is_not_adaptation(self):
        a=ChangingMedium(5,plasticity=True,shock=False)
        b=ChangingMedium(5,plasticity=False,shock=False)
        for t in range(90):
            a.step(t%8);b.step(t%8)
        self.assertGreater(a.drift(),.07)
        self.assertEqual(b.drift(),0.)
        self.assertGreater(np.linalg.norm(a.W-b.W),.02)

    def test_shock_affects_same_ping(self):
        before=ChangingMedium(11,plasticity=False)
        after=before.clone()
        after.change_connectivity()
        self.assertTrue(after.shocked)
        self.assertGreater(after.change_magnitude,.2)
        self.assertGreater(np.linalg.norm(before.step(0)-after.step(0)),.05)

    def test_shock_independent_of_noise_rng(self):
        x=ChangingMedium(9)
        y=ChangingMedium(9)
        for t in range(40):
            x.step((t*3)%8)
            y.step(t%8)
        x.change_connectivity();y.change_connectivity()
        np.testing.assert_allclose(x.input,y.input)
        self.assertTrue(np.all(x.mask == y.mask))

    def test_reference_fork_is_not_mutated(self):
        m=ChangingMedium(6)
        state=m.h.copy()
        frozen=m.W.copy()
        response=m.clone().step(3)
        self.assertEqual(response.shape,(4,))
        np.testing.assert_allclose(m.h,state)
        np.testing.assert_allclose(m.W,frozen)

class TestTracking(unittest.TestCase):
    def test_forgetting_tracks_changed_return(self):
        x=np.array([0.,0.,0.,0.])
        fast=AdaptiveRLS(gamma=.94)
        slow=AdaptiveRLS(gamma=1.)
        for t in range(130):
            y=np.full(4,.15 if t<70 else .65)
            fast.update(x,0,y);slow.update(x,0,y)
        fast_error=np.linalg.norm(fast.predict(x,0)-.65)
        slow_error=np.linalg.norm(slow.predict(x,0)-.65)
        self.assertLess(fast_error,slow_error)

    def test_paired_run_evaluates_all_policies(self):
        rows,ev=paired_run(seed=3,steps=147,change_at=48,eval_every=12,
                           plasticity=True,shock=True)
        self.assertEqual(len(rows),len(POLICIES)*2*3)
        self.assertEqual(len(ev),len(POLICIES)*2)
        self.assertTrue(all(np.isfinite(r.ref_rmse) for r in rows))
        self.assertTrue(all(r.shock_size>0 for r in rows))
        self.assertTrue(all(any(t==48 for t,e in vs) for vs in ev.values()))

if __name__=="__main__":unittest.main()
