import unittest
import numpy as np
from brahsol_v3 import (N, MASKS, BINARY, initial_tm, tm_step,
  final_templates, diag_contrast, information, episode, replay_posterior,
  train_stop_threshold, gate_read_wipe)

class TestDynamics(unittest.TestCase):
    def test_all_tm_states_bounded(self):
        u,r=initial_tm()
        for i in range(40):
            u,r,_=tm_step(u,r,MASKS[i%len(MASKS)],dt=3)
        self.assertTrue(np.all(u>=0) and np.all(u<=1))
        self.assertTrue(np.all(r>=0) and np.all(r<=1))

    def test_linear_common_reads_preserve_class_contrast(self):
        c0=diag_contrast(final_templates('linear',0)[0])
        for k in [1,3,8,12]:
            self.assertAlmostEqual(diag_contrast(final_templates('linear',k)[0]),c0,places=11)

    def test_tm_reads_erase_distinguishability(self):
        c0=abs(diag_contrast(final_templates('TM',0)[0]))
        c12=abs(diag_contrast(final_templates('TM',12)[0]))
        self.assertLess(c12/c0,.05)

    def test_global_sum_identity_is_symmetry(self):
        u,r=initial_tm()
        _,_,x=tm_step(u,r,np.ones(N,bool),dt=3)
        self.assertLess(np.ptp(x.sum(axis=1)),1e-14)

    def test_fixed_masks_cover_three_binary_bits(self):
        addresses=[tuple(int(BINARY[t,j]) for t in range(3)) for j in range(N)]
        self.assertEqual(len(set(addresses)),8)
        self.assertTrue(np.all(BINARY.sum(axis=1)==4))

class TestInference(unittest.TestCase):
    def test_information_is_nonnegative(self):
        means=np.arange(N)[:,None]*np.ones((N,len(MASKS)))*.002
        prior=np.ones(N)/N
        res=information(means,prior,.012)
        self.assertTrue(np.all(res >= -1e-8))

    def test_efference_replay_exact(self):
        for seed in [4,8,19,31]:
            e=episode(seed,'random',5,.012)
            post=replay_posterior(e['masks'],e['observations'],.012,None)
            self.assertEqual(int(np.argmax(post)),e['prediction'])

    def test_history_ablation_differs(self):
        e=episode(77,'random',5,.012)
        full=replay_posterior(e['masks'],e['observations'],.012,None)
        abl=replay_posterior(e['masks'],e['observations'],.012,0)
        self.assertGreater(float(np.linalg.norm(full-abl)),.05)

    def test_observer_not_given_label(self):
        e=episode(6,'global',5,.012)
        self.assertEqual(e['prediction'],0)  # tie-break, not true identification
        # No hypothesis-dependent summed response in global mode.

    def test_stop_is_trained_separately(self):
        a,values=train_stop_threshold(range(5000,5010),'fixed',.012,5,.025)
        self.assertIn(a,values)
        self.assertTrue(0.5<=a<=1.)

    def test_result_reproducibility(self):
        a=episode(10,'active',5,.012)
        b=episode(10,'active',5,.012)
        self.assertEqual(a['prediction'],b['prediction'])
        self.assertEqual(a['reads'],b['reads'])
        self.assertEqual(a['observations'],b['observations'])

if __name__=='__main__':unittest.main()
