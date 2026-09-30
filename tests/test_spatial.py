import math
import unittest
import numpy as np
import pandas as pd
import torch
from models.spatial_model import SpatialFactorAE, LogisticPrior, distortion
from experiments.posthoc_spatial import nearest, held_assignments, code_bits, coding, counts
from utils.spatial_metrics import sequence_stats, Metrics
from experiments.diagnose_spatial import projectors


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_quantized_only_decode_and_gradients(self):
        torch.manual_seed(0)
        m = SpatialFactorAE(rank=4, spatial_dim=8, temporal_dim=8)
        x = torch.randn(2, 20, 256)
        o = m(x)
        (o["y"].square().mean()+o["rate"].mean()).backward()
        for name, param in m.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        m.eval()
        with torch.no_grad():
            o = m(x)
            self.assertTrue(torch.equal(o["zs"], o["zs"].round()))
            self.assertTrue(torch.equal(o["zt"], o["zt"].round()))
            a = m.decode_spatial(o["zs"])
            s = m.temporal_decoder(o["zt"])
            self.assertTrue(torch.allclose(o["y"], m.combine(a, s)))
            self.assertTrue(torch.allclose(a.norm(dim=2), torch.ones_like(a[:, :, 0]), atol=1e-5))
            self.assertTrue(torch.allclose(o["rate"]*x[0].numel(), o["bits_spatial"]+o["bits_temporal"]))

    def test_logistic_mass_and_extreme_gradient(self):
        p = LogisticPrior(1)
        z = torch.linspace(-8, 8, 101).reshape(1, 1, -1)
        scale = torch.nn.functional.softplus(p.log_scale)+.001
        expected = -torch.log2(torch.sigmoid((z+.5)/scale)-torch.sigmoid((z-.5)/scale))
        self.assertTrue(torch.allclose(p.bits(z), expected, atol=.02))
        extreme = torch.tensor([[[-1000., 1000.]]], requires_grad=True)
        p.bits(extreme).sum().backward()
        self.assertTrue(torch.isfinite(extreme.grad).all())
        self.assertTrue((extreme.grad.abs() > 0).all())

    def test_hold_assignment_matches_bruteforce(self):
        torch.manual_seed(1)
        b, c, k, j, l = 2, 3, 2, 4, 8
        s, x = torch.randn(b, k, j*l), torch.randn(b, c, j*l)
        d = torch.randn(5, c, k)
        labels = held_assignments(x, s, d, l, 2)
        for n in range(b):
            for block in range(j//2):
                sl = slice(block*2*l, (block+1)*2*l)
                costs = ((d@s[n, :, sl]-x[n, :, sl]).square()).sum((1, 2))
                self.assertEqual(int(labels[n, block]), int(costs.argmin()))
        mask = torch.rand_like(x) < .3
        labels = held_assignments(x, s, d, l, 2, mask)
        for n in range(b):
            for block in range(j//2):
                sl = slice(block*2*l, (block+1)*2*l)
                costs = ((d@s[n, :, sl]-x[n, :, sl]).square()*(~mask[n, :, sl])).sum((1, 2))
                self.assertEqual(int(labels[n, block]), int(costs.argmin()))

    def test_dictionary_identity_and_fixed_temporal(self):
        a, s = torch.randn(1, 4, 3, 2), torch.randn(1, 2, 16)
        d = a[0].clone()
        labels = nearest(a, d)
        self.assertTrue(torch.equal(labels, torch.arange(4)[None]))
        self.assertTrue(torch.equal(SpatialFactorAE.combine(a, s), SpatialFactorAE.combine(d[labels], s)))

    def test_subspace_rotation_invariance(self):
        torch.manual_seed(4)
        a = torch.randn(2, 5, 20, 4)
        rotation = torch.linalg.qr(torch.randn(4, 4))[0]
        before, rank = projectors(a)
        after, _ = projectors(a@rotation)
        self.assertTrue(torch.allclose(before, after, atol=2e-6))
        self.assertTrue((rank == 4).all())

    def test_markov_rate_and_boundary_reset(self):
        count = counts(4)
        labels = np.array([[0, 0, 1], [3, 3, 2]])
        sequence_stats(labels, None, None, count, [])
        self.assertEqual(count["transition"].sum(), 4)
        self.assertEqual(count["transition"][1, 3], 0)
        bits = code_bits(torch.tensor(labels), coding(counts(4), .5))
        self.assertTrue(torch.equal(bits, torch.full((2,), 6.)))
        self.assertEqual(float(code_bits(torch.zeros(1, 10, dtype=torch.long), coding(counts(1), .5))), 0.)

    def test_masked_loss_and_patient_aggregation(self):
        x, y = torch.ones(3, 2, 256), torch.ones(3, 2, 256)
        mask = torch.zeros_like(x, dtype=torch.bool)
        y[0, :, 0] = 100
        mask[0, :, 0] = True
        self.assertEqual(float(distortion(x[:1], y[:1], mask[:1])), 0.)
        # Two perfect recordings from patient A; one silence reconstruction from B.
        y[0] = 1; y[2] = 0
        batch = dict(mask=mask, sha256_id=["a1", "a2", "b1"], patient_id=["a", "a", "b"], start=[0, 0, 0])
        m = Metrics(256)
        m.add(batch, x, y, torch.zeros(3), torch.zeros(3))
        self.assertAlmostEqual(m.finish()["wave_nmse"], .5)


if __name__ == "__main__":
    unittest.main()
