import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from models.model import EEGRateDistortionAE, waveform_huber
from models.temporal_model import SlowFastTemporalAE, build_rate_model
from utils.experiment_utils import evaluate


class Windows(Dataset):
    n_channels = 2
    sfreq = 256

    def __init__(self):
        self.x = torch.randn(3, 2, 256)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": self.x[index]}


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.model = SlowFastTemporalAE(n_channels=2, slow_dim=2, fast_dim=3, width=8,
                                        channel_mode="independent")
        self.x = torch.randn(2, 2, 256)

    def test_decode_rate_and_residual_contract(self):
        self.model.eval()
        seen = []
        hook = self.model.fast.encoder.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0]))
        with torch.no_grad():
            out = self.model.forward_components(self.x)
            hook.remove()
            self.assertEqual(out["slow_latent"].shape, (2, 2, 2, 4))
            self.assertEqual(out["fast_latent"].shape, (2, 2, 3, 16))
            for name in ("slow_latent", "fast_latent"):
                self.assertTrue(torch.equal(out[name], out[name].round()))
            self.assertTrue(torch.allclose(seen[0].reshape_as(self.x), self.x - out["slow"]))
            decoded, slow, fast = self.model.decode(out["slow_latent"], out["fast_latent"])
            self.assertTrue(torch.equal(decoded, out["reconstruction"]))
            self.assertTrue(torch.equal(decoded, slow + fast))
            bits = sum(branch.prior.bits(out[name].flatten(0, 1)).sum()
                       for branch, name in ((self.model.slow, "slow_latent"), (self.model.fast, "fast_latent")))
            self.assertTrue(torch.allclose(out["rate"] * self.x.numel(), bits))

    def test_channel_independence(self):
        self.model.eval()
        with torch.no_grad():
            original = self.model(self.x)[0]
            changed = self.x.clone()
            changed[:, 1] *= 10
            self.assertTrue(torch.equal(original[:, 0], self.model(changed)[0][:, 0]))
            self.assertTrue(torch.allclose(original.flip(1), self.model(self.x.flip(1))[0]))

    def test_joint_default_mixing_decode_and_gradients(self):
        model = SlowFastTemporalAE(n_channels=20, slow_dim=2, fast_dim=3, width=8)
        self.assertEqual(model.channel_mode, "joint")
        self.assertEqual(model.slow.encoder[0].weight.shape, (8, 20, 7))
        self.assertEqual(model.fast.encoder[0].groups, 1)
        x = torch.randn(2, 20, 256)
        out = model.forward_components(x)
        (waveform_huber(x, out["reconstruction"]) + .01 * out["rate"]).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        model.eval()
        with torch.no_grad():
            out = model.forward_components(x)
            self.assertEqual(out["slow_latent"].shape, (2, 2, 4))
            self.assertEqual(out["fast_latent"].shape, (2, 3, 16))
            self.assertTrue(torch.equal(model.decode(out["slow_latent"], out["fast_latent"])[0],
                                        out["reconstruction"]))
            bits = (model.slow.prior.bits(out["slow_latent"]).sum()
                    + model.fast.prior.bits(out["fast_latent"]).sum())
            self.assertTrue(torch.allclose(out["rate"] * x.numel(), bits))
            changed = x.clone()
            changed[:, 1] += 10
            self.assertFalse(torch.allclose(model.slow.encoder(x), model.slow.encoder(changed)))
            self.assertFalse(torch.allclose(model(x, quantize=False)[0][:, 0],
                                            model(changed, quantize=False)[0][:, 0]))
        config = dict(architecture="slow-fast", n_channels=20, slow_dim=2, fast_dim=3,
                      width=8, channel_mode="joint")
        restored = build_rate_model(config).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(restored(x)[0], out["reconstruction"]))

    def test_training_gradients_and_continuous_mode(self):
        out = self.model.forward_components(self.x)
        loss = (waveform_huber(self.x, out["reconstruction"])
                + .25 * waveform_huber(self.x, out["slow"]) + .01 * out["rate"])
        loss.backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        for branch in (self.model.slow, self.model.fast):
            self.assertGreater(branch.encoder[-1].weight.grad.abs().sum().item(), 0)
            self.assertGreater(branch.prior.log_scale.grad.abs().sum().item(), 0)
        y, rate = self.model(self.x, quantize=False)
        self.assertEqual(y.shape, self.x.shape)
        self.assertIsNone(rate)

    def test_checkpoint_factory(self):
        config = dict(architecture="slow-fast", n_channels=2, slow_dim=2, fast_dim=3, width=8)
        restored = build_rate_model(config).eval()
        restored.load_state_dict(self.model.state_dict())
        self.model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(self.model(self.x)[0], restored(self.x)[0]))
        self.assertIsInstance(build_rate_model(dict(n_channels=2, latent_dim=4)), EEGRateDistortionAE)
        self.assertIn("architecture", config)

    def test_validation_accounting_with_partial_batch(self):
        dataset = Windows()
        result = evaluate(self.model, DataLoader(dataset, batch_size=2), 1., torch.device("cpu"))
        self.assertEqual(result["n_windows"], 3)
        self.assertAlmostEqual(result["estimated_bits_per_channel_sample"],
                               result["slow_estimated_bits_per_channel_sample"] +
                               result["fast_estimated_bits_per_channel_sample"], places=6)
        with torch.no_grad():
            out = self.model.forward_components(dataset.x)
        expected = ((dataset.x - out["slow"]).square().sum() / dataset.x.square().sum()).item()
        self.assertAlmostEqual(result["slow_nmse"], expected, places=5)
        self.assertAlmostEqual(result["fast_nmse_gain"], result["slow_nmse"] - result["nmse"])

    def test_invalid_settings(self):
        for config in (dict(slow_stride=16), dict(fast_stride=3), dict(width=7), dict(slow_dim=0)):
            with self.assertRaises(ValueError):
                SlowFastTemporalAE(**config)
        for x in (torch.randn(2, 3, 256), torch.randn(2, 2, 255), torch.randn(2, 2, 0)):
            with self.assertRaises(ValueError):
                self.model(x)


if __name__ == "__main__":
    unittest.main()
