"""Two-level residual temporal compression with configurable channel mixing."""
from __future__ import annotations

import torch
from torch import nn

from models.model import EEGRateDistortionAE, ResidualBlock
from models.spatial_model import LogisticPrior, quantize


class TemporalBranch(nn.Module):
    def __init__(self, latent_dim: int, stride: int, width: int, channels: int = 1):
        super().__init__()
        encoder = [nn.Conv1d(channels, width, 7, padding=3)]
        for _ in range(stride.bit_length() - 1):
            encoder.extend([nn.Conv1d(width, width, 5, stride=2, padding=2),
                            ResidualBlock(width)])
        encoder.extend([ResidualBlock(width, dilation=2), nn.Conv1d(width, latent_dim, 1)])
        self.encoder = nn.Sequential(*encoder)
        decoder = [nn.Conv1d(latent_dim, width, 1)]
        for _ in range(stride.bit_length() - 1):
            decoder.extend([ResidualBlock(width),
                            nn.ConvTranspose1d(width, width, 4, stride=2, padding=1)])
        decoder.extend([nn.SiLU(), nn.Conv1d(width, channels, 7, padding=3)])
        self.decoder = nn.Sequential(*decoder)
        self.prior = LogisticPrior(latent_dim)


class SlowFastTemporalAE(nn.Module):
    """Slow prediction plus a fast-coded residual; joint EEG channels by default.

    Joint latents are [B, D, L]; independent latents are [B, C, D, L].
    Decoding needs only the two transmitted latents, never the original signal.
    Slow/fast describe update rates, not guaranteed spectral disentanglement.
    The model is offline/noncausal (symmetric convolutions and GroupNorm).
    """
    def __init__(self, n_channels=20, slow_dim=4, fast_dim=4,
                 slow_stride=64, fast_stride=16, width=32, channel_mode="joint"):
        super().__init__()
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (n_channels, slow_dim, fast_dim, slow_stride, fast_stride, width)):
            raise ValueError("Channels, dimensions, strides and width must be positive integers")
        if width % 8:
            raise ValueError("width must be divisible by 8 for GroupNorm")
        if any(s < 2 or s & (s - 1) for s in (slow_stride, fast_stride)):
            raise ValueError("Strides must be powers of two, at least 2")
        if slow_stride <= fast_stride:
            raise ValueError("slow_stride must exceed fast_stride")
        if channel_mode not in ("joint", "independent"):
            raise ValueError("channel_mode must be joint or independent")
        self.channel_mode = channel_mode
        self.n_channels = n_channels
        self.slow_stride, self.fast_stride = slow_stride, fast_stride
        self.slow_dim, self.fast_dim = slow_dim, fast_dim
        channels = n_channels if channel_mode == "joint" else 1
        self.slow = TemporalBranch(slow_dim, slow_stride, width, channels)
        self.fast = TemporalBranch(fast_dim, fast_stride, width, channels)

    def decode(self, slow_latent, fast_latent):
        """Reconstruct from mode-specific latents; zero fast codes are not an ablation.

        For a slow-only ablation use the returned slow waveform directly,
        since the fast decoder may have nonzero biases.
        """
        if self.channel_mode == "joint":
            if (slow_latent.ndim != 3 or fast_latent.ndim != 3
                    or slow_latent.shape[0] != fast_latent.shape[0]
                    or slow_latent.shape[1] != self.slow_dim
                    or fast_latent.shape[1] != self.fast_dim
                    or slow_latent.shape[-1] * self.slow_stride
                    != fast_latent.shape[-1] * self.fast_stride):
                raise ValueError("Incompatible joint slow and fast latent shapes")
            slow = self.slow.decoder(slow_latent)
            fast = self.fast.decoder(fast_latent)
            return slow + fast, slow, fast
        if (slow_latent.ndim != 4 or fast_latent.ndim != 4
                or slow_latent.shape[:2] != fast_latent.shape[:2]
                or slow_latent.shape[1:3] != (self.n_channels, self.slow_dim)
                or fast_latent.shape[1:3] != (self.n_channels, self.fast_dim)
                or slow_latent.shape[-1] * self.slow_stride
                != fast_latent.shape[-1] * self.fast_stride):
            raise ValueError("Incompatible slow and fast latent shapes")
        b, c = slow_latent.shape[:2]
        slow = self.slow.decoder(slow_latent.flatten(0, 1)).reshape(b, c, -1)
        fast = self.fast.decoder(fast_latent.flatten(0, 1)).reshape(b, c, -1)
        return slow + fast, slow, fast

    def forward_components(self, x, quantized=True):
        if (x.ndim != 3 or x.shape[1] != self.n_channels or x.shape[0] < 1
                or x.shape[-1] < self.slow_stride or x.shape[-1] % self.slow_stride):
            raise ValueError("Expected nonempty [B, channels, T], T divisible by slow_stride")
        b, c, t = x.shape
        flat = x if self.channel_mode == "joint" else x.reshape(b * c, 1, t)
        zs = self.slow.encoder(flat)
        zs = quantize(zs, self.training) if quantized else zs
        slow = self.slow.decoder(zs)
        # Both training and inference form the residual from the decoded code.
        # Gradients remain end-to-end through the slow prediction.
        residual = flat - slow
        zf = self.fast.encoder(residual)
        zf = quantize(zf, self.training) if quantized else zf
        fast = self.fast.decoder(zf)
        if quantized:
            rs = self.slow.prior.bits(zs).sum() / x.numel()
            rf = self.fast.prior.bits(zf).sum() / x.numel()
        else:
            rs = rf = None
        return dict(reconstruction=(slow + fast).reshape(b, c, t),
                    slow=slow.reshape(b, c, t), fast=fast.reshape(b, c, t),
                    residual=residual.reshape(b, c, t),
                    slow_latent=zs if self.channel_mode == "joint" else zs.reshape(b, c, self.slow_dim, -1),
                    fast_latent=zf if self.channel_mode == "joint" else zf.reshape(b, c, self.fast_dim, -1),
                    slow_rate=rs, fast_rate=rf,
                    rate=rs + rf if quantized else None)

    def forward(self, x, quantize=True):
        output = self.forward_components(x, quantized=quantize)
        return output["reconstruction"], output["rate"]


def build_rate_model(model_config):
    """Load legacy single-stream checkpoints as well as new temporal models."""
    config = dict(model_config)
    architecture = config.pop("architecture", "baseline")
    if architecture == "baseline":
        return EEGRateDistortionAE(**config)
    if architecture == "slow-fast":
        # Initial slow-fast checkpoints predate this field and were independent.
        config.setdefault("channel_mode", "independent")
        return SlowFastTemporalAE(**config)
    raise ValueError(f"Unknown architecture: {architecture}")
