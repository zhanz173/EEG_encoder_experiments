"""Small joint-channel 1D ResNet autoencoder with an estimated rate bottleneck."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FactorizedLogisticPrior(nn.Module):
    """Learned per-channel logistic prior for unit-width quantization bins."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.location = nn.Parameter(torch.zeros(1, channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1))

    def bits(self, z: torch.Tensor) -> torch.Tensor:
        scale = F.softplus(self.log_scale) + 1e-3
        upper = torch.sigmoid((z + 0.5 - self.location) / scale)
        lower = torch.sigmoid((z - 0.5 - self.location) / scale)
        probability = (upper - lower).clamp_min(1e-9)
        return -torch.log2(probability)


class EEGRateDistortionAE(nn.Module):
    """16x temporal downsampling, joint 20-channel input, ~4.5 s encoder context.

    Four dilated bottleneck blocks (dilations 1, 2, 4, 8) provide a
    theoretical encoder receptive field of 1,147 samples at 256 Hz.
    """

    def __init__(self, n_channels: int = 20, latent_dim: int = 64) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive")
        widths = (32, 48, 64, 96, 128)
        encoder = [nn.Conv1d(n_channels, widths[0], 7, padding=3)]
        for in_width, out_width in zip(widths[:-1], widths[1:]):
            encoder.extend(
                [
                    nn.Conv1d(in_width, out_width, 5, stride=2, padding=2),
                    ResidualBlock(out_width),
                ]
            )
        encoder.extend(ResidualBlock(widths[-1], dilation=d) for d in (1, 2, 4, 8))
        encoder.append(nn.Conv1d(widths[-1], latent_dim, 1))
        self.encoder = nn.Sequential(*encoder)

        decoder = [nn.Conv1d(latent_dim, widths[-1], 1)]
        for in_width, out_width in zip(widths[:0:-1], widths[-2::-1]):
            decoder.extend(
                [
                    ResidualBlock(in_width),
                    nn.ConvTranspose1d(in_width, out_width, 4, stride=2, padding=1),
                ]
            )
        decoder.extend([ResidualBlock(widths[0]), nn.GroupNorm(8, widths[0]), nn.SiLU()])
        decoder.append(nn.Conv1d(widths[0], n_channels, 7, padding=3))
        self.decoder = nn.Sequential(*decoder)
        self.prior = FactorizedLogisticPrior(latent_dim)
        self.n_channels = n_channels
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor, quantize: bool = True):
        if x.ndim != 3 or x.shape[1] != self.n_channels:
            raise ValueError(f"Expected [batch, {self.n_channels}, time]")
        if x.shape[-1] % 16:
            raise ValueError("Window length must be divisible by 16")
        z = self.encoder(x)
        if quantize:
            if self.training:
                z_hat = z + torch.empty_like(z).uniform_(-0.5, 0.5)
            else:
                z_hat = torch.round(z)
            bits = self.prior.bits(z_hat)
            rate = bits.sum() / (x.shape[0] * x.shape[1] * x.shape[2])
        else:
            z_hat = z
            rate = None
        reconstruction = self.decoder(z_hat)
        return reconstruction, rate


class EEGContinuousAE(EEGRateDistortionAE):
    """The same encoder/decoder with a real-valued latent and no rate model."""

    def __init__(self, n_channels: int = 20, latent_dim: int = 64) -> None:
        super().__init__(n_channels=n_channels, latent_dim=latent_dim)
        del self.prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstruction, _ = super().forward(x, quantize=False)
        return reconstruction

def waveform_nmse(x: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    error = (x - reconstruction).square().sum(dim=dims)
    energy = x.square().sum(dim=dims).clamp_min(1e-8)
    return (error / energy).mean()


def waveform_huber(x: torch.Tensor, reconstruction: torch.Tensor,
                   delta: float = 1.0) -> torch.Tensor:
    """Mean per-window Huber distortion in units of each input window's RMS.

    Multiplying by two makes the quadratic region equal per-window NMSE.
    Only the loss residual is normalized; the encoder still sees the original
    fixed-scale waveform and retains its absolute amplitude information.
    """
    if x.ndim != 3 or x.shape != reconstruction.shape:
        raise ValueError("Input and reconstruction must have matching [batch, channel, time] shapes")
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("Huber delta must be positive and finite")
    dims = tuple(range(1, x.ndim))
    energy = x.square().sum(dim=dims, keepdim=True).clamp_min(1e-8)
    rms = (energy / x[0].numel()).sqrt()
    residual = (reconstruction - x) / rms
    absolute = residual.abs()
    quadratic = absolute.clamp(max=delta)
    per_element = 0.5 * quadratic.square() + delta * (absolute - quadratic)
    return 2 * per_element.mean(dim=dims).mean()
