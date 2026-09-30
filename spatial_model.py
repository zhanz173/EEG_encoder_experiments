"""Quantized bilinear EEG factorization and a compatible single-stream control."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from model import EEGRateDistortionAE


class LogisticPrior(nn.Module):
    """Stable discrete logistic likelihood; entropy arithmetic always uses float32."""
    def __init__(self, channels):
        super().__init__()
        self.location = nn.Parameter(torch.zeros(1, channels, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1))

    def bits(self, z):
        with torch.autocast(device_type=z.device.type, enabled=False):
            scale = F.softplus(self.log_scale.float()) + 1e-3
            upper = (z.float() + .5 - self.location.float()) / scale
            lower = (z.float() - .5 - self.location.float()) / scale
            # sigmoid(u)-sigmoid(l) = sigmoid(u)*sigmoid(-l)*(1-exp(l-u))
            log_mass = F.logsigmoid(upper) + F.logsigmoid(-lower)
            log_mass = log_mass + torch.log(-torch.expm1(-1.0 / scale))
            return -log_mass / math.log(2)


def quantize(z, training):
    # Quantization precision is independent of convolution autocast precision.
    z = z.float()
    return z + torch.empty_like(z).uniform_(-.5, .5) if training else z.round()


class SpatialFactorAE(nn.Module):
    def __init__(self, n_channels=20, rank=8, spatial_dim=16, temporal_dim=32,
                 spatial_stride=16, **unused):
        super().__init__()
        if not (1 <= rank <= n_channels) or min(spatial_dim, temporal_dim, spatial_stride) < 1:
            raise ValueError("Invalid rank, latent dimension, or spatial stride")
        self.n_channels, self.rank, self.spatial_stride = n_channels, rank, spatial_stride
        temporal = EEGRateDistortionAE(n_channels, temporal_dim)
        self.temporal_encoder = temporal.encoder
        self.temporal_decoder = temporal.decoder
        self.temporal_decoder[-1] = nn.Conv1d(32, rank, 7, padding=3)
        self.spatial_encoder = nn.Sequential(nn.Linear(n_channels * spatial_stride, 128),
                                             nn.SiLU(), nn.Linear(128, spatial_dim))
        self.spatial_decoder = nn.Sequential(nn.Linear(spatial_dim, 128), nn.SiLU(),
                                             nn.Linear(128, n_channels * rank))
        self.spatial_prior = LogisticPrior(spatial_dim)
        self.temporal_prior = LogisticPrior(temporal_dim)

    def decode_spatial(self, z):
        b, _, j = z.shape
        a = self.spatial_decoder(z.transpose(1, 2)).reshape(b, j, self.n_channels, self.rank)
        return F.normalize(a.float(), dim=2, eps=1e-6)

    @staticmethod
    def combine(a, s):
        b, j, _, k = a.shape
        if s.shape[1] != k or s.shape[-1] % j:
            raise ValueError("Spatial frames and temporal samples are incompatible")
        blocks = s.reshape(b, k, j, -1)
        return torch.einsum("bjck,bkjl->bcjl", a, blocks).flatten(2)

    def forward(self, x):
        b, c, t = x.shape
        if c != self.n_channels or t % 16 or t % self.spatial_stride:
            raise ValueError("Input must have matching channels and time divisible by 16 and spatial stride")
        patches = x.unfold(-1, self.spatial_stride, self.spatial_stride)
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        zs = quantize(self.spatial_encoder(patches).transpose(1, 2), self.training)
        zt = quantize(self.temporal_encoder(x), self.training)
        a = self.decode_spatial(zs)
        s = self.temporal_decoder(zt).float()
        # Keep bilinear synthesis in float32 so evaluating a frozen dictionary
        # does not change numerical precision relative to the original model.
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.combine(a, s)
        bs = self.spatial_prior.bits(zs).sum((1, 2))
        bt = self.temporal_prior.bits(zt).sum((1, 2))
        return dict(y=y, a=a, s=s, zs=zs, zt=zt, bits_spatial=bs, bits_temporal=bt,
                    rate=(bs + bt) / (c * t))


class SingleStreamAE(nn.Module):
    def __init__(self, n_channels=20, latent_dim=64, **unused):
        super().__init__()
        base = EEGRateDistortionAE(n_channels, latent_dim)
        self.encoder, self.decoder = base.encoder, base.decoder
        self.prior = LogisticPrior(latent_dim)
        self.n_channels = n_channels

    def forward(self, x):
        z = quantize(self.encoder(x), self.training)
        y = self.decoder(z).float()
        bits = self.prior.bits(z).sum((1, 2))
        return dict(y=y, zt=z, bits_spatial=torch.zeros_like(bits), bits_temporal=bits,
                    rate=bits / x[0].numel())


def make_model(config):
    return (SpatialFactorAE if config["architecture"] == "factorized" else SingleStreamAE)(**config)


def distortion(x, y, mask, kind="nmse", delta=1.0):
    x, y = x.float(), y.float()
    keep = ~mask
    energy = (x.square() * keep).sum((1, 2)).clamp_min(1e-12)
    count = keep.sum((1, 2)).clamp_min(1)
    if kind == "nmse":
        return ((x-y).square() * keep).sum((1, 2)) / energy
    rms = (energy / count).sqrt()[:, None, None]
    r = ((x-y) / rms).abs()
    q = r.clamp(max=delta)
    return (2 * (.5*q.square() + delta*(r-q)) * keep).sum((1, 2)) / count
