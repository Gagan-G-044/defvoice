"""
The trainable back-end that sits on top of the frozen SSL encoder.

Kept in its own module because ml/train_cm_head.py and the inference path MUST
instantiate the identical architecture -- a silent shape mismatch here is a
state_dict load error at best and garbage scores at worst.

Deliberately small (~0.9M params for H=1024): the whole point of freezing the
encoder is that this fits comfortably on a 4GB laptop GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentiveStatsPool(nn.Module):
    """Mean + std pooling over time, weighted by a learned attention map.

    Better than plain mean pooling for spoofing cues, which are often localised
    to a few frames (onsets, plosives, unvoiced segments) rather than spread
    evenly across the utterance.
    """

    def __init__(self, in_dim: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_dim, bottleneck, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, in_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H) -> (B, 2H)
        h = x.transpose(1, 2)                       # (B, H, T)
        w = torch.softmax(self.attn(h), dim=2)
        mean = torch.sum(w * h, dim=2)
        var = torch.sum(w * h * h, dim=2) - mean * mean
        std = torch.sqrt(var.clamp(min=1e-8))
        return torch.cat([mean, std], dim=1)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_classes: int = 2,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = AttentiveStatsPool(in_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(2 * in_dim),
            nn.Linear(2 * in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, H) hidden states from the frozen encoder."""
        return self.net(self.pool(feats))


class WeightedLayerSum(nn.Module):
    """Optional: learn which encoder layer matters instead of hard-coding one.

    The blueprint pinned layers 5-6. That is a reasonable prior -- lower-middle
    layers carry more low-level artefact information than the top layers, which
    have specialised toward phonetic content -- but it is a guess. This module
    lets the data decide, at the cost of caching all layers instead of one.
    """

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_layers))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (L, B, T, H)
        w = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return torch.sum(w * hidden_states, dim=0)
