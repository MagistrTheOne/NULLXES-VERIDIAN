from __future__ import annotations

import math

import torch
from torch import nn

from veridian.config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(start_dim=-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.scaling_type = config.rope_scaling.rope_type if config.rope_scaling else "default"
        self.scaling_factor = config.rope_scaling.factor if config.rope_scaling else 1.0
        self.original_max_position_embeddings = (
            config.rope_scaling.original_max_position_embeddings
            if config.rope_scaling and config.rope_scaling.original_max_position_embeddings
            else config.max_position_embeddings
        )

    def _scaled_positions(
        self,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if self.scaling_type != "yarn" or self.scaling_factor <= 1.0:
            return positions, 1.0

        scale = self.scaling_factor
        mscale = 0.1 * math.log(scale) + 1.0

        if positions.numel() == 0:
            return positions, mscale

        threshold = float(self.original_max_position_embeddings)
        scaled = torch.where(
            positions <= threshold,
            positions,
            threshold + (positions - threshold) / scale,
        )
        return scaled, mscale

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
        else:
            position_ids = position_ids.to(device=device, dtype=torch.float32)

        scaled_positions, mscale = self._scaled_positions(position_ids)
        freqs = torch.einsum("i,j->ij", scaled_positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype) * mscale
        sin = emb.sin().to(dtype=dtype) * mscale
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k

