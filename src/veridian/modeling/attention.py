from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from veridian.config import ModelConfig
from veridian.modeling.rope import RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        bias = config.use_bias
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=bias,
        )
        self.rotary_emb = RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_ids is None:
            pos = torch.arange(seq_len, device=hidden_states.device)
        else:
            pos = position_ids[0] if position_ids.dim() == 2 else position_ids
        cos, sin = self.rotary_emb(seq_len, hidden_states.device, query_states.dtype, pos)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        min_value = torch.finfo(query_states.dtype).min
        causal_mask = torch.zeros(
            seq_len,
            seq_len,
            device=hidden_states.device,
            dtype=query_states.dtype,
        )
        causal_mask = causal_mask.masked_fill(
            torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool).triu(1),
            min_value,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            padding_mask = ~attention_mask[:, None, None, :].to(dtype=torch.bool)
            causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len).clone()
            causal_mask = causal_mask.masked_fill(padding_mask, min_value)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        return self.o_proj(attn_output)
