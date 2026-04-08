from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from veridian.config import ModelConfig


class ExpertMLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        bias = config.use_bias
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


@dataclass(slots=True)
class MoEOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    z_loss: torch.Tensor
    router_logits: torch.Tensor


class SparseTopKMoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_local_experts
        self.num_shared_experts = config.num_shared_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertMLP(config) for _ in range(config.num_local_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [ExpertMLP(config) for _ in range(config.num_shared_experts)]
        )

    def _load_balance_loss(
        self,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        dispatch_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
        dispatch_mask = dispatch_mask.sum(dim=1) / float(self.top_k)
        tokens_per_expert = dispatch_mask.mean(dim=0)
        router_prob_per_expert = router_probs.mean(dim=0)
        return self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)

    def forward(self, hidden_states: torch.Tensor) -> MoEOutput:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_states = hidden_states.reshape(batch_size * seq_len, hidden_size)

        router_logits = self.router(flat_states)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(flat_states.dtype)
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        routed_output = torch.zeros_like(flat_states)

        for expert_index, expert in enumerate(self.experts):
            token_positions, expert_slot = torch.where(topk_indices == expert_index)
            if token_positions.numel() == 0:
                continue

            expert_input = flat_states.index_select(0, token_positions)
            expert_output = expert(expert_input)
            gate = topk_weights[token_positions, expert_slot].unsqueeze(-1)
            routed_output.index_add_(0, token_positions, expert_output * gate)

        if self.shared_experts:
            shared_output = torch.stack(
                [expert(flat_states) for expert in self.shared_experts],
                dim=0,
            ).mean(dim=0)
            routed_output = routed_output + shared_output

        aux_loss = self._load_balance_loss(router_probs.float(), topk_indices)
        z_loss = torch.mean(torch.logsumexp(router_logits.float(), dim=-1).pow(2))
        return MoEOutput(
            hidden_states=routed_output.view(batch_size, seq_len, hidden_size),
            aux_loss=aux_loss,
            z_loss=z_loss,
            router_logits=router_logits.view(batch_size, seq_len, self.num_experts),
        )

