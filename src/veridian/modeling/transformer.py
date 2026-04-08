from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from veridian.config import ModelConfig
from veridian.modeling.attention import GroupedQueryAttention
from veridian.modeling.moe import MoEOutput, SparseTopKMoE
from veridian.modeling.rmsnorm import RMSNorm


@dataclass(slots=True)
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    z_loss: torch.Tensor | None = None


class VeridianDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GroupedQueryAttention(config)
        self.moe = SparseTopKMoE(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        checkpoint_attention: bool = False,
    ) -> tuple[torch.Tensor, MoEOutput]:
        if checkpoint_attention and self.training:
            def attention_forward(states: torch.Tensor) -> torch.Tensor:
                return self.self_attn(
                    self.input_layernorm(states),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )

            attn_output = checkpoint(attention_forward, hidden_states, use_reentrant=False)
        else:
            attn_output = self.self_attn(
                self.input_layernorm(hidden_states),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        hidden_states = hidden_states + self.residual_dropout(attn_output)
        moe_output = self.moe(self.post_attention_layernorm(hidden_states))
        hidden_states = hidden_states + self.residual_dropout(moe_output.hidden_states)
        return hidden_states, moe_output


class VeridianModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [VeridianDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            std = self.config.initializer_range
            if module.out_features == self.config.hidden_size:
                std = std / (2 * self.config.num_hidden_layers) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        aux_loss = hidden_states.new_zeros(())
        z_loss = hidden_states.new_zeros(())

        for layer in self.layers:
            hidden_states, moe_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                checkpoint_attention=self.gradient_checkpointing,
            )
            layer_aux = moe_output.aux_loss
            layer_z = moe_output.z_loss
            aux_loss = aux_loss + layer_aux.to(hidden_states.dtype)
            z_loss = z_loss + layer_z.to(hidden_states.dtype)

        hidden_states = self.norm(hidden_states)
        return hidden_states, aux_loss, z_loss


class VeridianForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = VeridianModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=config.initializer_range,
            )

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.model.set_gradient_checkpointing(enabled)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        hidden_states, aux_loss, z_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            loss = (
                ce_loss
                + self.config.router_aux_loss_coef * aux_loss
                + self.config.router_z_loss_coef * z_loss
            )

        return CausalLMOutput(logits=logits, loss=loss, aux_loss=aux_loss, z_loss=z_loss)
