from __future__ import annotations

import math

import torch

from veridian.config import TrainingConfig


def build_optimizer(
    model: torch.nn.Module,
    train_config: TrainingConfig,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_epsilon,
        weight_decay=train_config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_config: TrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < train_config.warmup_steps:
            return float(step + 1) / max(1, train_config.warmup_steps)

        progress = (step - train_config.warmup_steps) / max(
            1, train_config.num_train_steps - train_config.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = train_config.min_learning_rate / train_config.learning_rate
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

