from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RopeScalingConfig:
    rope_type: str = "default"
    factor: float = 1.0
    original_max_position_embeddings: int | None = None
    beta_fast: float = 32.0
    beta_slow: float = 1.0


@dataclass(slots=True)
class ModelConfig:
    name: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_local_experts: int
    num_shared_experts: int
    num_experts_per_tok: int
    max_position_embeddings: int
    target_max_position_embeddings: int
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 10000.0
    rope_scaling: RopeScalingConfig | None = None
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 1.0e-4
    capacity_factor_train: float = 1.25
    capacity_factor_eval: float = 1.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    use_bias: bool = False
    use_cache: bool = True

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads


@dataclass(slots=True)
class DatasetSourceConfig:
    name: str
    path: str
    split: str
    weight: float = 1.0
    text_field: str = "text"
    config: str | None = None
    sample_weight: float | None = None
    translation_pair: dict[str, Any] | None = None


@dataclass(slots=True)
class DataConfig:
    name: str
    datasets: list[DatasetSourceConfig]
    streaming: bool = True
    shuffle_buffer_size: int = 10000
    sequence_length: int = 2048
    min_document_tokens: int = 0


@dataclass(slots=True)
class TokenizerConfig:
    name: str
    vocab_size: int
    model_type: str
    special_tokens: list[str]
    datasets: list[DatasetSourceConfig]
    min_frequency: int = 2
    russian_merge_budget: int = 42000
    normalization: str = "NFC"
    lowercase: bool = False
    byte_fallback: bool = False
    whitespace_mode: str = "isolated"


@dataclass(slots=True)
class TrainingConfig:
    name: str
    seed: int
    device: str
    dtype: str
    global_batch_size_tokens: int
    micro_batch_size_sequences: int
    gradient_accumulation_steps: int
    num_train_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    grad_clip_norm: float
    log_every: int
    eval_every: int
    save_every: int
    checkpoint_dir: str
    bf16: bool = True
    use_tf32: bool = True
    activation_checkpointing: bool = True
    compile_model: bool = False


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_dataset(entry: dict[str, Any]) -> DatasetSourceConfig:
    return DatasetSourceConfig(**entry)


def load_model_config(path: str | Path) -> ModelConfig:
    raw = _load_yaml(path)
    rope_scaling = raw.pop("rope_scaling", None)
    return ModelConfig(
        **raw,
        rope_scaling=RopeScalingConfig(**rope_scaling) if rope_scaling else None,
    )


def load_data_config(path: str | Path) -> DataConfig:
    raw = _load_yaml(path)
    datasets = [_parse_dataset(item) for item in raw.pop("datasets")]
    return DataConfig(datasets=datasets, **raw)


def load_tokenizer_config(path: str | Path) -> TokenizerConfig:
    raw = _load_yaml(path)
    datasets = [_parse_dataset(item) for item in raw.pop("datasets")]
    return TokenizerConfig(datasets=datasets, **raw)


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig(**_load_yaml(path))

