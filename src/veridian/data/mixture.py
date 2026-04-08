from __future__ import annotations

import random
from collections.abc import Iterator

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset

from veridian.config import DataConfig, DatasetSourceConfig


def format_record(source: DatasetSourceConfig, record: dict) -> str | None:
    if source.translation_pair:
        translation = record.get(source.text_field)
        if not isinstance(translation, dict):
            return None
        source_key = source.translation_pair["source_key"]
        target_key = source.translation_pair["target_key"]
        template = source.translation_pair["template"]
        src = translation.get(source_key)
        tgt = translation.get(target_key)
        if not src or not tgt:
            return None
        return template.format(source=src, target=tgt)

    value = record.get(source.text_field)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    return str(value).strip() or None


def _iter_source(source: DatasetSourceConfig, streaming: bool) -> Iterator[str]:
    dataset = load_dataset(
        path=source.path,
        name=source.config,
        split=source.split,
        streaming=streaming,
    )
    for record in dataset:
        text = format_record(source, record)
        if text:
            yield text


def _iter_source_records(source: DatasetSourceConfig, streaming: bool) -> Iterator[dict]:
    dataset = load_dataset(
        path=source.path,
        name=source.config,
        split=source.split,
        streaming=streaming,
    )
    for record in dataset:
        yield record


def weighted_record_iterator(
    sources: list[DatasetSourceConfig],
    streaming: bool,
    seed: int,
) -> Iterator[dict]:
    rng = random.Random(seed)
    iterators = [_iter_source_records(source, streaming) for source in sources]
    weights = [source.weight for source in sources]

    while True:
        index = rng.choices(range(len(sources)), weights=weights, k=1)[0]
        try:
            yield next(iterators[index])
        except StopIteration:
            iterators[index] = _iter_source_records(sources[index], streaming)
            yield next(iterators[index])


def weighted_text_iterator(
    sources: list[DatasetSourceConfig],
    streaming: bool,
    seed: int,
    weight_attr: str = "weight",
) -> Iterator[str]:
    rng = random.Random(seed)
    iterators = [_iter_source(source, streaming) for source in sources]
    weights = [getattr(source, weight_attr) or 1.0 for source in sources]

    while True:
        index = rng.choices(range(len(sources)), weights=weights, k=1)[0]
        try:
            yield next(iterators[index])
        except StopIteration:
            iterators[index] = _iter_source(sources[index], streaming)
            yield next(iterators[index])


def render_chat_sample(record: dict) -> str | None:
    if "messages" in record and isinstance(record["messages"], list):
        chunks: list[str] = ["<bos>"]
        for message in record["messages"]:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if not role or not content:
                continue
            role_token = {
                "system": "<|system|>",
                "user": "<|user|>",
                "assistant": "<|assistant|>",
                "tool": "<|tool|>",
            }.get(role)
            if role_token is None:
                continue
            chunks.append(f"{role_token}\n{content}\n<|end_of_turn|>")
        chunks.append("<eos>")
        return "\n".join(chunks)

    instruction = record.get("instruction") or record.get("prompt")
    response = (
        record.get("output")
        or record.get("response")
        or record.get("answer")
        or record.get("completion")
    )
    if instruction and response:
        system = record.get("system") or "Ты полезный русскоязычный ассистент."
        user = instruction
        if record.get("input"):
            user = f"{instruction}\n\n{record['input']}"
        return (
            "<bos>\n"
            f"<|system|>\n{system}\n<|end_of_turn|>\n"
            f"<|user|>\n{user}\n<|end_of_turn|>\n"
            f"<|assistant|>\n{response}\n<|end_of_turn|>\n"
            "<eos>"
        )

    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def build_assistant_loss_mask(token_ids: list[int], tokenizer: Tokenizer) -> list[int]:
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    end_turn_id = tokenizer.token_to_id("<|end_of_turn|>")
    eos_id = tokenizer.token_to_id("<eos>")
    boundary_ids = {
        token_id
        for token_id in (
            tokenizer.token_to_id("<|system|>"),
            tokenizer.token_to_id("<|user|>"),
            tokenizer.token_to_id("<|tool|>"),
            assistant_id,
            end_turn_id,
            eos_id,
        )
        if token_id is not None
    }
    if assistant_id is None:
        return [1] * len(token_ids)

    mask = [0] * len(token_ids)
    in_assistant = False
    for index, token_id in enumerate(token_ids):
        if token_id == assistant_id:
            in_assistant = True
            continue
        if in_assistant and token_id in boundary_ids:
            in_assistant = False
            continue
        if in_assistant:
            mask[index] = 1
    if sum(mask) == 0:
        return [1] * len(token_ids)
    return mask


class PackedCausalLMDataset(IterableDataset):
    def __init__(
        self,
        data_config: DataConfig,
        tokenizer_path: str,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.tokenizer_path = tokenizer_path
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        eos_token_id = tokenizer.token_to_id("<eos>")
        if eos_token_id is None:
            raise ValueError("Tokenizer must define <eos> token")

        buffer: list[int] = []
        text_iter = weighted_text_iterator(
            self.data_config.datasets,
            streaming=self.data_config.streaming,
            seed=self.seed,
        )

        for text in text_iter:
            token_ids = tokenizer.encode(text).ids
            if not token_ids:
                continue
            buffer.extend(token_ids)
            buffer.append(eos_token_id)

            while len(buffer) >= self.data_config.sequence_length + 1:
                window = buffer[: self.data_config.sequence_length + 1]
                del buffer[: self.data_config.sequence_length]
                input_ids = torch.tensor(window[:-1], dtype=torch.long)
                labels = torch.tensor(window[1:], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }


class PackedSFTDataset(IterableDataset):
    def __init__(
        self,
        data_config: DataConfig,
        tokenizer_path: str,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.tokenizer_path = tokenizer_path
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        tokenizer = Tokenizer.from_file(self.tokenizer_path)
        eos_token_id = tokenizer.token_to_id("<eos>")
        if eos_token_id is None:
            raise ValueError("Tokenizer must define <eos> token")

        record_iter = weighted_record_iterator(
            self.data_config.datasets,
            streaming=self.data_config.streaming,
            seed=self.seed,
        )
        token_buffer: list[int] = []
        loss_buffer: list[int] = []

        for record in record_iter:
            rendered = render_chat_sample(record)
            if rendered is None:
                continue

            ids = tokenizer.encode(rendered).ids
            if not ids:
                continue
            mask = build_assistant_loss_mask(ids, tokenizer)

            token_buffer.extend(ids + [eos_token_id])
            loss_buffer.extend(mask + [0])

            while len(token_buffer) >= self.data_config.sequence_length + 1:
                window_ids = token_buffer[: self.data_config.sequence_length + 1]
                window_mask = loss_buffer[: self.data_config.sequence_length + 1]
                del token_buffer[: self.data_config.sequence_length]
                del loss_buffer[: self.data_config.sequence_length]

                input_ids = torch.tensor(window_ids[:-1], dtype=torch.long)
                labels = torch.tensor(window_ids[1:], dtype=torch.long)
                loss_mask = torch.tensor(window_mask[1:], dtype=torch.bool)
                labels = labels.masked_fill(~loss_mask, -100)
                attention_mask = torch.ones_like(input_ids)
                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask,
                }
