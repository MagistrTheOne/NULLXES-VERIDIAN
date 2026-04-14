from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
import yaml

from veridian.config import DatasetSourceConfig
from veridian.data.mixture import format_record, render_chat_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VERIDIAN byte-level BPE tokenizer.")
    parser.add_argument("--tokenizer-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--corpus-output", default=None)
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _source_to_text(source: DatasetSourceConfig, record: dict[str, Any]) -> str | None:
    if source.text_field == "messages":
        return render_chat_sample(record)
    return format_record(source, record)


def _iter_source_records(source: DatasetSourceConfig) -> Iterator[str]:
    dataset = load_dataset(
        path=source.path,
        name=source.config,
        split=source.split,
        streaming=True,
    )
    for record in dataset:
        text = _source_to_text(source, record)
        if text:
            yield text


def _weighted_text_iterator(raw_config: dict[str, Any], seed: int) -> Iterator[str]:
    sources = [DatasetSourceConfig(**entry) for entry in raw_config["datasets"]]
    weights = [source.sample_weight or 1.0 for source in sources]
    iterators = [_iter_source_records(source) for source in sources]
    rng = random.Random(seed)

    while True:
        index = rng.choices(range(len(sources)), weights=weights, k=1)[0]
        try:
            yield next(iterators[index])
        except StopIteration:
            iterators[index] = _iter_source_records(sources[index])
            yield next(iterators[index])


def _iter_synthetic_lines(raw_config: dict[str, Any]) -> Iterator[str]:
    for injection in raw_config.get("synthetic_injections", []):
        repeats = int(injection.get("repeats", 0))
        for _ in range(repeats):
            for line in injection.get("lines", []):
                yield line


def training_corpus(raw_config: dict[str, Any], seed: int, max_documents: int | None) -> Iterator[str]:
    for line in _iter_synthetic_lines(raw_config):
        yield line

    written = 0
    for text in _weighted_text_iterator(raw_config, seed):
        if not text:
            continue
        yield text.replace("\r\n", "\n").replace("\r", "\n")
        written += 1
        if max_documents is not None and written >= max_documents:
            break


def dump_corpus(raw_config: dict[str, Any], seed: int, max_documents: int | None, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in training_corpus(raw_config, seed, max_documents):
            handle.write(line)
            handle.write("\n")
    return output_path


def main() -> None:
    args = parse_args()
    raw_config = _load_yaml(args.tokenizer_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_documents = args.max_documents
    if max_documents is None:
        max_documents = raw_config.get("max_documents")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=bool(raw_config.get("digit_split", True))),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = list(raw_config.get("special_tokens", []))
    forced_tokens = list(raw_config.get("forced_tokens", []))
    trainer = trainers.BpeTrainer(
        vocab_size=int(raw_config["vocab_size"]),
        min_frequency=int(raw_config.get("min_frequency", 2)),
        show_progress=True,
        special_tokens=special_tokens + forced_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=int(raw_config.get("max_token_length", 64)),
    )

    corpus_iter = training_corpus(raw_config, args.seed, max_documents)
    tokenizer.train_from_iterator(corpus_iter, trainer=trainer)

    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokenizer.model.save(str(output_dir), "tokenizer")

    if args.corpus_output:
        dump_corpus(raw_config, args.seed, max_documents, args.corpus_output)

    print(f"saved tokenizer.json -> {tokenizer_path}")
    print(f"saved vocab/merges -> {output_dir}")


if __name__ == "__main__":
    main()
