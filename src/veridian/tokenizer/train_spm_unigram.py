from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path
from typing import Any

import sentencepiece as spm
from datasets import load_dataset
import yaml

from veridian.config import DatasetSourceConfig
from veridian.data.mixture import format_record, render_chat_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VERIDIAN SentencePiece Unigram tokenizer.")
    parser.add_argument("--tokenizer-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-documents", type=int, default=5_000_000)
    parser.add_argument("--corpus-output", default=None)
    return parser.parse_args()


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _source_to_text(source: DatasetSourceConfig, record: dict[str, Any]) -> str | None:
    if source.text_field == "messages":
        return render_chat_sample(record)
    return format_record(source, record)


def _iter_source_records(source: DatasetSourceConfig):
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


def _weighted_text_iterator(raw_config: dict[str, Any], seed: int):
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


def build_corpus(config_path: str, seed: int, max_documents: int, output_path: str | Path) -> Path:
    raw_config = _load_yaml(config_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text_iter = _weighted_text_iterator(raw_config, seed)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        brand_cfg = raw_config.get("brand_injection", {})
        brand_lines = brand_cfg.get("lines", [])
        brand_repeats = int(brand_cfg.get("repeats", 0))
        for _ in range(brand_repeats):
            for line in brand_lines:
                handle.write(line.strip())
                handle.write("\n")

        while written < max_documents:
            text = next(text_iter)
            if not text:
                continue
            handle.write(text.replace("\r\n", "\n").replace("\r", "\n"))
            handle.write("\n")
            written += 1

    return output_path


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.tokenizer_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = output_dir / "tokenizer"

    if args.corpus_output:
        corpus_path = Path(args.corpus_output)
        build_corpus(args.tokenizer_config, args.seed, args.max_documents, corpus_path)
    else:
        with tempfile.TemporaryDirectory(prefix="veridian_spm_") as tmp_dir:
            corpus_path = Path(tmp_dir) / "corpus.txt"
            build_corpus(args.tokenizer_config, args.seed, args.max_documents, corpus_path)

            spm.SentencePieceTrainer.train(
                input=str(corpus_path),
                model_prefix=str(model_prefix),
                model_type=config["model_type"],
                vocab_size=int(config["vocab_size"]),
                character_coverage=float(config["character_coverage"]),
                normalization_rule_name=config.get("normalization_rule_name", "nfkc"),
                shuffle_input_sentence=bool(config.get("shuffle_input_sentence", True)),
                input_sentence_size=int(config.get("input_sentence_size", 0)),
                seed_sentencepiece_size=int(config.get("seed_sentencepiece_size", 1_000_000)),
                max_sentence_length=int(config.get("max_sentence_length", 16384)),
                num_threads=int(config.get("num_threads", 8)),
                train_extremely_large_corpus=bool(config.get("train_extremely_large_corpus", False)),
                byte_fallback=bool(config.get("byte_fallback", False)),
                user_defined_symbols=config.get("user_defined_symbols", []),
                unk_id=int(config.get("unk_id", 0)),
                bos_id=int(config.get("bos_id", 1)),
                eos_id=int(config.get("eos_id", 2)),
                pad_id=int(config.get("pad_id", 3)),
                unk_piece=config.get("unk_piece", "<unk>"),
                bos_piece=config.get("bos_piece", "<bos>"),
                eos_piece=config.get("eos_piece", "<eos>"),
                pad_piece=config.get("pad_piece", "<pad>"),
            )
            return

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        model_type=config["model_type"],
        vocab_size=int(config["vocab_size"]),
        character_coverage=float(config["character_coverage"]),
        normalization_rule_name=config.get("normalization_rule_name", "nfkc"),
        shuffle_input_sentence=bool(config.get("shuffle_input_sentence", True)),
        input_sentence_size=int(config.get("input_sentence_size", 0)),
        seed_sentencepiece_size=int(config.get("seed_sentencepiece_size", 1_000_000)),
        max_sentence_length=int(config.get("max_sentence_length", 16384)),
        num_threads=int(config.get("num_threads", 8)),
        train_extremely_large_corpus=bool(config.get("train_extremely_large_corpus", False)),
        byte_fallback=bool(config.get("byte_fallback", False)),
        user_defined_symbols=config.get("user_defined_symbols", []),
        unk_id=int(config.get("unk_id", 0)),
        bos_id=int(config.get("bos_id", 1)),
        eos_id=int(config.get("eos_id", 2)),
        pad_id=int(config.get("pad_id", 3)),
        unk_piece=config.get("unk_piece", "<unk>"),
        bos_piece=config.get("bos_piece", "<bos>"),
        eos_piece=config.get("eos_piece", "<eos>"),
        pad_piece=config.get("pad_piece", "<pad>"),
    )


if __name__ == "__main__":
    main()
