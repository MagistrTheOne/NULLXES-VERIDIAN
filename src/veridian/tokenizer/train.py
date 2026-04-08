from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers.trainers import BpeTrainer

from veridian.config import load_model_config, load_tokenizer_config
from veridian.data.mixture import weighted_text_iterator
from veridian.tokenizer.spec import build_bpe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VERIDIAN tokenizer from scratch.")
    parser.add_argument("--tokenizer-config", required=True)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-documents", type=int, default=5_000_000)
    return parser.parse_args()


def _text_iterator(config_path: str, seed: int, max_documents: int):
    tokenizer_config = load_tokenizer_config(config_path)
    iterator = weighted_text_iterator(
        tokenizer_config.datasets,
        streaming=True,
        seed=seed,
        weight_attr="sample_weight",
    )
    for index, text in enumerate(iterator):
        if index >= max_documents:
            break
        yield text


def main() -> None:
    args = parse_args()
    tokenizer_config = load_tokenizer_config(args.tokenizer_config)
    model_config = load_model_config(args.model_config) if args.model_config else None
    tokenizer = build_bpe_tokenizer(tokenizer_config, model_config=model_config)

    trainer = BpeTrainer(
        vocab_size=model_config.vocab_size if model_config else tokenizer_config.vocab_size,
        min_frequency=tokenizer_config.min_frequency,
        special_tokens=tokenizer_config.special_tokens,
        show_progress=True,
        continuing_subword_prefix="",
        end_of_word_suffix="",
    )
    tokenizer.train_from_iterator(
        _text_iterator(args.tokenizer_config, args.seed, args.max_documents),
        trainer=trainer,
        length=args.max_documents,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_dir / "tokenizer.json"))


if __name__ == "__main__":
    main()

