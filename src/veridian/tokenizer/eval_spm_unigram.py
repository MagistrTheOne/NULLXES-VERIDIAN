from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


EXAMPLES = {
    "russian": "Веридиан строит multilingual токенайзер для русских и английских текстов.",
    "english": "VERIDIAN builds a multilingual tokenizer for technical and web-scale language data.",
    "mixed": "NULLXES-интеграция_v2 uses VERIDIAN for RU+EN assistant workflows.",
    "technical": "POST /api/v2/agents?model=VERIDIAN&source=NULLXES returned HTTP 200 in 184ms.",
    "code": "def build_router(name='VERIDIAN'):\n    return {'owner': 'NULLXES', 'name': name}",
}

BRAND_TOKENS = ["VERIDIAN", "NULLXES", "Веридиан"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VERIDIAN SentencePiece tokenizer.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def format_pieces(sp: spm.SentencePieceProcessor, text: str) -> tuple[list[str], list[int]]:
    pieces = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)
    return pieces, ids


def is_atomic(sp: spm.SentencePieceProcessor, token: str) -> bool:
    pieces = sp.encode(token, out_type=str)
    return len(pieces) == 1 and pieces[0].lstrip("▁") == token


def main() -> None:
    args = parse_args()
    sp = spm.SentencePieceProcessor(model_file=args.model)

    lines: list[str] = []
    token_counts = []

    lines.append(f"model: {args.model}")
    lines.append(f"vocab_size: {sp.vocab_size()}")
    lines.append("")
    lines.append("Brand token checks:")
    for token in BRAND_TOKENS:
        pieces = sp.encode(token, out_type=str)
        atomic = is_atomic(sp, token)
        lines.append(f"- {token}: atomic={atomic} pieces={pieces}")

    lines.append("")
    lines.append("Example segmentations:")
    for name, text in EXAMPLES.items():
        pieces, ids = format_pieces(sp, text)
        token_counts.append(len(pieces) / max(len(text.split()), 1))
        lines.append(f"[{name}]")
        lines.append(f"text: {text}")
        lines.append(f"pieces: {pieces}")
        lines.append(f"ids: {ids[:32]}{'...' if len(ids) > 32 else ''}")
        lines.append("")

    avg_token_length = sum(token_counts) / len(token_counts)
    lines.append(f"average_tokens_per_whitespace_word: {avg_token_length:.3f}")
    lines.append("")
    lines.append("Edge cases:")
    for text in [
        "VERIDIAN/NULLXES bridge",
        "NULLXES-интеграция_v2",
        "C++::router<=VERIDIAN",
        "Веридиан_v2_beta",
    ]:
        pieces = sp.encode(text, out_type=str)
        lines.append(f"- {text}: {pieces}")

    report = "\n".join(lines)
    print(report)

    if args.report_path:
        Path(args.report_path).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
