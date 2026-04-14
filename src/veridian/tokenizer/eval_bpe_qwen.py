from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer


EXAMPLES = {
    "russian": "Веридиан строит устойчивый токенайзер для русских технических и естественных текстов в 2026 году.",
    "english": "VERIDIAN builds a robust tokenizer for multilingual technical and production text in 2026.",
    "mixed": "NULLXES-интеграция_v2 uses VERIDIAN for RU+EN assistant workflows with api_response.status_code=200.",
    "code": "def route_user(user_id=8472): return {'model': 'VERIDIAN', 'owner': 'NULLXES'}",
    "json": "{\"service\":\"VERIDIAN\",\"status\":\"ok\",\"latency_ms\":184,\"owner\":\"NULLXES\"}",
    "brand": "NULLXES VERIDIAN Веридиан NULLXES_AI ARACHNE-X",
}

BRAND_TOKENS = ["NULLXES", "VERIDIAN", "Веридиан", "NULLXES_AI", "ARACHNE-X"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VERIDIAN byte-level BPE tokenizer.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def encode(tokenizer: Tokenizer, text: str):
    encoded = tokenizer.encode(text)
    return encoded.tokens, encoded.ids


def is_atomic(tokenizer: Tokenizer, token: str) -> bool:
    pieces, _ = encode(tokenizer, token)
    return len(pieces) == 1


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)

    lines: list[str] = []
    chars_per_token: list[float] = []

    vocab_size = tokenizer.get_vocab_size()
    lines.append(f"tokenizer: {args.tokenizer}")
    lines.append(f"vocab_size: {vocab_size}")
    lines.append("")
    lines.append("Brand token checks:")
    for token in BRAND_TOKENS:
        pieces, ids = encode(tokenizer, token)
        lines.append(f"- {token}: atomic={is_atomic(tokenizer, token)} pieces={pieces} ids={ids}")

    lines.append("")
    pieces_2026, _ = encode(tokenizer, "2026")
    lines.append(f"digit_split_2026: {pieces_2026}")

    lines.append("")
    lines.append("Example segmentations:")
    for name, text in EXAMPLES.items():
        pieces, ids = encode(tokenizer, text)
        chars_per_token.append(len(text) / max(len(pieces), 1))
        lines.append(f"[{name}]")
        lines.append(f"text: {text}")
        lines.append(f"pieces: {pieces}")
        lines.append(f"ids: {ids[:48]}{'...' if len(ids) > 48 else ''}")
        lines.append("")

    avg_chars_per_token = sum(chars_per_token) / max(len(chars_per_token), 1)
    lines.append(f"average_chars_per_token: {avg_chars_per_token:.3f}")
    lines.append("")
    lines.append("Edge cases:")
    for text in [
        "NULLXES-интеграция_v2",
        "user_id=8472_session_active",
        "api_response.status_code=200",
        "<div data-model=\"VERIDIAN\">NULLXES</div>",
        "ARACHNE-X routed request_id=20260414",
    ]:
        pieces, _ = encode(tokenizer, text)
        lines.append(f"- {text}: {pieces}")

    report = "\n".join(lines)
    print(report)

    if args.report_path:
        Path(args.report_path).write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
