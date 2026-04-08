from __future__ import annotations

from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFC, Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import Digits, Punctuation, Sequence as PreTokenizerSequence, Split
from tokenizers.processors import TemplateProcessing

from veridian.config import ModelConfig, TokenizerConfig


def build_bpe_tokenizer(
    tokenizer_config: TokenizerConfig,
    model_config: ModelConfig | None = None,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    normalizers = [NFC()]
    if tokenizer_config.lowercase:
        normalizers.append(Lowercase())
    tokenizer.normalizer = NormalizerSequence(normalizers)

    tokenizer.pre_tokenizer = PreTokenizerSequence(
        [
            Split(Regex(r"(\r\n|\r|\n\n|\n|\t| {1,4})"), behavior="isolated"),
            Split(
                Regex(r"(==|!=|<=|>=|->|=>|::|&&|\|\||\*\*|//|/\*|\*/|```|`{1,3})"),
                behavior="isolated",
            ),
            Punctuation(behavior="isolated"),
            Digits(individual_digits=False),
        ]
    )

    if "<bos>" in tokenizer_config.special_tokens and "<eos>" in tokenizer_config.special_tokens:
        bos_id = tokenizer_config.special_tokens.index("<bos>")
        eos_id = tokenizer_config.special_tokens.index("<eos>")
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <eos> $B:1 <eos>:1",
            special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
        )

    if model_config is not None and model_config.vocab_size != tokenizer_config.vocab_size:
        tokenizer_config.vocab_size = model_config.vocab_size

    return tokenizer

