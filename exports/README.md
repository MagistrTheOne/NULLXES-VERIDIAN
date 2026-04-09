---
license: mit
language:
  - ru
  - en
tags:
  - causal-lm
  - bilingual
  - moe
  - pytorch
  - experimental
pipeline_tag: text-generation
library_name: pytorch
---

# VERIDIAN beta-0

VERIDIAN is an experimental bilingual causal language model developed by NULLXES.

This repository contains an early beta checkpoint exported from the custom VERIDIAN training stack.

## Status

- Stage: early beta
- Languages: Russian, English
- Architecture: decoder-only sparse MoE
- Format: custom PyTorch checkpoint exported to `safetensors`

## Included files

- `veridian_beta_0.safetensors`: model weights
- `config.json`: Hugging Face style metadata config
- `tokenizer.json`: tokenizer
- `tokenizer_config.json`: tokenizer metadata
- `special_tokens_map.json`: special token definitions

## Limitations

- This is not yet a native Transformers-integrated model repo.
- Some names and brand tokens may be split poorly by the tokenizer.
- Output quality is still unstable across prompts and domains.

## Notes

This release is primarily intended as a beta artifact and research checkpoint.
