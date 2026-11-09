# Qwen3.8 chat-template reference

The compiled formatter is pinned to `Qwen/Qwen3.8-27B` revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
The official `Qwen/Qwen3.8-Flash-Next` template at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` has the same SHA-256.
The two pinned `tokenizer.json` files are also byte-identical.
Both default to `enable_thinking=true`, `reasoning_effort="xhigh"`, and
`preserve_thinking=true`. Native efforts are `low`, `medium`, and `xhigh`;
medium adds no effort instruction. Explicit thinking-off suppresses the
effort instruction and closes the generation prompt's thinking block.

- Official `chat_template.jinja` SHA-256:
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`
- Repository copy with its terminating LF SHA-256:
  `514d5304fcac63bda60e6faf860d5657010cf43cbc4c633c8bf678ee184e7a37`
- Official `tokenizer.json` SHA-256:
  `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`
- Recognized Unsloth Qwen3.8 GGUF template SHA-256:
  `12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`

`chat_template.jinja` is reference data only. Gufo validates recognized
artifact hashes and executes the bounded C++ implementation.

Full rendered-byte goldens cover tools before system instructions, the Unsloth
artifact's leading developer-message extension, tool-loop reasoning retention,
Unicode trimming around images and optional `Picture N:` numbering.
The reference rejects late system/developer messages; Gufo hoists them into
the leading system turn in their original order. Tool/system cases also compare
complete token sequences using the real GGUF tokenizer.

The tokenizer applies NFC normalization and the pinned Unicode split pattern,
including Unicode case-insensitive contractions, letter/mark runs and
trailing-whitespace backtracking. Ten raw-text
goldens check exact token IDs for spacing, code, accents, scripts with combining
marks, Unicode number categories and special-token boundaries on both models.

## Qwen3.6 dialect

`qwen36_chat_template.jinja` is the official `Qwen/Qwen3.6-35B-A3B` template
at revision `345d467b117d1fd5eda9b879a5988de9ae5f47a7` (SHA-256
`1b8361565cc35d47551d342214cff341c0f1a9685ce529c7ce31de0ed3616248`; the
repository copy adds a terminating LF). Unsloth's 35B template renders the same
bytes for every golden. Artifacts that are neither Qwen3.8 nor carry a pinned
Qwen3.8 template render this dialect, which differs from Qwen3.8 in two places:

- It has no reasoning-effort instruction, so no effort adds a system turn.
- An assistant message without `reasoning_content` may carry its thinking
  inline: the text before the first `</think>` (after any `<think>`) is the
  thought and the text after the last `</think>` is the answer.

Its template makes `preserve_thinking` opt-in. Gufo keeps earlier thinking by
default for both dialects: the model is trained to use it, and it keeps earlier
turns byte-stable for the continuation cache. `--preserve-thinking off` or
`chat_template_kwargs.preserve_thinking=false` restores the template default.
