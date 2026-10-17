# Prompt corpus: notice and checksum

`bench_corpus.txt` is the frozen text the benchmark cuts its long prompts from. Every engine gets byte-identical
prompts from it, and its sha256 keys the reference prompt-token counts in `bench.py` (4,133 tokens for the ~4K
prompt and 16,286 for the ~16K prompt with the Qwen3.6 / Qwen3.8 chat templates, thinking off).

| | |
| --- | --- |
| File | `bench_corpus.txt` |
| Size | 145,652 bytes (144,500 characters, UTF-8) |
| sha256 | `1a9bb103b3f6d39ee5b9f5bf7207e304c5ce92788c273685f597508c7d2bde28` |

Check it with `shasum -a 256 bench_corpus.txt`. `bench.py` records the sha256 in every result and warns when it
differs; with another corpus the prompt-token check is reported as unchecked.

## Source and license

The corpus consists of excerpts of eight source files of [Hugging Face Transformers](https://github.com/huggingface/transformers)
(`src/transformers/models/*/modeling_*.py`, byte-identical to the files in Transformers v5.16.1 and v5.17.0),
licensed under the Apache License, Version 2.0. Each excerpt keeps the file's own copyright and license header.
The Apache License 2.0 text is in this repository's `LICENSE` file and at <https://www.apache.org/licenses/LICENSE-2.0>.

Changes made: each file was cut to its first 20,000 characters and preceded by a `### FILE: <name>` heading and a
code fence; the concatenation was cut to 144,500 characters, so the last excerpt is shorter. The text is used only
as prompt material for benchmarks; it is not executed or imported.

| Order | File | Copyright notices in the excerpt |
| ---: | --- | --- |
| 1 | `modeling_granite_speech.py` | Copyright 2025 The HuggingFace Inc. team. |
| 2 | `modeling_modernvbert.py` | Copyright 2026 Illuin Technology and contributors, and The HuggingFace Inc. team. |
| 3 | `modeling_altclip.py` | Copyright 2022 The BAAI Teams Authors and The HuggingFace Inc. team. |
| 4 | `modeling_xcodec.py` | Copyright 2025 The HuggingFace Inc. team. |
| 5 | `modeling_t5.py` | Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team. |
| 6 | `modeling_ctrl.py` | Copyright 2018 Salesforce and HuggingFace Inc. team. Copyright (c) 2018, NVIDIA CORPORATION. |
| 7 | `modeling_emu3.py` | Copyright 2024 HuggingFace Inc. team. |
| 8 | `modeling_videoprism.py` (first 4,134 characters) | Copyright 2026 The HuggingFace Inc. team. |
