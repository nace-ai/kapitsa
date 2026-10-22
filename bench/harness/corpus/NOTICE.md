# Prompt corpus: notice and checksum

`bench_corpus.txt` is the frozen text the benchmark cuts its long prompts from. Every engine gets byte-identical
prompts from it, and its sha256 keys the reference prompt-token counts in `bench.py` (4,133 tokens for the ~4K
prompt and 16,286 for the ~16K prompt with the Qwen3.6 / Qwen3.8 chat templates, thinking off).

| | |
| --- | --- |
| File | `bench_corpus.txt` |
| Size | 653,464 bytes (650,000 characters, UTF-8): enough for a 128K-token prompt (544,000 characters at `bench.py`'s 4.25 per token) |
| sha256 | `58a7756fd94fc48a7b7306d330b2c44663bdf22faa943f82430966c2878081b2` |
| First 144,500 characters | sha256 `1a9bb103b3f6d39ee5b9f5bf7207e304c5ce92788c273685f597508c7d2bde28`: the original corpus (V1), byte for byte |

Check it with `shasum -a 256 bench_corpus.txt`, or rebuild it from the source files with `python3 make_corpus.py`
(downloads them, rebuilds the corpus and compares it, and its first 144,500 characters, with the checksums above).

**Extension (22 Oct 2026).** The corpus was 144,500 characters (V1, sha256 `1a9bb103…`), about 34,000 tokens, too
short for a 64K or 128K prompt. It was extended by appending only: V1 is its first 144,500 characters unchanged, so
every prompt cut from the first 144,500 characters, the 1K-32K prompts and the warm-up included, is byte-identical to
before, and results with V1's sha256 compare with this corpus' up to that length. `bench.py` and `kapitsa bench` accept
both checksums as the frozen corpus.

`bench.py` records the sha256 in every result and warns when it
differs; with another corpus the prompt-token check is reported as unchecked.

## Source and license

The corpus consists of excerpts of 33 source files of [Hugging Face Transformers](https://github.com/huggingface/transformers)
(`src/transformers/models/*/modeling_*.py`; the first eight byte-identical to the files in Transformers v5.16.1 and
v5.17.0, all of them to v5.17.0), licensed under the Apache License, Version 2.0. Each excerpt keeps the file's own copyright and license header.
The Apache License 2.0 text is in this repository's `LICENSE` file and at <https://www.apache.org/licenses/LICENSE-2.0>.

Changes made: each file was cut to its first 20,000 characters and preceded by a `### FILE: <name>` heading and a
code fence; the concatenation was cut to 650,000 characters (V1: 144,500), so the last excerpt is shorter. Files whose
license header is not near their top (`modeling_mistral.py`, `modeling_qwen2.py`, `modeling_gpt_neox.py`) were left out. The text is used only
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
| 8 | `modeling_videoprism.py` (V1 ends after its first 4,134 characters) | Copyright 2026 The HuggingFace Inc. team. |
| 9 | `modeling_bart.py` | Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. |
| 10 | `modeling_bert.py` | Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team. Copyright (c) 2018, NVIDIA CORPORATION. |
| 11 | `modeling_bloom.py` | Copyright 2022 HuggingFace Inc. team and BigScience workshop. |
| 12 | `modeling_clip.py` | Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. |
| 13 | `modeling_deberta_v2.py` | Copyright 2020 Microsoft and the Hugging Face Inc. team. |
| 14 | `modeling_detr.py` | Copyright 2021 Facebook AI Research The HuggingFace Inc. team. |
| 15 | `modeling_electra.py` | Copyright 2019 The Google AI Language Team Authors and The HuggingFace Inc. team. |
| 16 | `modeling_falcon.py` | Copyright 2023 the Falcon authors and HuggingFace Inc. team. |
| 17 | `modeling_gemma.py` | Copyright 2024 Google Inc. HuggingFace Inc. team. |
| 18 | `modeling_gemma2.py` | Copyright 2024 Google Inc. HuggingFace Inc. team. |
| 19 | `modeling_gpt2.py` | Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team. Copyright (c) 2018, NVIDIA CORPORATION. |
| 20 | `modeling_led.py` | Copyright 2021 Iz Beltagy, Matthew E. Peters, Arman Cohan and The HuggingFace Inc. team. |
| 21 | `modeling_llama.py` | Copyright 2022 EleutherAI and the HuggingFace Inc. team. |
| 22 | `modeling_longformer.py` | Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team. |
| 23 | `modeling_mamba.py` | Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team. |
| 24 | `modeling_mbart.py` | Copyright 2021, The Facebook AI Research Team and The HuggingFace Inc. team. |
| 25 | `modeling_mixtral.py` | Copyright 2023 Mistral AI and the HuggingFace Inc. team. |
| 26 | `modeling_opt.py` | Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. |
| 27 | `modeling_pegasus.py` | Copyright 2021, Google and The HuggingFace Inc. team. |
| 28 | `modeling_phi3.py` | Copyright 2024 Microsoft and the HuggingFace Inc. team. |
| 29 | `modeling_qwen2_moe.py` | Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. |
| 30 | `modeling_qwen3.py` | Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. |
| 31 | `modeling_roberta.py` | Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team. Copyright (c) 2018, NVIDIA CORPORATION. |
| 32 | `modeling_swin.py` | Copyright 2022 Microsoft Research and The HuggingFace Inc. team. |
| 33 | `modeling_vit.py` (first 8,523 characters) | Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. |
