# Results, 2026-10-28: Clover-35B-A3B on the MacBook Pro M5 Max, 128 GB

DiffuSpace's [Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview), a block-diffusion MoE
tuned from Qwen3.6-35B-A3B, on Kapitsa after [#773](https://github.com/nace-ai/kapitsa/pull/773) (its kernels),
[#774](https://github.com/nace-ai/kapitsa/pull/774) (its converter) and [#776](https://github.com/nace-ai/kapitsa/pull/776)
(its backend). MacBook Pro (16-inch, 2026, `Mac17,6`), M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power in
High Power mode. Weights: the Hub checkpoint at `4b0402b` (every shard's sha256 checked against the Hub's), converted by
`tools/convert_clover.py` into the 35B's 4-bit recipe (MLX affine 4-bit group 64, the router and shared-expert gate
8-bit), 19.5 GB. Every run is `kapitsa bench` (the protocol of [bench/harness](../../harness/README.md): 512-token coding
answers at T 0.6 and greedy, cold 4K and 16K prompts, follow-ups, tool calls) on that folder.

**The poster's cell** is the full run at 12:21 UTC on `80740c0`, #776's pre-merge head (#776 was squash-merged as
`250c7c9`; its final head `a9fa5ae` added only a log line, so the decoding is the same): **6,002 / 319** tok/s (prefill 4K / decode T 0.6), greedy 332, 16K prompt 5,313, follow-ups 0.04 and
0.05 s, peak 21.1 GB. Its decoding is Kapitsa's default for Clover (`NI_CLOVER_DECODE=auto`): Clover's causal pathway
has the 35B-A3B's shape, so the 35B's DFlash2 drafter (from the installed `qwen3.6-35b-a3b` pack) drafts and KFlash
verifies, as for the 35B. No other engine runs Clover on a Mac (DiffuSpace's own engines: SGLang on CUDA, a llama.cpp
backend not yet released), so the poster's other cells are ×.

| Started (UTC) | Decoding | Build | Prefill 4K / 16K (tok/s) | Decode T 0.6, median of 5 (range) | Greedy | Flags |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 12:21 | `auto`: DFlash2 drafter, KFlash | `80740c0` | 6,002 / 5,313 | **319.2** (249.7-334.1) | 332.4 | repetition loop (0.43): see below |
| 12:01 | `auto`: DFlash2 drafter, KFlash | `b0e7160` (stamped 909e282) | 5,883 / 5,275 | 288.7 (282.3-343.7) | 293.2 | contention (Google Chrome, 4.5% mean, in greedy and 4K) |
| 11:41 | `spec`: self-speculative (quick: 3 runs, 4K only) | `b0e7160` (stamped 909e282) | 5,958 / – | 213.5 (204.2-221.3) | 210.4 | repetition loop (0.46): see below |

The binary of the two `b0e7160` runs was built in a tree configured before the branch's commits, so its stamp reads the
base, `909e282`; its source was #776's first head (the review commit `80740c0` changed docs, tests and `clover_bench`,
not the decoding).

**The repetition flags** come from the answer's style, not a loop: the bench's distinct-4-gram counts each `-` of the
`# ------` separator lines Clover writes between sections as a word. Replays of the bench's request with bench-style
nonces read 0.58-0.67 at T 0.6 (`auto`) and 0.62-0.74 greedy (`spec`); their most repeated lines are `pass`, `"""` and
a separator (4-6 times each in 58-71 lines), and none repeats a block.

**Clover's own decoding** (the two modes DiffuSpace ships), for comparison, single requests of the bench's coding
prompt through `kapitsa serve`: self-speculative (`spec`, a denoising pass drafts 7 tokens and the causal 8-row verify
checks them, SGLang's threshold rule) keeps 4.0-4.7 tokens a cycle there at 21 ms a cycle, 187-224 tok/s, about plain
decoding of the causal pathway (212 tok/s at T 0.6 on a prose-and-code prompt); block diffusion (`diffusion`, the Hugging
Face implementation's LowConfidence at 0.95) fills 1.25 positions a 32-row denoising pass on the 4-bit weights, 45.6
tok/s greedy. `build/clover_bench` (no server) on a short C++ answer: 5.98 tokens a cycle, 276 tok/s, 27 of 50 cycles
keeping all 8 rows. Every line behind these numbers: [requests.md](requests.md).

**The `spec` record's `decoding` field** reads `plain` ("plain decoding, KFlash (none)"): `kapitsa bench` reports
the engine's drafter, and knows nothing of `NI_CLOVER_DECODE`, under which that run's server decoded self-speculatively
(its log: "8-row draft and verify" cycles).
