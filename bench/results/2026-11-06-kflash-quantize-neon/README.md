# kflash-quantize-neon: reproduction and evidence

Reference: `0cb86eefcbbd91ef98110ee67cc59a6fbc28cb34`.
Candidate implementation: `0f92834788846394200eed4627faad000e3d0608`.

Mac Studio (2025), Apple M3 Ultra (60 GPU cores, 96 GiB), macOS 26.4.1 (25E253).
Release builds use `-O3`. The model is Qwen3.5-4B-4bit with its six-layer DFlash BF16 checkpoint.
Drafter SHA256: `1eb221d36abb13a5f1b972f8d031a9723fad8cbb7d275abe548b60e77577eb42`.
Model config SHA256: `f3efc81b2ea8d96a45301037d3ccccbcccdef44a961845c87f286aaddbc6eaaa`.
Tokenizer SHA256: `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.

The CPU checker compiles the implementation from each revision with the same flags. It does not load a model
onto the GPU. Run from a checkout of the candidate that also contains the reference commit.

```sh
python3 bench/results/2026-11-06-kflash-quantize-neon/reproduce_quant.py . 0cb86eef 0f928347 /path/to/dflash2/model.safetensors
```

The optional safetensors input is the 4B drafter checkpoint above. Without it, the synthetic checks and timings
still run. Real-weight timings exclude file I/O and allocation: both implementations receive the same resident
BF16 row pointers and preallocated output storage. All rows are also checked in reversed order. W/S/B bytes
are compared before timing. Both 4/8-bit formats and both rounding modes are covered. Each timed row reports
four interleaved reference/candidate pairs; the output SHA256 covers W, then S, then B.

For actual drafter outputs, build the existing `qmm_draft_check` in each revision. To select 8-bit as well as 4-bit,
use an identical out-of-tree copy of that checker on both sides, replacing `DFlash2 draft(target, drafter, 4)`
with `DFlash2 draft(target, drafter, 0)` and set `NI_KFLASH_BITS=4` or `8`, `NI_KFLASH_LOWRANK=0`. Compare candidate
with `--from` pointing at the same-width reference. These checks cover fused and debug/unfused passes.

The standard complete-model A/B uses `kapitsa bench --model MODEL --quick --json --out RESULT` in A B A B order.
It exercises the default 4-bit drafter; no NI setting is passed through by that benchmark. Raw result files include
hardware conditions and flags. The isolated CPU quantization timings do not establish faster inference.

All GPU work holds the project's GPU lock. The target logit gate uses the standing `logit_dump` and
`logit_compare.py`; direct drafter comparisons additionally exercise the changed path. No Metal source changed.

Full HTTP regression has the same two timing failures on reference and candidate (47 pass each): the counting
reply stops at 693 tokens before the delayed side request, and the slot holder ends around 17 seconds before a
second 15-second keepalive. Their unmodified failure details are recorded rather than calling the suite all-pass.

`draft.ids` is the frozen prompt for the 270-buffer direct drafter comparisons.

The first campaign is A1 B1 A2 B2. `bench-02-quant` flags a sampled repetition loop; its result is retained, and
`bench-05-reference` / `bench-06-quant` are the additional pair, both without flags. The sampled benchmark does not
fix a seed. No consistent full-model throughput or TTFT improvement is claimed.
