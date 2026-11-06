# kflash-lowrank-scratch: reproduction and evidence

Reference: `0cb86eefcbbd91ef98110ee67cc59a6fbc28cb34`.
Candidate implementation: `4cbb4a0a111e39fe4722eb7f6fd4c20beabf29df`.

Mac Studio (2025), Apple M3 Ultra (60 GPU cores, 96 GiB), macOS 26.4.1 (25E253).
Release builds use `-O3`. The model is Qwen3.5-4B-4bit with its six-layer DFlash BF16 checkpoint.
Drafter SHA256: `1eb221d36abb13a5f1b972f8d031a9723fad8cbb7d275abe548b60e77577eb42`.
Model config SHA256: `f3efc81b2ea8d96a45301037d3ccccbcccdef44a961845c87f286aaddbc6eaaa`.
Tokenizer SHA256: `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.

The CPU checker compiles the implementation from each revision with the same flags. It does not load a model
onto the GPU. Run from a checkout of the candidate that also contains the reference commit.

```sh
python3 bench/results/2026-11-06-kflash-lowrank-scratch/reproduce_lowrank.py . 0cb86eef 4cbb4a0a
```

Each process receives the same deterministic packed Q4 codes and BF16 coefficients. It reports factorization time,
process peak RSS (bytes on macOS), and full A/P output size. The runner hashes the complete output and requires
all arms and repetitions to match. Small cases include a partial final 4096-row tile; the 2560/4096-wide cases
have four interleaved pairs. The 4096-wide case is a synthetic operator input, not a complete 9B model run.

For real weights, give each revision a new `KAPITSA_HOME`, set `NI_KFLASH_LOWRANK=768` and run
`qmm_draft_check MODEL PROMPT_IDS OUTPUT`, adding `--from REFERENCE_OUTPUT` for the candidate. Each arm must log
that the factors were computed. Hash `KAPITSA_HOME/kflash/dflash2-lowrank-*.bin` after both runs: the output check
uses fresh caches rather than reading the same saved factorization twice.

The complete-model A/B uses fresh factor-cache homes on each arm and `ni_run MODEL @PROMPT_IDS 1024 0 --dflash auto`
with `NI_KFLASH_LOWRANK=768`, under `.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 3 --label lowrank-ab -- /usr/bin/time -l ...`
(the `...` is the `ni_run` command above; the wrapper holds the GPU lock and sets `NI_LOCK_HELD=1`). Every row records nonempty text, token count, EOS count, drafter load information,
accepted draft tokens/cycle and the complete generated-ID hash. `/usr/bin/time -l` supplies whole-process peak RSS.
The file system's model pages are not flushed; a fresh factor cache does not mean cold file-system I/O.

All GPU work holds the project's GPU lock. The target logit gate uses the standing `logit_dump` and
`logit_compare.py`; direct drafter comparisons additionally exercise the changed path. No Metal source changed.

Full HTTP regression has the same two timing failures on reference and candidate (47 pass each): the counting
reply stops at 693 tokens before the delayed side request, and the slot holder ends around 17 seconds before a
second 15-second keepalive. Their unmodified failure details are recorded rather than calling the suite all-pass.

`draft.ids` is the frozen prompt for the 270-buffer direct drafter comparisons.
`continuation.ids` and `continuation.txt` are the frozen 1024-token generation prompt (the text includes its chat template).
