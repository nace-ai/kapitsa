# Fast profile: TATTN hi + lo on Flash-Next, the bar, and the candidates

Evidence for PR #986 and [docs/profiles.md](../../../docs/profiles.md). Mac: MacBook Pro (16-inch, 2026), M5 Pro
(20-core GPU), 48 GB, macOS 26.5.2, 5 and 6 Nov 2026. Raw logits are not committed (71 MB per 72-row dump); every
report names its dumps' builds.

## Output check (`reports/`)

`tools/logit_compare.py` between `logit_dump --paths step` dumps. Flash-Next: Qwen3.8-Flash-Next Q2_0, 72 rows over
16K, 32K and 64K windows (24 each), prefill chunk 2048 unless named. 35B: Qwen3.6-35B-A3B 4-bit, 192 rows over 8
windows of 512 to 3,200 tokens. Kapitsa dumps: build 4d9a9ec8 (this branch before its rebase onto e09e4f85; the
TATTN kernel, `src/kernels/tensor/qsattn_t.metal`, is byte-identical at the PR head). Splash dumps: Splash 1.5.0
(cabbad27) with `splash-logits.patch`, which adds teacher-forced scoring to `backend-benchmark` (`--logits-from`) and
`SPLASH_LOGITS_KEEP_CACHE=1`; each dump's `meta.json` says so.

| Report | Rows | Top-1 flips | Mean KL |
|---|---|---|---|
| `flashnext-fast2048-vs-precise.txt` (what ships) | 72 | 4 | 8.8e-3 |
| `flashnext-fast4096-vs-precise.txt` | 72 | 4 | 8.8e-3 |
| `flashnext-fast2048-vs-fast4096.txt` | 72 | every row bitwise identical | 0 |
| `flashnext-splash-cached-vs-precise.txt` (the bar) | 72 | 16 | 0.46 |
| `flashnext-splash-cold-vs-cached.txt`, `flashnext-splash-worst8cold-vs-cached.txt` | 24 + 8 | 0 + 1 | 2.6e-3, 1.7e-2 |
| `flashnext-fast-i8-vs-precise.txt` (fast + `NI_QMT_I8`) | 72 | 6 | 2.1e-2 |
| `flashnext-fuse-vs-precise.txt` (`NI_QSA_FUSE`) | 72 | every row bitwise identical | 0 |
| `35b-splash-vs-precise.txt` (the 35B bar, cold scoring) | 192 | 4 | 2.1e-3 |
| `35b-fast-{step,verify,prefill}-vs-precise.txt` (no 35B member) | 192 each | every row bitwise identical | 0 |
| `35b-i8-step-vs-precise.txt` (`NI_QMT_I8`) | 192 | 5 | 4.0e-3 |
| `35b-kvi8-{step,verify}-vs-precise.txt` (`NI_KV_I8`) | 192 | 3 | 1.1e-3 |
| `35b-exactmoe0-verify-vs-precise.txt` (`NI_DFLASH_EXACT_MOE=0`) | 192 | 0, passes the floor's gate | 1.8e-4 |
| `lightning-attnq8-vs-precise.txt` (`NI_NH_ATTN_Q8`) | 192 | 1, a near-tie | 6.5e-4 |

The Splash cold and cached rows agree on top-1 for 31 of 32, so the bar's Splash rows are Splash's own numerics, not
a cache fault in the scoring patch.

## Speed (`legs/`)

`legs.py RUNDIR RUNLOG` writes one CSV row per leg from `ni_run`'s output; the `.run.log` next to each CSV is the
script's own log with each leg's verdict (load, AC, memory pressure). Every run alternates its arms within a round and
reverses the order in the next round. Legs marked `void-*` are left out of every median.

- `fn_ab5-try1`: build 4187b609 (#986's head after its rebase onto 7381db44; the merged head c8f4ba9b changes only comments and docs) as built, precise against fast, 4
  legs each, expert store pinned at 24 GB. Fast picked 2048-row chunks itself (the prefill group's slots log -0.50 GB,
  as `NI_CHUNK=2048` did in fn_ab3). 32K: median wall ms per draft cycle 54.1 precise, 53.6 fast (1% faster); prefill
  43.7 s against 39.5 s (10% faster). The 1% decode gap is inside the legs' spread (53.6 to 55.6 ms
  precise, 52.6 to 55.6 fast); tokens per second are 4% lower for fast because its greedy text drafts 1.95 tokens per
  cycle against 2.05. One leg per arm logged memory pressure. The 64K legs were voided: the first
  precise 64K leg left used swap at 8.4 GB, and the guard (used swap over 8 GB or free disk under 40 GB) stopped the
  rest.
- `fn_ab6-try1`: the same build and setup, 64K only, rerun right after. Median wall ms per draft cycle 59.4 precise,
  57.5 fast (3% faster); prefill 93.3 s against 87.2 s (7% faster). Every fast leg and 2 of 4 precise legs logged
  memory pressure; no leg tripped the guard.

Earlier runs, kept for the record:

- `fn_ab3-try1`: build 4d9a9ec8, expert store pinned at 24 GB, three arms: precise, fast at 4096-row chunks, fast at
  2048 (`NI_CHUNK=2048`), 4 legs each at 32K and 64K. Median ms per draft cycle: precise 54.2 and 60.1, fast at 2048
  54.2 and 57.6, fast at 4096 58.7 and 64.5, 7 of 8 4096 legs under memory pressure (the clean one, 56.3 ms at 32K, still 4% slower). Prefill against precise: 2048 8%
  and 10% faster, 4096 15% and 17%. This run set the fast profile's chunk at 2048.
- `fn_ab4-try1`: the same without the store pin. The run went into swap (past 8 GB) and was stopped after 15 legs;
  every leg logged memory pressure. Not used.
- `fn_ab2-try1`: precise against fast at 4096, during a CPU job; 4 of 16 legs void. Not used.
- `b35ab`: 35B, `NI_QMT_I8` prefill at 4K and 16K and `NI_DFLASH_EXACT_MOE=0` decode at 4K, 4 pairs each. I8 prefill
  3.5% slower at 4K, about the same at 16K; EXACT_MOE=0 decode about the same.
- `queue3_ab-try1`: 35B `NI_KV_I8` decode at 4K (117.6 against 116.8 tok/s, 4 pairs; the greedy text differs, 43 decoded tokens against 41) and Flash-Next `NI_QSA_FUSE`
  decode at 32K (40.5 against 42.9 tok/s, 2 legs against 1; 1 void). Lightning skipped: its model is on an external
  drive that was not mounted.

## Metal validation (`validation/`)

`tattn_validate.sh`, build 4187b609, fast profile, a 10,240-token prompt (its last 2048-row chunk passes 8,192 keys,
so `qattn_t_sparse_g12d2` runs) and 8 tokens of decode. With `MTL_SHADER_VALIDATION=1` the run completed with no
validation message (`shader-validation.out`). Adding `MTL_DEBUG_LAYER=1` aborts on the dense `attn_prefill11r`
(40,960 B as the layer counts it, against 32,768 B) before the sparse kernel, as AGENTS.md describes for any prefill. The same prompt with `NI_PROFILE=1` and no validation (`ni-profile.out`, build
ef3dc9aa, the same kernels) shows `qattn_t_sparse_g12d2` dispatched: 1.7 calls per step, 2.8% of GPU time.

## Not run

- Any Mac but this M5 Pro; Flash-Next's 4-bit pack.
- Lightning `NI_NH_ATTN_Q8` speed.
- `NI_QSA_FUSE` at 64K.
