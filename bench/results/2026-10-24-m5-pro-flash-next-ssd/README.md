# M5 Pro, 48 GB: Qwen3.8-Flash-Next 180B-A7B streamed from the SSD, Kapitsa against llama.cpp, 2026-10-24

Is Kapitsa the fastest engine for Flash-Next on a 48 GB Mac, where the model does not fit in memory? Measured on one
MacBook Pro M5 Pro, 48 GB (`Mac17,8`, macOS 26.5.2, High Power mode, AC power). Every model file was on the internal
disk for the timed legs.

- **Kapitsa** main `4b087c8` (`build/ni_run`), the Q2_0 pack (43 GB on disk): routed experts streamed through the
  expert store, which was **pinned at 24 GB** (`NI_EXPERT_STORE=24`) for memory headroom during the run. The auto
  size on this Mac is about 29.9 GB, so these Kapitsa numbers come from a smaller store than a user gets. Four runs at
  the auto size were stopped at memory pressure level 2 (see Caveats). GPU heater
  on (the default). No speculative decoding.
- **llama.cpp** `0a9ef00`, `llama-server -ngl 99 -fa on -c 33280 -np 1` on ISTA's Q2_0 GGUF (66.4 GB, mmap), no
  speculative decoding. A side row runs it with `--cpu-moe` (routed experts on the CPU).
- **MLX was not measured.**

Prompts are token ids ([prompts/](prompts/)): 85 tokens (`dec`), 1K, 4K, 16K, 32K, then 128 greedy tokens.
Prefill tok/s is read at that length and decode tok/s at that context. Warm legs run ABBA per length, two legs per
engine, each in its own hold of the GPU lock with 30 s between legs. Cold starts are approximate: `sudo purge` was not
available, so before each cold leg the other engine's files were read through the page cache. Scripts are in
[scripts/](scripts/) and every leg's log is in [legs/](legs/), along with the aborted and refused attempts.

**Verdict: Kapitsa is not the fastest everywhere.** Its prefill is behind at 85 tokens (79% slower) and at 1K (28%
slower), level at 4K (-2.0%) and 16K (+1.0%), both inside the legs' spread, and slightly ahead at 32K (+2.7%: both Kapitsa legs, 541.7
and 558.0 tok/s, are above both llama.cpp legs, 535.5 and 535.2, one pair of legs per engine). Its decode
is ahead up to 4K (+4.0%, +6.3%, +8.0%) and behind at 16K (-5.1%) and 32K (-8.2%). Cold start to the first token is
level (Kapitsa's cold leg 13.9 s, llama.cpp 13.2 s; Kapitsa's second leg is warm, see below).

## Summary (all model files on the internal disk; ± = half the range over the mean)

| engine | length (prompt tok) | prefill tok/s | decode tok/s | TTFT ms (warm, prompt) |
|---|---:|---:|---:|---:|
| Kapitsa main 4b087c8 (Q2_0 pack, expert-store streaming, store pinned 24 GB, not the auto default, heater on) | dec (85) | 52.7 (n 2, ±0.6%) | 40.5 (n 2, ±0.2%) | 1613.0 (n 2, ±0.6%) |
| llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 -fa on) | dec (85) | 255.0 (n 2, ±2.8%) | 39.0 (n 2, ±1.1%) | 333.6 (n 2, ±2.8%) |
| llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 --cpu-moe -fa on) | dec (85) | 34.8 (n 2, ±17.4%) | 13.2 (n 2, ±1.0%) | 2518.5 (n 2, ±17.4%) |
| Kapitsa main 4b087c8 (Q2_0 pack, expert-store streaming, store pinned 24 GB, not the auto default, heater on) | c1k (1024) | 445.0 (n 2, ±20.0%) | 40.8 (n 2, ±0.4%) | 2396.5 (n 2, ±20.0%) |
| llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 -fa on) | c1k (1024) | 617.5 (n 2, ±1.1%) | 38.3 (n 2, ±0.3%) | 1658.5 (n 2, ±1.1%) |
| Kapitsa main 4b087c8 (Q2_0 pack, expert-store streaming, store pinned 24 GB, not the auto default, heater on) | pf4 (4132) | 558.2 (n 2, ±4.8%) | 40.0 (n 2, ±0.1%) | 7420.0 (n 2, ±4.8%) |
| llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 -fa on) | pf4 (4132) | 569.4 (n 2, ±0.3%) | 37.1 (n 2, ±0.1%) | 7257.2 (n 2, ±0.3%) |
| llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 --cpu-moe -fa on) | pf4 (4132) | 132.9 (n 2, ±7.5%) | 14.6 (n 2, ±3.1%) | 31258.0 (n 2, ±7.5%) |
| Kapitsa main 4b087c8 (Q2_0 pack, expert-store streaming, store pinned 24 GB, not the auto default, heater on) | pf16 (16287) | 555.0 (n 2, ±2.9%) | 34.7 (n 2, ±0.1%) | 29372.0 (n 2, ±2.9%) |
| llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 -fa on) | pf16 (16287) | 549.7 (n 2, ±0.0%) | 36.5 (n 2, ±0.5%) | 29628.3 (n 2, ±0.0%) |
| Kapitsa main 4b087c8 (Q2_0 pack, expert-store streaming, store pinned 24 GB, not the auto default, heater on) | c32k (32768) | 549.9 (n 2, ±1.5%) | 32.7 (n 2, ±0.2%) | 59606.0 (n 2, ±1.5%) |
| llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 -fa on) | c32k (32768) | 535.4 (n 2, ±0.0%) | 35.6 (n 2, ±0.5%) | 61206.1 (n 2, ±0.0%) |

## Cold start (the other engine's files read through the page cache first; prompt dec, 85 tok)

| engine | load s (engine's own) | TTFT ms (prompt) | launch to first token s (wall: process time - decode time) |
|---|---:|---:|---:|
| Kapitsa main 4b087c8 (Q2_0 pack, expert-store streaming, store pinned 24 GB, not the auto default, heater on) | 8.1 (n 2, ±48.8%) | 1661.0 (n 2, ±0.3%) | 9.9 (n 2, ±39.4%) |
| llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 -fa on) | 11.8 (n 2, ±0.8%) | 378.3 (n 2, ±9.9%) | 13.2 (n 2, ±0.0%) |

## Fastest? (Kapitsa vs the best llama.cpp configuration at each length)

- dec prefill: Kapitsa 52.7 vs llama.cpp 255.0 tok/s -> fastest: no, by 79.3%
- dec decode: Kapitsa 40.5 vs llama.cpp 39.0 tok/s -> fastest: yes, by 4.0%
- c1k prefill: Kapitsa 445.0 vs llama.cpp 617.5 tok/s -> fastest: no, by 27.9%
- c1k decode: Kapitsa 40.8 vs llama.cpp 38.3 tok/s -> fastest: yes, by 6.3%
- pf4 prefill: Kapitsa 558.2 vs llama.cpp 569.4 tok/s -> fastest: no, by 2.0%
- pf4 decode: Kapitsa 40.0 vs llama.cpp 37.1 tok/s -> fastest: yes, by 8.0%
- pf16 prefill: Kapitsa 555.0 vs llama.cpp 549.7 tok/s -> fastest: yes, by 1.0%
- pf16 decode: Kapitsa 34.7 vs llama.cpp 36.5 tok/s -> fastest: no, by 5.1%
- c32k prefill: Kapitsa 549.9 vs llama.cpp 535.4 tok/s -> fastest: yes, by 2.7%
- c32k decode: Kapitsa 32.7 vs llama.cpp 35.6 tok/s -> fastest: no, by 8.2%
- cold start to first token, per leg (s): Kapitsa c1 13.85, c2 6.02; llama.cpp c1 13.21, c2 13.22 (read each leg: a leg whose weights load far faster than the others was warm, not cold; no percentage is derived from the mean)

## Generated tokens (greedy, first 128): Kapitsa vs llama.cpp, common prefix

- dec: 20 of 128 tokens equal before the first difference
- c1k: 2 of 128 tokens equal before the first difference
- pf4: 9 of 128 tokens equal before the first difference
- pf16: 3 of 48 tokens equal before the first difference
- c32k: 63 of 128 tokens equal before the first difference

## Caveats

- Kapitsa store sizes seen: 24.00 GB (70.1%).
- Same Mac: M5 Pro 48 GB; all model files on the internal disk during timed legs (ISTA GGUF copy removed after the run, sha256-checked against the drive; df 170 GiB free). Kapitsa heater default-on (main 4b087c8). MLX not measured.
- Kapitsa expert store pinned at 24 GB (NI_EXPERT_STORE=24) for memory headroom during the run; the auto default on this Mac is ~29.9 GB, so these Kapitsa numbers are from a smaller store than a user gets.
- Cold starts are approximate: no sudo purge; the other engine's files were read through the page cache before each cold leg. Kapitsa's second cold leg (c2) was warm: its weights loaded in 4.1 s against 12.0 s in c1, and the eviction step took 3 s, so its pack stayed cached. The table's mean of the two Kapitsa legs (9.9 s, ±39.4%) is not a cold start.
- Four attempts at Kapitsa's default store size (28-29.9 GB) were stopped by the harness at memory pressure level 2 and are kept in `legs/aborted/auto0_..auto3_*` (swap +0 MB, +1,847 MB, +0 MB, +44 MB; auto3 logged that the weights are no longer kept wired while idle). So the default store size was not measured here, and it was not tested whether a user's session would hit the same pressure.
- pf16: Kapitsa stopped at EOS after 47 decoded tokens, llama.cpp decoded 128 (`ignore_eos`); decode rates are over the same 16K context but not the same token count.
- Kapitsa's 1K prefill legs read 356 and 534 tok/s (±20%); both are below llama.cpp's 610 and 625.
- Greedy outputs differ between engines (different quantized weights/kernels; common prefix 2-63 tokens), so decode speed is compared, not text.
- cpu-moe leg 2 preflight counted file-backed page cache as available (same GGUF re-read). It also covered the pf4_1 rerun (gate read 40.3 GB). Threshold 28 GB and every abort unchanged; rv_leg.sh now has this off by default (opt-in RV_AVAIL_FILEBACKED=1). Every other leg used the old preflight.
- llama --cpu-moe pf4_1 first ran right after a 1-min foreign GPU bench (lock-serialized, no overlap): 122.6 tok/s; rerun at the end of the run: 123.0 tok/s (row uses the rerun); pf4_2 142.8 tok/s, so that row's spread is real leg-to-leg variance, not the bench. Side row, not in the verdict.

## Legs

| engine | prompt | leg | ok | prompt tok | prefill tok/s | prompt ms | decode tok/s | load s | store | swap Δ MB |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|
| kap | c1k | 1 | ok | 1024 | 356.2 | 2875 | 40.60 | 2.96 | 24.00 GB (70.1%) | 0 |
| llama-gpu | c1k | 1 | ok | 1024 | 610.5 | 1677 | 38.23 | 8.15 | - | 0 |
| kap | c1k | 2 | ok | 1024 | 533.9 | 1918 | 40.90 | 2.78 | 24.00 GB (70.1%) | 0 |
| llama-gpu | c1k | 2 | ok | 1024 | 624.5 | 1640 | 38.46 | 9.27 | - | 0 |
| kap | c32k | 1 | ok | 32768 | 541.7 | 60487 | 32.60 | 3.00 | 24.00 GB (70.1%) | 0 |
| llama-gpu | c32k | 1 | ok | 32768 | 535.5 | 61190 | 35.73 | 1.62 | - | 0 |
| kap | c32k | 2 | ok | 32768 | 558.0 | 58725 | 32.70 | 2.79 | 24.00 GB (70.1%) | 0 |
| llama-gpu | c32k | 2 | ok | 32768 | 535.2 | 61222 | 35.38 | 11.20 | - | 0 |
| kap | dec | 1 | ok | 85 | 53.0 | 1604 | 40.60 | 2.80 | 24.00 GB (70.1%) | 0 |
| llama-cpumoe | dec | 1 | ok | 85 | 40.8 | 2081 | 13.37 | 1.74 | - | 0 |
| llama-gpu | dec | 1 | ok | 85 | 247.9 | 343 | 38.53 | 11.62 | - | 0 |
| kap | dec | 2 | ok | 85 | 52.4 | 1622 | 40.40 | 2.96 | 24.00 GB (70.1%) | 0 |
| llama-cpumoe | dec | 2 | ok | 85 | 28.8 | 2956 | 13.11 | 2.27 | - | 0 |
| llama-gpu | dec | 2 | ok | 85 | 262.1 | 324 | 39.37 | 1.49 | - | 0 |
| kap | dec | c1 | ok | 85 | 51.0 | 1666 | 36.40 | 12.02 | 24.00 GB (70.1%) | 0 |
| llama-gpu | dec | c1 | ok | 85 | 204.3 | 416 | 38.60 | 11.68 | - | 0 |
| kap | dec | c2 | ok | 85 | 51.3 | 1656 | 40.00 | 4.14 | 24.00 GB (70.1%) | 0 |
| llama-gpu | dec | c2 | ok | 85 | 249.5 | 341 | 38.16 | 11.88 | - | 0 |
| kap | pf16 | 1 | ok | 16287 | 538.8 | 30226 | 34.60 | 2.97 | 24.00 GB (70.1%) | 0 |
| llama-gpu | pf16 | 1 | ok | 16287 | 549.5 | 29640 | 36.72 | 1.89 | - | 0 |
| kap | pf16 | 2 | ok | 16287 | 571.1 | 28518 | 34.70 | 2.81 | 24.00 GB (70.1%) | 0 |
| llama-gpu | pf16 | 2 | ok | 16287 | 549.9 | 29617 | 36.33 | 10.62 | - | 0 |
| kap | pf4 | 1 | ok | 4132 | 531.2 | 7779 | 40.00 | 2.97 | 24.00 GB (70.1%) | 0 |
| llama-cpumoe | pf4 | 1 | ok | 4132 | 123.0 | 33588 | 15.07 | 2.45 | - | 0 |
| llama-gpu | pf4 | 1 | ok | 4132 | 571.2 | 7234 | 37.12 | 1.41 | - | 0 |
| kap | pf4 | 2 | ok | 4132 | 585.2 | 7061 | 40.10 | 2.78 | 24.00 GB (70.1%) | 0 |
| llama-cpumoe | pf4 | 2 | ok | 4132 | 142.8 | 28928 | 14.18 | 2.50 | - | 0 |
| llama-gpu | pf4 | 2 | ok | 4132 | 567.5 | 7281 | 37.06 | 10.44 | - | 0 |


## Erratum (2026-10-26)

**What was wrong.** Kapitsa's decode legs above did not stop at the end of the turn (`ni_run` does not stop at
`<|im_end|>`), and the 1K and 32K prompts have no assistant header. Kapitsa's first end-of-turn token comes at output
index 2 (1K), 31 (4K) and 63 (32K), so the rest of the 128 timed tokens is text generated after the answer. At 16K
Kapitsa stopped after 48 ids (its log reports "decode 47 tok") and llama.cpp decoded 128. This voids every decode
comparison in this file, including the ledger line `dec decode: Kapitsa 40.5 vs llama.cpp 39.0 tok/s -> fastest: yes,
by 4.0%` above.

**Corrected value.** The decode column is not comparable. This run has no valid decode measurement, and a re-measure
is pending. The prefill columns are unaffected.

**Evidence.** The `ids:` line of each committed Kapitsa leg in `legs/` (248046 is `<|im_end|>`, 248044 is
`<|endoftext|>`):

| Legs | Ids | First 248046 | First 248044 |
|---|---|---|---|
| `kap_c1k_1.log`, `kap_c1k_2.log` | 128 | index 2 | none |
| `kap_pf4_1.log`, `kap_pf4_2.log` | 128 | index 31 | none |
| `kap_pf16_1.log`, `kap_pf16_2.log` | 48 | index 43 | index 45 |
| `kap_c32k_1.log`, `kap_c32k_2.log` | 128 | index 63 | none |
| `kap_dec_1.log`, `kap_dec_2.log`, `kap_dec_c1.log`, `kap_dec_c2.log` | 128 | none | none |

**Every remaining decode comparison in this file is superseded (appended, not edited above).** The erratum above names
one of them, the `dec decode` ledger line; six more places above still carry the retracted percentages, and this line is
their correction. It is appended rather than written over them because the file is frozen and its earlier text may not
be edited:

- the **Verdict**, "Its decode is ahead up to 4K (+4.0%, +6.3%, +8.0%) and behind at 16K (-5.1%) and 32K (-8.2%)";
- four more decode rows of the **Fastest?** ledger: `c1k decode` (yes, by 6.3%), `pf4 decode` (yes, by 8.0%),
  `pf16 decode` (no, by 5.1%) and `c32k decode` (no, by 8.2%);
- the **Caveats** line "Greedy outputs differ between engines (different quantized weights/kernels; common prefix 2-63
  tokens), so decode speed is compared, not text".

With the answer ending at id 2, 31 and 63 of the timed 128, decode speed is not compared at all until the re-measure:
the decode column carries no comparison, the prefill columns are the only compared speed in this file, and the Verdict's
decode sentence, those five decode ledger rows and that Caveat line are void as written. The prefill percentages, the
prefill ledger rows and the cold-start rows are unaffected.
