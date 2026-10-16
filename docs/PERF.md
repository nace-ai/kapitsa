# nace-infer performance notes (M4 Pro 20-core GPU, 273 GB/s, macOS 15.3.1)

Targets: Splash (incoai) is the reference for end-to-end speed but cannot run on macOS 15 (Metal 4,
macOS 26.4+). Its published Qwen3.6-35B-A3B numbers are 210 tok/s decode (M5 Pro, DFlash2
speculative decoding) and 2,011 tok/s prefill at 32K (M5 neural accelerators). Scaled to M4 Pro the
decode target is ~150-180 tok/s on code. MLX/oMLX are measured here as the same-hardware floor.

## Decode (Qwen3.6-35B-A3B 4-bit, no speculation)
| version | tok/s | notes |
|---|---|---|
| 0.1 | 98 | first correct build |
| 0.2 | 105 | threadgroup-staged matvec (norm once per threadgroup), separate 5.5 us top-k |
| 0.3 | 116-123 | MTP speculative decoding, depth 1 (harness: 117.5-122.6) |

Verify cost (rows of one matvec, N=8192 K=2048 4-bit, isolated): 1 row 42.7 us (221 GB/s), 2 rows 44.6 us,
3 rows 64.5 us (best of several register layouts), 4 rows 89 us. A simdgroup-matrix version (qmvm/qmvm2,
Splash-style integer codes in 8x8 MMAs) was slower here (80-107 us). Expert dedupe across rows (NI_DEDUPE):
gate/up 97-121 us vs 78 per-row; slower. Depth 2 on repetitive text: 2.38 tok/cycle but 101 tok/s.
Bandwidth budget: 1.79 GB/token -> 128 tok/s at 230 GB/s. Per-token profile (NI_PROFILE=1):
DeltaNet in-proj 39 us/layer, MoE gate/up 50 us, down 29 us, out_proj 34 us, lm_head 1.2 ms.

## Prefill
| version | 4k tok/s | 16k tok/s |
|---|---|---|
| 0.1 | 604 | 445 |
| 0.2 (2048 chunks, 16-row expert tiles, attention v10) | 785 | 606 |
| MLX / oMLX (same Mac) | 824-923 | 769-875 |

Dense quantized GEMM: 6.0-6.2 TFLOP/s (MLX 6.2-6.6). Expert GEMM: ~4.2 TFLOP/s.
Tile anomaly: 32x64 and 32x128 tiles (2 row fragments per simdgroup) run at 2-3 TFLOP/s: they
keep 2.3-4.4 KB of fragments on the stack (see Attention below).

Attention (D=256, 16 q / 2 kv heads), 2048 queries x 16K keys: ours 200 ms (2.56 TFLOP/s), MLX
steel attention 97 ms (5.3). Tried: GQA heads as fragment rows (v2), in-register softmax (v3/v4,
slower: spills), head dim split over simdgroup pairs like MLX (v5), K^T tile (v6), V load overlapped
with softmax (v7), 16-byte tile loads (v8, -10%), 8 queries/16-key tiles (v9, slower), per-lane
fragment loads (v10). Ablation: device K/V loads ~25%, Q.K ~30%, P.V ~15%, the rest bookkeeping.
Open: find what MLX does differently (it runs 32 rows of ONE head per threadgroup).

Found (compile-time evidence; `NI_ARCHIVE_DIR=<dir> ni_compile <kernels>` prints machine-code size,
per-thread stack and GPRs from a Metal binary archive): v1-v10 keep their fragment arrays on the
stack. The compiler does not fully unroll the 16 x 4 fragment loop nests, so o[], qf[], s[] are
indexed at run time and demoted to memory: v10 6912 B of stack per thread with 47 GPRs (v1-v9:
6.4-13 KB). Every MMA then moves its operands through memory, which fits the ablation (loads and
both products all expensive). MLX unrolls every tile loop. The same v10 with only
`#pragma clang loop unroll(full)` added (attn_prefill10u_q4_k32): 96 B stack, 96 GPRs.
Not the cause: `max_total_threads_per_threadgroup` (v10 compiles to byte-identical code with it).
MLX's `simdgroup_barrier(mem_none)` fences around fragment loads raise an isolated P.V loop from
52 to 92 GPRs plus spills here, so v11 leaves them out. The compiler caps at ~92-96 GPRs and
spills beyond that.
The same thing explains the GEMM tile anomaly: qmm_b4_m32_n64 2304 B, qmm_b4_m32_n128 4352 B,
qmm_moe_b4_m32_n64 2304 B of stack; m64_n64 / m16_n64 / m32_n32 have none.
Plain-decode matvecs (qmv2/qmv3) have no stack; the MTP-verify ones spill a little at the
96-GPR cap: qmvr_multi_t2/t3 16/48 B, qmvr4_b8_t2 16 B, and 224-304 B for qmvr_b*_t3/t4 (R >= 3).

v11 (`NI_ATTN=attn_prefill11`, opt-in): MLX's layout, 32 queries of ONE query head per
threadgroup (grid ceil(T/32) x Hq), 4 row groups x 2 head-dim halves, everything unrolled. Also:
no padded rows for G = 6 (27B; v10 wastes a quarter of its MMAs there), causal mask only on
diagonal tiles, exp2 softmax with a finite sentinel, fp32 P, contiguous 64-byte K row fetches,
heaviest query blocks first. Variants: `_x` (own 8 KB exchange buffer, 2 barriers fewer per
tile), `_k16` / `_k16x` (16-key tiles, 12 / 16 KB). All 92 GPRs, 64-128 B residual spill (v10: 6912).
Not yet timed (GPU was busy): run bench_attn with v10, v10u and the v11 variants (it now checks
every kernel against a CPU reference) and `ni_run --dump-logits` A/B before switching the default.

Gated DeltaNet prefill (NI_GDN_PAR=1, opt-in, not measured yet). Default: gdn_prep loops over T per
channel (C/128 = 64 threadgroups on the 35B, two barriers per token for the q/k norm), gdn_step one 1024-thread
threadgroup per value head (32 on the 35B) with a threadgroup reduction per token; gdn_step was ~1.5 ms
per 512 tokens, ~6% of a 4k prefill. NI_GDN_PAR: gdn_prep_par (a simdgroup per 8 tokens x 128
channels, norm via simd_sum), gdn_step_par_l{4,8,16,32} (value rows of a head are independent:
spread over simdgroups/threadgroups, no barriers; L lanes per row, 128/L state floats per lane;
NI_GDN_STEP picks L, default 4; NI_GDN_SGS simdgroups per threadgroup, default 2), gdn_gnorm (gated
RMSNorm, writes out_proj's half input, replacing rows_to_half). Sums keep the default kernels' order
in the source, but fast math reassociates them per kernel (the machine code of gdn_prep, gdn_step,
rows_to_half and every new kernel changes under `#pragma clang fp reassociate(off)`), so the paths
should agree only to rounding by default and bit for bit with NI_MATH_SAFE=1 (safe math, A/B checks
only). Estimated instructions per state element per token ~5.5 (L=4) vs ~11 in gdn_step. Compiled
register use: gdn_step_par_l4 96 GPRs + 64 B spilled to stack, l8 69, l16 50, l32 33 (gdn_step 55),
so l8 may beat the l4 default. Check/measure: `bench_gdn 512 32 16` (35B), `bench_gdn 512 48 16` (27B).


## Qwen3.8-27B 4-bit (dense, 64 layers, hidden 5120)
| mode | decode tok/s | tok/cycle |
|---|---|---|
| plain | 16.0 | 1 |
| MTP depth 1 | 25.3 (harness 23.9-24.4) | 1.88 |
| MTP depth 2 | 23.6 | 2.39 |
| MTP depth 3 | 21.6 | 3.04 |
Plain decode reads 15.1 GB/token: 16.0 tok/s = 242 GB/s. Prefill 111 tok/s at 4k (dense GEMM ~6 TFLOP/s).
Small-batch kernels on M4 Pro are FMA-bound beyond 2 rows: the simdgroup-matrix route pads rows to 8 and runs on
the same ALUs, and the prefill GEMM kernel reaches only ~70 GB/s at 4-16 rows (tile staging per step).

## 8-row verify (DFlash2 target side; `src/model/qwen35_v8.cpp`, `src/kernels/verify8.metal`)
`encode_verify` with R >= 5 rows (anchor + up to 7 drafts) runs the 8-row path; R <= 4 (MTP depths 1-3) keeps
the MTP-era kernels unchanged (`NI_V8=1` sends R >= 2 to the 8-row path for A/B). `kMaxVerify` is 8; the MTP
history limit (`kMaxMtpRows` = 4) and the engine's MTP depth clamp (3) are unchanged. macOS 27.0.1, M4 Pro,
every number from runs under the GPU lock (A/B interleaved within one locked session; the GPU's speed drifted by
up to ~10% between sessions, so compare within a table only).

Kernels (all new, the default path dispatches none of them):
* `qmm8*`: Y^T = W X^T on 8x8x8 simdgroup MMAs, weight = A operand as exact integer codes (`1024 + q` in half,
  two integer ops per two weights; idea from Splash's Apple9 register tile), one fp32 epilogue per 64-group
  (`y += s * dot + (b - 1024 s) * sum x`), MLX weight layout read directly (no repacked copy). Inputs are
  "X8 tables" (half, already in the B-fragment order, plus per-group sums) written by `x8_prep2`
  (K/256 threadgroups, coalesced stores; fused RMSNorm). Up to 4 segments per dispatch (GDN qkv|z|b|a,
  attention q|k|v, router|shared gate), in-threadgroup K split with a fixed-order sum (deterministic).
  Default inner loop `qmm8pf` (next group's loads issued before the current MMAs), 2 fragments per simdgroup.
* `qmm8_gu1_b4`: gate + up + SwiGLU for 8 simdgroups x (1 gate + 1 up fragment) = one 64-column group of h
  per threadgroup, writing the down GEMM's X8 table and sums directly (no fp32 h, no prep dispatch).
* MoE: `moe8_plan` (one threadgroup) groups the 64 (row, expert) routes by distinct expert (ascending ids)
  and writes indirect grids; experts used by >= 3 rows and the shared expert run as 8-row MMA tiles
  (`qmm8_gu1_b4`, `moe8_down_nf2`), experts used by 1-2 rows as register-dequant matvecs over just those rows
  (`moe8s2_gu/down`); `moe8_combine` sums the 8 routes + shared in slot order (fp32, deterministic). Router
  and shared gate: `qmm8x2_b8` (two-term hi+lo half inputs, exact codes: ~22-bit inputs, fp32-level routing).
* Attention: `attn8m_partial` (one threadgroup per history chunk x kv head serves all G heads x 8 rows on
  MMAs, attn_prefill11's inner loop, each K/V tile read once), reduced by `attn_rows_reduce`.
* GDN: `gdn_prep_verify` (unchanged) + `gdn_step_verify8` (gdn_step_verify's math, one normalization pass
  after the 8-row scan instead of two barriers per row); the tape and pend buffers hold 8 rows.
* `capture8`: optional bf16 copy of target layer outputs for a drafter (`set_verify_capture`).

GEMM vs the 1-row decode matvec (`bench_v8 sweep <model>`: ~1 GB of rotating weights, best of 4; error vs an
fp64 CPU reference on the same half-rounded inputs, 2-4e-5):

| 27B shape (N x K) | 1-row us | 8-row us | 8-row GB/s | ratio | ratio, 6 sessions |
|---|---:|---:|---:|---:|---:|
| GDN in_proj qkv\|z\|b\|a 16480 x 5120 | 205.1 | 233.8 | 203 | 1.14 | 1.13-1.25 |
| attention q\|k\|v 14336 x 5120 | 181.6 | 213.6 | 193 | 1.18 | 1.06-1.26 |
| GDN out / attention o 5120 x 6144 | 80.5 | 94.0 | 188 | 1.17 | 1.09-1.17 |
| MLP gate+up 34816 x 5120 | 406.0 | 508.7 | 197 | 1.25 | 1.12-1.35 |
| MLP down 5120 x 17408 | 225.2 | 268.3 | 187 | 1.19 | 1.13-1.32 |
| LM head 248320 x 5120 | 2938 | 3467 | 206 | 1.18 | 1.09-1.27 |

| 35B shape (N x K) | 1-row us | 8-row us | 8-row GB/s | ratio |
|---|---:|---:|---:|---:|
| GDN in_proj 12352 x 2048 | 64.7 | 78.5 | 181 | 1.21 |
| attention q\|k\|v 9216 x 2048 | 47.2 | 58.0 | 183 | 1.23 |
| GDN out / attention o 2048 x 4096 | 23.7 | 32.7 | 144 | 1.38 |
| one expert's gate+up 1024 x 2048 | 8.2 | 11.9 | 99 | 1.44 |
| one expert's down 2048 x 512 | 4.7 | 8.1 | 73 | 1.71 |
| router 256 x 2048, 8-bit | | 10.2 | | |
| LM head 248320 x 2048 | 1104 | 1347 | 212 | 1.22 |

A single expert matrix is launch-bound in isolation (the 1-row kernel reaches only 120-145 GB/s on it too); inside
the MoE dispatch an 8-row expert tile costs ~5 us for gate+up (1.18 MB) and ~2.7 us for down.
The bench times `qmm8pf` on the concatenated N; in the model the MLP gate+up runs as `qmm8_gu1_b4` (SwiGLU
epilogue), measured 531-557 us per 27B layer in the profiles.

Inner-loop variants (27B gate+up 34816x5120): base 507-537 us, prefetching (default) 497-525, split MMA chains
(`qmm8s`, `qmm8pfs`) 512-532 (within noise of the default), 4 fragments per simdgroup 519-535, threadgroup-staged
input table (cs simdgroups share each chunk, `qmm8t`) 533-600. Ablations: without MMAs 487 us, without DRAM streaming (weights cache-resident) 472, without
table reads 512: no single resource dominates; the MMA work alone (2.85 GFLOP at the 7.41 TFLOP/s MMA ceiling)
is 385 us, the 1-row time 405 us, and the loop overlaps both at ~82% efficiency.

Teacher-forced check (`ni_v8check <model> @ids n R`): plain greedy decode of n tokens (logits kept), then a
fresh prefill and encode_verify of the same tokens R rows at a time (all rows committed, so every verify also
folds 8 tape rows), each row's logits compared with the plain step predicting the same position:

| model | prompt | rows | argmax = plain | \|logit diff\| at the plain argmax: median / p95 / max | old R = 4 kernels (same test) |
|---|---|---:|---:|---|---|
| 27B | p700 | 64 | 64/64 | 0.0017 / 0.0054 / 0.023 | 0.0014 / 0.0049 / 0.015 |
| 27B | code_27b | 64 | 64/64 | 0.0007 / 0.0032 / 0.026 | |
| 27B | p4000 | 32 | 32/32 | 0.0017 / 0.021 / 0.067 | |
| 35B | p700 | 64 | 64/64 | 0.061 / 0.23 / 1.34 | 0.046 / 0.31 / 0.63 |
| 35B | code_35b | 64 | 64/64 | 0.018 / 0.13 / 0.37 | |
| 35B | p4000 | 32 | 32/32 | 0.0097 / 0.13 / 0.43 | |

The 35B's larger spread is its MoE routing (a top-8 boundary moved by rounding), at the level of the existing
R = 4 verify; the 27B's is ~1e-3. No argmax differed in any of the 320 rows.

Full verify (embedding, all layers, LM head, top-k/sampling for every row) vs one plain decode step, GPU time
per command buffer:

| model | prompt (context) | plain step ms | 8-row verify ms | ratio | old kernels, R = 4 |
|---|---|---:|---:|---:|---:|
| 27B | p700 | 72.6 / 73.1 | 90.1 / 91.4 | 1.24 / 1.25 | 150.0 ms (2.11x) |
| 27B | code_27b (85 + 64) | 71.2 | 89.6 | 1.26 | |
| 27B | p4000 | 72.6 | 92.5 | 1.27 | |
| 35B | p700 | 10.04 / 10.13 | 24.9 / 24.4 | 2.48 / 2.41 | 27.6 ms (2.75x) |
| 35B | code_35b (85 + 64) | 10.07 | 22.6 | 2.24 | |
| 35B | p4000 | 10.44 | 26.8 | 2.57 | |

Same binary, same session per model (the 27B plain step drifted between 63.5 and 73 ms over the day; within a
session the ratio stayed 1.22-1.33). The 35B verify reads every touched expert once: 39.0 distinct routed experts
per layer on p700, 34.1 on code_35b, 41.2 on p4000 (of 64 routes), i.e. ~4.3 GB per verify against 1.79 GB per
plain step; its ratio is that byte ratio (2.4x) plus ~5%. Improvements along the way (35B p700 / 27B p700,
interleaved A/B): prefetching GEMM loop -1.2 ms / -0.5 ms; gate/up as 8 simdgroups x (1+1) fragments -0 / -2.5 ms;
MMA attention -0.6 to -1.3 ms (p4000: 26.8 vs 28.6 ms on the 35B); scalar tiles for 1-2-row experts -0.8 ms
(35B); the remaining switches were within noise.

Per-kernel profile of one 8-row verify (`NI_PROFILE=1`, isolated dispatches, so small kernels are inflated by
~5 us each), 27B p700: `qmm8pf` 46.7 ms (193 dispatches), `qmm8_gu1_b4` 35.7 (64), `x8_prep2` 4.0 (193),
`gdn_step_verify8` 2.6 (48), `attn8m_partial` 1.2 (16), `gdn_prep_verify` 1.0, rest < 1 ms; 92.3 ms in total.
35B p700 (40 routed experts per layer): scalar expert gate/up `moe8s2_gu` 7.6 ms + down `moe8s2_down` 4.4 (the
~27 experts with 1-2 rows), MMA expert tiles `qmm8_gu1_b4` 2.1 + `moe8_down_nf2` 1.0 (~13 + the shared expert),
dense `qmm8pf` 6.2 (81), `gdn_step_verify8` 1.1, `x8_prep2` 1.0 (121), `attn8m_partial` 0.5, router `qmm8x2_b8` 0.45,
`moe8_plan` 0.39, `moe8_combine` 0.27, rest < 0.5; 26.1 ms in total. Experts are ~58% of the verify.

MTP (R <= 4) is untouched: `ni_run --spec D` ids identical to 45705ec for D = 0..3 on code_* (128 tokens), same
speeds. `ni_run --spec D` now also runs D = 4..7 through the 8-row verify (partial commits of the tape every
cycle): greedy output identical to plain decoding over 256 tokens on both models (D = 7 and 4). On the 27B,
MTP D = 7 gives 31.2-33.8 tok/s at 4.05 tok/cycle (code_27b, 256 tokens) against 25.0-27.4 for D = 2 (the default)
and 14.0-14.7 plain; on the 35B, D = 7 runs at 76-81 tok/s and D = 4 at 92-97, below D = 1 (107): the MTP head's
acceptance is too low there to pay for the expert union, which a block drafter's (DFlash2) is meant to fix.

Tried and dropped: scalar experts reading their inputs from device memory per 2 columns (29.0 vs 24.9 ms; the
staged version is the default), the staged-table GEMM and the ablation kernels (templates kept in verify8.metal,
not instantiated). Left opt-in: `NI_V8_GU=2` (4 simdgroups x 2+2 fragments: +2.5-3 ms on the 27B), `NI_V8_GEMM=base|s|pfs`, `NI_V8_GU_PF=1`,
`NI_V8_PREP=1` (8-threadgroup prep), `NI_V8_GDN=0`, `NI_V8_ATTN=0` (scalar attention with K/V tiles shared by 4-5
rows), `NI_V8_MOE_SMALL=0` (all experts as MMA tiles), `NI_V8_ROUTER_X2=0` (half-input router: 0.4 ms faster,
p95 logit noise 0.45 vs 0.23).

## Register spills (`ni_kstats --stats`)
The AGX pipeline object records each kernel's temporary registers and stack spill per thread;
`ni_kstats --stats [kernels]` prints them (no GPU work). The 32-row GEMM "tile anomaly" is a spill:
qmm_tile's store loop (simdgroup_barrier inside) stays rolled for FM=2, so `acc[i][j]` is indexed
dynamically and the whole accumulator array (plus the A fragments) lives in stack memory: m32_n64
spills 2304 B/thread (8 x 256 B accumulators + 2 x 128 B), m32_n128 4352 B; m16/m64 tiles and
m32_n32 do not. Full-unroll pragmas on the two store loops remove it (m32_n64: 44 registers, m32_n128:
59, both 0 spill; m64_n64 unchanged at 56). The bench-only attn_prefill9_q8_k32 fails pipeline creation
(33280 B threadgroup memory > 32 KB), so `ni_kstats --stats` over the whole library exits 1.
Every attention prefill kernel spills too (attn_prefill10_q4_k32: 6912 B, older variants up to
13 KB): a likely cause of the 2x gap to MLX's steel attention. Several qmv/qmvr verify kernels hit
the 96-register ceiling and spill 16-300 B.

## MoE prefill v2 (opt-in, `NI_MOE_FUSED`), not yet measured
`src/kernels/qmm_moe.metal`. Routed experts: `moe_bucket2` -> `qmm_moe_gu` (gate and up from one X
tile, SwiGLU in the epilogue, half h in expert-sorted rows) -> `qmm_moe_dn` (-> fp32 rows by pair).
`NI_MOE_FUSED=1` folds the shared expert in as expert E over the T token rows (two dispatches for
all expert work; the dense sg/su/swiglu/sd dispatches go away); `=2` keeps it on the dense path.
Tiles: spill-free, 64 rows by default (`NI_MOE_BM=16|32|64`), row fragments past the tile's rows
skipped per simdgroup, fragments interleaved over simdgroup rows; for full tiles the gate/up tile
does the same work per K step as the dense m64_n64 GEMM (6 TFLOP/s). Outputs should equal the
original path bit for bit (same dequantized halfs, same MMA order): `bench_moe` checks this and
times both paths on one synthetic layer. Registers: gu m64 82, dn m64 71 (dense m64_n64: 56).
Estimate at 2048-token chunks: expert GEMMs 23 -> ~17-19 ms/layer, swiglu pass and fp32 hg/hu
(64 MB written + 80 MB swiglu traffic per layer) gone: prefill +5-10% (4k ~800-840 tok/s).

## Same Mac, other engines (tok/s, M4 Pro)
nace-infer from the tables above (ni_run); the others from research/inference/REPORT.md (servers,
TTFT on real 4,133 / 16,285-token prompts, MTPLX-optimized 4-bit MLX weights; llama.cpp on GGUF
UD-Q4_K_M). Not one harness, so treat differences under ~5% as noise.

| Qwen3.6-35B-A3B 4-bit | decode | prefill 4k | prefill 16k |
|---|---|---|---|
| nace-infer plain | 105 | 765-785 | 600-606 |
| nace-infer MTP depth 1 | 117-123 | (same) | (same) |
| mlx-lm 0.32.0 server | 63-67 (raw generate 81) | 923 | 824 |
| oMLX 0.7.1.dev1 | 84-99 | 840-915 | 769-875 |
| MTPLX 2.12.2, MTP depth 1 | 66-76 | 890-902 | 788-792 |
| llama.cpp b11429 (Metal) | 53 | 733 | 681 |
| llama.cpp + draft-mtp | 66-73 | 719 | 652 |

| Qwen3.8-27B 4-bit | decode | prefill 4k | prefill 16k |
|---|---|---|---|
| nace-infer plain | 16.0 | 111 | - |
| nace-infer MTP depth 1 | 23.9-25.3 | (same) | - |
| mlx-lm | 10.7-11.4 | 125 | 101 |
| oMLX | 11.8-12.0 | 128 | (memory guard) |
| MTPLX (MTP) | 20.6-30.8 | 113-125 | 101-110 |

Attention share of nace-infer prefill (35B, 10 attention layers, v10 at 2.56 TFLOP/s): ~0.5 s of
5.3 s at 4k, ~8.6 s of 27 s at 16k. At MLX's 5.3 TFLOP/s that becomes ~0.26 s / ~4.2 s, i.e.
about 805 tok/s at 4k and 715 at 16k (projection, not measured).
