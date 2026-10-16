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
fp64 CPU reference on the same half-rounded inputs, 2-4e-5), with the model's tiling (`qmm8pf`, 2 fragments per
simdgroup, threadgroup = 2 column blocks x 4 k-slices; 8 k-slices for small grids), two runs in one session:

| 27B shape (N x K) | 1-row us | 8-row us | 8-row GB/s | ratio |
|---|---:|---:|---:|---:|
| GDN in_proj qkv\|z\|b\|a 16480 x 5120 | 197.7-199.9 | 223.0-223.7 | 212 | 1.12-1.13 |
| attention q\|k\|v 14336 x 5120 | 173.1-183.5 | 197.7-198.0 | 209 | 1.08-1.14 |
| GDN out / attention o 5120 x 6144 | 79.0-79.5 | 89.6-89.8 | 197 | 1.13-1.14 |
| MLP gate+up 34816 x 5120 | 403.2-403.7 | 456.3-464.7 | 216-220 | 1.13-1.15 |
| MLP down 5120 x 17408 | 220.3-222.4 | 255.6-255.9 | 196 | 1.15-1.16 |
| LM head 248320 x 5120 | 2851-3186 | 3169-3205 | 223-226 | 0.99-1.12 |

| 35B shape (N x K) | 1-row us | 8-row us | 8-row GB/s | ratio |
|---|---:|---:|---:|---:|
| GDN in_proj 12352 x 2048 | 58.7-66.8 | 74.3-78.2 | 182-192 | 1.11-1.33 |
| attention q\|k\|v 9216 x 2048 | 45.0-45.9 | 57.5-57.6 | 185 | 1.25-1.28 |
| GDN out / attention o 2048 x 4096 | 23.4-23.5 | 30.6-31.2 | 151-154 | 1.30-1.33 |
| one expert's gate+up 1024 x 2048 | 7.9-8.0 | 13.5-13.9 | 85-88 | 1.71-1.74 |
| one expert's down 2048 x 512 | 4.7-4.8 | 7.4-7.6 | 78-79 | 1.59-1.60 |
| router 256 x 2048, 8-bit | | 7.2-7.3 | | |
| LM head 248320 x 2048 | 1088-1107 | 1294-1297 | 221 | 1.17-1.19 |

The first plan (4 column blocks per threadgroup, k-slices only until the grid had 1280 simdgroups) measured
gate+up 500-507 us and the LM head 3354-3380 us in the same tiling sweep (cs 2 / ks 4: 452-457 and 3141-3199).
A single expert matrix is launch-bound in isolation (the 1-row kernel reaches only 120-150 GB/s on it too); inside
the MoE dispatch an 8-row expert tile costs ~5 us for gate+up (1.18 MB) and ~2.7 us for down.
The bench times `qmm8pf` on the concatenated N; in the model the MLP gate+up runs as the SwiGLU kernel
(`qmm8_gu1k2_b4`: 8 column slices x 2 k-slices, 512 threads; vs `qmm8_gu1_b4` without the K split: 27B p700
78.2 vs 79.6 ms per verify, 35B 21.7 vs 22.1, interleaved).

Inner-loop variants (27B gate+up 34816x5120, first tiling plan: 4 column blocks, no K split): base 507-537 us, prefetching (default) 497-525, split MMA chains
(`qmm8s`, `qmm8pfs`) 512-532 (within noise of the default), 4 fragments per simdgroup 519-535, threadgroup-staged
input table (cs simdgroups share each chunk, `qmm8t`) 533-600. Ablations: without MMAs 487 us, without DRAM
streaming (weights cache-resident) 472, without table reads 512: no single resource dominates; the MMA work alone
(2.85 GFLOP at the 7.41 TFLOP/s MMA ceiling of bench_mma) is 385 us and the 1-row time 405 us; with every other
instruction of the loop included the kernel runs at ~80% of that MMA ceiling. The 2 x 4 tiling then took
gate+up to 452-465 us (1.13-1.15x).

Teacher-forced check (`ni_v8check <model> @ids n R`): plain greedy decode of n tokens (logits kept), then a
fresh prefill and encode_verify of the same tokens R rows at a time (all rows committed, so every verify also
folds 8 tape rows), each row's logits compared with the plain step predicting the same position:

| model | prompt | rows | argmax = plain | \|logit diff\| at the plain argmax: median / p95 / max | old R = 4 kernels (same test) |
|---|---|---:|---:|---|---|
| 27B | p700 | 64 | 64/64 | 0.0014 / 0.0044 / 0.011 | 0.0014 / 0.0049 / 0.015 |
| 27B | code_27b | 64 | 64/64 | 0.0010 / 0.0025 / 0.051 | |
| 27B | p4000 | 32 | 32/32 | 0.0018 / 0.012 / 0.087 | |
| 35B | p700 | 64 | 64/64 | 0.026 / 0.21 / 0.41 | 0.046 / 0.31 / 0.63 |
| 35B | code_35b | 64 | 64/64 | 0.0098 / 0.14 / 0.79 | |
| 35B | p4000 | 32 | 32/32 | 0.0051 / 0.13 / 0.13 | |

The 35B's larger spread is its MoE routing (a top-8 boundary moved by rounding), at the level of the existing
R = 4 verify; the 27B's is ~1e-3. No argmax differed in any of the 320 rows.

Full verify (embedding, all layers, LM head, top-k/sampling for every row) vs one plain decode step, GPU time
per command buffer:

| model | prompt (context) | plain step ms | 8-row verify ms | ratio | old kernels, R = 4 |
|---|---|---:|---:|---:|---:|
| 27B | p700 | 64.21 / 63.55 | 79.70 / 77.91 | 1.24 / 1.23 | 135.8 ms (1.98x) |
| 27B | code_27b (85 + 64) | 63.64 | 80.23 | 1.26 | |
| 27B | p4000 | 66.26 | 82.90 | 1.25 | |
| 35B | p700 | 9.58 / 9.61 | 23.32 / 23.42 | 2.43 / 2.44 | 26.56 ms (2.77x) |
| 35B | code_35b (85 + 64) | 9.47 | 21.62 | 2.28 | |
| 35B | p4000 | 9.96 | 25.41 | 2.55 (scalar attention: 26.95) | |

Final binary, one locked session per model and table row group (the 27B plain step drifted between 62.4 and 73 ms
over the day across sessions; within a session the 27B ratio stayed 1.22-1.33 through the whole tuning). The 35B
verify reads every touched expert once: 39.0 distinct routed experts per layer on p700, 34.1 on code_35b, 41.2 on
p4000 (of 64 routes), i.e. ~4.3 GB per verify against 1.79 GB per plain step; its ratio is that byte ratio (2.4x)
plus ~5%. Improvements along the way (35B p700 / 27B p700, interleaved A/B): prefetching GEMM loop -1.2 ms / -0.5
ms; gate/up as 8 simdgroups x (1+1) fragments -0 / -2.5 ms, plus the 2-way K split -0.4 / -1.4 ms; MMA attention
-0.6 to -1.3 ms (p4000: 26.8 vs 28.6 ms on the 35B); scalar tiles for 1-2-row experts -0.8 ms (35B); the remaining
switches were within noise.

Per-kernel profile of one 8-row verify (`NI_PROFILE=1`, isolated dispatches, so small kernels are inflated by
~5 us each), final binary. 27B p700: `qmm8pf` 40.6 ms (193 dispatches: GDN in/out, attention q|k|v/o, MLP down,
LM head), `qmm8_gu1k2_b4` 31.9 (64), `gdn_step_verify8` 2.5 (48), `x8_prep2` 2.3 (193), `attn8m_partial` 1.1 (16),
`gdn_prep_verify` 0.8, rest < 0.5; 79.6 ms in total, 91% in the GEMMs. 35B p700 (40 routed experts per layer):
scalar expert gate/up `moe8s2_gu` 7.7 ms + down `moe8s2_down` 3.7 (the ~27 experts with 1-2 rows), MMA expert
tiles `qmm8_gu1k2_b4` 2.4 + `moe8_down_nf2` 1.0 (~13 + the shared expert), dense `qmm8pf` 5.9 (81),
`gdn_step_verify8` 1.1, `x8_prep2` 0.9 (121), `attn8m_partial` 0.5, router `qmm8x2_b8` 0.44, `moe8_plan` 0.39,
`moe8_combine` 0.25, rest < 0.5; 25.2 ms in total, experts 59%.

MTP (R <= 4) is untouched: `ni_run --spec D` ids identical to 45705ec for D = 0..3 on code_* (128 tokens), same
speeds. `ni_run --spec D` now also runs D = 4..7 through the 8-row verify (partial commits of the tape every
cycle): greedy output identical to plain decoding over 256 tokens on both models (D = 7 and 4). On the 27B, MTP D =
7 gives 31.2-35.9 tok/s at 4.05 tok/cycle (code_27b, 256 tokens; 35.9 with the final binary) against 25.0-27.5 for
D = 2 (the default) and 14.0-15.1 plain; on the 35B, D = 7 runs at 76-84 tok/s and D = 4 at 92-101, below D = 1
(107-112): the MTP head's acceptance is too low there to pay for the expert union, which a block drafter's
(DFlash2) is meant to fix.

Tried and dropped: scalar experts reading their inputs from device memory per 2 columns (29.0 vs 24.9 ms; the
staged version is the default), the staged-table GEMM and the ablation kernels (templates kept in verify8.metal,
not instantiated). Left opt-in: `NI_V8_GU=1|2|1k4|2k2|2k4` (`2` = 4 simdgroups x 2+2 fragments: +2.5-3 ms on the
27B; the K-split variants other than 1k2 within 1-3 ms), `NI_V8_PLAN=1`, `NI_V8_GEMM=base|s|pfs`, `NI_V8_GU_PF=1`,
`NI_V8_PREP=1` (8-threadgroup prep), `NI_V8_GDN=0`, `NI_V8_ATTN=0` (scalar attention with K/V tiles shared by 4-5
rows), `NI_V8_MOE_SMALL=0` (all experts as MMA tiles), `NI_V8_ROUTER_X2=0` (half-input router: 0.1-0.4 ms faster,
p95 logit noise 0.39-0.45 vs 0.21-0.23).

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

## DFlash2 block drafter (opt/dflash2-draft; opt-in `ni_run --dflash <drafter dir>` / `NI_DFLASH`)

Spec and oracle: `research/inference/dflash2/SPEC.md`, `dflash2_ref.py` (MLX). Code: `src/model/dflash2.{h,cpp}`,
`src/kernels/spec_dflash.metal`, capture hook in `Qwen35` (`CaptureHook`, `set_capture`), tools
`dflash_check` (oracle dumps, timing, `--sim` replay), `bench_qmm8`, `dflash_attn_test`, `dflash_qtest`.

* **Load**: config checked against SPEC §1; every 2-D projection quantized at load to MLX affine 4-bit
  (or 8-bit, `NI_DFLASH_BITS=8`) g64. Bit-exact with `mx.quantize`: MLX's Metal kernel (IEEE fp32,
  round half away from zero; what wrote the q4 dumps) by default, MLX's CPU backend (half to even)
  with `NI_DFLASH_QROUND=even` (`dflash_qtest` + numpy: 0 differing words/scales/biases). k|v of all
  layers stacked into one [NL*2048, H] matrix (one GEMM per context commit). Norms, conv bases and
  both codebooks stay bf16; embedding and LM head are the target's. 35B 0.51 GB resident (8-bit 0.71),
  27B 1.32 GB (8-bit 2.22); load 0.5-1.4 s.
* **Capture**: after each `T_ids` layer of `encode_step`/`encode_prefill`/`encode_verify`, rows in
  the armed position window are copied fp32 -> bf16 into `[rows, n*H]`. Unarmed it adds nothing:
  logits after prefill bitwise identical to main 45705ec and generated ids identical (p700, 64
  tokens, plain and MTP, both models).
* **Context path**: captures -> half (pre-scaled by 1/16, exact for bf16 values) -> `fc` -> RMSNorm
  (x16 folded back) -> stacked k|v GEMM -> `dfl_ctx_commit` (k_norm, RoPE at the row's position, bf16)
  into a 2048-slot ring per layer.
* **Pass**: one encoder, ~85 dispatches: embedding + norm, per layer {P_a GEMM, conv prepare,
  q|kv GEMM, q/k norm + RoPE, attention + split reduce, o GEMM, conv finish + residual + norm,
  P_m GEMM, conv prepare, fused gate/up/SwiGLU GEMM, down GEMM, conv finish + norm}, LM head +
  selector GEMM, two-stage top-16, edge lattice, walk (greedy or sampled with q).
* **8-row GEMM `dfl_qmm8`**: weights straight to registers (opt/shortpf qmms lane layout), exact
  codes as half MMA operands (Splash idea), fp32 per-group epilogue `s.C + b.sum(x)`, X as a half
  table with per-64 sums written by the producers. Interleaved sweep (`bench_qmm8`, warm GPU):
  175-207 GB/s and 5.0-5.9 TFLOP/s on the drafter shapes (LM head 248320x2048 1.38 ms = 206 GB/s,
  248320x5120 3.5 ms = 200 GB/s). 16 rows per simdgroup (`NI_DFLASH_RF=2`) never wins; a K split
  helps every shape with < 1024 threadgroups (heuristic: the largest of 1/2/4/8 leaving >= 8 groups
  per simdgroup). Fused gate/up gets its own K split inside its 64-column threadgroup.

| D1 vs the oracle q4 dumps (`code_*_q4`, 3 cycles) | 35B | 27B |
|---|---|---|
| ring K / V after the prompt commit (bf16) | 1.7e-3 / 1.7e-3 | 1.7e-3 / 1.7e-3 |
| c(j) | 9.6e-7 | 3.2e-6 |
| per-layer h (worst layer, worst cycle) | 2.5e-3 | 2.4e-3 |
| final hidden / draft logits | 1.7-2.1e-3 / 1.3-1.6e-3 | 2.1-2.8e-3 / 2.0-2.6e-3 |
| top-16 sets | 112/112 every cycle | 111/112, 112/112, 112/112 |
| edges (matched by token ids) | 0.9-1.0e-3 | 1.3-2.2e-3 |
| proposal chains | identical 3/3 | identical 3/3 |
| sampled walk (T 0.6/1.0, 3 seeds) | q = softmax(S/T) to 1.6e-7; index = inverse CDF of u in 42/42 | same, 3.1e-7 |

All numbers are relative L2 errors; the inputs were the dump's own captures and block embedding. The
intermediates of every layer (n, D, a, q, k, v, o, u, n2, D2, mm, hgu, f) are 3e-8 to 2.1e-3. With
8-bit weights against the bf16 dump: per-layer h 1.3-1.9e-2 (weight rounding), chains still identical.

| tokens per cycle, all 7 proposals verified (`dflash_check --sim N`, code_*, greedy, plain-decode numerics) | 35B | 27B |
|---|---|---|
| ours, 4-bit drafter, 256 tokens | **3.26** | **4.33** |
| ours, 8-bit drafter, 256 tokens | 3.30 | 4.22 |
| oracle serial verify, fp32 math, bf16 weights, 256 (SPEC §15.2) | 3.24 | 4.15 |
| oracle batched, q4 drafter, 256 (SPEC §15.3) | 3.19 | 4.42 |
| ours, 4-bit drafter, **512 tokens** | **3.72** | **4.79** |
| oracle 512 fp32 / q4; Splash (same ids, q4) | 3.65 / 3.62; 3.61 | 4.79 / 4.83; 4.83 |

Acceptance by draft position at 512: 35B 0.84 0.79 0.69 0.67 0.74 0.55 0.82 (oracle 0.79 0.81 0.76 0.65 0.70
0.52 0.81), 27B 0.86 0.88 0.79 0.86 0.84 0.83 0.76 (oracle 0.84 0.88 0.81 0.86 0.87 0.77 0.86). 4-bit is the
default: 8-bit buys nothing measurable and costs 26% (35B) / 38% (27B) more drafting time.

Drafting pass GPU time (one command buffer per pass, median of 10-20 runs, under the GPU lock; the range is
across lock slots, which differ by up to ~10% on this shared, warm Mac):

| | 35B-A3B | 27B |
|---|---|---|
| bytes read per pass (4-bit drafter + target LM head) | 0.49 GB | 1.65 GB |
| pass at P = 96 | 3.0-3.5 ms (min 3.02) | 9.3-10.0 ms (min 9.25) |
| pass at P = 4096 (full 2047-row window) | 3.5-4.0 ms (scalar attention 4.5-5.4) | 9.9-10.4 ms (scalar 10.9-11.4) |
| + context commit of the m kept rows (decode) | 0.4-0.9 ms | 0.6-0.8 ms |
| prompt context commit, 85 rows (8-row chunks / prefill GEMM) | 2.3-3.8 / 3.4-5.9 ms | 6.9-7.2 / 8.8-9.2 ms |

Profile of the 35B pass (`NI_PROFILE=1`, isolated dispatches, 3.4-3.6 ms): 8-row GEMMs 2.4-2.5 ms (LM head
~1.4), fused gate/up 0.51-0.53, attention + reduce 0.16 (0.73 at the full window before the 256-key splits),
top-16 0.09-0.16, conv/norm kernels 0.14, q/k prep 0.03 (0.09 before the per-pass RoPE table), lattice + walk 0.02
(0.05 as one threadgroup). 27B (10.0 ms): GEMMs 6.45 (LM head ~3.5, down ~1.6), gate/up 2.7 (0.54 ms per layer =
184 GB/s), the rest 0.8. The pass streams ~140-175 GB/s overall against a ~205 GB/s ceiling of the GEMM kernel.

End to end with the existing verify (`ni_run --dflash`, at most `kMaxVerify - 1 = 3` of the 7 proposals; greedy;
the full 8-row verify + speculative-sampling accept are other branches' work):

| prompt, 256 tokens unless noted | output vs plain greedy | tok/cycle | GPU/cycle | tok/s (plain) |
|---|---|---|---|---|
| 35B code_35b | identical 256/256 | 2.77 | 28.8 ms | 93.1 (107.9) |
| 35B p4000, 64 tokens (prompt > 2047: ring window, 2047-row commit) | identical 64/64 | 3.71 | 33.8 ms | 107.2 (94.8) |
| 27B code_27b | identical 256/256 | 3.15 | 152 ms | 20.6 (14.7) |

The 4-row verify of the existing path (R = 4: 2x the bytes-time of a decode step on the dense model, ~25 ms on
the MoE) dominates the cycle: with an 8-row verify near 1.15-1.35x decode the same drafts give the SPEC §11 cycle.
(Superseded by the full loop below: 8-row verify, on-GPU acceptance, double-buffered cycles.)

## DFlash2 decode loop (opt/dflash2 = opt/dflash2-draft + opt/dflash2-verify8 + the loop)

Opt-in: `ni_run <model> @ids n T --dflash <drafter dir|auto>`, server `--dflash <dir|auto>` (or `NI_DFLASH`).
`auto` = `DFlash2::find`: `<model>/dflash2`, `<model>/drafter`, then `*-DFlash2` directories next to the model, in
`<parent>/drafters` and in `~/.cache/nace-infer-models/drafters` whose name minus `-DFlash2` (case-insensitive) is the
model directory name up to a `-` and whose config is a `DFlash2DraftModel` with the target's hidden size, vocabulary
and layer count (qwen3.6-35b-a3b-4bit -> Qwen3.6-35B-A3B-DFlash2, qwen3.8-27b-4bit-mtp -> Qwen3.8-27B-DFlash2). When a
drafter loads it replaces MTP; without the flag nothing changes (G3 below).

**One cycle** (anchor `y` at position `P`, SPEC §3), no host round trip inside:
1. drafting half: ring commit of the rows the previous acceptance kept (their verify captures -> fc -> norm ->
   stacked k|v GEMM -> K/V ring), then the drafting pass from the anchor (7-token chain; q at T > 0);
2. verify half: `dfl_depth` (MoE targets, below), the 8-row verify of `[y, d_0 .. d_6]` with the drafter's
   `CaptureHook` armed (the verify branch's `capture8` / `set_verify_capture` were folded into it), `dfl_dist_rows`
   (T > 0) and `dfl_accept`, which writes the bonus / correction token to `tokens[P+1+a]`;
3. the host reads `a`, emits `tokens[P+1 .. P+1+a]` (cut at a stop token / the budget: `m` rows), `commit_rows(s, m)`
   (the next verify folds `m` tape rows), `P += m`.
* **Double-buffered** (`NI_DFLASH_CB=3`, default in ni_run and the engine): `dfl_accept` also writes a drafter state
  (`gst`: next anchor `P + a + 1`, rows `a + 1`, their first position) and every position-dependent drafter kernel
  (embedding, RoPE table, attention split plan, lattice, walk, commit) reads it when the host passes the sentinel
  position, so the NEXT cycle's drafting half is committed right behind each verify, before the host waits for it;
  the host then encodes the following verify (~600 dispatches) while that drafting half runs. If the host cuts `m`
  below `a + 1` the generation ends, and the extra ring rows past the host's `d.ctx` are never read. `NI_DFLASH_CB=2`:
  drafting and verify halves as two command buffers committed back to back (host-driven positions); `=1`: one buffer.
  35B code_35b 512 greedy, interleaved: wall per cycle = GPU time + 0.04 ms (mode 3), + 0.25-0.30 (mode 2), + 0.69 (1).
* **Greedy acceptance**: longest prefix of the chain equal to the verify rows' argmax (the verify's own
  `topk_stage1_rows` / `sample_stage2_rows`, ties to the lowest id, as plain decoding), bonus = argmax of row `a`.
* **Sampled acceptance** (exact speculative sampling, SPEC §8.2): `dfl_dist_rows` rebuilds each verify row's
  distribution exactly as the plain sampler does (`sample_stage2_rows`: top-k ordered (value desc, id asc), top-p on
  the untempered probabilities, then temperature on the kept prefix); accept `d_i` iff `u_i q_i(d_i) < p_i(d_i)`;
  after a rejection draw from `max(0, p_a - q_a)` over the kept set (or `p_a` if empty), after `K` acceptances from
  `p_K`. Uniforms `pcg_hash(seed ^ pcg_hash(pos) ^ salt)`, separate salts for walk / acceptance / residual; a
  position's acceptance and residual uniforms are used at most once per generation.
* **Row-independent MoE numerics at T > 0** (`Qwen35::set_verify_moe_exact`, `NI_DFLASH_EXACT_MOE=0` off): the 8-row
  verify sends experts used by 1-2 rows through register-dequant matvecs and the others through MMA tiles, which round
  differently, so a row's logits move (up to ~0.1 on the 35B) with how the other, drafted, rows route; the sampled
  acceptance needs `p(. | prefix)` itself. Sampled cycles therefore run every routed expert as an MMA tile. Not slower
  (35B code_35b 512, T = 0.6, interleaved: 132.6 / 136.5 tok/s vs 130.6 / 130.5), and the S2 test on the same start
  point went from p = 0.086 / 0.039 (joint / first token, 2000 trials) to 0.202 / 0.162 (3000 trials).
* **Dynamic verify depth on MoE targets** (default tau 0.5 at T = 0 and 0.3 at T > 0; `NI_DFLASH_DYN=tau`, 0 = off;
  `NI_DFLASH_DYN_T` for T > 0; dense targets: off): the 35B verify's cost is the union of routed experts over its rows (SPEC §15.7: 31.5 per
  layer for 8 rows, 23.4 for 5), so `dfl_depth` decides on the GPU how many proposals to verify from the drafter's
  confidence `c_j` (the walk's probability of its choice: softmax of `S_j` at T = 0, `q_j(d_j)` at T > 0);
  `moe8_plan` routes only the live rows (a GPU-side row limit) and `dfl_accept` stops there.
  - T = 0: verify proposal `p` while `prod_{j <= p} c_j >= tau` (the greedy output is the verify argmax whatever is
    verified, so the proposal's own confidence may be used).
  - T > 0: verify `p` iff `prod_{j < p} c_j >= tau`. The decision about `d_p` must not depend on `d_p`: given the
    accepted prefix it is then fixed and every position is an ordinary speculative step or the bonus draw from
    `p_p`, so the emitted law is exact. The first version used the inclusive product at T > 0 too, which skips
    unlikely draws of `q_p` and draws from `p_p` instead: biased at positions >= 1 of a cycle; S2's first-token
    test cannot see it and its start point rarely accepted `d_0`, so `dflash_accept_test` now has dynamic-depth
    cases, including that rule, which it rejects. Measured before the fix (12 seeds, server, T = 0.6): no visible
    effect on the text (distinct-4-gram ratio 0.990 vs 0.992 for plain sampling, Mann-Whitney p = 0.56). Exact rule,
    35B code_35b 512 T = 0.6, tau 0 / 0.3 / 0.5 / 0.7: 139.2 / 149.4 / 147.8 / 140.3 tok/s (single runs).
  - Calibration (469 cycles, code / p700 / prose, 512 tokens, all 7 verified): c >= 0.99 -> 97% accepted,
    0.95-0.99 -> 81%, 0.7-0.8 -> 50%, < 0.5 -> 13-27%. Offline simulation with the measured cost (0.38 ms per expert of
    U) predicted +8-11%; measured (greedy, 512, interleaved): code 135.1 / 136.9 tok/s at tau 0 -> 144.4 / 146.7 at
    0.3 -> 147.0 / 146.6 at 0.5; prose 93.1 -> 111.8 -> 112.2. Sweep on the final loop: code 160.7 / 159.9 / 159.0 /
    154.8 tok/s at tau 0.4 / 0.5 / 0.6 / 0.7, p700 189.1 / 199.4 / 185.9 / 184.8, prose 115.3 / 119.6 / 109.9 / 113.1.
    Static depths are worse: 5 of 7 gave 126.7 / 130.0 vs 120.0 / 117.6 for 7 (256 tokens).
* **Prefix reuse** (engine): the drafter ring is per slot; on a checkpoint restore `DFlash2::rewind` keeps the entries
  of the reused prefix that are still in the 2048-slot window (`valid_lo`, the draft attention starts there), and the
  prompt capture covers only the rows that can still be seen, `[max(start, n - 2047), n)`. The end of a generation
  commits the last kept rows (unless the committed drafting half already did) and folds the tape.
* **27B drafter on the verify GEMMs** (`NI_DFLASH_V8`, default for hidden >= 4096): its MLP and LM head run on the
  verify path's `qmm8_gu1k2` / `qmm8pf` (inputs permuted into their fragment-order table by `dfl_x8_to_v8`; the per-64
  sums have the same layout). `bench_v8 dfl <model>` (same rotating weights, best of 6): 27B gate+up 509.6 vs 606.3 us,
  down 255.7 vs 300.2, LM head 3430 vs 3661, o 62.5 vs 66.2; 35B within +-5% either way (kept on `dfl_qmm8`). Pass:
  8.73 vs 9.19 ms (P = 96), 9.04 vs 9.54 (full window); D1 parity unchanged (worst per-layer h 2.4e-3).
* Tried and dropped: `gdn_step_verify8` with every per-row input staged in threadgroup memory (27B verify 80.4 / 80.7
  vs 79.6 / 80.5 ms, 35B 21.4 / 21.5 vs 21.6 / 21.4; identical logits); the pass's RoPE table (512 precise sin/cos,
  50-100 us) over 8 threadgroups instead of one (35B 49.7 -> 61.8 us, 27B 97.6 -> 97.9; the final-matrix binary still
  had it, bitwise the same values); static verify depths (above).

**Correctness** (final build unless noted; every run under the GPU lock):

| check | 35B-A3B | 27B |
|---|---|---|
| G1 code_* 512 greedy (`--ignore-eos`) vs plain greedy | identical 512/512 (6 runs of the final matrix too) | identical 512/512 (6 runs) |
| G1 p4000 + 256 greedy | first difference at token 197: plain top-2 gap 0.020; one multi-row prefill of the prefix picks the DFlash token (gap -0.170); the verify row -0.151 (G2: the target's own batching numerics). An earlier build (no dynamic depth) split at token 119: plain gap 0.273, verify row a tie (-0.0002), prefill +0.128 (the 35B's verify-vs-plain noise: p95 0.14, max 0.79 on code_35b) | identical 256/256 |
| chat prompt of 5349 tokens (engine_turns turn 3) | first difference at token 73: plain gap 0.0042, one prefill picks the DFlash token | |
| G3 default path vs 45705ec (base build) | logits after prefill bitwise identical, plain 64 and MTP depth 1 (128) ids identical | logits bitwise identical, plain 32 and MTP depth 2 (96) ids identical |
| D1 drafter parity (q4 dump) | chains 3/3, worst per-layer h 2.49e-3 | chains 3/3, 2.41e-3 (MLP/LM head on the verify GEMMs) |
| server tests/test_server.py with `--dflash auto` | 13/13 (T = 0.6 default; also with the double-buffered loop) | 13/13 |
| engine_turns (5168-token prompt, 2 follow-ups reusing 5161 tokens, T = 0) | 3.88 / 3.61 / 3.12 tok/cycle (the ring survives the rewind); turns 1-2 identical to plain decoding, turn 3 a near-tie (above) | |

* **S1** (`dflash_accept_test`, synthetic prefix-conditional p and q, joint law of the first two emitted tokens,
  40000-100000 trials per case): every case passes (random p/q, q = p, q disjoint from p, point masses, p(d) = 0, flat
  q, K = 1/3/7, T = 0.6/1.0, dynamic depth with the exact rule); `dfl_dist_rows` equals a CPU transcription of the
  plain sampler bit for bit; the depth-sensitive regime rejects the old inclusive T > 0 rule (chi2 446, 3 dof).
* **S2** (`dflash_stat`, real models: after a fixed prompt, a greedy start point with >= 3 kept tokens and top
  probability <= 0.8; DFlash2 cycles until two tokens are out, state restored per trial; chi-square against p(t1)
  p(t2 | t1) from the same 8-row verify path): all pass.

| model, start point | T | trials | MoE path | joint p | first token p |
|---|---:|---:|---|---:|---:|
| 35B prose (+23) | 0.6 | 2000 | 1-2-row experts on the scalar path | 0.086 | 0.039 |
| 35B prose (+23) | 1.0 | 2000 | scalar path | 0.432 | 0.086 |
| 35B prose (+23) | 0.6 | 3000 | row-independent (default at T > 0) | 0.202 | 0.162 |
| 35B prose (+23) | 1.0 | 3000 | row-independent | 0.658 | 0.165 |
| 35B code (+32) | 0.6 | 2000 | row-independent | 0.496 | 0.496 |
| 35B prose, dynamic depth on | 0.6 | 3000 | row-independent | 0.255 | 0.162 |
| 27B prose (+23) | 0.6 | 1200 | dense | 0.316 | 0.361 |

**Performance** (ni_run, `code_*` ids = chat-templated coding request, 512 tokens, `--ignore-eos`, the same seed
for every sampled run; configurations interleaved inside one locked session; tok/s = decode tokens / decode wall time):

| model, T | plain | current best (MTP) | DFlash2 | tok/cycle (DFlash2) | ms/cycle (GPU) |
|---|---:|---:|---:|---:|---:|
| 35B, greedy | 107.4 | depth 1: 123.3 / 123.3 / 125.2 | **161.7 / 162.9 / 161.7** (+31%) | 3.41 (3.59 verified) | 20.9-21.0 |
| 35B, T = 0.6 | | depth 1: 119.2 / 119.3 / 119.3 | **151.0 / 150.7 / 150.9** (+26%) | 3.50 (5.34 verified) | 23.1-23.2 |
| 27B, greedy | 15.9 | depth 2: 31.0 / 30.4 / 31.4 | **55.2 / 55.0 / 55.1** (+78%) | 4.78 | 86.4-86.8 |
| 27B, T = 0.6 | | depth 2: 31.2 / 31.1 / 31.2 | **47.7 / 47.7 / 48.0** (+53%) | 4.15 | 86.4-87.0 |

All 12 greedy runs (MTP and DFlash2) are identical to plain greedy over the 512 tokens. Splash on this Mac (SPLASH_RUN.md,
same ids, greedy, 512): 35B 124.6-132.4 tok/s at 3.61 tok/cycle, 27B 54.7 at 4.83. The 35B's tok/cycle is below the
oracle's 3.65 on purpose (dynamic depth; all 7 verified: 3.70); the 27B's 4.78 matches (oracle 4.79, Splash 4.83).
Longer context, p4000 + 256 greedy: 35B 183.9 tok/s (4.72 tok/cycle, 25.6 ms/cycle), 27B 80.1 tok/s (7.29 tok/cycle,
90.6 ms/cycle; a raw code continuation, plain 13.5).

Server (`research/inference/scripts/bench.py` decode phases against `nace-infer-server`, the app's streaming API:
middle-window decode rate, median of 3 x 512 tokens, thinking off; T = 0.6 / top_p 0.95 / top_k 20 and T = 0):

| model | default server (MTP) T = 0.6 / T = 0 | `--dflash auto` T = 0.6 / T = 0 |
|---|---|---|
| 35B | 123.3 (118.4-125.4) / 125.1 (123.4-127.0) | **151.6** (135.3-160.8) / **165.5** (156.3-167.9) |
| 27B | 30.5 (30.4-31.8) / 30.9 (30.6-31.4) | **52.4** (49.2-56.9) / **57.6** (57.2-57.7) |

(T = 0.6 sampled runs vary with the text drawn; the 27B's earlier DFlash2 server run, same code but before the
allocation fix below: 52.0 / 56.0.) Memory: the drafter adds its resident size to the process footprint, +0.48 GB
(35B) / +1.27 GB (27B) at load, plus a 48 / 40 MiB context ring per slot; after one request the servers measured
20 / 21 GB (35B, MTP / DFlash2) and 18 / 19 GB (27B) (`footprint -p`). Before `Gpu::alloc` got its own autorelease
pool the drafter's bf16 checkpoint (1.05 / 3.85 GB) was never freed after the 4-bit quantization (Metal keeps new
buffers in the caller's pool, and the C++ callers have none; same fix as opt/ane c2de003): +1.46 / +4.7 GB.

Profile of a cycle (`NI_PROFILE=1`, isolated dispatches, so small kernels are inflated by a few us each):
35B, code 96 tokens: 21.9 ms (first loop: 27.4): dense verify GEMMs (`qmm8pf`) 5.8, routed experts 8.6 (scalar 1-2-row
passes 6.2, MMA tiles 2.4; 13.2 before the dynamic depth), drafter 3.0 (8-row GEMMs incl. the 1.4 ms LM head 2.4, gate/up
0.5), GDN 1.5, x8 tables 0.85, router 0.42, plan 0.31, verify attention 0.28, draft attention 0.23. 27B, code 64
tokens: 90.7 ms: the 8-row GEMMs of verify and drafter (`qmm8pf` 46.5, `qmm8_gu1k2` 35.4) are 90%; GDN 3.4, x8 tables
2.4, remaining drafter GEMMs 1.7, attention 0.6. Both are bandwidth-bound: the 27B cycle reads ~16.7 GB (verify 15.0,
drafter 1.7), the 35B's ~3.6 GB at 3.6 rows.

Tools: `ni_run ... --dflash auto [--ignore-eos] [--ref <plain output>]` (`--ref`: at the first token that differs from a
reference run, the logits of both tokens from the verify row, plain decoding and one prefill of the prefix; `NI_TRACE`,
`NI_DFLASH_CONF=<file>` per-cycle confidences; `NI_DFLASH_CB`, `NI_DFLASH_VERIFY`, `NI_DFLASH_DYN[_T]`,
`NI_DFLASH_EXACT_MOE`, `NI_DFLASH_V8` switches), `dflash_accept_test [trials] [K] [stream] [case]` (S1),
`dflash_stat <model> @ids [trials] [T]` (S2), `engine_turns <model> <text> [--dflash auto] [--dump-ids p]` (multi-turn
prefix reuse through `Engine::chat`), `bench_v8 dfl <27b|35b>` (drafter GEMM shapes), `dflash_check` (D1, unchanged).

Open: the 27B cycle at a ~4k context is 90.6 ms (target 88; 86.4-86.8 at 85 + 512); further 27B gains are in the
8-row GEMMs (90% of the cycle, already ~1.15x a 1-row matvec) and small fusions (writing the X8 tables of the GDN /
attention outputs from their kernels: ~64 dispatches per verify). The GPU-driven drafting pass splits the attention
context in fixed 256-key chunks (the host-driven plan balances them; same math, other fp32 rounding). The drafter's
target-LM-head pass is ~40% of a 27B pass and ~45% of a 35B pass.

## 27B DFlash2 cycle (opt/dfl27 = 4cdc5a0 + the changes below)

Goal: Qwen3.8-27B decode with DFlash2 clearly faster than Splash, greedy and T = 0.6. Every number under the GPU lock;
the Mac's speed drifted by up to ~15% between sessions (other agents' jobs, thermals), so changes were measured with
`ni_run ... NI_AB=<knobs>` (below: two kernel settings alternate per decode cycle inside one run, so both see the same
GPU state; GPU ms per cycle by parity, 53 cycles each) and base vs new as interleaved separate runs.

**Result** (ni_run `code_27b`, 512 tokens, `--ignore-eos`, one session, base = 4cdc5a0 with only `NI_SEED` added to
ni_run, runs interleaved):

| | base 4cdc5a0 | opt/dfl27 | |
|---|---:|---:|---:|
| 27B greedy, tok/s (3 runs) | 51.8 / 54.7 / 54.7 | **61.7 / 60.4 / 61.8** | +14.1% |
| 27B greedy, GPU ms / cycle | 88.8 (mean) | 77.8 | 4.776 tok/cycle both |
| 27B T = 0.6, 10 seeds (42, 1-9), mean tok/s | 53.3 (46.7-59.4) | **60.6** (52.8-65.2) | +13.7% |
| 27B T = 0.6, seed 42 (PERF's single seed) | 46.7 (4.154 tok/cycle) | 59.3 (4.604 tok/cycle) | other text |
| 27B T = 0.6, GPU ms / cycle | 88.4 | 78.3 | |

This session's base ran 2.5% below the "DFlash2 decode loop" table above (55.0-55.2 greedy); scaled to that state the
new greedy is ~62.9 tok/s. Splash on this Mac: 54.7 tok/s on the same ids (SPLASH_RUN.md), 57.7 greedy / 55.2 T = 0.6
through the server harness.

35B (`code_35b`, same session, interleaved; every change below applies to its dense GEMMs, expert gate/up tiles, GDN
step and draft head):

| | base 4cdc5a0 | opt/dfl27 | |
|---|---:|---:|---:|
| 35B greedy, tok/s (3 runs) | 158.6 / 157.9 / 154.3 | **167.7 / 168.8 / 170.3** | +7.6% |
| 35B greedy, GPU ms / cycle, tok/cycle | 21.4-22.0, 3.407 | 19.7-20.0, 3.362 | |
| 35B T = 0.6, 14 seeds (1-14), mean tok/s | 154.5 (tok/cycle 3.552, 22.95 ms) | **163.5** (3.487, 21.29 ms) | +5.8% |
| 35B T = 0.6, seed 42 (3 runs) | 149.7 / 150.6 / 150.0 (3.500) | 136.0 / 136.4 / 136.3 (2.887) | other text |

At seed 42 the new verify numerics draw another text, one with lower acceptance: the new build with the full draft
head gives 2.903 tok/cycle there (so not the head), and with every new knob at its old value and the full head it
reproduces the base's output bit for bit (seed 42 T = 0.6: 3.500 tok/cycle, 146 cycles; also 27B code T = 0.6 and 35B
p4000 greedy: identical outputs). Over 14 other seeds tokens per cycle are within 2% and decode +5.8%.

**T = 0.6 vs greedy (lead 1).** No sampled-path overhead to remove: GPU ms per cycle is the same at T = 0.6 and T = 0
(98.64 vs 98.57 ms in one session; the sampled-only kernels, stage-1 top-20 + `dfl_dist_rows` + the residual draw, cost
~0.05 ms in the pipeline: a top-32 stage 1 from sorted runs and simdgroup selection, ids identical, measured -0.05 ms
and was dropped). The 13% gap of the table above is the text seed 42 happens to draw: tokens per cycle at T = 0.6 over
seeds 1-6 are 4.60 / 4.87 / 5.01 / 4.37 / 5.01 / 4.91 (mean 4.80, greedy 4.78); over seeds 42 + 1-9 above 4.72 (base)
and 4.74 (new). The walk's temperature is a free choice (the acceptance only needs q = the law d was drawn from;
`NI_DFLASH_QT` = T_q / T): T_q = 0.5 T gave 4.53 and the argmax chain (q one-hot) 4.42 over the same 6 seeds, so q =
softmax(S / T) (the spec, Splash, z-lab) stays the default.

**Reduced draft head (lead 3).** The drafter's top-16 now comes from the target LM head rows of the 65536 highest
ranked ids plus every id >= the first end-of-sequence id (added tokens and padding rows), gathered at load into its own
4-bit matrix (189 MB on the 27B, 75 MB on the 35B); the ids map back in `dfl_top16` (shared id map). Only the drafts
change; the verify always uses the full head. Ranking: Pulsar 1.1.7's `data/head-ranked.u32` (248320 ids, specials first,
then by frequency; Apache-2.0, sha256 9eca7a0f...c2c5), read from `<drafter>/head-ranked.u32` or `NI_DFLASH_HEAD_IDS`;
installed next to both drafters on this Mac. Without a list the full head is used (`NI_DFLASH_HEAD=n` forces n rows, in
id order without a list). Greedy 512 tokens, tok/cycle (tok/s), one session per model:

| rows | 27B code | 27B prose | 27B p700 | 35B code | 35B prose | 35B p700 |
|---|---|---|---|---|---|---|
| full 248320 | 4.776 | 2.555 (32.0) | 5.162 (61.8) | 3.430 (160.5) | 2.433 (126.3) | 4.154 (179.9) |
| 98304 | 4.776 | 2.555 (32.9) | 5.162 (63.1) | 3.430 (167.9) | 2.433 (130.1) | 4.154 (186.7) |
| **65536 (default)** | 4.776 | 2.568 (33.2) | 5.162 (63.7) | 3.362 (166.1) | 2.410 (131.2) | 4.154 (189.6) |
| 49152 | 4.776 | 2.481 (32.1) | 5.059 (62.8) | 3.362 (166.9) | 2.231 (123.9) | 4.121 (189.8) |
| 32768 | 4.645 | | | | | |
| id order 65536 / 32768 | 4.522 / 4.258 | | | | | |

27B code: GPU ms per cycle 110.1 / 106.3 / 104.9 / 103.7 / 103.1 for full / 98k / 65k / 49k / 32k in a slow session
(-5.2 ms at 65k; ~-3 ms in normal sessions). D1 (`dflash_check`) forces the full head.

**8-row verify GEMMs (lead 2).** `bench_v8 cmp <N> <K> <reps> <kernel[:nf:cs:ks]> ...` alternates kernels on the same
rotating ~1 GB of weights (min / median of 12); then each change by `NI_AB` in the real loop (B - A, ms per cycle):
* **CM 1, exact subnormal code operands** (`V8Knobs::cm`, `qmm8sn*`): the MMA's A operand is the bare code `q` read as
  a half subnormal (`q * 2^-24`, exact; Apple's half MMA keeps subnormals), the epilogue scales by `s * 2^24`. One
  integer op fewer per MMA (no OR) and no `1024 * sum(x)` term to cancel: GEMM error vs fp64 2-5e-7 instead of 2-3e-5;
  teacher-forced 27B verify vs plain decode: max |logit diff| at the plain argmax 0.025 instead of 0.051, max over the
  vocabulary 0.062 instead of 0.126. -0.67 / -0.80 ms.
* **Two groups per inner step** (`pair`, `qmm8snb*`): the scale and bias of groups g, g+1 in one 4-byte load each, half
  the loop iterations. -0.94 / -0.74 ms (needs an even number of groups per row).
* **Tiling** (`kplan`): 8 k-slices x 1 column block per threadgroup whenever a slice keeps >= 8 groups (K >= 4096), 16
  k-slices (512 threads, `qmm8snb16`) for K >= 16384. cmp (us, cs1/ks8 vs the cs2/ks4 plan): gate+up 427 vs 446, GDN
  in_proj 209 vs 223, attention o 84 vs 88, LM head 3038 vs 3103, down 232 (ks16) / 240 (ks8) vs 256. Back to the old
  plan: +3.9 ms (ks8 instead of ks16 for down: +0.2 / +0.3 ms). The 35B's shapes keep their plans (K = 2048) except
  the K = 4096 ones, which already ran 8 x 1.
* **MLP gate/up as two GEMM segments** (`gusplit`): gate and up as two segments of one `qmm8` dispatch (fp32 rows) and
  `swiglu_x8` building the down GEMM's table, instead of the fused `qmm8_gu1k2` (8 column slices x 2 k-slices of one
  64-column group, which streamed ~10% slower than `qmm8pf` on the same bytes). -2.67 / -2.47 ms (-2.73 / -2.75 on top of
  the GEMM changes); the drafter's MLP takes the same path.
* Together (cm + pair + old kplan): -2.70 / -2.53 ms, i.e. additive.
* Tried, no gain: CM 2 (odd fragments take the nibbles at bits 4-7 in place, x / 16 in the table: 10 integer ops per 8
  MMAs instead of 14; in cmp within noise of CM 1, and the table change would reach every producer), a prefetching
  two-group loop (96 GPRs), 4 fragments per simdgroup, nf 1 with ks 8/16, the drafter's conv / attention projections on
  these GEMMs (+0.27 / -0.06 ms).

**Other verify kernels.**
* **GDN step over value-row blocks** (`gdnsplit`, default 256 threads = 32 rows): a head's 128 value rows are
  independent, so `gdn_step_verify8s` spreads them over 4 threadgroups (192 instead of 48 threadgroups of 1024
  threads); the per-head tape rows k, g are written by the block that arrives last at a per-head counter (all blocks
  have read the old ones by then); `gdn_gnorm_x8` applies the gated RMSNorm and writes the out_proj's table with
  x8_prep2's summation order. Logits bitwise identical on both models (`ni_v8check` dumps, `cmp`). -0.66 / -0.74 ms.
* **x8_prep2 row norms** (`prepu`): four independent load chains. -0.38 / -0.46 ms.
* Tried, neutral or slower: GDN prepare with all rows in parallel (bitwise identical; +0.14 / -0.25), the GDN and
  attention outputs written straight into the next GEMM's table (bitwise identical; +0.02 / +0.03: dispatch boundaries
  cost ~0.7 us each, measured with `V8Knobs::pad` empty dispatches: +0.18 ms for 268), a GDN step with 4 lanes per value
  row (96 GPRs, +0.7 to +1.1 ms), a latency-scheduled GDN step (next row's operands before the stores; bitwise
  identical, -0.12 / +0.05).

**In-pipeline cost per 27B cycle** (`NI_AB=skip=<bit>`: odd cycles skip a kernel group; outputs garbage, timing valid
because the dense cycle's work does not depend on token values; before the GEMM, GDN and prep changes, reduced head on,
cycle ~87.5 ms): MLP gate/up 33.4 ms (192 GB/s), MLP down 16.6 (192), GDN in_proj 11.2 (204), GDN out + attention o
5.2 (217), LM head + sampling 3.4 (209), attention q|k|v 3.0 (223), GDN step 3.2-3.5, normed X8 preps 1.1, GDN prep
0.5, attention core 0.45; drafter: MLP gate/up 2.3, down 1.1, conv / attention projections 1.0-1.2, ring commit
0.5-0.6, LM head (65k rows) 0.5. After the changes the cycle is 77-78 ms: ~16.3 GB per cycle at ~210 GB/s.

**Correctness** (final build):
* G1: 27B `code_27b` 512 and p4000 + 256 greedy identical to plain decoding; 35B `code_35b` 512 identical. 35B p4000 +
  256: first difference at output token 101: plain top-2 gap 0.110 (20.1403 vs 20.0302), the verify row -0.0055 (20.0580
  / 20.0635), one prefill of the prefix +0.024 (20.0427 / 20.0192): all three computations of that row are within ~0.1
  of each other, the 35B's multi-row noise at this context (the base build split this prompt at token 197, gap 0.020).
* D1 (`dflash_check` on the q4 dumps, full head): chains identical 3/3 on both drafters, worst per-layer h 2.42e-3
  (27B) / 2.49e-3 (35B); the sampled walk matches softmax(S / T) to 2.3e-7.
* S1 (`dflash_accept_test`): all cases pass, the biased depth rule is still rejected (chi2 362).
* S2 (`dflash_stat`, joint law of the first two tokens vs p(t1) p(t2 | t1), chi-square p): 27B prose (+23) T = 0.6
  1200 trials joint 0.307 / first token 0.357 (base 0.316 / 0.361), T = 1.0 1200 trials 0.212 / 0.119; 35B prose (+23)
  T = 0.6 2000 trials 0.148 / 0.038 and 3000 trials 0.298 / 0.169, code (+32) T = 0.6 2000 trials 0.518. All pass.
* `tests/test_server.py` with `--dflash auto` (`NI_DFLASH=auto bench/shortpf/serve_test.sh`, main's script): 13/13 on
  both models (the reduced head loaded from `<drafter>/head-ranked.u32`).
* Memory: the reduced head adds 189 MB (27B) / 75 MB (35B) to the drafter (resident 1.51 / 0.59 GB), the split MLP
  2 x 557 KB of scratch.

Tools: `ni_run ... NI_AB=key=value,...` (the `V8Knobs` keys: cm, pair, kplan, gusplit, gdnsplit, prepu, and timing-only
skip / pad: odd cycles run the given knobs, prints GPU ms per cycle of both halves), `NI_SEED` (sampled runs),
`NI_DFLASH_QT` (walk temperature factor), `NI_DFLASH_HEAD` / `NI_DFLASH_HEAD_IDS` (draft head rows / ranked list),
`NI_V8_*` (knob defaults), `bench_v8 cmp` (interleaved kernel / tiling comparison), `ni_v8check` with
`NI_V8CHECK_DUMP=<file>` (verify logits for bitwise A/B with `cmp`).
