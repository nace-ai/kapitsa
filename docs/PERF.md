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
