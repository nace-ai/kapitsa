# Kapitsa (formerly nace-infer) performance notes (M4 Pro 20-core GPU, 273 GB/s, macOS 15.3.1)

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

## Dense prefill GEMM v2: qmr (opt-in `NI_QMM=1`, + `NI_QMM_GU=1`), macOS 27.0.1
`src/kernels/qmm3.metal`, selected in `Qwen35::enc_qmm` / `enc_qmm_gu`. All numbers below on macOS
27.0.1 (MSL 3.1 source, runtime-compiled). The GPU was shared all day and its speed drifted by up to
~20% between sessions, so compare numbers within one row/session only (all A/B runs interleaved).

simdgroup-matrix ceiling (`bench_mma`: 8x8x8 MMAs on registers only, 2048 x 128 threads): half x half
-> float 7.41 TFLOP/s, float x float 7.44-7.59, half -> half 7.84 (fp32 FMA loop 6.75). The MMAs plus
the B-fragment loads of the qmr kernel alone (ablation below) reach 7.51.

Kernel TFLOP/s, M = 2048 (bench_qmm medians; MLX 0.32.4 `mx.quantized_matmul`, same 4-bit g64 data,
fp16 / bf16 activations, 3 timed loops after warm-up):
| shape (27B) | qmm_b4_m64_n64 (old default) | qmm2_b4_m64_n64 | MLX fp16 / bf16 | qmr_k32_w14_m32_n128 |
|---|---|---|---|---|
| session A (morning, cool) | | | | |
| 2048x17408x5120 gate/up | 6.15-6.20 | 6.34-6.51 | 6.26-6.58 / 6.23-6.32 | 7.10-7.13 |
| 2048x5120x17408 down | 5.90-6.01 | 6.23-6.37 | 6.44-6.48 / 6.43-6.49 | 6.99-7.04 |
| 2048x5120x6144 out_proj | 5.98-6.15 | 5.99-6.45 | 6.59-6.65 / 6.61-6.64 | 6.88-7.13 |
| 2048x16384x5120 qkv+z | 6.06-6.18 | 6.27-6.61 | 6.57-6.62 / 6.59-6.67 | 7.16-7.17 |
| session B (11:35, one locked script, GPU ~15% slower) | | | | |
| gate/up | 5.30 | 5.62 | 5.29-5.85 / 5.00-5.24 | 6.19 |
| down | 5.02 | 5.34 | 4.97-5.07 / 4.96-5.10 | 5.96 |
| out_proj | 5.07 | 5.39 | 5.03-5.18 / 5.14-5.20 | 6.00 |
| qkv+z | 5.07 | 5.36 | 5.20-5.27 / 5.32-5.36 | 5.95 |
35B shapes (session A, us): 2048x8192x2048 10896 -> 9498 (6.31 -> 7.24 TFLOP/s), 2048x4096x2048
5497 -> 4799, 2048x2048x4096 5704 -> 4893, 2048x512x2048 764 -> 639, 2048x2048x512 716 -> 622,
2048x1024x5120 3664 -> 3080, 2048x12288x5120 42925 -> 36982; N = 48 (GDN b/a): 578 -> 265
(qmr_k64_w14_m16_n64, picked for N <= 64).

What MLX runs (v0.32.4, `affine_qmm_t`). The M4 Pro is `applegpu_g16s`: `is_nax_available()` needs
macOS 26.2+ AND architecture generation >= 17, so the macOS 26/27 NAX (neural accelerator) kernels are
never used on this GPU, and macOS 27 changes nothing in MLX's choice. `QuantizedMatmul::eval_gpu` ->
`qmm_splitk` -> split_k = 512 / (N/32 * M/32) = 1 for these shapes -> `qmm`:
* tile BM = BN = BK = 32, 128 threads as 2 x 2 simdgroups of 16 x 16 (2 x 2 fragments each);
  threadgroup memory Xs[32 x 40] + Ws[32 x 40] of T (5 KB); no double buffering, no register
  prefetch, no swizzle (grid x = N tiles: neighbouring threadgroups share the X tile).
* X: steel `BlockLoader`, one 8-element vector per thread per K step.
* W: `QuantizedBlockLoader` dequantizes into threadgroup memory: each thread reads 4 bytes (8 values)
  and one scale/bias (the pointer advances every second step: group 64 / BK 32), computes in float
  `s * (b & 0x0f) + bias` and `(s / 16) * (b & 0xf0) + bias` (no shifts), stores 8 scalars of T.
* K loop: barrier, load X, dequantize W, barrier, `BlockMMA::mma`: per 8-wide K slice, per-lane loads
  of the A and B fragments from threadgroup memory converted to float (A/B are `float` simdgroup
  matrices), `simdgroup_barrier(mem_none)` fences, 4 float MMAs.
* epilogue: each lane stores its 2 accumulator elements straight to device memory (as T).
Its dequantization work per MMA is twice ours (32 instead of 64 rows share a W tile) and every operand
passes through threadgroup memory, but the small footprint (5 KB, few registers) keeps many threadgroups
per core, which hides barriers and dequant: 85-90% of the MMA ceiling. The old `qmm_b4_m64_n64` has
the same structure with a 4x larger tile (10 KB, 52 GPRs) but 8-byte X loads, 2-byte W stores, a
scale/bias reload per word and an epilogue staged through threadgroup memory with simdgroup barriers;
`qmm2_b4_m64_n64` (16-byte loads/stores, unrolled, register epilogue) is at MLX's level.

qmr keeps W out of threadgroup memory:
* C = W_tile * X_tile^T with W as the A operand. In an 8x8 A fragment a lane holds row r and columns
  2c, 2c+1. The K order inside each 32-block is permuted (fragment f, column 2c+e <-> nibble (f%4)+4e of
  word c) so that a lane's A elements for the 4 K-slices of a step are exactly the 8 nibbles of ONE
  packed word of its row. `((w >> 4j) & 0x000F000F) | 0x34003400` is the half2
  (0.25 + q_j/4096, 0.25 + q_j+4/4096); minus 0.25, times 4096*scale, plus bias (half2 FMA) gives the
  dequantized pair: ~2.5 instructions per weight, each weight dequantized once per threadgroup
  (simdgroups split N) straight into registers. The 4096 factor keeps both factors of q*s normal halfs
  (a plain half scale is subnormal below 6.1e-5: 2162 scales of the 27B, 14818 of the 35B's dense
  projections; checked with `QMM_SCALE=1e-3 / 1e-4`).
* X is staged once per threadgroup in the same permuted K order (4 x 4-byte stores per 16-byte chunk)
  and read as the transposed B operand with `simdgroup_load`.
* tile 32 (M) x 128 (N), BK 32, 4 simdgroups along N, each 32 x 32 (4 x 4 fragments: per K slice 4
  dequantized A fragments + 4 B loads feed 16 MMAs). 80 GPRs, no stack, 5 KB threadgroup memory.
  Register epilogue (fp32, optional +=). M < 64 or N <= 64: `qmr_k64_w14_m16_n64` (44 GPRs);
  M <= 8: `qmr_k64_w12_m8_n64`.
* Ablation (gate/up and down shapes, session B-like GPU, med TFLOP/s): full 6.89 / 6.42; no dequant
  math 7.22 / 6.78; no X staging 7.31 / 6.81; no X staging or barriers 7.35 / 6.88; no W loads 7.32 /
  6.80; only MMAs + B loads 7.51 / 7.05. Overhead left: ~8% over the MMA ceiling, spread over the
  dequant (~5%), X staging and W load latency.
Tried, not better (within ~1-2% or slower): BK 64 (93 GPRs) and 128 (spills), 64 x 64 tiles (1 x 4 or
2 x 2 simdgroups), 64 x 128 and 32 x 256 with 8 simdgroups, two X buffers with one barrier per step,
register prefetch of the next step's X and W (96 GPRs + spill) or of W only (92 GPRs), M-tile swizzle
(2 or 3 tile rows walked together). `qmr_k64_w12_m32_n64` (2 simdgroups) is as fast but sits at 96
GPRs with 16 B of stack.

Short prefills (old path: qmm_b4_m32_n32 below 64 rows), us: N=17408 K=5120 at M = 8 / 16 / 32 / 48:
1201 / 1213 / 1234 / 2339 -> 302 (m8) / 479 / 889 / 1294-1317 (m16_n64); M=63 N=5120 K=17408 2647 ->
1799; M=16 N=5120 K=17408 1325 -> 543; M=48 N=5120 K=6144 908 -> 488.

`NI_QMM_GU=1` (needs NI_QMM): `qmr_gu_*` computes gate and up from one X tile (simdgroup tile 32 x
(16 gate + 16 up) rows) and writes half(silu(g) * u): no fp32 hg/hu (2 x 143 MB per 27B layer at 2048
rows) and no swiglu pass; also used for the 35B's dense shared expert. Kernel: 6.95-6.98 TFLOP/s
counting both projections vs 6.88 for qmr on each (2048 x 17408 x 5120, same session); identical
fp32 accumulation order, so outputs are bit-identical to NI_QMM=1 alone (checked on 27B and 35B logits).

End to end (ni_run n_gen 1, tok/s, interleaved in one locked script per row; the first run of a
script pays a warm-up, marked *):
| model, prompt | default | NI_QMM=1 | NI_QMM=1 NI_QMM_GU=1 |
|---|---|---|---|
| 27B p4000, 08:35 | 102.8, 101.6 | 122.6, 125.6 | |
| 27B p4000, 08:50 (with --dump-logits) | 108.7 | 131.4 | |
| 27B p4000, 13:34 (slow GPU) | 90.5* | 108.7 | 109.1 |
| 27B p4000, reverse order | (pending) | | |
| 27B p16000 | (pending) | | |
| 35B p4000, 08:40 | 939.3, 940.2 | 1009.4, 1011.9 | |
| 35B p4000, 14:47 | 771.7*, 932.7 | 984.6, 992.7 | 1000.8, 987.5 |
| 35B p16000, 14:48 | 738.4, 701.2 | | 753.0, 727.5 |
MLX 0.32.4 (mlx_lm.stream_generate prompt_tps, same weights and ids), 27B p4000 at 08:30: 122.3, 124.7.

Correctness vs the default path:
* 27B code_27b, 256 greedy tokens: identical (NI_QMM=1 and NI_QMM=1 NI_QMM_GU=1).
* 27B p4000 logits: max |diff| 0.016, rms 0.0035, KL 1.8e-9, same top-1, top-2 margin 7.709 vs 7.707.
* 35B code_35b, 256 greedy tokens: identical. 35B p4000 logits: max |diff| 0.79, rms 0.17, KL 2.0e-5,
  same top-1 (margin 7.54 vs 7.46). That is the 35B's own rounding-noise band (routing near-ties
  amplify it): the default path against itself with `NI_MATH_SAFE=1` differs by max 0.86, rms 0.165,
  KL 2.5e-5. Against MLX 0.32.4 (bf16) logits: default rms 0.301 / KL 1.0e-4, qmr rms 0.307 / KL 1.5e-4.
* bench_qmm error (max relative over 64 sampled outputs vs a double reference on the dequantized
  halfs) 2.5e-6 - 3.9e-5 for qmr, same range as qmm_b4 / qmm2; qmr_gu 2.4e-4 - 4.5e-4 (half output).

Tools on macOS 27: `ni_kstats --stats` prints "regs n/a" (the AGX pipeline object layout changed);
`NI_ARCHIVE_DIR=<dir> ni_compile <kernels>` still reports GPRs and stack.

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

## Short prefills and cached-turn TTFT (`NI_SHORTPF`, default on; macOS 27.0.1)
The cached-TTFT column of the benchmark harness is the second chat turn on a cached 4k/16k prefix:
a short new suffix (27B: 21-28 tokens, because its template renders the previous answer exactly as
generated, so the live state continues; 35B: 37-65 tokens, because the Qwen3.6 template drops the
empty think block of earlier answers, so the turn-1 checkpoint is restored and the answer
re-prefilled). At 45705ec nace-infer took 0.83 / 0.94 s (27B, 4k / 16k) and 0.25 / 0.32 s (35B) there;
Ollama 0.52 / 0.60 and 0.24 / 0.28, mlx-lm 0.57 / 0.90 (27B), llama.cpp 0.18 / 0.20 (35B).
`NI_SHORTPF=0` restores the previous behaviour; the parts:

1. **One pass per suffix.** The engine split the suffix at the turn-boundary checkpoint into two
   prefill passes, each reading every weight. `encode_prefill(..., snap, snap_row)` now captures the
   checkpoint inside the pass: `gdn_prep_par_snap` copies the conv window as it stands after
   `snap_row` tokens (before rolling it) and `gdn_step_par_snap_l*` writes the recurrent state after
   token `snap_row - 1` (two loops, no per-token branch). Cold prompts save a pass too (their last
   chunk holds the boundary). `ni_run --prefix P --split B` compares the snapshot with a separate
   prefill of [P, B): max|d| = 0 when both passes use the same kernels (27B at 23 rows); otherwise
   rounding-level (the attention split count depends on T).
2. **Short-prefill GEMMs** (passes of <= 128 rows, `NI_SHORTPF_MAX`). Two kernel families, picked
   per projection group by `short_qmms()` from bench_qmms data (`NI_SHORTPF_GEMM=auto`, or
   `qmms` / `tile` / `qmr` to force one):
   * `qmms_b{4,8}_t{1..8}` (qmms.metal, from the WIP branch): a simdgroup owns 8 weight rows and
     dequantizes them as half2 straight into simdgroup-matrix A operands; the B operand is read from
     the row-major half X in a matching permuted k order; K split over 4/8 simdgroups (fixed-order
     sum); no threadgroup tiles. Wins at <= 16 rows everywhere and on small weights (all 35B shapes,
     the fused gate/up) up to 56 rows; the only short kernel for 8-bit weights (35B router + shared
     gate: 5-19x the old `qmm_b8_m32_n32`, which ran at 3.5 GB/s).
   * qmr with one row tile holding all rows (qmm3.metal `qmr_k64_w14_m{24,40,48,56}_n64` next to
     main's m16/m32/m64; two tiles from 49 rows, `Qwen35::short_tile`), multi-projection `qmrs_*`
     (one dispatch for GDN qkv/z/b/a or attention q/k/v), `qmr_gu_k64_w14_m24_n64`. Wins on the big
     27B groups past 16 rows (compute bound: the 16-row tiles of main paid 32 rows for 23).
   * Projections that share an input run in one dispatch (`enc_proj`: GDN in-proj, attention q/k/v,
     router + shared gate, the MTP head's q/k/v); `enc_qmm_large` keeps the large-prefill choice.
3. **MoE experts in 8-row tiles** (`qmms_moe_gu_b4` / `qmms_moe_dn_b4`, tile lists from
   `moe_bucket2` with BM = 8; passes of <= 128 rows, `NI_SHORTPF_MOE_MAX`). At 37-65 rows the 35B
   routes ~1-2 rows per expert; the 64-row fused tiles paid barriers and threadgroup staging per K
   step for them. Blocking 2-8 row blocks per simdgroup (X gathered once) measured +-1%: dropped.
4. **Attention over a long cached prefix** (passes of <= 128 queries, `NI_SHORTPF_ATTN_MAX`): rows
   are (query, head) pairs of one KV head (`attn_prefill11g`: a 32-row block holds 32/G queries x G
   heads instead of padding T to 32 per head), keys split over ~100 threadgroups in all
   (`attn_prefill11gs` + `attn11_reduce`; `NI_ATTN_SPLIT` overrides).
5. CPU side: the turn boundary comes out of the one tokenizer pass (`Tokenizer::encode` reports the
   id index where an added token starts at the conversation's end) instead of a second encode of the
   whole conversation; the server sets TCP_NODELAY on its SSE streams (`NI_TCP_NODELAY=0` reverts) so
   a first token never waits for the client's delayed ACK; the `chat:` log line shows the
   template/tokenize time.

bench_qmms (isolated GEMMs, weights rotated over copies past the SLC, median of interleaved
trials): sum over one pass of a layer stack's dense projections (35B: no routed experts), ms:
| model, rows | main (qmr m8/m16 tiles, per projection) | qmms | one-tile qmr | auto rule | best per shape |
|---|---|---|---|---|---|
| 27B, 8 | 105.9 | 74.6 | (qmms) | 74.6 | 74.6 |
| 27B, 16 | 153.2 | 134.9 | 153.2 | 134.9 | 129.1 |
| 27B, 23 | 278.3 | 198.8 | 193.7 | 190.8 | 190.2 |
| 27B, 32 | 275.2 | 272.9 | 258.5 | 252.5 | 247.1 |
| 27B, 48 | 396.5 | 406.5 | 372.9 | 368.3 | 366.7 |
| 35B, 8 | 19.2 | 5.6 | (qmms) | 5.6 | 5.6 |
| 35B, 23 | 29.0 | 13.1 | 16.3 | 13.1 | 13.1 |
| 35B, 37 | 36.7 | 21.1 | 31.4 | 21.1 | 21.1 |
| 35B, 55 | 41.3 | 34.0 | 35.9 | 34.0 | 31.1 |
At 65-128 rows two qmr row tiles beat main's m32_n128 large tile up to ~96 rows on both models
(35B GDN in-proj at 65 rows 632 vs 1038 us, 27B 2089 vs 3083) and tie at 128.

bench_attn (median of 9 x 8 back-to-back calls), ms per layer:
| model, queries | 4k keys: v11 / GQA split | 16k keys: v11 / GQA split |
|---|---|---|
| 27B (G=6), 8 | 1.03 / 0.26-0.31 | 4.11 / 0.98-1.05 |
| 27B, 23 | 1.04 / 0.60-0.66 | 4.12 / 2.25-2.41 |
| 27B, 37 | 1.55 / 0.88-0.90 | 6.14 / 3.33-3.34 |
| 35B (G=8), 23 | 0.68 / 0.36-0.41 | 2.77 / 1.46-1.48 |
| 35B, 37 | 1.03 / 0.55-0.62 | 4.13 / 2.14-2.35 |
| 35B, 55 | 1.04 / 0.80-0.85 | 4.08 / 3.10-3.28 |
| 35B, 128 | 2.03 / 1.74 | 8.08 / 6.80-6.90 |

End to end, prefill of a cached-turn suffix (ni_run --prefix/--split, restore + one pass incl. the
MTP history rows, median of 5), ms; base = NI_SHORTPF=0 (two passes, main's kernels):
| case | base | in-pass only | + short GEMMs/MoE/attention (default) |
|---|---|---|---|
| 35B, 4k prefix + 37 (boundary 7 before the end) | 170-172 | 128 | 75-76 |
| 35B, 4k + 55 | 204 | | 100-101 |
| 35B, 4k + 65 | 224-227 | 192 | 127 (185 with the 64-row limits) |
| 27B, 700 + 23 | 409-417 | 296 | 210-211 |
| 27B, 700 + 37 | 415 | | 335 |
| 27B, 4k + 23 | 443 | | 219 |
Cold short prompts (one pass, no boundary): 35B 8 / 23 tokens 53 / 87 -> 27 / 48 ms; 27B 118 / 288
-> 88 / 207 ms. The 27B is compute bound here (~8.8 ms per suffix token at ~6 TFLOP/s on 8-row MMA
granularity): its cached turn costs what its suffix length costs.

Through the server, the harness's pattern (bench/shortpf/cached_ttft.py: same corpus, system prompt,
questions, sampling, thinking off, 1 s turn gap; median of 3 follow-ups; `--slots 1 --ctx 32768`,
default MTP), seconds to the first token. before = main cc776e5 (qmr + fused gate/up, which already
took the 27B from 0.83 to 0.35 s), after = this branch; competitors from the harness runs of the same
day (research/inference/results):
| cached turn TTFT | 45705ec (harness) | before (cc776e5) | after | Ollama | llama.cpp | mlx-lm |
|---|---|---|---|---|---|---|
| 27B, 4k prefix | 0.829 | 0.347 | 0.247 (0.242) | 0.50-0.52 | 0.60-0.65 | 0.573 |
| 27B, 16k prefix | 0.941 | 0.447 | 0.280 (0.272) | 0.58-0.60 | 0.64-0.69 | 0.898 |
| 35B, 4k prefix | 0.254 | 0.232 | 0.117 | 0.22-0.24 | 0.18-0.21 | |
| 35B, 16k prefix | 0.320 | 0.264 | 0.163 | 0.26-0.28 | 0.20-0.23 | |
27B follow-up suffixes 23/28/21 tokens (prefill 244/302/237 ms after vs 344/464/338 before), 35B
62/54/37 (129/114/93 ms after vs 238/229/195 before at 65/56/37). In parentheses: an earlier session
of the same 27B code paths (its cold 4k prompt ran 9% faster: the GPU speed drifts between sessions).
Cold short prompts (three 30-47-token prompts, median of 3 each): 27B 0.42 / 0.46 / 0.62 ->
0.29 / 0.39 / 0.47 s (0.27 / 0.33 / 0.40 in the faster session); 35B 0.147 / 0.158 / 0.173 ->
0.064 / 0.074 / 0.086 s. Template + tokenizer: 1-5 ms per request at 4k-16k.
A new build's first requests compile its kernels (the shader cache is keyed by the library):
before the load-time prewarm (`prewarm_pipelines`, NI_PREWARM=0 skips) the first cached turns of the
35B took 253-334 ms of prefill and the first 2.9k prompt 9.0 s instead of 3.3 s.

Correctness:
* prefix reuse stays exact: turn-2 output with the cache == the same conversation sent cold at T=0
  (server, 4k, after 3 cached follow-ups): equal on both models, before and after.
* greedy continuation (128 tokens, ni_run, with MTP and without) after a cached-suffix prefill
  (35B 4k + 37 / 55, 27B 700 + 23 / 37) and after cold prompts (23 tokens on both models, 100 on the
  35B): identical ids to NI_SHORTPF=0.
* logits of the last position vs NI_SHORTPF=0 (cached suffixes and cold prompts): 27B max|d|
  0.004-0.011, KL 2e-10 - 7e-8; 35B max|d| 0.37-1.18, KL 4e-5 - 5e-4; argmax always the same. That is
  the 35B's own rounding band (routing near-ties amplify it): its default path against
  NI_MATH_SAFE=1 on the same prompts gives KL 3e-5 - 1.6e-3, max|d| 0.08-1.05 (4k + 55: 5.4e-4 vs the
  new path's 4.8e-4), and changes the greedy text of the 100-token prompt at token 121; the new
  path does not.
* tests/test_server.py: 13/13 on both models.
* bench_qmms / bench_attn check every kernel against double-precision CPU references (GEMM max rel.
  err 1e-6 - 3e-5, fused gate/up 3-4e-4 (half output); attention max err / max |ref| 2-3e-4, as v11).
