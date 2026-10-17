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

## 27B prefill with the Neural Engine (opt-in `NI_ANE=1`)
M4 Pro (20-core GPU, 16-core ANE), macOS 27.0.1 (26A434), Qwen3.8-27B MLX 4-bit g64, 2026-10-16. Code:
`src/ane/`, `src/kernels/ane_ffn.metal`,
`src/model/qwen35_ane.cpp`. Each dense FFN layer of a prefill chunk (>= 512 rows) is split by intermediate
channel between the GPU (4-bit GEMMs) and the ANE (one W8A8 MIL program, Hadamard-rotated int8, private
`AppleNeuralEngine.framework`); the ANE's partial down projection is joined in fp32. Schedules:
- plain (`NI_ANE_WAVE=0`): per layer, mixer on the GPU, then the FFN split with GPU and ANE concurrent. The
  ANE idles during the mixer, so the best share balances the FFN: 27 of 34 units here.
- wavefront (the default): each chunk runs as two row halves in a skewed schedule, so the ANE computes the FFN
  of one half while the GPU runs the mixer of the other. The ANE is then hidden under the mixer and takes the
  whole FFN (34/34 units); the GPU runs the mixer, rotate/pack/join and the int8 staging. Bitwise equal to
  the plain split at the same share.

Cold prefill, `ni_run`, `NI_NO_MTP=1`, n_gen 1 (marked runs: n_gen 64/65), every run under the GPU lock and
reported; A/B blocks interleave modes. Thermal state (`NSProcessInfo.thermalState`) at the end of each run.

| block (start) | prompt | GPU only (default) | NI_ANE=1 | speedup |
|---|---|---|---|---|
| plain 25 units (16:20, n_gen 4) | p700 | 109.7 | 186.0 (fair) | 1.70x |
| plain 25 units (16:58, n_gen 64) | p4000 | 107.9 (nominal) | 204.2 (nominal) | 1.89x |
| plain 27 units, block 1 (18:30) | p4000 | 111.0 (nominal) | 217.4 (nominal) | 1.96x |
| | p16000 | 106.5 (fair) | 195.5 (fair) | 1.84x |
| plain 27 units, block 2 (18:43, reversed) | p16000 | 101.1 (fair) | 200.1 (nominal) | 1.98x |
| | p4000 | 105.4 (fair) | 206.3 (fair) | 1.96x |
| plain 27 units (18:56, n_gen 65) | p4000 | 111.5 | 217.9 (nominal) | 1.95x |
| wavefront 25 / 29 / 33 units (18:27, n_gen 64) | p4000 | 107.9 (16:58) | 213.7 (fair) / 234.7 (fair) / 298.8 (nominal) | 1.98 / 2.18 / 2.77x |
| wavefront 34 units (18:56, n_gen 65) | p4000 | 111.5 | 320.5 (nominal) | 2.87x |
| **wavefront 34 units, block 1 (19:08)** | p4000 | 110.4 (fair) | **317.8** (fair) | **2.88x** |
| | p16000 | 101.7 (fair) | **246.7** (fair) | **2.43x** |
| **wavefront 34 units, block 2 (19:17, reversed)** | p16000 | 100.7 (fair) | **264.5** (nominal) | **2.63x** |
| | p4000 | 98.6 (fair) | **277.3** (fair) | **2.81x** |

The last p4000 pair ran after two 16k prompts (throttled: the GPU alone fell to 98.6). Decode is unchanged
(14.3-14.6 tok/s plain in both modes); the 35B MoE never uses the split.

Calibration (`ane_calib`; split layers on real weights, random post-norm input, layers 8-15, median of 2):
| units | GPU alone FFN (2048 rows) | split's GPU part | ANE alone | split layer |
|---|---|---|---|---|
| 21 | 203.4 ms | 79.2 | 40.7 | 77.7 |
| 23 | 191.8 | 65.2 | 44.4 | 67.1 |
| 25 | 199.6 | 55.7 | 47.9 | 56.9 |
| 26 | 192.2 | 48.3 | 50.3 | 54.3 |
| **27** | 205.7 | 45.5 | 52.0 | **54.1** |
| 29 | 192.8 | 32.1 | 55.3 | 56.8 |
ANE 1.93 ms per 512-channel unit at 2048 rows (17 TOPS), GPU part 6.2 ms per GPU unit; the split layer is
max(ANE, GPU part) + 2-4 ms. 512-row chunks gain too (15.2 vs 48.2 ms at 27 units). `ane_calib --wave`
(whole 2048-row chunks, 64 layers): GPU alone 18.6 s (110.1 tok/s); 31 units 8.40 s (243.9), 33 units 7.40 s
(276.8), **34 units 7.20 s (284.3, 2.58x)**. Both results are stored in
`~/.cache/nace-infer-models/ane-cache/calibration.txt` (keyed by chip, macOS build and model) and used by
`NI_ANE=1`.

Where the wavefront's time goes (4k, 34 units): about 98-113 ms per layer per 2048-row chunk, against about
290 ms on the GPU alone, of which the FFN is about 190 ms. The GPU now runs only the mixer plus the split's small
kernels, so the mixer (GDN/attention and their projections, ~100 ms) is the limit. The ANE is busy 9.2 s of 12.6 s:
256 evaluations of 1024 rows x 17408 channels, 35.8 ms each, 15.3 TOPS. At 16k, attention grows and the gain
falls to 2.4-2.6x.

Accuracy (`kl.py` / `klall.py`: KL(GPU alone || split) of the next-token distribution; "65 positions" = the
last prompt position plus 64 greedy decode steps, whose contexts are identical because the tokens agree):
| case | KL | greedy tokens |
|---|---|---|
| p4000, plain 27 units, 65 positions | mean 9.1e-6, median 4.5e-7, max 1.6e-4 | 64/64 identical |
| p4000, wavefront 34 units, 65 positions | mean 1.4e-5, median 7.2e-7, max 2.9e-4 | 64/64 identical |
| p4000 last position: 25 / 27 / 29 / 33 / 34 units | 6.5e-6 / 8.6e-6 / 2.3e-5 / 3.5e-5 / 7.2e-7 | 64/64 for 25, 29, 33 (wave) |
| p16000 last position: plain 27 / wavefront 34 | 3.4e-9 / 2.5e-8 | - |
| code_27b (85 tokens, split forced: `NI_ANE_MIN_ROWS=64 NI_ANE_FNS=128`), 65 positions | mean 1.1e-4, median 3.1e-6, max 3.0e-3 (last prompt position) | 256/256 identical (25 units), 64/64 (27) |
| p700 last position, plain 25 | 5.8e-8 | 4/4 |
The GPU alone is chunk-size invariant (NI_CHUNK=1024 or 64: bitwise-identical logits), so there is no GPU
noise floor to compare with. Startup check (2 layers, 512 and 2048 rows, NaN-filled ANE output): the split
differs from the GPU alone by 0.75-0.88% of the ANE's contribution. Split runs are bitwise reproducible.
Fault injection (`NI_ANE_FAULT=10`: the ANE job of layer 10 never starts): the 2 s watchdog retires the split, the
chunk reruns on the GPU from the recurrent-state checkpoint, and logits and tokens equal the GPU alone bit for bit.

Memory (p4000 / p16000): split surfaces 454 MB (25 units), 486 MB (27), 629 MB (wavefront 34: two activation
lanes); process footprint 16,975 -> 17,965 MB and 17,492 -> 18,481 MB (+0.99 GB: surfaces + the 157 MB
recurrent-state checkpoint for reruns + rotation scratch); Metal allocated +0.91 GB. The ANE service's own
buffers are not in any process footprint: system wired memory rose 43-55 MB at the program load and 0.95-1.13 GB
by the end of the first split chunk, which also wires the surfaces and checkpoint (0.79 GB), leaving roughly
0.17-0.34 GB for the service (noisy: other processes).

Server (`nace-infer-server`, 27B default MTP depth 2, two ~4,575-token chat requests with 48 new tokens each):
21.0 s per request with `NI_ANE=1` against 45.1 / 44.7 s on the GPU alone, with identical replies (the engine
waits for each split chunk and would rerun a failed one).

Startup: compiling the 13-function program takes 31-40 s (MIL 115-133 KB, 13-18 MB rotation constant); a later
process finds it in the ANE daemon's cache (1-3 ms) and loads it in 44-78 ms, but the daemon evicts it:
twice today, after 10-40 minutes (other processes use the ANE too), so a cold start can pay the compile again. Row scales: 0.1 s computed, cached on disk (4-5 MB per share). Startup check 0.65-1.0 s.

### Review: re-measurement and fixes (2026-10-16, 19:47-21:00; fixes 01638f7, c2de003)
An independent re-run under the GPU lock, every run reported. `ni_run`, `NI_NO_MTP=1`, n_gen 65 with
`--dump-all-logits` (the decode steps then run synchronously; prefill timing is unaffected). The fix builds give
bitwise the same logits as 05da0cd in both modes (p4000 and p16000, 65 positions). Thermal state at the end of each run.

| block (start) | build | prompt | GPU only | NI_ANE=1 | speedup |
|---|---|---|---|---|---|
| 1 (19:47) | 05da0cd | p4000 | 110.3 (45705ec: 106.8) | 317.2 (fair) | 2.88x |
| 2 (19:59) | 05da0cd | p16000 | 95.5 | 266.5 (fair) | 2.79x |
| 3 (20:11, reversed; after another agent's run) | 05da0cd | p4000 | 89.6 (45705ec: 93.7) | 239.0 (fair) | 2.67x |
| 4 (20:14, reversed) | 05da0cd | p16000 | 97.5 | 234.3 (fair) | 2.40x |
| 5 (20:41) | 01638f7 | p4000 / p16000 | 116.6 / 109.9 | 338.6 (nominal) / 304.5 (fair) | 2.90x / 2.77x |
| 6 (20:47, reversed) | 01638f7 | p16000 / p4000 | 110.6 / 111.9 | 270.7 / 307.6 (fair) | 2.45x / 2.75x |
| 7 (20:55) | c2de003 | p4000 / p16000 | 119.7 / 107.4 | 330.2 (nominal) / 289.9 (fair) | 2.76x / 2.70x |

Other runs: plain split (`NI_ANE_WAVE=0`, 27 units) p4000 228.4; p1000 338.2 vs 116.3 GPU only; p2348 (a 2048-row
wavefront chunk, then 300 rows on the GPU) 268.3 vs 118.0; p2650 (then a 602-row plain split) 280.7 vs 116.2;
`NI_CHUNK=1024` p4000 324.7 / 331.9 vs 119.5; `ane_calib --wave --units 33,34 --reps 1` (c2de003's rebuild path):
GPU alone 113.0, 33 units 286.6, 34 units 308.1 tok/s per 2048-row chunk.

KL(GPU only || NI_ANE=1) over the last prompt position + 64 greedy steps. The noise floor is the GPU alone with
`NI_MATH_SAFE=1` against the default (fast-math) build:
| case | KL mean / median / max | greedy tokens |
|---|---|---|
| p4000, wavefront 34 units | 1.36e-5 / 7.2e-7 / 2.9e-4 | 65/65 |
| p4000, plain 27 units | 9.1e-6 / 4.5e-7 / 1.6e-4 | 65/65 |
| p1000 (halves of 512 and 488 rows) | 5.8e-5 / 3.4e-6 / 9.1e-4 | 65/65 |
| p2348 / p2650 | 3.4e-5 / 3.3e-6 / 3.3e-4; 1.6e-4 / 2.6e-5 / 2.4e-3 | 65/65; 65/65 |
| p16000 | last prompt position 2.5e-8; the 42 steps before the divergence 1.4e-4 / 2.5e-5 / 1.8e-3 | 43 of 65: step 43 is a near-tie (0.439 vs 0.412; 0.463 vs 0.403 with the ANE), KL 5.3e-3 |
| noise floor p4000 | 2.5e-8 / 6.6e-10 / 6.6e-7 | 65/65 |
| noise floor p16000 | last prompt position 4.1e-12 | 6 of 65 (step 6: top-1 p 0.31) |
So W8A8 on the whole FFN costs about 550x the GPU's own rounding noise in mean KL at 4k (still below the 1e-4 gap
between the default build and MLX bf16), some low-confidence positions exceed 1e-3, and greedy tokens are not always
identical: on p16000 they split at a near-tie, which the GPU alone with exact math already splits at step 6.
Default path bitwise equal to 45705ec. NI_ANE=1 is bitwise reproducible run to run and across chunk sizes
(`NI_CHUNK=1024` = 2048). Fault injection: `ni_run` p4000 bitwise equal to the GPU alone over 65 positions; server
(one 4,568-token request, MTP on): GPU only 44.7 s, `NI_ANE=1 NI_ANE_FAULT=10` 51.2 s with the identical reply
(the engine's rerun path).

Checked with small Metal programs under the lock: a command buffer blocked in `encodeWaitForEvent` holds back the
later command buffers of the same queue, even independent ones (so `stage_layer`, which overwrites a weight set at the
start of the command buffer after the one that waited for that set's last ANE job, is ordered); an `MTLResidencySet`
keeps a released buffer alive and resident until it is removed; a buffer allocated without an autorelease pool is
never freed. Fixed:
* `Gpu::release` now removes the buffer from the residency set (01638f7), and `Gpu::alloc`/`wrap` run in their own
  autorelease pool (c2de003). Before, an `FfnSplit` teardown (a failed startup check, `ane_calib` rebuilds) freed
  nothing. With `NI_ANE_FAULT=0` (the check fails, GPU only), Metal at the end: 18,197 MB before, 17,442 MB after,
  the same as without NI_ANE. A normal NI_ANE=1 run now also frees the check's 126 MB of scratch buffers: process
  footprint at p4000 17,838 MB (was 17,964).
* An `FfnSplit` constructor that fails after allocating (program compile/load, bindings) frees what it made.
* The startup check and `ane_bench` used 2048 rows whatever the prefill chunk: with `--prefill-chunk`/`NI_CHUNK`
  below 2048 the GPU wrote up to 71 MB past `hg_`, `hu_` and `hhalf_` (silent in the `NI_CHUNK=1024` run). They now
  stay within the chunk; a chunk below the split's minimum turns NI_ANE off, and chunks above 2048 rows are logged
  as GPU-only.
* The check's scratch buffers on its failure path, and `ane_bench_chunk`'s per-call sequence, are released.

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
nace-infer from the tables above (ni_run); the others from the cross-engine harness (servers,
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
| nace-infer `NI_ANE=1` (GPU + Neural Engine, wavefront) | (same) | 277-321 | 247-265 |
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
day:
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

## 35B prefill v3 and cached turns (branch opt/moe35, macOS 27.0.1, 2026-10-16/17)
Starting point (main 2e5bf23, fair server harness 21:17): cold prefill 4k / 16k 1014 / 842 tok/s (Splash 1021 / 879,
oMLX 970 / 882); cached-turn TTFT at 4k / 16k 0.12 / 0.15 s (MTPLX 0.10 / 0.17). Every GPU run below under the
shared lock; A/B runs interleaved in one locked script; ni_run numbers are `--pf-reps 3` (reps 1-2 after a warm-up
rep), which run the prefill of p4000 (3999 ids) / p16000 (15999 ids) with the MTP history pass, as the server does.

### Where the time went (NI_PROFILE=1, isolated dispatches; they sum to ~1.3-1.4x the real time)
| kernel | 4k | 16k |
|---|---|---|
| dense projections (qmr_k32_w14_m32_n128) | 46.4% | 39.9% |
| routed experts (qmm_moe_gu + qmm_moe_dn) | 30.8% | 27.0% |
| attention (attn_prefill11) | 5.0% | 17.8% |
| GDN recurrence (gdn_step_par_l16) | 5.5% | 4.7% |
| 8-bit GEMMs (qmm_b8_m32_n32: router, shared gate, the MTP head's projections) | 3.5% | 3.0% |
| everything else (shared expert, top-k, combine, rows_to_half, GDN prep/gnorm, ...) | 8.8% | 7.6% |
The small kernels cost much less in the pipeline than isolated: `NI_DUP=<kernel>` (new, timing aid: every dispatch
of the kernel runs twice; only for kernels that do not read their own output) gives their in-context cost.

### Routed experts on the qmr design (qrm, qmm_moe3.metal; NI_MOE_QMR=32 default, 0 = qmm_moe_gu/_dn)
qmr's register dequantization (W as the A operand straight from the packed words, X staged once per K step in the
permuted K order) on moe_bucket's sorted pair rows: gate/up 64 + 64 weight rows x 32 rows per threadgroup (SwiGLU
epilogue, half h), down 128 x 32 (fp32 rows by pair id). A tile holds at most 32 rows of one expert; the kernel
branches once per threadgroup on its live 8-row fragments and runs a K loop specialized for that count, so partial
tiles pay MMAs and X staging only for their rows. 88 / 85 GPRs, no stack. bench_moe (T = 2048, one synthetic layer,
routed experts only, TFLOP/s):
| routing | qmm_moe_gu / _dn (old default) | qrm gu / dn | layer total |
|---|---|---|---|
| balanced (64 rows per expert) | 6.50 / 6.26 | 7.06-7.17 / 6.97-7.07 | 16.1 -> 14.5-14.7 ms |
| skewed (median 36, max 527 rows) | 5.13-5.75 / 4.95-5.51 | 6.14-6.68 / 6.16-6.65 | 20.4 -> 16.8 ms (1.21x) |
| uniform random (44-85 rows) | 4.98-5.78 / 4.79-5.59 | 5.89-6.58 / 5.75-6.56 | 1.14-1.15x |
Full tiles reach the dense qmr's 7.0-7.2 TFLOP/s; partial tiles cost ~10%. Error vs a double reference equals the old
path's (h 1.1-1.9e-4 (half output), y 0.8-1.3e-7). Tried, slower: 64-row tiles with 32 + 32 / 64 weight rows (all
three routings and e2e, 3695 vs 3645 ms at 4k), 16-row tiles, BK = 64 (48 / 32 B of stack), down with 64 rows.
Folding the shared expert in (NI_MOE_FUSED=1) is bitwise identical and as fast as keeping it dense.

### 8-bit GEMMs on qmr (qmr8_*, qmm3.metal; NI_QMM8=1 default: the MTP head)
Byte j of a lane's word e is the 4-bit kernel's k for fragment j, column e, so X is staged as before; 0x3400 | q << 2
is the half 0.25 + q/1024 and one half2 FMA with (1024 s, b) dequantizes. bench_qmm (TFLOP/s, qmr8 vs
qmm_b8_m32_n32): 2048x8192x2048 6.90 vs 4.65; 2048x2048x4096 6.48 vs 4.39; 2048x256x2048 5.86 vs 4.38; gate/up
6.79 (qmr8_gu); errors as qmm_b8. Used for the MTP head's projections (bf16 in the file, 8-bit at load; they never
reach the trunk's logits); NI_QMM8=2 also moves the trunk's 8-bit router and shared gate (p4000 KL 2.1e-4 from routing
near-ties for ~0.3%, so not the default).

### GDN recurrence with four value rows per lane (gdn_step_parr4_l16, default; NI_GDN_STEP overrides)
The prefetching variant (next token's q/k/v/g/beta loaded during the current step) changed nothing (3874 vs 3866 ms),
so the step is not waiting for loads but for its own chain: per token a decay, dot, a 16-lane shuffle reduction,
update, dot and reduction, one dependent chain per lane. Holding R rows of the same head per lane (shared q/k loads,
R independent chains): bench_gdn 2048 tokens x 32 heads, step ms: par_l16 4.51, parr2_l16 2.91, **parr4_l16 2.75**,
parr4_l32 3.10, parr8_l32 3.06 (par_l4 5.70, l8 5.26, l32 5.90). Same arithmetic per row in the same order:
bit-identical outputs and state under NI_MATH_SAFE=1, and identical logits in e2e. 4k e2e 3620-3639 -> 3524-3528 ms
(parr2).

### Attention: skip unit rescales (attn_prefill11r, default; 11g / 11gs too)
When no row of a simdgroup's 8 queries raises its running max on a key tile, corr is exactly 1 and the O rescale is
x * 1.0: a uniform `simd_all` branch skips it. Bitwise neutral; 2048 x 16k 92.4 -> 91.4 ms, 2048 x 4k 18.05 -> 17.87.
Tried, slower: the K tile row-major with transposed simdgroup_loads (attn_prefill12: 95.2 -> 99.0 ms).

### Large prefill, end to end (ni_run, ms per prefill)
| build / flags | 4k | 16k |
|---|---|---|
| base (all new paths off) | 3863-3868 | 18798-18810 (851 tok/s) |
| + qrm experts | 3642-3660 | |
| + qrm + qmr8 (MTP head) | 3620-3635 | 17615-17632 (907-908) |
| + GDN rows per lane (parr2) + attention 11r | 3524-3528 | 17188 (931) |
Logits vs base: qrm KL 2.08e-5, max|d| 0.66, same top-1 (the qmr dense change was 2.0e-5; the 35B's NI_MATH_SAFE
noise band 2.5e-5); every other change bitwise identical (p4000 and p16000). No gain at large T, measured and left
as opt-in or default only where bitwise neutral: moe_bucket3 (parallel tile scan, 3-5x faster kernel), moe_topk_rows8
(8 tokens per 256-thread threadgroup), NI_QMRS (one dispatch for qkv/z/b/a: +1.1%), GDN prefetch (+1.2%),
gdn_step_par_l8/l32.

### Prefill chunk size (measured; the default stays 2048)
`--prefill-chunk 4096` (`NI_CHUNK` in ni_run) gives bitwise-identical logits (p4000, p16000) and is faster: p4000
3500-3513 vs 3582-3587 ms (two interleaved pairs, -2.1%); p16000 in alternating single runs on a warm GPU 17906 /
17924 / 18867 vs 18531 / 19806 / 19542 ms (-3.4 / -9.5 / -3.5%). The activation buffers scale with the chunk: Metal
21196 -> 21923 MB, process footprint 20999 -> 21584 MB at load. Memory is one of the harness's metrics (Kapitsa has
the smallest footprint of the engines), so the default stays 2048 and 4096 is the app's option.

### Cached turn (the harness pattern: 4k context, 37-62-token follow-up suffixes)
ni_run with the harness's real suffixes (bench prompts regenerated with the chat template: 4123 + 62, 4178 + 53,
4224 + 37), `--prefix P --split B --pf-reps 6`, median of reps 1-5; base 94.5-95.5 ms for 53 tokens. Isolated
profile of that pass: trunk experts ~45 ms (bandwidth: ~10 GB of expert weights for ~150 distinct experts per
layer), dense projections ~30 ms (qmms at 5.2-5.8 TFLOP/s; bench_qmms keeps qmms the best kernel at 37-62 rows),
attention 7.8, GDN step 3.7, the MTP head's MoE 2.7 (its experts are group-32 and fell to the 64-row large tiles),
router 4.5, bucket 2.2, small kernels the rest.
* **Experts of <= 2 rows by FMAs** (qmms_moe2_*, NI_SHORTPF_MOE_FMA, default): at these lengths most experts get
  1-2 rows, which the 8-row MMA path pads to 8 rows (8 multiply-adds per weight on the same ALUs). An expert with at
  most 2 rows takes a matvec form on the MMA path's own half-rounded weights (qs_dequant), fp32 products and sums;
  the others keep the MMA path. Two versions were rejected on the way: (1) any tile of <= 2 rows qualified, but
  which pairs land in the last tile of a larger expert depends on moe_bucket's atomic scatter order, so it was not
  run-to-run deterministic (KL 1.5e-6 between two runs): only an expert's first and only tile qualifies now;
  (2) MLX's pre-scaled-x trick without the half rounding of the weights moved the cached-turn path further from
  the large-prefill path's numerics (a cold check failed once), so it now uses exactly the MMA path's weights and
  differs only in the order of the fp32 sums (bench_moe error 7e-8 like the MMA path). Also takes group-32
  experts, so the MTP head stops falling back to the large tiles. Cached turns 62 / 53 / 37 tokens: 101.0 / 95.2 /
  74.6 -> 94.7 / 88.7-88.9 / 69.1-69.9 ms (all-MMA vs default, same binary); two runs bitwise equal; KL vs the
  all-MMA path 3.5e-4 - 9.4e-4 (routing near-ties; the 35B's band).
* **Token-block split of small qmms dispatches** (NI_QMMS_TY, default): the router + shared gate (N = 257: 17
  threadgroups) and the shared expert's gate/up (32) split their 8-row token blocks over grid.y; bitwise equal.
* moe_bucket3, moe_topk_rows8, gdn_step_parr4_snap_l16, attn_prefill11g/gs with the rescale skip: bitwise equal.
* **MTP history after the first token** (engine, NI_MTP_DEFER): the last chunk's MTP history rows run in their
  own command buffer queued behind the trunk's, so the first token is out before them; the first speculative cycle
  is queued after them (no change to any output).
Tried, slower or no gain: the experts in three dispatches by tile class (1 / 2 / 3-8 rows: 1855 vs 1721 us),
qmms with 32-bit row offsets (no fewer GPRs), NI_QMMS_TY=2 (t4 row groups: 87.7 vs 87.1 ms), the shared expert
folded in as expert E (NI_SHORTPF_FOLD=1: 86.9-88.8 vs 89.1 ms, noisy; opt-in), a GPU wake-up dispatch at request
arrival and user-interactive QoS for the request thread (no change in the server, removed).

The server pays an idle penalty the back-to-back reps do not: the harness waits 1 s between turns. ni_run
`--pf-gap MS` (new) idles the GPU before each rep: 53-token pass 94.5 ms back to back, 95 at 100 ms, 103 at 300 ms,
112-114 at 1 s (GPU time +19%), 131-167 at 2 s (another 25-40 ms between commit and the GPU's start/finish being
seen). A 25 ms keep-alive dispatch during the gap did not help (106-108 vs 107-110 ms at 1 s). Through the server
(gap 1 s) the same turns took 116-144 ms against 76-123 ms with no gap; one run, during heavy load from other
processes, 207-217 ms. The engine log now prints the last chunk's encode time, commit -> GPU start, GPU time and
GPU end -> wake-up.

### Server, the harness pattern (bench/shortpf/serve_ttft.sh --sizes 4000,16000 --greedy --cold-check)
main 2e5bf23 against this branch, interleaved, every run reported (prefill tok/s = prompt tokens / turn-1 TTFT;
cached = median TTFT of the 3 follow-ups; cold check = the last follow-up sent again cold gives the same text):
| run (start) | build | 4k cold | 16k cold | cached 4k | cached 16k | cold check 4k / 16k |
|---|---|---|---|---|---|---|
| 1 (23:44) | main | 1027 | 863 | 0.114 (0.145, 0.114, 0.094) | 0.164 (0.180, 0.164, 0.105) | yes / no |
| 1 | branch (FMA v2) | 1125 | 935 | 0.111 (0.139, 0.111, 0.094) | 0.145 (0.160, 0.145, 0.123) | yes / yes |
| 2 (23:54, slower GPU) | branch (FMA v2) | 1042 | 867 | 0.117 (0.133, 0.117, 0.090) | 0.124 (0.168, 0.107, 0.124) | no / yes |
| 2 | main | 960 | 802 | 0.128 (0.150, 0.128, 0.095) | 0.127 (0.182, 0.127, 0.124) | yes / no |
| 3 (00:00) | branch (final FMA) | 1100 | 930 | 0.112 (0.131, 0.112, 0.094) | 0.143 (0.176, 0.143, 0.107) | yes / yes |
Branch / main: cold prefill +8-10% at both sizes in each pair. In the server every follow-up's command buffer
starts 4-14 ms after its commit (the new log field) and runs 10-15% longer than back to back; a wake-up dispatch at
request arrival did not shorten that. The suffixes are 37-65 tokens because the template drops the empty think block
of earlier answers, so the previous answer is prefilled again (MTPLX's server keeps it: its suffixes are 19-26).

### History prefill (engine, NI_HISTORY_PREFILL, default on)
After a reply that ends with "stop", the next request is a user query, and the template renders the answer without
the think block (here the empty `<think>\n\n</think>\n\n` of the generation prompt) that the live state holds, so
the follow-up used to restore the turn-boundary checkpoint and prefill the answer again with its own message
(37-65 tokens, two passes' worth of weights for one). Now a thread started after the reply waits until the server
has been idle for NI_HISTORY_IDLE_MS (250), then renders the conversation as the next request will (the answer, then
a placeholder user turn, cut at that turn's `<|im_start|>`), restores the checkpoint and prefills the rendered answer
except its last token (so the follow-up's own pass still covers the MTP pair of that row), with its MTP history.
The follow-up then continues the live state with only its new message (20-27 tokens). It gives up when a request
arrives during the wait (a benchmark's next run, a quick tool, another conversation), when another request used the
slot, after "length" (benchmark decode runs; cut turns are rarely continued as they stand) and after tool calls (the
next message is a tool response, which keeps the think block, so the live state already matches it). Work is
serialized with requests by the engine lock: a request that lands while it runs waits for it (50-110 ms for
one-sentence answers, 360-490 ms for 256-token ones). The first version started at once and delayed the harness's
cold 4k turn, which follows the warm-up's tool call immediately, by 57 ms; the idle wait removed that.

Harness pattern (serve_ttft.sh, same binary, `--seed` pairs the prompts; cached = median of 3 follow-ups):
| run | mode | 4k cold | 16k cold | cached 4k | cached 16k | cold check 4k / 16k |
|---|---|---|---|---|---|---|
| job22 (start at once) | on | 976 | 850 | 0.065 (0.064, 0.074, 0.065) | 0.080 (0.080, 0.086, 0.071) | yes / no |
| job22 | off | 1035 | 867 | 0.131 (0.135, 0.131, 0.099) | 0.146 (0.167, 0.146, 0.131) | yes / yes |
| seed 11 (idle wait) | on | 1031 | 867 | 0.066 (0.066, 0.076, 0.060) | 0.076 (0.074, 0.088, 0.076) | yes / no (char 63) |
| seed 11 | off | 1033 | 868 | 0.115 (0.143, 0.115, 0.095) | 0.151 (0.164, 0.151, 0.102) | yes / no (char 63) |
| seed 22 | off | 1114 | 928 | 0.105 (0.129, 0.105, 0.091) | 0.163 (0.163, 0.177, 0.105) | yes / no (char 70) |
| seed 22 | on | 1113 | 928 | 0.062 (0.061, 0.072, 0.062) | 0.074 (0.074, 0.085, 0.069) | no (char 735) / yes |
Cold prefill is unchanged with the idle wait (pairs within 0.2%; job22's 976 is the delayed turn). Cold checks
fail at the same rate in both modes (the same conversation, seed 11, splits at the same character in both): the
cached path and a cold prefill take different chunkings, and these answers have near-ties.

Is the state equal to the old cached path's? Offline replays of one greedy conversation (`ni_run --plan`, new:
the server's exact passes, checkpoints and restores; plain decoding so only the prompt state differs): for each
follow-up prompt, cold (one prefill), off (restore + suffix) and on (restore + rendered answer, then the follow-up
from there), 32 greedy tokens with every step's logits. Cold is bitwise reproducible (two runs equal).
| follow-up | KL(cold, off) first / mean / max | KL(cold, on) first / mean / max | tokens vs cold |
|---|---|---|---|
| 4k 1 | 5.8e-4 / 6.1e-4 / 3.8e-3 | 1.2e-4 / 5.7e-4 / 4.0e-3 | both split at step 10 (cold margin 0.069; off 0.006, on 0.005 the other way) |
| 4k 2 | 3.0e-5 / 3.8e-5 / 4.5e-4 | 3.5e-4 / 5.9e-5 / 3.5e-4 | both identical |
| 4k 3 | 2.4e-4 / 1.5e-4 / 1.8e-3 | 1.3e-4 / 2.0e-4 / 2.5e-3 | both identical |
| 16k 1 | 3.6e-7 / 5.5e-5 / 9.9e-4 | 2.2e-7 / 7.4e-5 / 1.3e-3 | both identical |
| 16k 2 | 3.6e-6 / 9.8e-4 / 1.2e-2 | 1.5e-6 / 7.2e-4 / 9.6e-3 | both identical |
| 16k 3 | 6.2e-3 / 3.1e-4 / 6.2e-3 | 4.1e-3 / 2.4e-4 / 4.1e-3 | both identical |
The history-prefilled state is as close to a cold prefill as the old cached path's.

### Fair harness (compare_engines.py, 2026-10-17 01:58-03:22, final build 57df0c2)
Every run of the session, in order, each in its own GPU-lock span (server start, warm-up, 15 s quiet GPU, bench.py,
stop); Kapitsa main = 2e5bf23. The Mac was warmer than for the 2026-10-16 baseline (main 958-1002 / 819-839 here
against 1014 / 842 then). Decode tok/s (median of 5, T = 0.6 / greedy); prefill tok/s = prompt / cold TTFT; cached =
median TTFT of 3 follow-ups (samples).
| run (start) | decode | prefill 4k / 16k | cached 4k / 16k s | peak GB |
|---|---|---|---|---|
| Kapitsa opt/moe35 (01:58) | 118.2 / 120.4 | **1077 / 902** | **0.067** (0.066, 0.067, 0.060) / **0.074** (0.074, 0.088, 0.068) | 20.3 |
| Splash 1.3.2 (02:06) | 130.7 / 132.7 | 1010 / 859 | 0.160 (0.160, 0.185, 0.152) / 0.205 (0.226, 0.205, 0.205) | 22.0 |
| Kapitsa main (02:22) | 118.8 / 121.1 | 958 / 819 | 0.116 (0.143, 0.116, 0.090) / 0.120 (0.183, 0.120, 0.114) | 20.4 |
| oMLX + DFlash2 (02:36; 5.75% foreign GPU load flagged) | 126.2 / 125.9 | 990 / 856 | 0.269 (0.298, 0.269, 0.238) / 0.434 (0.434, 0.466, 0.378) | 23.4 |
| Kapitsa opt/moe35 (02:48) | 120.3 / 123.2 | **1070 / 914** | **0.065** (0.065, 0.071, 0.056) / **0.072** (0.072, 0.083, 0.070) | 20.3 |
| MTPLX (03:03) | 77.2 / 72.4 | 889 / 778 | 0.102 (0.096, 0.102, 0.306) / 0.136 (0.136, 0.139, 0.127) | 30.5 |
| Kapitsa main (03:12) | 120.7 / 122.2 | 1002 / 839 | 0.120 (0.147, 0.120, 0.104) / 0.180 (0.180, 0.228, 0.112) | 20.4 |
| Kapitsa opt/moe35, `--prefill-chunk 4096` (03:20) | 120.6 / 117.3 | 1086 / 915 | 0.067 / 0.074 | 21.0 |
Branch against main in the adjacent pairs: prefill +12.4% / +10.1% and +6.8% / +8.9% (4k / 16k), cached turns
0.116 -> 0.067 and 0.120 -> 0.065 s (4k), decode equal (within the runs' spread). Against the same session's
competitors: 16k prefill 902-914 vs Splash 859 and oMLX 856, 4k 1070-1077 vs 1010 / 990, cached 4k 0.065-0.067 s vs
MTPLX 0.102. Tool calls pass everywhere. The 4096-row chunk run is within the 2048 runs' spread here (+0.7%) for
+0.7 GB, which confirms keeping 2048. Server logs: the history prefill ran after each "stop" follow-up (47-98 ms),
not after the "length" ones, and the 16k cold turn and the tools phase never waited behind it.

### Correctness of the branch defaults against main
* code_35b, 256 greedy tokens: identical, with MTP and with plain decoding.
* p4000 logits KL 2.1e-5 (max|d| 0.66), p16000 KL 3.6e-7 (max|d| 1.14), same top-1.
* Cached-turn suffixes (ni_run --prefix/--split): logits KL 3.6e-5 - 1.9e-3, same top-1; 128 greedy tokens after
  them identical for the 62- and 53-token suffixes; the 37-token one splits at generated token 37, a near-tie
  (main p = 0.602 vs 0.395, logit margin 0.42; the branch 0.579 vs 0.419), which the branch with the FMA experts off
  splits the same way, so it comes from the qrm prefix (KL 2e-5) through a routing near-tie.
* tests/test_server.py: 13/13 (serve_test.sh), before and with the history prefill (final build).
* Every bitwise-neutral change verified bitwise in e2e logits (GDN parr, attention 11r, bucket3, topk8, qmms
  token-block split, folding the large-T shared expert).

## DFlash2 block drafter (opt/dflash2-draft; opt-in `ni_run --dflash <drafter dir>` / `NI_DFLASH`)

Checked against a Python/MLX reference implementation. Code: `src/model/dflash2.{h,cpp}`,
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

Server (the cross-engine harness's decode phases against `nace-infer-server`, the app's streaming API:
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

## DFlash2 review: independent re-check (2026-10-17, build of 4cdc5a0 + dd9025c)

Everything under `lockf -k -t 540 ~/.cache/nace-gpu.lock`; this lock slot ran ~7% (35B) and ~12% (27B) slower than
the loop section's slots, so compare within a table only.

**Greedy equality (G1)**, `ni_run <model> @<prompt> 512 0 --ignore-eos [--dflash <drafter>] [--ref <plain>]`:

| prompt | 35B-A3B | 27B |
|---|---|---|
| code_* | identical 512/512 | identical 512/512 |
| p700 | first difference at token 241 (pos 941): plain top-2 gap 0.035, verify row -0.42, one prefill of the prefix +0.021 | identical 512/512 |
| p4000 | first difference at token 197 (pos 4197): plain gap 0.020, verify row -0.151, one prefill -0.170 (takes the DFlash token) | identical 512/512 |

The 35B p700 split is the 8-row verify's numerics, not the loop: it is unchanged with `NI_DFLASH_DYN=0`,
`NI_V8_MOE_SMALL=0` and `NI_DFLASH_CB=1`, and the teacher-forced verify without any drafter (`ni_v8check` 512 rows)
flips at the same position, as does the MTP-era R = 4 verify; MTP depth 1 (R = 2) stays identical over the 512.
`ni_v8drift` (new) splits the verify's deviation from plain decoding into the per-step error of the kernels on the
plain state ("fresh") and the drift when the verify path builds the KV/GDN state ("accumulated", what a speculative
loop does), over 512 generated rows; |change of the plain top-2 gap| median / p95 / max, argmax flips:

| | fresh | accumulated |
|---|---|---|
| 35B p700, R = 8 (DFlash2) | 0.0023 / 0.25 / 0.70, 0 | 0.095 / 0.56 / 1.35, 5 (gaps 0.010-0.92) |
| 35B p700, R = 8, `NI_V8_MOE_SMALL=0` | 0.0023 / 0.15 / 0.64, 0 | 0.096 / 0.57 / 2.77, 4 |
| 35B p700, R = 4 (MTP depth 3 kernels) | 0.0013 / 0.13 / 1.31, 0 | 0.107 / 0.59 / 2.03, 4 |
| 35B p700, R = 2 (MTP depth 1 kernels) | 0.0000 / 0.0002 / 0.0009, 0 | 0.0032 / 0.057 / 0.42, 0 |
| 35B code_35b, R = 8 | 0.0015 / 0.081 / 1.08, 0 | 0.038 / 0.26 / 4.79, 0 |
| 27B p700, R = 8 | 0.0011 / 0.0043 / 0.11, 0 | 0.0014 / 0.0052 / 0.064, 0 |

So on the dense 27B the 8-row verify tracks plain decoding at the 1e-3 level; on the MoE 35B any multi-row verify
with half-precision inputs (R = 4 as much as R = 8) has a heavy-tailed per-step error (top-8 routing boundary flips)
that accumulates in the state, and greedy DFlash2 output leaves plain greedy at the first near-tie it meets (2 of 3
prompts within 512 tokens here). Both splits seen are near-ties (gaps 0.035, 0.020), but this is above SPEC §12's
"<= 0.02" noise target and the 35B's default (MTP depth 1) does not have it.

**Default path (G3)** vs a build of 45705ec: logits after the p700 prefill bitwise identical; plain greedy (64), MTP
greedy and T = 0.6 (128) ids identical on both models.

**Sampled exactness through the real loop** (`dflash_seedtest`, new: many seeds through the engine's
double-buffered loop, tested against the exact law of the plain sampler and, with `--law-out`, against the law
whose 2nd/3rd token comes from the 8-row verify rows; prose prompt, T = 0.6 unless noted, top_k 20, top_p 0.95,
start point = prompt + 24 greedy tokens, chi-square p of (t1), (t1, t2), (t1, t2, t3)):

| model, arm | trials | first-cycle a = 0 / 1-4 / 5-7 | t1 | t1,t2 (plain / verify law) | t1,t2,t3 (plain / verify law) |
|---|---:|---|---:|---|---|
| 35B plain sampler (reference) | 4000 | | 0.013 | 0.011 / 0.011 | 0.17 / 0.17 |
| 35B DFlash2, seeds A | 4000 | 2637 / 199 / 1164 | 0.996 | 0.047 / 0.052 | 0.054 / 0.056 |
| 35B DFlash2, seeds C | 4000 | 2628 / 221 / 1151 | 0.69 | 0.62 / 0.63 | 0.41 / 0.41 |
| 35B DFlash2, T = 1.0 | 4000 | 2256 / 648 / 1096 | 0.87 | 0.41 / 0.40 | 0.40 / 0.38 |
| 27B plain sampler (reference) | 1500 | | 0.19 | 0.27 / 0.28 | 0.80 / 0.81 |
| 27B DFlash2, two seed sets | 1600 | 1153 / 119 / 328 | 0.20 | 0.72 / 0.71 | 0.57 / 0.56 |

At this start point the verify law and the plain law differ by <= 0.004 per probability (27B <= 0.001). No test falls
below 0.01; the borderline 35B batch A did not reproduce with fresh seeds (C), and the plain sampler's own arm shows a
comparable fluctuation. `dflash_accept_test` (S1, the GPU kernels on synthetic p/q) with two new seed streams
(`40000 3 7`, `40000 7 11`): all 26 cases pass (p 0.029-1.0), the biased depth rule is still rejected (chi2 345 / 387).

**Performance** (`ni_run`, code_* ids, 512 tokens, `--ignore-eos`, the four configurations interleaved, 3 rounds):

| | MTP default | DFlash2 | gain | DFlash2 tok/cycle, GPU ms/cycle |
|---|---|---|---:|---|
| 35B greedy | depth 1: 114.9 / 113.7 / 114.5 | 150.2 / 150.3 / 150.0 | +31% | 3.407 (3.59 verified), 22.6 |
| 35B T = 0.6 | 111.3 / 106.9 / 110.5 | 140.2 / 139.3 / 139.8 | +27% | 3.500 (5.34 verified), 25.0 |
| 27B greedy | depth 2: 26.9 / 27.1 / 27.6 | 48.1 / 48.0 / 49.3 | +78% | 4.776, 96.8-99.3 |
| 27B T = 0.6 | 26.9 / 26.9 / 27.5 | 41.4 / 41.8 / 41.7 | +54% | 4.154, 99.2-100.2 |

Same ratios as the loop section; the absolute 27B number of this slot (48-49 tok/s) is below Splash's 54.7 from
another day, so "on par with Splash" for the 27B holds only against same-slot conditions, which were not re-measured.

**Fixed** (dd9025c): with a stop token or max_tokens inside the last accepted chain, the committed drafting half
had already written all a + 1 ring rows; the rows past `d.ctx` overwrote the slots of the window's oldest
positions, which `DFlash2::rewind` still counted as valid on the next request (prefix reuse, e.g. a regenerate of a
prompt longer than 2048 tokens): up to 7 wrong context rows at the far end of the window. Draft quality only; the
engine now raises `valid_lo` past them. tests/test_server.py with `--dflash auto` (35B): 13/13.

**Merging into main** (d6d2ed0 at the time of the review): `git merge-tree` reports conflicts in docs/PERF.md,
src/engine/engine.{cpp,h}, src/metal/gpu.mm (the same autorelease-pool change on both sides), src/model/qwen35.cpp,
src/server/main.cpp and tools/ni_run.cpp. The substantive one is the prefill capture: main's dense prefill can run the
FFN on the Neural Engine (`enc_mlp_prefill_split`, the wave schedule `enc_prefill_wave`), captures checkpoints inside
the pass, and re-encodes a chunk on the GPU when the Neural Engine half fails; the drafter's capture hook
(`enc_capture` after each target layer) and `encode_prefill_commit` (after the last chunk) have to follow the final
residual of every layer in those paths (or DFlash2 has to keep the prefill on the GPU), then G1 (with p4000 for the
ring window), G3 against main, S1 and the sampled test must be re-run.

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

## DFlash2 on main: every prefill path, on by default, Kapitsa's own draft head (merge/dflash2, 2026-10-17)

main c3bcc53 + opt/dflash2 cc8e556 (drafter, 8-row verify, double-buffered loop, review fixes) + opt/dfl27 9fbadd6
(27B cycle: reduced draft head, faster verify GEMMs, split GDN verify), then the changes below; main's 8a05d3a (tied
embeddings for the small Qwen3.5 models, f32 parameter tensors to bf16) merged in at the end, after which the quick
checks were re-run on the final build against a build of 8a05d3a: the DFlash2-off bitwise table below (all 24 checks, both
models), greedy code_* 512 identical to plain (35B 174.7 tok/s, 27B 61.4), S1 and test_server 13/13 on both models. Every run under
`lockf -k -t 3000 ~/.cache/nace-gpu.lock` in spans under ~10 minutes (one 27B multi-turn span took 11), from frozen
binaries; the reference is a clean build of c3bcc53. macOS 27.0.1, M4 Pro; other agents shared the lock, so absolute
speeds drift between spans: compare within a table.

**The drafter's capture in every prefill path.** The hook copies each target layer's final output (mixer and FFN
added) of the prompt rows the drafter can still see, `[max(start, n - 2047), n)`; the prompt commit projects them into
the ring after the last chunk.

| prefill path | capture point | check |
|---|---|---|
| plain layer loop (GPU; short prefills; in-pass checkpoint chunks) | after the FFN of each target layer | logits after prefill with the hook armed bitwise equal to c3bcc53 (both models; code / p700 / p4000) |
| NI_ANE plain split (27B chunks of >= 512 rows) | after `encode_layer`'s join | 27B p700 (one split chunk): the same; the wavefront check below |
| NI_ANE wavefront (27B default: two row halves, skewed) | half A after join FFN_i(A); half B after join FFN_i(B), one skew step later; the last layer after the final join | prompt captures and logits bitwise equal to the plain split at the same share (34/34 units), p4000 and p16000 |
| a chunk rerun after a failed split | the hook stays armed until the prompt commit, which is encoded after the rerun | `NI_ANE_FAULT=10`: ni_run p4000 captures, logits and ids bitwise equal to `NI_ANE=0`; through the engine (rerun in `chat()`, and in the history prefill) text and tok/cycle equal to `NI_ANE=0` |
| work after the first token (`NI_MTP_DEFER`) | with DFlash2 there are no MTP history rows; the prompt commit (and the ring snapshot below) takes their place behind the trunk's last command buffer, encoded after the trunk is committed | the first drafting pass is queued after it |
| idle history prefill (`NI_HISTORY_PREFILL`) | the ring goes back to the restored checkpoint (with its snapshot) and the prefilled answer is captured and committed like prompt rows; split chunks are now checkpointed, waited for and rerun as in `chat()` (it never called `ane_finish_chunk`) | multi-turn checks below |

**Ring and checkpoints.** The ring has 2048 slots per draft layer, so after more than ~2048 committed rows it no longer
holds the window before an older checkpoint, and `DFlash2::rewind` left the drafter only the rows prefilled after the
restore. Each engine checkpoint now has a ring snapshot (`DFlash2::encode_ring_copy`, 50 / 42 MB per checkpoint on the
35B / 27B, a ~0.3 ms GPU copy), taken right after the prompt commit of the request that saved the checkpoint and
restored with it in `chat()` and in the history prefill. 27B, thinking on, three turns of 3000 tokens (each restoring
the first turn's boundary): turn 2 3.60 -> 3.67 tok/cycle (a cold prefill of the same prompt: 3.675), turn 3 3.95 ->
4.04 (outputs cut inside the reasoning, not comparable with the cold run).

**Defaults.** `kapitsa serve` uses DFlash2 when `DFlash2::find` finds a drafter for the model (`<model>/dflash2`,
`<model>/drafter`, or `*-DFlash2` next to the model, in `<parent>/drafters` and in `~/.cache/nace-infer-models/drafters`
whose name minus `-DFlash2` is the model directory's name up to a `-`, with the loader's SPEC §1 checks against the
target: `qwen3.6-35b-a3b-4bit` -> `Qwen3.6-35B-A3B-DFlash2`, `qwen3.8-27b-4bit-mtp` -> `Qwen3.8-27B-DFlash2`), MTP
otherwise. `--dflash none` / `NI_DFLASH=0` keep MTP, `--spec 0` turns both off, `--dflash <dir>` names a drafter. The
branch's dynamic verify depth defaults stay (MoE 0.5 / 0.3, dense off) and so does the Neural Engine prefill split.
The drafter's and the 8-row verify's 82 pipelines are created at load (`DFlash2::prewarm_pipelines`, 1-12 ms with a warm
shader cache). `ni_run` stays plain unless `--spec` / `--dflash` is given.

**Reduced draft head: Kapitsa's own list.** opt/dfl27 drafts from the target LM head rows of a ranked id list read from
`<drafter>/head-ranked.u32` (Pulsar's frequency ranking, a third-party file) and used the full head without one. The
default needs no file now: the lowest 98304 token ids, i.e. the byte-level BPE tokenizer's merge order (a frequency
proxy, as for the MTP draft head), plus every added / special token, gathered at load (283 MB on the 27B, 113 MB on the 35B). `NI_DFLASH_HEAD_IDS=<file>`
still takes a ranked list (65536 rows by default), `NI_DFLASH_HEAD=n` sets the rows, `0` the full head. Id order needs
more rows than a frequency ranking; tokens outside each set:

| text | Pulsar 65536 | ids < 65536 | ids < 81920 | ids < 98304 |
|---|---|---|---|---|
| greedy outputs, 512 tokens (code / p700 / p4000, both models) | 0.00-1.17% | 0.78-3.71% | 0.20-1.37% | 0.00-0.20% |
| Python (bench corpus, 35k tokens) | 0.22% | 2.81% | 1.30% | 0.03% |
| C++ (Kapitsa sources, 78k) | 0.73% | 1.29% | 0.36% | 0.05% |
| English technical docs (PERF.md + SPEC.md, 44k) | 0.41% | 1.41% | 0.67% | 0.16% |
| TypeScript (an app's sources, 2.2M) | 0.36% | 0.97% | 0.37% | 0.02% |

Interleaved ni_run, 512 tokens `--ignore-eos`; tok/s (tok/cycle); greedy = 2 rounds, T = 0.6 = seeds 1-4 (35B) / 1-3
(27B):

| model, prompt, T | Pulsar 65536 | **Kapitsa 98304** | id order 81920 | full head |
|---|---|---|---|---|
| 27B code_27b, 0 | 60.7 (4.776) | **61.1 (4.776)** | 61.0 (4.688) | 60.5 (4.776) |
| 27B p700, 0 | 63.2 (5.010) | **63.1 (5.010)** | 63.0 (4.961) | 61.2 (5.010) |
| 27B prose, 0 | 33.7 (2.594) | **33.5 (2.594)** | 33.7 (2.594) | 32.5 (2.581) |
| 27B code_27b, 0.6 | 57.1 (4.748) | **58.3 (4.869)** | | |
| 27B p700, 0.6 | 59.4 (5.007) | **58.9 (5.007)** | | |
| 27B prose, 0.6 | 32.8 (2.758) | **33.0 (2.804)** | | |
| 35B code_35b, 0 | 174.5 (3.340) | **177.5 (3.430)** | 176.7 (3.384) | 170.2 (3.430) |
| 35B p700, 0 | 201.2 (4.294)* | **204.3 (4.405)** | 188.6 (3.992)* | 196.5 (4.405) |
| 35B prose, 0 | 127.1 (2.222)* | **123.2 (2.174)** | 121.0 (2.129)* | 117.5 (2.174) |
| 35B prose, 0, the 184 tokens all outputs share | 143.5 (2.614) | **142.2 (2.614)** | | 135.2 (2.614) |
| 35B code_35b, 0.6 | 163.7 (3.518) | **164.1 (3.542)** | | |
| 35B p700, 0.6 | 194.2 (4.594) | **192.8 (4.613)** | | |
| 35B prose, 0.6 | 121.3 (2.455) | **121.2 (2.476)** | | |

\* output differs from the full head's from token 500 (p700) / 184 (prose) on: on the MoE 35B the verify's numerics,
and with them near-tie outputs, depend on the drafted rows. On the 27B the outputs do not, and Kapitsa's list drafts like
Pulsar's and the full head on all three prompts at +0.1-0.7 ms per cycle for the 50% larger head (-0.6 to +0.7% tok/s
greedy). On the 35B code (same output everywhere) it accepts more (3.430 vs 3.340, +1.7% tok/s); over prose's common
184 tokens it is -0.9%; sampled, the means are within -0.8 to +2.1%. So Kapitsa's list is within ~1% of Pulsar's
wherever the texts agree, and it is the default; the Pulsar files in the drafter directories of this Mac are no longer
read.

**Validation.** All on the final code, except that runs made before the last engine change (the ring snapshot) used a
build that differs from it only in `Engine`, which `ni_run` and the tools do not use, and the head comparison above
selected its lists explicitly.

*DFlash2 off: bitwise vs c3bcc53* (`ni_run <m> @<p> 64 0 --dump-logits`, `NI_DFLASH=0`; MTP at the model's default depth):

| | 35B code_35b / p700 / p4000 | 27B code_27b / p700 (one plain-split chunk) / p4000 (two wavefront chunks) |
|---|---|---|
| logits after prefill | bitwise identical, 3/3 | bitwise identical, 3/3 |
| plain greedy, 64 ids | identical, 3/3 | identical, 3/3 |
| MTP greedy, 64 ids (depth 1 / 2) | identical, 3/3 | identical, 3/3 |
| logits after prefill with the drafter loaded (`--dflash auto`: capture armed, prompt commit) | bitwise identical, 3/3 | bitwise identical, 3/3 |

*DFlash2 greedy vs plain greedy* (`--ignore-eos`, `--ref` = the plain run; G2 at a difference: the two tokens' logits
from plain decoding, from the DFlash2 verify row, and from one prefill of the prefix):

| model, prompt, tokens | result | DFlash2 tok/s (plain), tok/cycle |
|---|---|---|
| 35B code_35b, 512 | identical 512/512 | 164.8 (104.2), 3.430 |
| 35B p700, 512 | first difference at token 90: plain gap +0.118, verify row -0.054, one prefill +0.038 | 191.9 (102.5), 4.405 |
| 35B p4000, 256 | first difference at token 119: plain gap +0.160 (a third token 0.029 above the DFlash2 one), verify row -0.328, one prefill -0.002 to the third token | 202.5 (98.8), 5.312 |
| 27B code_27b, 512 | identical 512/512 | 56.1 (14.2), 4.776 |
| 27B p4000, 256 (wavefront prefill) | identical 256/256 | 83.4 (14.5), 7.286 |

The two 35B differences are near-ties at the 8-row verify's noise on this MoE target: `ni_v8drift` on 35B p700 (512 rows,
R = 8, final build): |change of the plain top-2 gap| median / p95 / p99 / max 0.0024 / 0.19 / 0.43 / 1.47 with the verify
on the plain state (0 argmax flips) and 0.092 / 0.39 / 0.68 / 3.18 when the verify builds the state (6 flips, at plain gaps
0.03-0.33); the review measured 0.095 / 0.56 / -, 1.35 and 5 flips on its build (MTP depth 1 stays identical).

*The capture in the Neural Engine schedules* (27B, DFlash2 greedy, `NI_DFLASH_DUMP_CAPT` = the 2047 prompt-capture rows
x 5 target layers):

| prompt, tokens | prefill: wavefront / plain split (34 units) / GPU alone | wavefront vs plain split | wavefront vs GPU alone | tok/cycle |
|---|---|---|---|---|
| p4000, 256 | 10.7 / 18.8 / 29.3 s | captures, logits, 256 ids bitwise identical | captures rel. L2 2.5e-3 (layer 5) .. 4.3e-2 (layer 61), the same in every 256-row block (a wrong layer: 0.95, rows off by one: 0.90); 256 ids identical | 7.286 / 7.286 / 7.286 |
| p16000, 256 | 47.2 / 79.3 / 121.7 s | captures, logits, 256 ids bitwise identical | 2.4e-3 .. 5.5e-2, uniform; outputs part at token 5, as plain decoding does after the two prefills (48 tokens, no drafter) | 6.071 / 6.071 / 6.375 (other text) |

*Neural Engine failure* (`NI_ANE_FAULT=10`: the ANE job of layer 10 of the first split chunk never starts, the 2 s
watchdog retires the split, the chunk reruns on the GPU, which then runs every later prefill alone):
ni_run p4000: prompt captures, logits and 64 ids bitwise identical to `NI_ANE=0` (7.875 tok/cycle both). engine_turns,
rerun of a `chat()` prefill (turn 3): three turns' text and tok/cycle (5.12 / 3.79 / 5.06) identical to `NI_ANE=0`.
engine_turns with thinking and the split from 192 rows, so the first split chunk is the idle history prefill of turn 1's
answer (499 rows): rerun there, then three turns' text and tok/cycle (6.06 / 3.31 / 4.13) identical to `NI_ANE=0`.

*S1* (`dflash_accept_test 40000 3 7` and `40000 7 11`): all cases pass (p 0.029-1.0), `dfl_dist_rows` equals the CPU
sampler, the biased depth rule is rejected (chi2 344.6 / 386.6).

*Sampled exactness through the engine's loop* (`dflash_seedtest`, prose prompt, start = prompt + 24 greedy tokens,
T = 0.6, top_k 20, top_p 0.95; chi-square p of (t1), (t1, t2), (t1, t2, t3) against the plain law, which the 8-row
verify law matches within 0.001 per probability here). 35B, 4000 trials per set:

| arm, seed base | t1 | t1, t2 | t1, t2, t3 |
|---|---:|---:|---:|
| DFlash2, 101 | 0.185 | 0.066 | 0.0006 |
| DFlash2, 1000000 | 0.062 | 0.136 | 0.014 |
| DFlash2, 2000000 | 0.386 | 0.453 | 0.281 |
| DFlash2, 5000000 | 0.252 | 0.794 | 0.957 |
| DFlash2, 6000000 | 0.736 | 0.287 | 0.729 |
| DFlash2, 7000000 | 0.359 | 0.628 | 0.389 |
| plain sampler, 303 / 3000000 / 4000000 | 0.136 / 0.432 / 0.585 | 0.261 / 0.304 / 0.013 | 0.094 / 0.137 / 0.138 |
| pooled: DFlash2 101 + 1e6 + 2e6 (12000) | 0.0097 | 0.018 | < 1e-4 |
| pooled: DFlash2 5e6 + 6e6 + 7e6 (12000) | 0.241 | 0.468 | 0.477 |
| pooled: plain sampler (12000) | 0.962 | 0.866 | 0.209 |

The first three sets were the low ones, and in them the first token, which the plain sampler draws in every arm from the
1-token prefill, is already skewed (pooled p 0.0097; Monte-Carlo-calibrated 0.003): replaying the seeds on the CPU
(`pcg_hash` uniform, the law's cumulative bounds) reproduces all 12000 first tokens, so that law is exact and the skew is
in those seeds' uniforms, which also drive the acceptance and residual draws. Three fresh sets pass at every level,
pooled too. A Monte-Carlo calibration of the same statistics (triples drawn from the law) gives the same p-values; the
plain sampler's own sets come out at 0.010 / 0.14 / 0.165 calibrated at the third token. 27B: DFlash2 seed base 201 (800 trials) 0.353 / 0.177 / 0.355, base 500000 (1600 trials) 0.360 /
0.551 / 0.881.

*Server* (`bench/shortpf/serve_test.sh`, defaults = DFlash2 found automatically): tests/test_server.py 13/13 on both
models (before and after the ring snapshot).

*Cached turns* (`bench/shortpf/serve_ttft.sh --sizes 4000,16000 --greedy --cold-check --seed 11`, history prefill on;
c3bcc53 = MTP default, then this build = DFlash2 default, back to back; prefill = turn-1 prompt / TTFT):

| model, build | prefill 4k / 16k tok/s | cached TTFT 4k / 16k s (3 follow-ups) | cold check 4k / 16k | 256-token follow-up: tok/s, tok/cycle (cold check run) |
|---|---|---|---|---|
| 35B c3bcc53 | 1107 / 919 | 0.0645 (0.065, 0.068, 0.061) / 0.0766 (0.077, 0.081, 0.072) | yes / no (char 63) | 125.7 (126.0) |
| 35B this build | 1141 / 947 | 0.0616 (0.062, 0.068, 0.053) / 0.0694 (0.069, 0.080, 0.069) | yes / no (char 63) | 208.3, 5.10 (199.9, 5.10) |
| 27B c3bcc53 | 374 / 326 | 0.234 (0.234, 0.296, 0.227) / 0.259 (0.259, 0.315, 0.245) | yes / yes | 33.2 (33.2) |
| 27B this build | 382 / 337 | 0.227 (0.227, 0.284, 0.227) / 0.261 (0.261, 0.318, 0.244) | yes / yes | 69.7, 5.80 (68.5, 5.80) |

The 35B's 16k cold check parts at the same character with c3bcc53 (main's PERF records the same split for seed 11: the
cached path and one cold prefill chunk differently at a near-tie). With DFlash2 the history prefill ran after every
"stop" reply (17-46 rows), and the cached 256-token follow-up drafts exactly as a cold prefill of the same prompt does
(same tok/cycle, same text). Prefill is +2-3%: the MTP history rows are gone, the drafter's commit runs after the
first token.

*Multi-turn* (`engine_turns`, `Engine::prefill_history` after each reply as the server runs it; then `ni_run` with one
prefill of each turn's prompt):
* 27B, thinking on, short answers, `NI_ANE=0 NI_SHORTPF=0` (the GPU's large-prefill kernels are chunk-size invariant,
  so the cached state equals a cold prefill's): turn 1 5.08 tok/cycle (cold 5.083); turn 2 continues the history-prefilled
  state 3.33 (cold 3.335); turn 3 restores turn 1's checkpoint with its ring snapshot 4.11 (cold 4.112); all three answers
  identical to the cold runs.
* 35B, short answers (history prefill after each): 3.58 / 2.64 / 4.64 tok/cycle, cold 3.583 / 2.875 (another text: the
  MoE cached path and a cold prefill part at a near-tie) / 4.643. 35B, 1500-token answers: 3.66 / 3.35 / 3.48, cold 3.656 /
  3.346 / 3.426.

*Speed*, interleaved (`ni_run`, code_* 512 tokens; c3bcc53 has no `--ignore-eos` and runs to 512 anyway):

| model, T | c3bcc53 MTP (default depth) | this build, MTP (`NI_DFLASH=0`) | this build, DFlash2 (default) | DFlash2 tok/cycle, ms/cycle |
|---|---|---|---|---|
| 35B, greedy | 121.2 / 121.8 / 121.5 | 121.1 / 121.7 / 122.4 | **170.4 / 171.0 / 170.5** (+40%) | 3.430, 20.0-20.1 |
| 35B, T = 0.6 | 122.1 / 121.6 / 121.0 | 121.0 / 120.9 / 121.5 | **152.4 / 151.2 / 152.6** (+25%) | 3.318, 21.7-21.9 |
| 27B, greedy | 29.7 / 31.1 | 31.1 / 31.1 | **60.2 / 61.5** (+98%) | 4.776, 77.5-79.2 |
| 27B, T = 0.6 | 30.6 / 30.6 | 30.6 / 30.8 | **52.8 / 53.3** (+73%) | 4.121, 77.2-77.9 |

Prefill (`ni_run`, n_gen 1; "+ DFlash2" times the capture and the prompt commit too, which the server runs after the
first token), ms:

| | 35B p4000 | 35B p16000 | 27B p4000 (wavefront) | 27B p16000 (wavefront) |
|---|---|---|---|---|
| c3bcc53 | 3466 / 3450 / 3457 | 16363 / 16351 / 16462 | 10239 / 10176 | 48142 / 47846 |
| this build, DFlash2 off | 3463 / 3430 / 3431 | 16374 / 16367 / 17253 | 10230 / 10168 | 48062 / 47863 |
| this build, + DFlash2 capture and prompt commit | 3488 / 3472 / 3474 | 16410 / 16388 / 16962 | 10379 / 10318 | 48174 / 48049 |

Memory: the drafter with its reduced head is 0.62 GB resident on the 35B (process footprint +0.59 GB at load) and
1.61 GB on the 27B (+1.47-1.55 GB); per slot the ring and its two snapshots add 151 / 126 MB and the prompt-capture
buffer 67 / 105 MB (35B / 27B). The MTP head stays loaded when the drafter replaces it (a possible saving).

## Turn-boundary checkpoints through tool loops (`NI_QUERY_CKPT`, default on; macOS 27.0.1)

Each slot keeps two recurrent-state checkpoints, and every request saves one at its turn boundary (the end
of the rendered conversation, before the assistant header). The Qwen3.6 template renders the assistant
turns after the last user query with their `<think>` block and the turns before it without, so when a new
user query arrives the prompt diverges from the live state three tokens after the previous query
(`<|im_start|>assistant\n`, then the think block the new render drops): only the checkpoint saved at that
query can be restored. In an agent loop (query -> tool call -> tool response -> ... -> final answer -> new
query) every request after the query ends with a tool response, and each saved its checkpoint over the
one with the lowest position: after two tool calls the query's checkpoint was gone, and the new query
prefilled the whole conversation from token 0. opt/moe35's history prefill (`Engine::prefill_history`,
which re-renders the answer in the idle gap after it) restores the same checkpoint and gave up there.

The change: a checkpoint records whether its request ended with a user query (the template's own test: a
user message not wrapped in `<tool_response>`; role `tool` and assistant continuations are not), and a new
boundary overwrites an invalid checkpoint first, then a tool-response one, then the oldest query one. A
slot thus holds the newest query checkpoint and the newest checkpoint, which is the one a tool loop's next
request restores when the client's re-render of the last assistant turn differs from the generated tokens
(for example a client that does not send the reasoning back). Conversations without tools choose exactly as
before (all their checkpoints are query ones), and so does `NI_QUERY_CKPT=0` in every case. Memory is
unchanged: two checkpoints per slot, 65.9 MB each on the 35B and 156.9 MB on the 27B. A third checkpoint
was not added: on its own it moves the cliff from two tool calls to three, and beside the new rule it would
only keep the second-newest tool response, which only a client that rewinds one tool step would restore.

The 27B is not affected: its template keeps every think block unless `preserve_thinking=false`, so its new
query continues the live state (4k, 1 / 2 / 3 tool calls: 0.30 s and only the query's own 32 tokens
prefilled, with the rule and without).

Through the server (bench/ckpt/tool_loop_ttft.py; Qwen3.6-35B-A3B 4-bit, greedy, thinking off, `--ctx 32768`
with the default 2 slots, MTP on): system prompt with 4k / 16k tokens of repository context plus two tools,
the query asks the model to read notes/step1.txt and follow the instruction at its end, each file (~600
tokens of code) names the next one and the last says to stop and answer, the model's replies are sent back
as they came (content + tool_calls, as Pi does), and the new query follows the answer after 1 s (the
harness's turn gap). The model made exactly the requested 1, 2 or 3 tool calls every time. Same seeded
conversations on every server, interleaved spans under the GPU lock; before = main d6d2ed0 and this branch
with `NI_QUERY_CKPT=0` (2 runs each), after = `NI_QUERY_CKPT=1` (3 runs). Seconds to the first token of
the new query, server prefill in the log within 10 ms of it:
| context, tool calls | new query: prompt tokens | cached before / after | before | after |
|---|---|---|---|---|
| 4k, 1 | 5,394 | 4,511 / 4,511 | 1.05-1.10 | 1.07-1.18 |
| 4k, 2 | 6,118 | 0 / 4,514 | 6.21-6.42 | 1.87-1.93 |
| 4k, 3 | 6,793 | 0 / 4,513 | 7.05-7.42 | 2.65-2.75 |
| 16k, 1 | 17,534 | 16,665 / 16,665 | 1.43-1.48 | 1.49-1.58 |
| 16k, 2 | 18,274 | 0 / 16,666 | 22.2-22.9 | 2.58-2.68 |
| 16k, 3 | 18,931 | 0 / 16,665 | 23.4-24.2 | 3.64-3.78 |
Every other request of these conversations has the same prompt and cached tokens in all variants (main and
`NI_QUERY_CKPT=0` request for request) and the same timing within the spans' drift (cold 16k prompt
19.9-20.5 s, a tool step 0.85-1.5 s: the tool loop continues the live state, since the reply is sent back as
generated). What the new query still pays is re-prefilling the tool turns without their think blocks:
~730 tokens per tool step (the call and its ~600-token result), ~0.8 s each at 4k and ~1.1 s at 16k. With one
tool call both rules restore the same checkpoint and run the same passes; a targeted A/B of that case (two
pairs of spans with paired seeds, off-on then on-off, 4 conversations per size and span) measured
1.047-1.109 s (off) vs 1.036-1.075 s (on) at 4k and 1.426-1.563 vs 1.436-1.492 s at 16k, the later span of each
pair the slower one on every request, cold prompts included: the spread in the table's 1-call rows is drift
between server sessions.

With thinking on (4k, 2 tool calls; the second conversation was asked for 3 but the model wrote its third call
inside an unclosed think block, which the parser leaves in the reasoning, so it stopped after 2): with the
reasoning not sent back, every tool step diverges at its think block and restores the previous tool
response's checkpoint, which both rules keep; the new query took 6.21 / 6.12 s before and 1.79 / 1.73 s
after. With `reasoning_content` sent back (Pi replays it), the tool steps continue the live state and the
new query took 8.03 / 7.64 s before (2 / 3 calls) and 2.07 / 3.03 s after.

With opt/moe35's history prefill (first measured on a scratch merge of 57df0c2, since merged into main): after
the answer the engine renders the conversation as the next query will and prefills it from the query's
checkpoint in the idle gap. With the old rule it found no checkpoint after 2-3 tool calls; with this one it
runs every time ("history: prefilled 2240 tokens of the rendered reply (from 16665) in 3280 ms") and the new
query prefills only its own 31 tokens (58-72 ms on the server):
| new query TTFT, s | 4k: 1 / 2 / 3 tool calls | 16k: 1 / 2 / 3 tool calls |
|---|---|---|
| history prefill, old rule | 0.25 / 5.88 / 6.79 | 0.61 / 21.76 / 23.60 |
| history prefill + this rule (1 s gap) | 0.26 / 0.98 / 1.69 | 0.57 / 1.60 / 2.60 |
| history prefill, old rule (5 s gap) | | 0.51 / 20.88 / 22.00 |
| history prefill + this rule (5 s gap) | | 0.50 / 0.50 / 0.18 |
With a 1 s gap the history prefill (0.95-3.28 s) is still running when the query arrives, so the query waits
for its rest. With 5 s it is done, and the query's own GPU work is 78-83 ms; the rest is the GPU waking up
after the idle gap ("GPU start +98 / +413 / +419 ms" in the log, the same with the old rule at 1 call).
Again on this branch merged with main 8a05d3a (history prefill on main), 5 s gap, new query after 1 / 2 / 3
tool calls: 16k 0.54 / 21.14 / 22.23 s with `NI_QUERY_CKPT=0` vs 0.51 / 0.11 / 0.13 s with the default, 4k
0.54 / 6.31 / 7.13 vs 0.52 / 0.53 / 0.54 s (the query's own GPU work 63-95 ms, the GPU's start after the pause
21-473 ms); replies identical, tests/test_server.py 13/13 on both models.

Correctness (greedy; every request's reasoning, content and tool calls compared for the same seeded
conversation across servers):
* 0 differences in 74 conversations: 24 thinking off (4k / 16k, 1-3 tool calls, 3 rounds: flag on vs main and
  flag off), 16 in the 1-call A/B, 8 with the default vs `NI_QUERY_CKPT=0` (2 + 6 on the merge), 4 thinking on
  (reasoning echoed and not), 10 whose new query asks for a detailed explanation after 2-3 tool calls (4k and
  16k, answers of 300-400 tokens, nine of them run to the limit: a cold re-prefill before, a restored
  checkpoint after), 3 on the 27B, 9 through the opt/moe35 merge. The restored path is the cached path every
  follow-up already takes, which is not bitwise a cold prefill (different chunking; opt/moe35 measured KL 1e-4
  to 6e-3 between them and saw greedy texts split at near-ties): none of these conversations hit one.
* main and `NI_QUERY_CKPT=0`: the same prompt and cached tokens for every request of every conversation.
* tests/test_server.py (bench/shortpf/serve_test.sh): 13/13 on the 35B with `NI_QUERY_CKPT=1` and `=0`, and
  with the new default (no variable) on the 35B and the 27B.
* bench/ckpt/kind_probe.py (one-token requests, NI_LOG=2): a plain query, a user query after a tool response,
  a user message that only starts with `<tool_response>` and multi-part text are query boundaries; role
  `tool`, a user message wrapped in `<tool_response>` (whitespace around it, or split over text parts) and an
  assistant continuation are not (all 8 as expected).

Made the default after these runs (`NI_QUERY_CKPT=0` restores "oldest first"): the default build against
`NI_QUERY_CKPT=0` on the same 4k conversations, new query after 2 / 3 tool calls: 1.95 / 2.76 s (restored the
query's checkpoint) vs 6.57 / 7.42 s (from token 0), replies identical.

## Tool calls written inside the reasoning (`NI_THINK_TOOLCALLS`, default on; macOS 27.0.1)

Qwen3.6 sometimes writes its tool call inside the think block and ends the turn without ever writing
`</think>`: "The user wants me to read notes/step3.txt next. Let me do that.\n\n<tool_call>...</tool_call>",
then `<|im_end|>`. The parser treated everything before `</think>` as reasoning, so the call went out as
reasoning_content and the reply had no content and no tool_calls: an agent saw an empty final answer and the
loop stopped. Found in bench/ckpt's thinking tool loops; the Qwen team says it cannot be fixed in the model yet
and points to the application layer (QwenLM/Qwen3.6#125).

What other engines do (default branches read on 2026-10-17): vLLM, mlx-lm and Ollama end the reasoning at the
first `<tool_call>`, so a call drafted in mid-thought is executed too and the rest of the reasoning becomes
content (vLLM #58227); SGLang does the same when streaming, but without streaming splits there only if no
`</think>` follows (its open PR #42030 holds the streamed text until `</think>` or the end); llama.cpp
constrains sampling (after a `<tool_call>` in the reasoning its grammar allows only more calls or the end);
oMLX, the app's other engine, promotes calls found in the reasoning at the end of the turn when the content has
none, filtered by tool name.

Kapitsa: a `<tool_call>` inside the reasoning is held back with the tokens after it instead of being streamed.
If the reasoning goes on (non-whitespace text, a call after `</think>`, content, a stray tag), the held tokens
are replayed through the parser as before: a drafted call stays reasoning, and the stream only differs in
arriving in one piece. If the turn ends with the call(s), with nothing after them but whitespace or `</think>`
and whitespace, and the model stopped by itself (not max_tokens, not a cancelled request), the complete calls
that name one of the request's tools are returned as tool calls (finish_reason `tool_calls`) and the reasoning
ends before them. An unknown function, an unterminated call or a cut turn stays reasoning. The parser moved to
src/engine/output_parser.h; `tools/parser_test <model dir>` runs 26 checks on real token ids (both models'
tokenizers, CPU only): unchanged streams for every turn without a call in the reasoning, the old results for
drafts, the call returned in the cases above, no call markup streamed as reasoning, deltas adding up.

How often (bench/ckpt/tool_loop_ttft.py --think --no-q2: thinking on, 4k-token context, loops of 3 tool calls,
the same seeded conversations with the switch off and on, 35B unless noted). A loop is cut short when a step's
reply is empty with the call left in the reasoning; the client then takes it as the final answer:
| sampling, history | conversations | cut short (off) | cut short (on) | calls returned from the reasoning (on) |
|---|---|---|---|---|
| greedy, reasoning not sent back | 8 | 3 | 0 | 3 |
| T = 0.6 (seeded per request), not sent back | 6 | 2 | 0 | 2 |
| greedy, reasoning sent back (as Pi does) | 6 | 0 | 0 | 0 |
| T = 0.6, sent back | 6 | 0 | 0 | 0 |
| 27B, greedy, sent back | 3 | 0 | 0 | 0 |
All five hits were the third call ("The user wants me to read notes/step3.txt next. Let me do that." and the
call, then `<|im_end|>`): with the earlier assistant turns rendered with empty think blocks (reasoning not sent
back), 5 of 14 loops broke; with the reasoning sent back, 0 of 15.

Correctness: every conversation without a hit has the same replies (every request's reasoning, content and
tool calls) with the switch off and on: 21 on the 35B, 2 of 3 on the 27B (the third splits in its second
request's reasoning because one server compiled the Neural Engine program at start, so that conversation's
first prefill ran on the GPU alone there and with the split on the other; no call was held in either). The
five with a hit are the same up to that step and then go on with the returned call. tests/test_server.py 14/14
(new: thinking + tools returns read_file as a tool call) on both models with the switch off, on and at the
default; parser_test 26/26 with both tokenizers. Made the default after these runs; `NI_THINK_TOOLCALLS=0`
leaves such calls in the reasoning.

Both switches re-checked on this branch merged with main b457a52 (DFlash2 on by default, history prefill),
before landing: tests/test_server.py 14/14 on the 35B and the 27B at the defaults; new user query after 2 / 3
tool calls (5 s pause) 16k 0.54 / 0.12 s and 4k 0.54 / 0.53 s by default vs 19.49 / 20.34 s and 5.76 / 6.42 s
with `NI_QUERY_CKPT=0` (the history prefill ran in 4 of 4 conversations vs 0), replies identical; thinking tool
loops (6 conversations, reasoning not sent back, greedy, DFlash2 decoding) 6 of 6 complete with 2 calls returned
from the reasoning vs 4 of 6 with `NI_THINK_TOOLCALLS=0`, the other 4 identical.

## Qwen3.5-4B and smaller Macs (branch opt/air, 2026-10-17)
Targets: Qwen3.5-4B, Qwen3.8-27B, Qwen3.6-35B-A3B on this MacBook Pro (M4 Pro 20-core GPU, 48 GB) and the MacBook
Airs (M4: 8/10-core GPU, 120 GB/s; M5: 10-core, ~153 GB/s; 16/24/32 GB). Every GPU run under the shared lock; A/B
runs interleaved. The GPU was shared with other sessions' jobs all morning, so absolute speeds drift between rounds.

### 4B correctness: A_log stored as f32
The 4B's MLX conversion (lmstudio-community, via mlx_vlm) stores each DeltaNet layer's `A_log` as f32 (bf16 values
widened: 0 of 768 have low mantissa bits); the 27B and 35B store bf16, and every GDN kernel reads bf16. Read as bf16
the f32 bytes gave wrong decay rates in all 24 DeltaNet layers: KL 5.8e-2 against mlx-lm at the code prompt's last
position (prefill and the token-by-token decode path alike: every Kapitsa variant agreed with every other), greedy
text off after a dozen tokens. Float parameters that are not bf16 now get an owned bf16 copy at load (exact here).
Against mlx-lm with the model cast to fp32 (`model.set_dtype(mx.float32)`, a higher-precision reference than its
default bf16 run):
| check | Kapitsa | mlx-lm default (bf16) |
|---|---|---|
| teacher-forced decode path, code prompt + 120 tokens (205 positions): KL mean / median / max | 9.3e-8 / 2.6e-8 / 3.7e-6 | 5.5e-4 / 2.6e-4 / 8.6e-3 |
| argmax equal to fp32 | 205/205 | 203/205 |
| prefill, last position: p700 / p4000 / p16000 KL | 3.2e-10 / 5.8e-11 / 5.2e-10 | 9.4e-8 / 2.6e-7 / 1.9e-7 |
| greedy 256 tokens equal to fp32 mlx-lm: code / prose | 256 / 256 | 7 / 226 |
The coordinator's "agrees with mlx-lm for 12 tokens": after the fix the first split from bf16 mlx-lm is at token 7,
a tie inside mlx-lm itself (0.457 vs 0.457); the fp32 reference and Kapitsa agree throughout.

### 4B speed against mlx-lm (same weights and token ids, raw generation, 2026-10-17 ~05:30)
| | Kapitsa (ni_run) | mlx-lm 0.32.2 (stream_generate, prefill step 2048) |
|---|---|---|
| prefill p4000 | 4440-4461 ms per rep (897-901 tok/s), GPU alone | 839-840 tok/s |
| prefill p16000 | 20688-20836 ms (768-773 tok/s), GPU alone | 725 tok/s |
| decode, code prompt, 512 greedy tokens | 83.0 / 83.1 tok/s plain | 80.0 |
Prefill is compute bound (7.1 GFLOP per token: 3.6 G non-embedding weights). The Neural Engine split now runs on the
4B too (below): 4k ~2000 tok/s. Decode reads 2.36 GB per token (the tied 4-bit embedding is the 357 MB output
layer): 83 tok/s = 196 GB/s; the 4B's small matrices (N 2560, K 2560-9216) reach 147-221 GB/s per kernel against
the 27B's ~240.

### 4B speculative decoding: the official MTP head
MLX conversions of the small Qwen3.5 models drop `mtp.*`. The official Qwen/Qwen3.5-4B checkpoint (Apache-2.0) has
the head as 15 bf16 tensors (241 MB of 9.3 GB); `tools/mtp_sidecar.py` fetches only those with HTTP range requests
and writes `mtp.safetensors` with the RMSNorm weights + 1 (Hugging Face stores them zero-centred: trunk layer 0
0.139 against the MLX file's 1.139) and the trunk hidden it reads in the header. ni_run, 256 greedy tokens:
| hidden / draft vocabulary | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| pre-norm, full vocabulary (code) | 106.8 tok/s, 1.73 tok/cycle, 0.74 accepted | 101.6, 2.16, 0.59 | 58.9, 2.34, 0.46 |
| post-norm, full vocabulary (code) | 110.3 / 115.8, 1.76, 0.76 | 106.6 / 108.9, 2.18, 0.59 | 64.1, 2.38, 0.47 |
| post-norm, ids < 32768 + specials (code) | **119.5**, 1.67, 0.67 | 116.6, 2.04, 0.52 | |
| post-norm, full (prose) | 112.0, 1.70, 0.71 | 102.8, 2.06, 0.53 | |
| post-norm, ids < 32768 + specials (prose) | **120.9**, 1.69, 0.70 | 112.4, 1.99, 0.50 | |
Plain decoding 83 tok/s. A depth-1 cycle is 13.9 ms against a 12.0 ms step: the 2-row verify (qmvrc1, two
single-row threadgroups per weight block) costs about one step, the draft ~1.1 ms. Depth 3 (a 4-row verify on the
FMA kernels) collapses. Defaults: depth 1 below a hidden size of 4096 (27B stays at 2); the full-vocabulary draft
only where the lm_head is < 10% of the weights (27B 4.5%, 4B 13%). Depth-1 output equals plain decoding for all 256
tokens on both prompts; depth 2 splits at token 81 on both, each a near-tie of plain decoding itself (top-2 logit
margins 0.0094 / 0.0053: the 3-row verify's half-precision products). Through the server (MTP by default):
tests/test_server.py 12/13; the failing check expects the tool-result turn to reuse the whole first prompt, but the
4B's own tool-call text is not the template's rendering of it, so the turn restores the turn-boundary checkpoint
(cached 371 of 378).
DFlash2 drafter: the community nchapman/Qwen3.5-4B-DFlash2 (1.32 GB bf16) declares no license and is the
`speculators` variant (5 full-attention layers, 16 q / 4 kv heads x 256, block 16, RoPE theta 1e4), not the z-lab /
inco geometry Kapitsa's DFlash2 kernels take (32 / 8 x 128, block 8, window 2048, RoPE 1e7); z-lab's
Qwen3.5-4B-DFlash is DFlash v1. Not used.

### Memory: KV caches that grow, a budget, a fit check
(Design: DEVELOPMENT.md, Memory.) Before: every slot's KV cache for the whole context at start (the app measured
35B 22.6 GiB and 27B 25.6 GiB with 2 slots x 64k: a slot's whole cache becomes resident with its first conversation,
since Metal wires a buffer whole when a command buffer binds it). Bitwise checks, ni_run, base 2006b21 against this
branch with the caches allocated whole and with `--kv-grow` (empty, grown before every chunk and step, the
engine's path): identical logits (prompt and every decode step) and greedy ids, plain and MTP, on the 4B (p4000, 64
tokens, depth 1), the 35B (p4000, 32, depth 1) and the 27B (p700 through the Neural Engine split, 16, depth 2).
Metal allocations (ni_run sizes its sequence at 8192 positions; base / trimmed buffers / grown to need):
4B 3901 / 3558 / 3407 MB, 35B 21196 / 21196 / 21103, 27B 18817 / 18163 / 17664; process footprint 27B 18074 /
17795 / 17321 MB (the 27B's fp32 gate/up rows, 2 x 143 MB, were bound by the MTP head's verify and so resident).

### GPU cores (MacBook Airs: 8-10)
`Gpu::core_count()` reads the AGXAccelerator's `gpu-core-count` from the IORegistry (20 here; `NI_GPU_CORES`
overrides). Host-side choices that sized grids for 20 cores now scale with it and keep their constants at 20 (so
nothing changes on this Mac): the decode and MTP-verify attention split-K (one chunk per core and kv head, >= 64
positions), the short-prefill attention key split (~5 threadgroups per core) and the qmms token-block split (below
24/5 x cores threadgroups, aiming at 32/5 x cores; bitwise neutral). Prefill GEMM grids are 100+ threadgroups at
any chunk and need no change. No kernel waits for another threadgroup (threadgroup-local atomics and one status flag
only), so fewer cores cannot deadlock; the Neural Engine handoff waits only on events signalled by committed command
buffers, with a 2 s watchdog. Limits: Apple's feature set tables give Apple10 (M5) the same 32 KB of threadgroup
memory and 1024 threads per threadgroup as Apple7-Apple9; `ni_kstats --limits` finds no production kernel above
32 KB (only the bench kernel attn_prefill9_q8_k32, 33280 B, already known) and every kernel below 1024 threads
dispatched within its limit; `Encoder::dispatch` now names a kernel dispatched above its pipeline's
maxTotalThreadsPerThreadgroup instead of failing the command buffer.

### Neural Engine: the 4B split and a default share per chip family
Before, the split had a share for the M4 Pro alone and `ane_rotate` needed hidden/128 to split over 8 simdgroups
(the 27B's 40 blocks). It now picks 1-8 simdgroups that split the blocks evenly (4B: 20 blocks, 5 x 4), and the
default share comes from the chip family and GPU size: M4 and M5 family, the wavefront with the whole FFN up to ~30
GPU cores (the measured M4 Pro balance: per layer and 2048-row chunk the ANE takes ~1.94 ms per 512-channel unit,
the GPU ~100 ms of mixer + 6.2 ms per unit it keeps, both GPU terms scaling with 20 / cores; so on an Air's 8-10
cores the GPU is even more the bottleneck and the whole FFN on the ANE also takes heat off the fanless GPU), ~31 of
34 units at 40 cores, the plain split scaled the same way; older chips GPU only unless `ane_calib --wave --write`
calibrated them. A calibration on the Mac still wins (the 27B here: 34/34 either way).
4B on this M4 Pro (ni_run, NI_NO_MTP, 18/18 units, two interleaved rounds, the second under other sessions' load;
ms per prefill rep):
| round | GPU alone p4000 | split p4000 | GPU alone p16000 | split p16000 |
|---|---|---|---|---|
| 1 | 4537, 4494 (887 tok/s) | 1996, 1995 (2004 tok/s, 2.25x) | 20306, 20621 (783 tok/s) | 13635, 16065 (1077 tok/s) |
| 2 | 6762, 6531 | 2869, 2924 (2.29x) | 27502, 26212 | 14237, 13936 (1.90x) |
Logits against the GPU alone, last prompt position + 32 greedy steps: p4000 KL mean 8.2e-5, median 1.6e-6, max
1.2e-3, 33/33 tokens; p16000 the tokens split at step 13 where the GPU alone has a near-tie (0.241 vs 0.219; split
0.243 vs 0.197), KL up to 1.7e-3 before it. For scale, mlx-lm's default bf16 path sits at KL mean 5.5e-4 from fp32
on this model. Startup check 2.1% of the ANE's part (27B 0.75-0.88%). Program: 13 functions, compiled in 15 s once
(the ANE daemon's cache afterwards), 188 MB of surfaces.

### With DFlash2, the tool-loop checkpoints and the eos fix (main b457a52, 73285c7, 30d430a merged into opt/air)
Bitwise checks, ni_run, main against opt/air after the merges, same session: 35B p4000 + 32 plain, 27B p700 + 16
plain (Neural Engine split) and 4B p4000 + 32 plain (GPU alone): logits of every step bitwise equal; MTP (35B code
128 tokens depth 1, 27B code 64 depth 2, 4B code 128 depth 1) and DFlash2 greedy (35B code 256, 27B code 128): token
ids equal (main's ni_run is byte-identical between 73285c7 and 30d430a). tests/test_server.py (14 checks) through
the server with the defaults, KV caches starting empty and growing, on the merge with 30d430a: 4B with MTP 14/14
(also GPU alone), 35B with DFlash2 14/14, 27B with DFlash2 14/14. Before main's eos fix the 4B failed two (its
replies ran on past <|im_end|>). parser_test passes on the merged build (35B and 4B tokenizers).

Speed A/B, main 30d430a against opt/air, ni_run, interleaved, two rounds (tok/s; prefill per rep):
| | main | opt/air | opt/air / main |
|---|---|---|---|
| 35B prefill p4000 (3 reps x 2) | 1083.7, 1126.9, 1127.4, 1095.6, 1123.4, 1129.9 (mean 1114.5) | 1098.3, 1126.0, 1129.2, 1110.5, 1128.1, 1129.2 (mean 1120.2) | 1.005 |
| 35B prefill p16000 (2 reps x 2) | 939.0, 944.2, 940.4, 944.2 | 942.4, 943.9, 939.3, 943.8 | 1.000 |
| 35B DFlash2 greedy, code 256 tokens | 149.3, 148.2 | 148.0, 149.0 | 0.998 |
| 35B MTP depth 1, code 256 tokens | 111.3, 112.2 | 110.8, 111.8 | 0.996 |
| 27B prefill p4000, Neural Engine split (2 reps x 2) | 385.4, 385.6, 386.0, 385.6 | 386.4, 386.0, 384.7, 385.1 | 1.000 |
| 27B DFlash2 greedy, code 128 tokens | 41.5, 40.4 | 41.1, 41.1 | 1.004 |
| 27B MTP depth 2, code 128 tokens | 25.6, 25.3 | 25.7, 25.8 | 1.012 |
(Absolute decode speeds are below the DFlash2 sections' 512-token runs: short runs, and a Mac shared with other
sessions' jobs all day; the A/B is interleaved.)

DFlash2 memory: the fit check's plan of the drafter (DFlash2::plan_bytes) missed the bf16 codebooks (254 MB), the
reduced draft head (98304 rows of the target's LM head: 35B 114 MB, 27B 284 MB), the first prefill's prompt-capture
and commit buffers (115 / 168 MB) and the ring scratch; it now sizes what the code allocates: 1.04 GB (35B) and
2.03 GB (27B) with 2 slots, not 0.67 / 1.41. At load the engine logs the plan against the allocation: 35B 0.623 GB
planned / 0.623 allocated, 27B 1.606 / 1.607 (the rings, 0.30 / 0.25 GB for 2 slots, are exact by construction).
`--dflash auto` keeps the drafter only where one full `--ctx` conversation fits beside it (before: also where the
model alone could not hold one, then taking most of what was left: 35B on a 32 GB Mac at 2/3 49.6k -> 19.5k tokens;
at a 17.4-18.8 GB budget the 27B would not have started).
Through the server, 35B `--ctx 65536 --slots 2`: `--mem-budget 21.99` (a 32 GB Mac at 2/3) leaves the drafter out,
takes 1024-row prefill chunks and decodes with MTP, KV room 49.5k tokens; `--mem-budget 25.73` (78%) loads it, KV
room 158k tokens.

Slot eviction: a slot whose KV cache was given up for another conversation kept its DeltaNet / conv state, and its
next prompt, starting at position 0 with no tokens left, prefilled on top of it. A dropped slot now starts from a
zero state. Server test (4B, GPU alone, a budget for one ~4k conversation and not two; A, B, A again, B again, each
evicting the other slot, greedy 48 tokens): before the fix the repeated replies differ from the first ones (A and B),
after it they are identical. A prompt that reuses nothing of its slot also gives up the slot's own cache before
other slots' (no copy, no transient), and a cache that starts empty needs no 256 MB copy margin.

DFlash2 with a growing KV cache: through the server (`--ctx 65536 --slots 2`, GPU alone), a ~3.4k-token prompt and a
long greedy answer, so the cache grows inside the DFlash2 loop (35B 0 -> 5120 -> 7168 positions over 2500 tokens,
27B the same over 1800): the replies are identical to main's, whose caches are allocated whole at start (35B 100.4
vs 102.3 tok/s, 27B 31.4 vs 31.5 on this essay prompt; the growth step took 8.5 / 17.5 ms).

Process footprint through the server (`--ctx 65536 --slots 2`, the app's settings; at load, after a 4k
conversation, after a second 4k one in the other slot, after a 15k one), main 30d430a against opt/air:
| model | main | opt/air |
|---|---|---|
| 4B (MTP) | 2.48 / 5.58 / 7.89 / 7.89 GiB | 2.48 / 3.64 / 3.84 / 4.26 GiB |
| 35B (DFlash2) | 19.70 / 21.82 / 23.15 / 23.15 GiB | 19.70 / 20.67 / 20.82 / 21.09 GiB |
| 27B (DFlash2, Neural Engine split) | 16.35 / 22.61 / 26.79 / 26.79 GiB | 16.35 / 18.66 / 19.08 / 19.90 GiB |
(main allocates both slots' 64k-token caches at start and a slot's becomes resident with its first conversation;
opt/air grows each cache to what its conversation uses.)

GPU cores, simulated with `NI_GPU_CORES` on this Mac (ni_run, GPU alone): at 10 and 8 cores against 20, greedy ids
are identical for the 4B and the 35B (p4000 + 64 tokens, plain and MTP depth 1) and the 27B (p700 + 32, plain and
MTP depth 2), and DFlash2 greedy for the 35B (code 256 tokens) and the 27B (code 128); logits differ only by
summation order (KL max: 4B 9e-9, 27B 4e-10, 35B 3.9e-4 where the split-K decode attention changes the MoE routing
inputs by rounding), the short-prefill suffix logits likewise (35B 5.4e-5, 4B 1.8e-10, 27B 1.5e-10). Speed on this
Mac does not change with the setting (DFlash2 35B 147.4 / 147.5 / 147.1 tok/s at 20 / 10 / 8): it only sizes grids.

