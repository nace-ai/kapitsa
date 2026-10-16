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
M4 Pro (20-core GPU, 16-core ANE), macOS 27.0.1 (26A434), Qwen3.8-27B MLX 4-bit g64, 2026-10-16. Design and
background: `research/inference/ane/ANE_DESIGN.md`; code: `src/ane/`, `src/kernels/ane_ffn.metal`,
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

### Cached turn (the harness pattern: 4k context, 37-62-token follow-up suffixes)
ni_run with the harness's real suffixes (bench prompts regenerated with the chat template: 4123 + 62, 4178 + 53,
4224 + 37), `--prefix P --split B --pf-reps 6`, median of reps 1-5; base 94.5-95.5 ms for 53 tokens. Isolated
profile of that pass: trunk experts ~45 ms (bandwidth: ~10 GB of expert weights for ~150 distinct experts per
layer), dense projections ~30 ms (qmms at 5.2-5.8 TFLOP/s; bench_qmms keeps qmms the best kernel at 37-62 rows),
attention 7.8, GDN step 3.7, the MTP head's MoE 2.7 (its experts are group-32 and fell to the 64-row large tiles),
router 4.5, bucket 2.2, small kernels the rest.
* **Experts of <= 2 rows by FMAs** (qmms_moe2_*, NI_SHORTPF_MOE_FMA): at these lengths most experts get 1-2 rows,
  which the 8-row MMA path pads to 8 rows (8 multiply-adds per weight on the same ALUs). An expert with at most 2
  rows takes a matvec form (codes converted once, MLX's pre-scaled-x trick, fp32 products: no half rounding of
  the weights); the others keep the MMA path. Only the expert's first tile qualifies: which pairs land in the last
  tile of a larger expert depends on moe_bucket's atomic scatter order, and the first version (any tile of <= 2
  rows) was not run-to-run deterministic (KL 1.5e-6 between two runs). Also takes group-32 experts, so the MTP head
  stops falling back to the large tiles. -5.5 ms (89.3-89.5 vs 95.0); KL vs base 4.5-4.9e-5.
* **Token-block split of small qmms dispatches** (NI_QMMS_TY, default): the router + shared gate (N = 257: 17
  threadgroups) and the shared expert's gate/up (32) split their 8-row token blocks over grid.y; bitwise equal.
* moe_bucket3, moe_topk_rows8, gdn_step_parr4_snap_l16, attn_prefill11g/gs with the rescale skip: bitwise equal.
* **MTP history after the first token** (engine, NI_MTP_DEFER): the last chunk's MTP history rows run in their
  own command buffer queued behind the trunk's, so the first token is out before them; the first speculative cycle
  is queued after them (no change to any output).
Tried, slower: the experts in three dispatches by tile class (1 / 2 / 3-8 rows, to give the 1-row path its own
56 instead of 91 GPRs: 1855 vs 1721 us), qmms with 32-bit row offsets (no fewer GPRs).

The server pays an idle penalty the back-to-back reps do not: the harness waits 1 s between turns. ni_run
`--pf-gap MS` (new) idles the GPU before each rep: 53-token pass 94.5 ms back to back, 95 at 100 ms, 103 at 300 ms,
112-114 at 1 s (GPU time +19%), 131-167 at 2 s (another 25-40 ms between commit and the GPU's start/finish being
seen). A 25 ms keep-alive dispatch during the gap did not help (106-108 vs 107-110 ms at 1 s). Through the server
(gap 1 s) the same turns took 116-144 ms against 76-123 ms with no gap; one run, during heavy load from other
processes, 207-217 ms. The engine log now prints the last chunk's encode time, commit -> GPU start, GPU time and
GPU end -> wake-up.
