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

## Short prefills (`NI_SHORTPF=1`, opt-in; macOS 27.0.1)
Cached-turn TTFT (second chat turn on a cached 4k/16k prefix = a short new suffix) and short cold
prompts. Two causes, both fixed behind the flag:
1. **The suffix was split at the turn-boundary checkpoint** into two prefill passes, each reading
   every weight (27B: ~370 ms per pass at <= 32 rows). Now `encode_prefill(..., snap, snap_row)`
   captures the checkpoint inside the pass: `gdn_prep_par_snap` copies the conv window as it stands
   after `snap_row` tokens (before rolling it) and `gdn_step_par_snap_l*` writes the recurrent state
   after token `snap_row - 1`. `ni_run --prefix P --split B` checks it against a separate prefill of
   [P, B): bit-identical (conv and state max|d| = 0) on both models.
2. **Small-T GEMMs**: `qmm_b4_m32_n32` (any T < 64) streams weights at 35-40 GB/s with 4-32 rows
   (bench_qmms, same on both models' shapes). `qmms_b{4,8}_t{1..8}` (src/kernels/qmms.metal):
   simdgroup owns 8 weight rows, each lane loads 16 contiguous codes of its row per 64-group and
   dequantizes them as half2 straight into simdgroup-matrix A operands (no threadgroup tiles, no
   barriers); the k order inside a group is permuted so the matching B operand (8 k x 8 tokens) is
   four 8-byte loads per lane from the row-major half X; K split over 4/8 simdgroups of a
   threadgroup (fixed-order sum). Up to 4 projections sharing X in one dispatch (GDN qkv/z/b/a,
   attention q/k/v, router + shared gate), fused gate/up+SwiGLU (`qmms_gu_*`, dense MLP and the
   MoE shared expert), MoE experts in 8-row tiles (`qmms_moe_gu_b4` / `qmms_moe_dn_b4`, tile lists
   of `moe_bucket2` with BM = 8). Beyond 64 rows, whole 64-row tiles stay on qmm and the remainder
   rows go to qmms (a partial 64-row tile cost a full one). Few queries over a long cached prefix:
   attention rows are (query, head) pairs of one KV head (`attn_prefill11g*`: 32-row blocks hold
   4 queries x 8 heads instead of padding T to 32) with the keys split over threadgroups
   (`attn_prefill11gs` + `attn11_reduce`).

bench_qmms (M4 Pro, isolated kernel, 4-bit g64; qmm = the engine's tile below 64 rows):
| shape (N x K) | T | qmm_b4_m32_n32 | qmms (ks) |
|---|---|---|---|
| 35B qkv 8192x2048 | 8 | 235 us (40 GB/s) | 54 us (174 GB/s, ks4) |
| 35B qkv 8192x2048 | 32 | 245 us (4.4 TFLOP/s) | 176 us (6.1, ks4) |
| 35B out 2048x4096 | 8 | 164 us | 30 us (ks8) |
| 27B qkv 10240x5120 | 8 | 789 us (37 GB/s) | 166 us (177 GB/s, ks4) |
| 27B gate 17408x5120 | 8 | 1296 us | 272 us (184 GB/s, ks8) |
| 27B gate 17408x5120 | 32 | 1335 us (4.3 TFLOP/s) | 1054 us (5.4, ks8) |
| 27B down 5120x17408 | 32 | 1467 us | 1296 us (ks8) |
| 27B gate 17408x5120 | 64 | m64_n64 2264 us (5.0) | 2175 us (5.3, ks8) |
Fused gate/up at 27B T=32: 1663 us for both matrices (two qmms: 2108). FMA tile matvecs for the
MoE experts (1-3 rows per expert tile, so the 8-row MMAs are mostly padding) were slower: 35B T=32
prefill 55 ms -> 71 (tiles of <= 2 rows) / 75 (<= 4); removed.
