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
Tile anomaly: 32x64 and 32x128 tiles (2 row fragments per simdgroup) run at 2-3 TFLOP/s.

Attention (D=256, 16 q / 2 kv heads), 2048 queries x 16K keys: ours 200 ms (2.56 TFLOP/s), MLX
steel attention 97 ms (5.3). Tried: GQA heads as fragment rows (v2), in-register softmax (v3/v4,
slower: spills), head dim split over simdgroup pairs like MLX (v5), K^T tile (v6), V load overlapped
with softmax (v7), 16-byte tile loads (v8, -10%), 8 queries/16-key tiles (v9, slower), per-lane
fragment loads (v10). Ablation: device K/V loads ~25%, Q.K ~30%, P.V ~15%, the rest bookkeeping.
Open: find what MLX does differently (it runs 32 rows of ONE head per threadgroup).

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

## nace-infer vs other engines (M4 Pro 48 GB, tok/s)
Other engines from research/inference/REPORT.md: server runs on real prompts (prefill = prompt
tokens / TTFT, 4k = 4,133 and 16k = 16,285 tokens), MLX weights except llama.cpp (GGUF UD-Q4_K_M).
nace-infer from ni_run (MLX 4-bit weights, not necessarily the same files as REPORT.md). Splash:
published numbers, M5 Pro, needs macOS 26.4+.

Qwen3.6-35B-A3B 4-bit:
| engine | decode | prefill 4k | prefill 16k |
|---|---|---|---|
| nace-infer 0.3, plain | 105 | 765-785 | 600-606 |
| nace-infer 0.3, MTP depth 1 | 116-123 | 765-785 | 600-606 |
| mlx-lm 0.32.0 server | 63-67 (raw generate 81) | 923 | 824 |
| oMLX 0.7.1.dev1 | 84-99 | 840-915 | 769-875 |
| MTPLX 2.12.2, MTP depth 1 | 66-76 | 890-902 | 788-792 |
| llama.cpp b11429 Metal | 53 | 733 | 681 |
| llama.cpp + draft-mtp | 66-73 | 719 | 652 |
| Splash (M5 Pro, reference only) | 210 | - | 2,011 at 32K |

Qwen3.8-27B 4-bit:
| engine | decode | prefill 4k | prefill 16k |
|---|---|---|---|
| nace-infer, plain | 16.0 | 111 | - |
| nace-infer, MTP depth 1 | 23.9-25.3 | 111 | - |
| mlx-lm | 10.7-11.4 | 125 | 101 |
| oMLX | 11.8-12.0 | 128 | (refused, memory guard) |
| MTPLX, MTP | 20.6-30.8 | 113-125 | 101-110 |

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
