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
