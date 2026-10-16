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
