# Results, 2026-11-04: the dense prefill GEMMs on a 128-row tensor tile (NI_QMT128), M5 Max

The Qwen dense prefill GEMMs on the M5's matrix units, `qmt_m64_n64` and the fused gate/up `qmtg_m64_n64`, now run
on a 128-row tile (`qmt_m128_n64`, `qmtg_m128_n64`, and `qmt_m128_n64_ld` for the Neural Engine split's GPU share)
when a chunk has 96 rows or more. The DFlash2 prompt commit's GEMM uses it too.

**The tile.** Four simdgroups of 32 token rows each cover the tile's 64 weight rows. The matmul reads X straight from
device memory, and only the dequantized weights go through threadgroup memory (8 KB, double-buffered). So a weight
is dequantized once for 128 token rows instead of 64. This is the layout of Splash's 128-row prefill tile (Splash
1.5.0, `gguf_linear.metal`), which prefilled the 27B and the 35B-A3B faster than Kapitsa in the 2026-10-25 table.

**The same bits.** Each simdgroup runs two 32 x 32 x 32 `matmul2d`s a step: the op the 64-row tile runs, on the same
elements, in the same K order. The last rows of a chunk run on the 32 rows that end at row M - 1 and store only their
own, so the matmul's view never passes the chunk.

**Two codegen traps.** Each of these cost about 17% of the whole kernel (`bench_qmm`):
- an epilogue loop that reads both accumulators at one index (the gate/up product formed in the store loop, or both
  accumulators stored by one loop);
- a per-item array of weight pointers, as `qmt_tile` has.

So the gate/up product goes into the up accumulator first, and each accumulator is stored by its own loop.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, under the GPU
lock, the session idle during the runs ([run_ab.sh](run_ab.sh), [run_ab.log](run_ab.log)).

- **Old:** main `fa0bc00` in its own Release build. Its binaries report `d9e5430`: the build's CMake cache held
  `KAPITSA_GIT` from an earlier `-D`. The code is `fa0bc00`'s.
- **New:** the branch, which is main plus this change, in its own Release build.
- **The runs:** `ni_run MODEL @prompt-ids.txt 8 0 --pf-reps 3 --dflash none`.
- **The prompt:** the 4,096 ids of `bench/quality/ids_qwen38` ([prompt-ids.txt](prompt-ids.txt), the same as #941's).
- **Chunks:** 2,048 rows, `ni_run`'s default. The server's single-slot 35B-A3B takes 4,096-row chunks; this run does
  not cover them.
- **The order:** rounds alternate old new / new old, with 45 s idle before every run.

## 4K prefill, old against new

| Model | Rounds | Old (ms, per round) | New (ms, per round) | Change |
| --- | ---: | --- | --- | ---: |
| Qwen3.6 35B-A3B | 6 | 659.6, 664.6, 663.9, 661.3, 667.5, 658.3 | 626.4, 625.8, 634.6, 627.3, 631.3, 622.3 | **-5.2%** (662.5 -> 627.9) |
| Qwen3.8 27B, rep 1 | 4 | 3,866.3, 3,855.7, 3,877.4, 3,815.6 | 3,551.3, 3,553.9, 3,608.6, 3,572.2 | **-7.3%** (3,853.7 -> 3,571.5) |
| Qwen3.8 27B, rep 2 | 4 | 4,036.4, 4,066.2, 4,099.3, 4,121.9 | 3,623.8, 3,631.1, 3,803.4, 3,781.1 | -9.1% (4,081.0 -> 3,709.8) |
| Clover 35B-A3B | 2 | 656.7, 657.6 | 622.7, 623.3 | -5.2% |

Each 35B-A3B and Clover cell is the mean of a run's reps 1 and 2. The 27B's reps are given apart: its prefill hands
16 of its MLP's 34 units to the Neural Engine and slows rep over rep (power), as in #941. The old and new ranges are
disjoint on every model.

## The GEMMs alone

`bench_qmm` ([bench_qmm/](bench_qmm)), M 2,048, interleaved, median TFLOP/s. `QMM_CMP=1` compares every output element
of each kernel with the first kernel's: every 128-row kernel below is bitwise identical to its 64-row tile, on every
shape.

| Shape (N x K) | 64-row tile | 128-row tile |
| --- | ---: | ---: |
| 27B gate/up 17408 x 5120 (`qmtg`) | 49.8 | 59.5 |
| 27B down 5120 x 17408 | 48.1 | 53.5 |
| 27B out_proj 5120 x 6144 | 47.8 | 53.4 |
| 27B GDN in_proj 16384 x 5120 | 47.1 | 55.1 |
| 27B split down, `_ld` view, 5120 x 17408 | 47.6 | 53.5 |
| 35B 8192 x 2048 | 45.3 | 52.5 |
| 35B 2048 x 4096 | 43.8 | 50.7 |
| 35B shared expert gate/up 512 x 2048 (`qmtg`) | 37.6 | 48.8 |

At chunk tails (M 100, 1,000 and 2,047) the 128-row tile is faster too, and the same bits. Below 96 rows the 64-row
tile stays: at M 33 (N 512, K 2,048) the 128-row tile takes 48 us against 39.

The probes that lost:
- **Wider tiles:** 8 simdgroups (256 rows) or 6 (192), and more than two accumulators a simdgroup (the register file
  spills).
- **A 64-K stage:** two 32-K matmuls a stage, half the barriers. +2% on the long-K shapes, -11% on gate/up
  ([probe-k64-stage.out](bench_qmm/probe-k64-stage.out)).

## Output check

`logit_dump` of the old build, then the new on the reference's prefixes, compared by `tools/logit_compare.py`
([output-check/](output-check)):

| Model | Paths | Result |
| --- | --- | --- |
| Qwen3.6 35B-A3B | prefill, step, verify, batch | every row bitwise identical; PASS (`qwen3.6-35b-a3b-4bit.m5.json`) |
| Qwen3.8 27B | prefill, step, verify, batch | every row bitwise identical; PASS (`qwen3.8-27b-4bit-mtp.m5.json`) |
| Clover 35B-A3B | prefill, step, verify, batch | every row bitwise identical; PASS (`clover-35b-a3b-4bit.m5.json`) |

The 27B's dumps ran its Neural Engine split (10 chunks, 1,152 evaluations each arm), so the `_ld` and split gate/up
kernels are covered.

## Validation layer

The 128-row kernels use 8,192 B of static threadgroup memory, 16,384 B as the layer counts it, so the layer
dispatches them. `bench_qmm` under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` ran M 33, 96, 100, 128 and 257 (N
64-192, K 64-256) on `qmt_m128_n64` and `qmtg_m128_n64`. Every run was clean of layer errors and bitwise identical to
the 64-row tile ([validation-qmt.out](bench_qmm/validation-qmt.out), [validation-qmtg.out](bench_qmm/validation-qmtg.out)).
M 1 differs, as the kernel requires M >= 32. The engine dispatches it only at 96 rows or more.

## Not measured

- **Other Macs:** the switch is on wherever the dense tensor path is (`qmt_on()`: the M5 family on macOS 26). Only the
  M5 Max was measured.
- **The 27B's Neural Engine share:** the default 16 of 34 units was set for the 64-row tile, and was not re-tuned
  for the faster GPU share.
- **The server's time to first token** (`bench/harness`), and the server's 4,096-row chunks on the 35B-A3B.
