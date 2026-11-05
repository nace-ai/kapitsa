# Results, 2026-11-05: the 128-row tile's outputs through the cooperative tensors' store, M5 Max

`qmt_m128_n64` and `qmt_m128_n64_ld` (#959) wrote each accumulator element on its own: `Y[o] = v`, or `Y[o] + v` when the
GEMM adds into its output. A simdgroup whose 32 rows and 64 columns lie inside Y now writes its two 32 x 32 accumulators
through the cooperative tensors' own `store` into a device tensor view of Y. In add mode it first `load`s the
destination tile into a cooperative tensor of the same layout and adds there. fp32 addition commutes, so `Y[o] + v`
has the same bits either way. The last rows of a chunk and a partial column tile keep the element stores.

The same values, written faster: 4-6% on the 35B-A3B's K-2048 projections.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, under the GPU
lock, the session idle during the runs.

## The GEMMs alone

`bench_qmm`, `QMM_CMP=1`: every output element of the 128-row tile against the 64-row tile's, median TFLOP/s:
- [plain.out](bench_qmm/plain.out): plain;
- [add.out](bench_qmm/add.out): `QMM_ADD=1`, add mode;
- [ld.out](bench_qmm/ld.out): the `_ld` view.

Every shape is bitwise identical in every mode, the chunk tails (M 100, 1,000, 2,047) included.

Before and after the change, the probe (both kernels in one run, `QMM_CMP=1`):

| Shape (M x N x K) | Element stores | Through `store` |
| --- | ---: | ---: |
| 35B-A3B GDN in_proj, 4096 x 12352 x 2048 | 55.2 | 58.4 |
| 35B-A3B attention q / k / v, 4096 x 9216 x 2048 | 54.9 | 58.3 |
| 35B-A3B out_proj, 4096 x 2048 x 4096 | 53.2 | 55.4 |
| 27B gate/up-sized, 2048 x 17408 x 5120 | 57.8 | 58.9 |
| 27B down, 2048 x 5120 x 17408 | 53.8 | 54.7 |

The gate/up kernel (`qmtg_m128_n64`, half outputs) was probed the same way, with a half cooperative tensor filled from
the fp32 one and stored. It came out level (58.8 against 59.0 TFLOP/s) and keeps its element stores.

## 4K prefill, old against new

[run_ab.sh](run_ab.sh), [run_ab.log](run_ab.log).

- **Old:** main's code (`446165e`, the 2026-11-05 cells' build: main `7a03626` with #959 and #961).
- **New:** this branch, main `90575b6` plus the change. The two differ in nothing else on the M5 Max: `90575b6` adds
  #929, which only touches an M4 Pro.
- **The runs:** `ni_run MODEL @prompt-ids.txt 8 0 --pf-reps 3 --dflash none` (4,096 ids, 2,048-row chunks), 4 rounds,
  old new / new old, 45 s idle before each run.

| Model | Old (ms, mean of reps 1-2, per round) | New | Change |
| --- | --- | --- | ---: |
| Qwen3.6 35B-A3B | 626.5, 627.8, 629.0, 627.9 | 615.7, 617.9, 615.7, 616.5 | **-1.8%** (627.8 -> 616.4) |
| Qwen3.8 27B | 3,180.8, 3,241.4, 3,259.9, 3,329.1 | 3,195.5, 3,204.4, 3,274.6, 3,237.2 | level (3,252.8 -> 3,227.9, inside the spread) |

The 35B-A3B's ranges are disjoint. The 27B's long-K projections gain least and its prefill waits on the Neural Engine
split, so it reads level.

## Output check

`logit_dump` of the old build, then the new on the reference's prefixes, `tools/logit_compare.py` with each model's M5
floor ([output-check/](output-check)): every row bitwise identical on the prefill, step, verify and batch paths of the
35B-A3B and the 27B (its dumps ran the Neural Engine split). PASS.

## Not measured

- **Other Macs:** only the M5 Max was measured.
- **The server's time to first token.**
