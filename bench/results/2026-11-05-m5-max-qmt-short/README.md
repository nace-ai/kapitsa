# Results, 2026-11-05: the 27B's GEMMs of 17-63 rows on the tensor tile, M5 Max

A prefill pass of fewer than 64 rows ran the Qwen3.8 27B's dense projections on the short-prefill simdgroup kernels:
- the `qmr` tiles for its large groups;
- `qmms` for the small ones.

Such passes are:
- a prompt's last chunk (the harness's 4K prompt is 4,129 tokens: 2,048 + 2,048 + 33 rows);
- a short turn's new message.

On the M5 Max the 64-row tensor tile (`qmt_m64_n64`, `qmtg_m64_n64`) runs those rows faster. The tile loads the rows
past M as zeros and stores nothing for them. `NI_QMT_MIN` now sets the fewest rows that take it. It is 17 for the 27B
on an Apple10 GPU of 40 cores or more, and 64 everywhere else, as before.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, under the GPU
lock, the session idle during the runs.

## The GEMMs alone

`bench_qmm`, the 27B's large projection groups, which keep the `qmr` tiles in a short pass. Each is timed on the
64-row tensor tile and on the short-prefill tile a pass of M rows takes (`short_tile`):
- one tile of 8 x ceil(M / 8) rows up to 48 rows (gate/up: up to 32);
- two tiles above that.
- [gate-up-17-63.out](bench_qmm/gate-up-17-63.out);
- [gate-up-9-24.out](bench_qmm/gate-up-9-24.out);
- [plain-9-48.out](bench_qmm/plain-9-48.out).

| Shape (N x K) | M | Short-prefill tile (us) | 64-row tensor tile (us) |
| --- | ---: | ---: | ---: |
| gate/up, 17408 x 5120 | 16 | 509 (`m16`) | 611 |
| gate/up | 17 | 638 (`m24`) | 514 |
| gate/up | 33 | 1,215 (`m24` x 2) | 527 |
| down, 5120 x 17408 | 16 | 295 (`m16`) | 330 |
| down | 24 | 400 (`m24`) | 334 |
| down | 33 | 623 (`m40`) | 337 |
| GDN qkv + z, 16384 x 5120 (one `qmrs` dispatch in the pass) | 33 | 525 (`m40`) | 288 |

From 17 rows the tensor tile is faster. At 16 rows and fewer the short tiles stay faster, and the floor stays there.
Decode and the 16-row KFlash verify are unaffected either way: they run `qmm8`, not these kernels.

**The validation layer:** `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, `qmt_m64_n64`, `qmtg_m64_n64` and
`qmt_m64_n64_ld` at M 17, 33, 48 and 63 ran clean, with errors at the reference's level
([validation.out](bench_qmm/validation.out)). The tile's 16,384 B of static threadgroup memory reads as 32,768 B under
the layer, at the cap. The kernel is main's, unchanged.

## Prefill, old against new

[run_ab.sh](run_ab.sh), [run_ab.log](run_ab.log), runs in [ab/](ab).

- **Old:** main `d9d479d`.
- **New:** this branch's first form, which set the floor to 17 for every model on the 40-core M5. The final form
  keeps 17 for the 27B alone, so the 27B runs the same kernels in both.
- **The runs:** `ni_run qwen3.8-27b-4bit-mtp @<ids> 4 0 --pf-reps R --dflash none`, rounds alternating old new / new
  old, 20 s idle before a short-prompt run and 45 s before a 4K run.

| Prompt | Old (ms, per round) | New | Change |
| --- | --- | --- | ---: |
| 33 ids ([prompt-ids-33.txt](prompt-ids-33.txt)), mean of reps 1-3 | 171.3, 171.0, 170.9 | 87.7, 87.7, 87.4 | **-48.8%** (171.1 -> 87.6) |
| 48 ids ([prompt-ids-48.txt](prompt-ids-48.txt)), mean of reps 1-3 | 189.5, 189.6, 189.4 | 89.3, 89.4, 89.3 | **-52.9%** (189.5 -> 89.3) |
| 4,129 ids ([prompt-ids-4129.txt](prompt-ids-4129.txt)), rep 0 | 3,266.0, 3,265.4, 3,264.4, 3,262.9 | 3,188.6, 3,179.7, 3,179.8, 3,180.2 | **-2.5%** (3,264.7 -> 3,182.1) |
| 4,129 ids, rep 1 | 3,276.4, 3,266.6, 3,268.8, 3,266.5 | 3,190.4, 3,181.1, 3,182.5, 3,189.2 | **-2.6%** (3,269.6 -> 3,185.8) |

The 4,129 ids are the 4,096 of `bench/quality/ids_qwen38` followed by 33 of them again (ids 100-132). That is the
harness's 4K prompt length (`prompt_tokens` 4,129 in the 2026-11-05 cells). They run as two 2,048-row Neural Engine
split chunks and a 33-row tail, which the GPU runs alone. The tail went from about 170 to 87 ms. Every range is
disjoint.

**The final form:** one more round on the final build ([final/](final), [output_check_48.sh](output_check_48.sh)):
- 33 ids 87.4 ms;
- 4,129 ids 3,183.2 / 3,182.3 ms.

## Output check

`logit_dump` at its default chunks never runs a pass of 17-63 rows. Main against the first form there
([output-check-default-chunks/](output-check-default-chunks)): every row bitwise identical on all four paths, for the
27B and the 35B-A3B. That says nothing about this change.

So the check that covers it ran `logit_dump --chunk 48`: every prefill pass is 48 rows, the short tiles on main, the
tensor tile here. It compared main `d9d479d` against the final form, with the 27B's M5 floor
([output-check-48/](output-check-48), `tools/logit_compare.py`):

| Path | Records | Top-1 | KL mean | KL p99 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| prefill | 8 | 1.0000, 0 flips | 1.48e-7 | 6.28e-7 | no row bitwise (the K order differs) |
| step | 192 | 1.0000, 0 flips | 2.95e-7 | 1.32e-6 | PASS |
| verify | 192 | 1.0000, 0 flips | 4.37e-7 | 4.91e-6 | PASS |
| batch | 192 | 1.0000, 0 flips | 5.88e-7 | 6.66e-6 | PASS |

Both kernels form the same half x half products in fp32. Only the K order of the sums differs, and the KLs are three
orders of magnitude under the floor's gate.

**Rows that read the same bits:** `NI_SHORTPF_EXACT` keeps a row's bits the same whatever the length of a short pass
it is read in. On the M5 that already held only below 64 rows for a dense model (`NI_SP_QMT`). For the 27B on the M5
Max it now holds within 1-16 rows and within 17 rows and up, where the tensor tiles give a row the same bits at any
M. A tail read as 16 + 24 rows can part from one read of 40 at a near-tie, as one of 70 could from 35 + 35 before.

The 35B-A3B and every other model keep the floor of 64. Their kernels did not change.

## Not measured

- **Other Macs and models:** only the 27B on the M5 Max. The M5 Pro, and the 4B and 9B on any M5, keep 64 until
  measured; `NI_QMT_MIN=<rows>` sets the floor for any model.
- **The server's time to first token** (`bench/harness`).
