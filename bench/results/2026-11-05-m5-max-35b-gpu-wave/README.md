# Results, 2026-11-05: the 35B-A3B's prefill as a two-half GPU wavefront, M5 Max

The 35B-A3B's prefill runs each layer's token mixer and then its MoE, one chunk at a time. Two of the mixer's kernels
leave most of the GPU idle:
- the DeltaNet scan, `gdn_step_parr4_l16`, about 12% of a 4K prefill: a few hundred threadgroups, each a long chain;
- the prefill attention, `qattn_t_g8d2s8`, about 6%.

`NI_PF_WAVE` runs a chunk as two halves on the GPU alone. In each layer:
1. half B's mixer runs beside half A's MoE;
2. then half A's next mixer runs beside half B's MoE.

The overlap comes from Encoder tapes, which are new in `src/metal/gpu.mm`. While a tape is set, the encoder records
what it would encode: pipelines, bindings, grids, and where each barrier falls. `play_pair` then encodes two tapes
together on the grouped encoder, keeping each chain's order. While its MoE layer is recorded, the MoE uses a second
`xh` / `hh` pair, because the mixer uses the first.

Each half's rows go through the kernels that a chunk of that size runs, and the second half follows the first as a
next chunk would. So the logits are those of half-size chunks.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, under the GPU
lock, the session idle during the runs.

## What to pair

`ni_run qwen3.6-35b-a3b-4bit @<4,129 ids> 8 0 --pf-reps 3 --dflash none`, the 4,129 ids in one chunk, one run each,
mean of reps 1-2 (scratch runs, not interleaved):

| Pairing | ms |
| --- | ---: |
| No wavefront (one 4,129-row chunk) | 591-600 |
| The two halves' tapes one after the other, no overlap | 624-630 |
| Every step beside the other chain's step | 737 |
| A DeltaNet scan or attention beside the other chain's largest GEMM step, the GEMM dispatched first | 665 |
| The same, the scan or attention dispatched first (the default) | **589** |
| Besides that, every other non-GEMM step beside a GEMM | 595 |

- **Two GEMMs side by side** contend for the matrix units and the caches, and lose.
- **The scan after the GEMM** fills the cores with the GEMM's threadgroups first, so the scan starts late.
- **The scan first** wins: its threadgroups all start at once, and the GEMM's fill the cores around them.

The overlap recovers about 35 ms. Splitting a 4K chunk into two 2,048-row halves costs about as much, because the
MoE reads every expert's weights once per half. So the wavefront pays only when each half has the rows of the chunk it
replaces. With 16,288 ids (`ni_run`, one run each), the 35B-A3B's server chunk of 4,096 rows reads 2,777 ms:
- 8,192-row chunks without the wavefront, 2,767;
- 4,096-row chunks with it, 2,767;
- 8,192-row chunks with it (halves of 4,096), **2,620**.

## The default

`NI_PF_WAVE` is on for the 35B-A3B shape (hidden 2048, 40 layers, 4-bit experts) on an Apple10 GPU of 40 cores or
more. There, the single-slot 35B-A3B takes 8,192-row chunks (8,448 with the MoE tail room) when its memory plan
fits (`auto_chunk`).

## Old against new

[run_ab.sh](run_ab.sh), [run_ab.log](run_ab.log), runs in [ab/](ab).

- **Old:** main's tree (`87d7dd5`, built as `7afb51d`).
- **New:** this branch.
- **Chunks:** each arm in the chunks its server picks. Old: 4,129 ids in one 4,224-row chunk, 16,288 in 4,096-row
  chunks. New: 8,192-row chunks.
- **The runs:** `--pf-reps 2`, rounds alternating old new / new old, 45 s idle before each run.

| Prompt | Old (ms, per round) | New | Change |
| --- | --- | --- | ---: |
| 16,288 ids ([prompt-ids-16288.txt](prompt-ids-16288.txt)), rep 1 | 2,777.0, 2,773.1, 2,777.5 | 2,622.1, 2,616.2, 2,615.8 | **-5.7%** (2,775.9 -> 2,618.0) |
| 16,288 ids, rep 0 (cold) | 2,810.5, 2,806.1, 2,812.1 | 2,670.4, 2,666.6, 2,665.4 | **-5.1%** |
| 4,129 ids ([prompt-ids-4129.txt](prompt-ids-4129.txt)), rep 1 | 592.3, 593.9, 592.3 | 586.6, 586.8, 587.6 | -1.0% (592.9 -> 587.0) |
| 4,129 ids, rep 0 (cold) | 1,489.2, 625.7, 625.2 | 1,526.1, 638.2, 637.3 | +2.1% (rounds 2-3: 625.4 -> 637.8) |

The 16,288 ids are the 4,096 of `bench/quality/ids_qwen38` four times over, cut to 16,288, the harness's 16K
length. Every run decoded the same first eight ids in both arms.

**The cold rep:** the first chunk after a load reads 12 ms slower, likely the lazy allocation of the MoE's second
`xh` / `hh` pair, about 35 MB each. The first round's rep 0 in both arms includes the process's first compile.

## Output check

`logit_dump` of main at its default 2048-row chunks, then this branch at `--chunk 8192` on the reference's prefixes,
`tools/logit_compare.py` with the 35B-A3B's M5 floor ([output-check/](output-check), [output_check.sh](output_check.sh)).
The result: every row bitwise identical on the prefill, step, verify and batch paths. PASS.

The dump's chunks split into the same 2,048-row pieces either way: the wavefront's halves of the candidate against
the reference's chunks. A chunk run whole would not match them bitwise, because of the attention and DeltaNet chunk
boundaries. So the identity shows the halves compute exactly what half-size chunks compute.

`ni_run` with the DFlash2 drafter armed (its prefill capture records each half's rows after their MoE layer)
decoded the same 64 tokens with and without the wavefront, at the same speed.

## The cell

Through the server, the harness's cell (`compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip
tools,agent,cache,sustain`):
- main's build (`7afb51d`) against this branch;
- interleaved old new / new old, two rounds, 60 s idle before each run, the session idle;
- [cell-ab/](cell-ab) ([run.sh](cell-ab/run.sh), [session.log](cell-ab/session.log)).

| | Old, round 1 / 2 | New, round 1 / 2 |
| --- | --- | --- |
| Prefill 4K (tok/s) | 6,516 / 6,518 | 6,557 / 6,553 (+0.6%) |
| Prefill 16K (tok/s) | 5,712 / 5,716 | **6,061 / 6,060 (+6.1%)** |
| Decode T 0.6 | 343.1 / 338.1 | 339.3 / 354.6 |
| Greedy | 375.5 / 372.2 | 375.1 / 370.7 |
| Decode footprint (GiB) | 21.56 / 21.55 | 22.75 / 22.75 |

- **The server's chunks:** the new server takes 8,448-row chunks (its log: `prefill: 8448-row chunks`).
- **Memory:** that costs 1.19 GiB of activation buffers from the first phase on; the peak to 16K is 23.5 GiB.
- **Decode** does not change: it runs no prefill kernel.
- **Contention:** the desktop app's GPU load was 0.5% or less in every phase.

**Two earlier sets of the new build alone** ([cell/](cell)) ran back to back with no idle gap between runs:
- `runs-set1`, the display allowed to sleep;
- `runs`, with `caffeinate -d`.

Both read 16K at 6,003-6,062, but their later runs' decode and 4K prefill fell (decode 253-330, 4K 4,256-5,623):
the chip after back-to-back runs. The interleaved set above, with gaps, is the measurement.

## Checked

- **CPU tests:** `api_test` (382 passed), `execution_session_test`, `prefix_cache_test` (67 passed).
- **Lint:** `prek run --all-files`.
- **No kernel changed:** the tapes replay main's kernels with their recorded bindings.

## Not measured

- **Other models and Macs:** the 27B is not affected (it has no MoE; its Neural Engine split has its own wavefront).
  Clover, Flash-Next and the 35B-A3B on other Macs keep their prefill (`NI_PF_WAVE=1` turns it on to measure them).
- **Memory beyond the cell's footprint.**
