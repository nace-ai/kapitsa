# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (Qwen decode probes that did not pay)

Probes of the Qwen decode cycles after #921, #922 and #923, recorded so that a retry starts from these numbers.
None is proposed: each is slower than what ships, or would change outputs. Each kernel probe is a patch on main
`8715427` behind a switch that is off by default (the base arm, the same binary), and its runs alternate base and
variant ([run.sh](run.sh)). Qwen3.8-Flash-Next 4-bit (Nace AI's MLX pack), Qwen3.6 35B-A3B 4-bit and Qwen3.8 27B 4-bit,
the README's decode prompt ([prompt-ids.txt](prompt-ids.txt), 56 tokens), greedy. MacBook Pro (16-inch, 2026),
`Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode, `caffeinate -is`, under the GPU
lock ([machine.json](machine.json)).

`NI_PROFILE=1` times every dispatch in a command buffer of its own (isolated dispatches): its sums rank kernels and
compare variants of one kernel; they are not decode speed. These runs, all but the depth pair's
([depth/logged/](depth/logged)), did not wait for a quiet GPU or log the foreign GPU load; the session sat idle while
they ran.

## Flash-Next's verify projections (`qmvs_multi`)

The MTP cycle's verify (3 rows) runs its 4- and 8-bit projections on `qmvs_multi`: a threadgroup per token row, 2
simdgroups of 4 weight rows each, the rows' threadgroups of a weight block adjacent so they share it through the
cache, each row's arithmetic the decode kernel's (bitwise). It is 22% of the cycle's dispatch time. Per cycle, 44
cycles of 96 tokens, two runs each:

| Variant (patch, switch) | ms a cycle | Outputs |
| --- | ---: | --- |
| `qmvs_multi`, as shipped (base) | 5.17-5.33 | bitwise the decode kernel's |
| the pair kernel `qmvp_multi` at 3 rows ([qmvp3.diff](qmvp3.diff), `NI_QMVP3=1`) | 5.94-5.99 | ~1e-7 off (fast math), as at 4 rows |
| every row in one simdgroup, each weight word unpacked once for the 3 rows ([qmvs-rows.diff](qmvs-rows.diff), `NI_QMVS_ROWS=1`) | 6.62-6.72 | up to 7.2e-7 off (outputs up to ~6); at 4 rows the 8-bit too |
| 2 weight rows a simdgroup ([qmvs-shape.diff](qmvs-shape.diff), `NI_QMVS_VAR=w2`) | 5.41 | bitwise |
| 8 weight rows a simdgroup (`w8`) | 5.93 | bitwise |
| 4 simdgroups a threadgroup (`t4`) | 6.19-6.30 | bitwise |
| 8 simdgroups a threadgroup (`t8`) | 7.68-7.69 | bitwise |

(The variants' times include the few `qmvs_multi` calls they leave, at other row counts.) The shipped shape is the
fastest. `qmvs_rows` was to be bitwise: each row's sum is `qmvs_row`'s expression in its order. It is not: the
compiler's fast math orders the 4-bit rows' sums differently once the unpacked words are shared, and
`#pragma clang fp reassociate(off)` on that loop made it differ more, on the 8-bit rows too (a run not kept
here; [qmvs-rows/fn-check.out](qmvs-rows/fn-check.out): `FN_ROWS=1 FN_VERBOSE=1 fn_check kernels`, the variant without the
pragma). Every Flash-Next run in this section wrote the same 95 tokens.

## Flash-Next's MTP depth

`ni_run`, 512 tokens (`--ignore-eos`), `--spec 2` (the default) against `--spec 3` (4 verify rows), the unpatched
build, 3 rounds alternating the arm order ([depth/logged/](depth/logged): `ab_lc.py`, `run_depth.sh`). Before each run
a quiet GPU (no foreign process above 5% for 15 s, waiting up to 900 s: `bench/harness/gpu_hogs.py`), then the 1-minute
load average below 2; the load average and the power source read just before and just after it, the foreign GPU load
sampled while it ran. A round counts when both its runs found the GPU quiet, saw 0.0% foreign GPU load, ended below
load 4, and drew from AC power before and after: rounds 0 and 1.

| Depth | tok/s | Tokens a cycle | ms of GPU a cycle |
| ---: | ---: | ---: | ---: |
| 2 | 108.1, 108.0 | 2.34 | 21.39, 21.40 |
| 3 | 102.9, 102.9 | 2.72 | 26.11, 26.09 |

Decode 108.05 -> 102.9 tok/s (-4.8%), 21.40 -> 26.10 ms of GPU a cycle, the medians: the fourth row's verify and the
third draft cost more than the 0.38 tokens a cycle they add. Every run wrote the same 511 tokens. Round 2's runs (107.9
and 102.9 tok/s, 0.06-0.11% foreign) both ended at load 14.2, with no process above 2.1% CPU then: they read their
weights in 17-21 s, the clean runs in 10 s, so the load was their own page-ins. The first pair ([depth/](depth), `d*.out`,
two runs each, alternated, without these checks) agrees: 107.2-107.3 against 101.8-102.0 tok/s.

## A repacked weight layout for the 8-row verify GEMM (`qmm8t`)

The 35B's and the 27B's 8-row verifies run their dense projections on `qmm8t*` (`src/kernels/tensor/verify8t.metal`),
which read each 64-group's codes from row-major rows and its scale and bias as two 2-byte loads from arrays of their
own. [bench_layout.cpp](bench_layout.cpp) times loads alone (no arithmetic) in `qmm8t`'s pattern (a simdgroup per 16
weight rows, one group a step, k split over `ks` simdgroups) for the 35B's GDN input projection (12352 x 2048, 4-bit:
14.2 MB), over 105 copies so that no read hits the cache, in three layouts: `split` as today; `sbpair` the scale and
bias as one 4-byte word; `tiled` each simdgroup's 16 rows of a group as one contiguous 576 B block (512 B of codes, 64
B of scale/bias pairs). Median of 7 ([layout/bench_layout.out](layout/bench_layout.out)):

| ks | split | sbpair | tiled |
| ---: | ---: | ---: | ---: |
| 1 | 391 GB/s | 272 GB/s | 478 GB/s |
| 2 | 460 GB/s | 464 GB/s | 473 GB/s |
| 4 | 481 GB/s | 449 GB/s | 518 GB/s |

Split over k as the kernels run, the tiled layout reads 4-8% faster (an earlier run of the same bench: 487 against 507
GB/s at `ks` 4), the scale/bias word nothing. That is the most a repack could give these GEMMs, and it would need
every kernel that reads the dense weights (decode, prefill, verify) rewritten for the new layout; not pursued.

## Where the 35B's and the 27B's decode cycles go

`NI_PROFILE=1`, 128 tokens, each model's default drafter (DFlash2 with KFlash, 8-row verifies), the unpatched build
([profiles/](profiles)), isolated dispatches a cycle:

- Qwen3.6 35B-A3B, 12.25 ms: the MoE's expert passes `moe8e_gu_r2` 26.6% and `moe8e_down_r2` 13.3%; the dense
  projections `qmm8t_b4_nt16` 17.2%, `qmm8tp_b4_nt16` 7.6%, `qmm8tp_b4_nt32` 6.1%; `moe8_plan` (one threadgroup, a
  thread per expert, the router top-k fused) 3.5%.
- Qwen3.8 27B, 39.97 ms: `qmm8tp_b4_nt16` 77.1%; `x8_prep2` (the GEMMs' input tables) 2.7% over 145 calls, 129 of
  them a row norm's (each layer's input and post-attention norms, the final norm), which a GEMM's threadgroups cannot
  compute, the other 16 the attention outputs' (0.3% of the cycle); the SwiGLU and the GDN outputs' tables are already
  written by the kernels before them (`swiglu_x8`, `gdn_gnorm_x8t`).

## Files

`run.sh`: every run of this folder, in order. `*.diff`: the probes' patches on `8715427`. `qmvp3/`, `qmvs-rows/`,
`qmvs-shape/`: each arm's `NI_PROFILE` output (`base-*`, `var-*`; `qmvs-shape/` named by variant) and `qmvs-rows/`'s
`fn_check`. `depth/`, `layout/`, `profiles/`: the other runs' outputs; `depth/logged/` the depth A/B with every run's conditions
(`ab.json`, `depth.out`). Local paths are replaced by placeholders (`~`,
`<repo>`: the checkout the patches were applied to, `<out>`: the outputs' folder).
