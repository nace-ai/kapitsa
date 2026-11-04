# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (Flash-Next's verify MoE reading its inputs through the caches)

The verify MoE of Qwen3.8-Flash-Next (2-4 rows: its MTP cycle's 3) runs each distinct expert's gate/up for the rows
that chose it (`moe4_gate_up`) and every row's down products (`moe4_down`). Both staged their row's input in
threadgroup memory first: x (2,560 floats, in a 16 KB array) and h (11 slots x 640 floats, 28 KB, which holds the down
kernel to one threadgroup a core). `moe4_gate_up_d` and `moe4_down_d` (`src/kernels/zqwen4v.metal`) read it from
device memory through the caches instead (`qmv2_rows_dx` in `src/kernels/quant.h`: `qmv2_rows` with a device pointer),
with the same arithmetic, so the outputs are the same to the bit. `NI_MOE4_DX`, on by default on the M5 Max (where it
was measured; `1` on the other Macs, `0` the staged kernels). Qwen3.8-Flash-Next 4-bit (Nace AI's MLX pack) at its
default, its MTP head at depth 2, MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4
(25E246), AC power, High Power mode, `caffeinate -is`, 2026-11-03 ([machine.json](machine.json)).

## Decode

`ni_run` ([ab/](ab)), one build, `NI_MOE4_DX=0` against the default, rounds alternating the arm order (old new, new
old, ...), the README's decode prompt ([ab/prompt-ids.txt](ab/prompt-ids.txt), 56 tokens), greedy, 512 tokens
(`--ignore-eos`), `--spec 2 --dflash none` (the server's default for this model). Every run waited for a quiet GPU (no
foreign process above 5% for 15 s: `bench/harness/gpu_hogs.py`) and sampled the foreign GPU load while it ran. Every
run wrote the same 511 tokens.

**Logged** ([ab/logged/](ab/logged), 2026-11-04, this branch at `83ad33a`, after main's merge brought #921's
`NI_HC_WIDE` in). Before each run the quiet wait (up to 900 s), then the 1-minute load average below 2; the load average
and the power source (`pmset -g batt`) read just before and just after the run. Three batches, 14 rounds.

`run2/` (4 rounds) and `run3/` (6) found the GPU quiet before every run. [pick.py](ab/logged/pick.py)
([pick.out](ab/logged/pick.out)) keeps the 5 rounds in which both runs were quiet, ended below load 4 and drew from AC
power before and after (in the other 5 a run ended at load 4.3-25.2). Their foreign GPU load was 0.0-0.13% mean (1.9%
at most):

| Arm | tok/s | ms of GPU a cycle | Tokens a cycle |
| --- | ---: | ---: | ---: |
| `NI_MOE4_DX=0` | 108.2 (107.3-108.3) | 21.37 (21.35-21.53) | 2.34 |
| default | 109.3 (108.3-109.5) | 21.16 (21.11-21.36) | 2.34 |

Medians: GPU time a cycle 21.37 -> 21.16 ms (-1.0%), decode 108.2 -> 109.3 tok/s (+1.0%), on top of #921; the default
is the faster arm in each of the 5 rounds (by 0.16-0.26 ms), and in each of the other 9. Only one round saw 0.0% foreign
load in both runs (`run2/` round 2): 21.35 -> 21.19 ms, 108.3 -> 109.1 tok/s.

The first batch (`dx_fn.out`, the runs' outputs and `ab.json` at the folder's top, 4 rounds) passed the load and power
checks (every run ended at load 1.43-2.23, on AC), but no wait found the GPU quiet: the Claude desktop app's helper
process held 2.9-3.8% mean foreign GPU load (6.5% at most) through both arms alike. It gave 21.42 -> 21.20 ms, 108.0 ->
109.2 tok/s, and is not cited. (`ab_lc.py` is the script as the later batches ran it: it since names the busiest
processes after a run that ends at load 4 or more and takes an arm's working folder, neither of which the first batch
needed. `run_lc.sh` ran the first batch after #922's A/B; its Flint step hung, in a folder without `bench/corpus.txt`,
and was stopped.)

The earlier rounds ran this branch at `c9bea88`, before main's merge.

**Idle** ([ab/idle/](ab/idle): 5 rounds in two batches, this session idle, each quiet wait up to 900 s). The round in
which both arms found the GPU quiet and saw 0.0% foreign load while they ran (batch 1's second):

| Arm | tok/s | ms of GPU a cycle | Tokens a cycle |
| --- | ---: | ---: | ---: |
| `NI_MOE4_DX=0` | 105.7 | 21.88 | 2.34 |
| default | 106.8 | 21.65 | 2.34 |

GPU time a cycle 21.88 -> 21.65 ms (-1.1%), decode 105.7 -> 106.8 tok/s (+1.0%). Batch 1's other two rounds were
quiet too, at 0.06-0.17% foreign load (`NI_MOE4_DX=0` 21.92, 21.88 ms; the default 21.61, 21.64): the median of its
three quiet rounds is the same, 21.88 -> 21.64 ms. Batch 2's two rounds, whose waits ran out at 900 s with the Claude
desktop app's helper process (which this session runs in) at 2.6-3.4% mean, agree: 21.87-21.90 against 21.65-21.69.

**The first A/B** ([ab/run1/](ab/run1), 3 rounds, each quiet wait up to 60 s): the helper held 4.9-5.5% mean (9.6% at
most) through both arms alike, so every wait ran out. The idle rounds replace it:

| Arm | tok/s | ms of GPU a cycle | Tokens a cycle |
| --- | ---: | ---: | ---: |
| `NI_MOE4_DX=0` | 105.1, 104.3, 104.7 | 22.04, 22.20, 22.11 | 2.34 |
| default | 106.1, 106.2, 105.9 | 21.84, 21.81, 21.86 | 2.34 |

GPU time a cycle 22.12 -> 21.83 ms (-1.3%), decode 104.7 -> 106.1 tok/s (+1.3%).

`NI_PROFILE=1` over 44 passes ([kernel/](kernel), isolated dispatches): `moe4_gate_up` 101.0 -> 98.6 us a call,
`moe4_down` 64.2 -> 60.4 us. (`profile-old.out` is `NI_MOE4_DX=0` on an earlier build of this branch, before
`moe4_gate_up_d` existed: the same staged kernels.)

## Checks

- `FN_ROWS=1 fn_check kernels` ([kernel/](kernel)): `moe4_gate_up_d` against `moe4_gate_up` and `moe4_down_d` against
  `moe4_down` (`moe4_down_s` under the layer), 1-4 rows, every output bitwise equal; the verify MoE against the decode
  MoE row by row, bitwise. The same under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` (the new kernels hold no
  threadgroup memory): "all kernels match".
- Output check ([output-check/](output-check)): `logit_dump --paths step,verify` with `NI_MOE4_DX=0` and the default:
  the prefill rows, 192 step rows and 192 verify rows (the MTP cycle's chain of 3) **every row bitwise identical**. No
  floor file exists for this 4-bit pack; bitwise identity needs none.
- The output check again on main's merge (`83ad33a`'s code, #921's `NI_HC_WIDE` in;
  [output-check/merged/](output-check/merged)): the prefill rows, 192 step rows and 192 verify rows, **every row
  bitwise identical**.

The first A/B's build read the switch as "`0` off, else on"; the idle rounds' build (the one the first two checks used)
reads it as committed, the per-Mac default: the code of `c9bea88` but for its comments.

## Not run

Flint, whose verify MoE keeps the staged kernels, the other Macs, and the server (`kapitsa serve`, T 0.6).

## Files

`ab/`: the A/B script, command, output and every run's `ni_run` output (`run1/`, `ab_fn.json` with each run's ids and
GPU load); `ab/logged/` the logged rounds (`ab_lc.py`, `dx_fn.json`, `run_lc.sh`, `run_dx2.sh`, `run_dx3.sh`,
`builds.txt`, `pick.py`; the first batch at its top, `run2/`, `run3/`); `ab/idle/`: the idle rounds the same way (the script with its 900 s wait, `run.sh`, `batch1/`, `batch2/`). `kernel/`: the `fn_check` and profile commands and outputs. `output-check/`: the dumps' commands and logs,
their `meta.json` and the comparison. Local paths are replaced by placeholders (`~`, `<repo>`: this branch's
checkout, `<scratch>`).
