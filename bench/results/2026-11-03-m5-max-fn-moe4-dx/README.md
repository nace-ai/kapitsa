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

`ni_run` ([ab/](ab)), one build, `NI_MOE4_DX=0` against the default, interleaved over 3 rounds (old new, new old, old
new), the README's decode prompt ([ab/prompt-ids.txt](ab/prompt-ids.txt), 56 tokens), greedy, 512 tokens
(`--ignore-eos`), `--spec 2 --dflash none` (the server's default for this model). Every run waited for a quiet GPU
(no foreign process above 5% for 15 s, at most 60 s: `bench/harness/gpu_hogs.py`) and sampled the foreign GPU load
while it ran.

| Arm | tok/s | ms of GPU a cycle | Tokens a cycle |
| --- | ---: | ---: | ---: |
| `NI_MOE4_DX=0` | 105.1, 104.3, 104.7 | 22.04, 22.20, 22.11 | 2.34 |
| default | 106.1, 106.2, 105.9 | 21.84, 21.81, 21.86 | 2.34 |

GPU time a cycle 22.12 -> 21.83 ms (-1.3%), decode 104.7 -> 106.1 tok/s (+1.3%). Every run wrote the same 511 tokens.
**GPU load:** nothing else ran, but the Claude desktop app's helper process, which this session runs in, held
4.9-5.5% mean (9.6% at most) through both arms alike, so every quiet wait ran out at 60 s.

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

The A/B's build read the switch as "`0` off, else on", the final code as above; on this Mac both run the same kernels.

## Not run

Flint, whose verify MoE keeps the staged kernels, the other Macs, and the server (`kapitsa serve`, T 0.6).

## Files

`ab/`: the A/B script, command, output and every run's `ni_run` output (`run1/`, `ab_fn.json` with each run's ids and
GPU load). `kernel/`: the `fn_check` and profile commands and outputs. `output-check/`: the dumps' commands and logs,
their `meta.json` and the comparison. Local paths are replaced by placeholders (`~`, `<repo>`: this branch's
checkout, `<scratch>`).
