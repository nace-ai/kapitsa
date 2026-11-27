# Results, 2026-11-04: the Qwen prefill on a grouped encoder (NI_PF_GROUPED), M5 Max

A Qwen prefill pass now runs on a grouped encoder (`Encoder::set_grouped`, scoped to the pass by
`Qwen35::GroupedPass`). Independent dispatches run side by side, while every other dispatch keeps a barrier before
it. The independent dispatches are:

- `enc_proj`'s projections: DeltaNet's qkv / z / b / a, attention's q / k / v, and the MoE router with the
  shared-expert gate.
- In the MoE (the fused path, shared expert not folded), the shared expert's gate / up beside the routed experts'
  gate / up, and its down projection beside theirs.

These are the same kernels on the same data, so the outputs are bitwise the serial encoder's.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power in High
Power mode, under the GPU lock, the session idle during the runs ([run_ab.sh](run_ab.sh), [run_ab.log](run_ab.log)).

- **Old:** main `25fdda6` (Kapitsa 0.2.27).
- **New:** the branch: main plus this change, built before its last edits, which touched comments and docs only.
- **The runs:** each arm's own Release build, `ni_run MODEL @prompt-ids.txt 8 0 --pf-reps 3 --dflash none`.
- **The prompt:** the 4,096 ids of `bench/quality/ids_qwen38` (chat, code_cpp, code_py and math, in that order:
  [prompt-ids.txt](prompt-ids.txt)).
- **What is timed:** the prefill reps 1 and 2 (rep 0 includes pipeline creation).
- **The order:** rounds alternate old new / new old, with 45 s idle before every run.

## 4K prefill, old against new

| Model | Rounds | Old (ms, per round) | New (ms, per round) | Change |
| --- | ---: | --- | --- | ---: |
| Qwen3.6 35B-A3B | 6 | 674.9, 671.4, 671.7, 671.7, 671.3, 672.5 | 659.8, 657.1, 658.2, 657.4, 658.4, 657.6 | **-2.1%** (671.7 -> 657.9) |
| Clover 35B-A3B | 2 | 671.0, 675.0 | 658.8, 657.1 | -2.2% |
| Qwen3.5 4B | 2 | 632.1, 631.9 | 618.8, 619.6 | -2.0% |
| Qwen3.8-Flash-Next 180B (4-bit, resident) | 2 | 1,941.9, 1,937.0 | 1,918.4, 1,918.9 | -1.1% |
| Qwen3.5 9B | 2 | 1,302.7, 1,304.7 | 1,296.9, 1,296.2 | -0.5% |
| Qwen3.8 27B | 3 | 3,827.5, 3,910.4, 3,864.4 | 3,816.1, 3,858.5, 3,856.5 | level (-0.5%, inside its spread) |

Each cell is the mean of a run's reps 1 and 2; the 27B's are its rep 1. The 27B's prefill hands half its MLP to the
Neural Engine, which is busy 79-100% of a prefill, and slows rep over rep (power): grouping its few independent GPU
dispatches does not show there. The old and new ranges are disjoint on every other model.

Where the 35B-A3B's gain comes from (`NI_PROFILE`, 4K), by share of its prefill:

- **Dense projections (`qmt_m64_n64`): 37%.** Their K is 2,048, so their tiles are short: 39-43 TFLOP/s against the
  27B's 48.
- **The routed experts' GEMMs: 28%.**
- **The shared expert:** gate / up `qmtg_m64_n64`, 512 threadgroups, 0.24 ms.
- **The router and shared gate:** 8-bit, 0.23 ms.

Run one after another, each of those small dispatches leaves most of the GPU idle at its tail.

## Output check

`logit_dump` of the old build, then the new on the reference's prefixes, compared by `tools/logit_compare.py` (the
floor from the dump's metadata; [output-check/](output-check)):

| Model | Paths | Result |
| --- | --- | --- |
| Qwen3.6 35B-A3B | prefill, step, verify, batch | every row bitwise identical; PASS (`qwen3.6-35b-a3b-4bit.m5.json`) |
| Clover 35B-A3B | prefill, step, verify, batch | every row bitwise identical; PASS (`clover-35b-a3b-4bit.m5.json`) |
| Qwen3.8 27B | prefill, step, verify, batch | every row bitwise identical; PASS (`qwen3.8-27b-4bit-mtp.m5.json`) |
| Qwen3.5 4B | prefill, step, verify, batch | every row bitwise identical (no M5 floor for it; reported only) |
| Qwen3.8-Flash-Next 180B | prefill, step, verify (it has no batch path) | every row bitwise identical (no floor for its 4-bit pack; reported only) |

`run_ab.sh`'s compare step passed a `--floor` flag that `logit_compare.py` does not take (the log shows the usage
error), and gave Flash-Next's `--paths step,verify` as one argument (`logit_dump` refused it). So the compares in
output-check/ were re-run by hand without the flag, and Flash-Next's two dumps were re-run with the arguments split.
Every run's first generated position also wrote its logits (`--dump-logits`): all old and new runs of a model are
byte-identical.

## Not measured

- Other Macs: the switch is on wherever the tensor-unit prefill is (the M5 family on macOS 26). Only the M5 Max was
  measured; `NI_PF_GROUPED=1` turns it on elsewhere.
- Flash-Next's Q2_0 packs, its expert store and split models stay on the serial encoder.
- The server's time to first token (`bench/harness`). `ni_run` times the same `encode_prefill` passes the engine
  encodes.
