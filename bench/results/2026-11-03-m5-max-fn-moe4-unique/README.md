# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (the verify MoE's distinct experts found by a threadgroup)

`moe4_unique` (`src/kernels/zqwen4v.metal`) lists the distinct experts of a verify pass's rows (their top-k) for the
distinct-expert MoE kernels (`moe4_gate_up`, `moe4_down`): Flash-Next's MTP verify (3 rows x 10 experts, every layer)
and Flint's verifies of 2-8 rows. It ran on one thread, walking the entries and searching the list it was writing in
device memory: 19 us a call at Flash-Next's shape, 48 calls a pass, on the pass's critical path. It now runs on the
32 threads it was always dispatched with: each entry finds its expert's first entry in threadgroup memory, a first
entry's place in the list is the count of first entries before it (the serial pass's numbering), and each first entry
gathers the rows and slots of every entry that chose its expert. The list, the masks, the slots and the count are the
serial pass's, so the outputs are the same to the bit. Qwen3.8-Flash-Next 4-bit (Nace AI's MLX pack) at its default,
its MTP head at depth 2 (3 verify rows), MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS
26.4 (25E246), AC power, High Power mode, `caffeinate -is`, 2026-11-03 ([machine.json](machine.json)).

## Decode

`ni_run` ([ab/](ab)), main `d93297d` (old) against this branch (new, the same commit plus the change), two
checkouts, rounds alternating the arm order (old new, new old, ...), the README's decode prompt
([ab/prompt-ids.txt](ab/prompt-ids.txt), 56 tokens), greedy, 512 tokens (`--ignore-eos`), `--spec 2 --dflash none`
(the server's default for this model). Every run waited for a quiet GPU (no foreign process above 5% for 15 s:
`bench/harness/gpu_hogs.py`) and sampled the foreign GPU load while it ran. Every run wrote the same 511 tokens.

**Idle** ([ab/idle/](ab/idle): 4 rounds in two batches, this session idle, each quiet wait up to 900 s). The rounds in
which both arms found the GPU quiet and saw 0.0% foreign load while they ran (batch 1's second and third):

| Arm | tok/s | ms of GPU a cycle | Tokens a cycle |
| --- | ---: | ---: | ---: |
| main | 105.5, 106.2 | 21.92, 21.79 | 2.34 |
| this branch | 107.9, 108.1 | 21.42, 21.38 | 2.34 |

Medians: GPU time a cycle 21.85 -> 21.40 ms (-2.1%), decode 105.9 -> 108.0 tok/s (+2.0%). The other two rounds agree
(main 21.92-21.93 ms, this branch 21.36-21.39): batch 1's first, quiet but at 0.28% and 0.04% foreign load, and
batch 2's, whose waits ran out at 900 s with the Claude desktop app's helper process (which this session runs in) at
3.2-3.5% mean.

**The first A/B** ([ab/run1/](ab/run1), 3 rounds, each quiet wait up to 60 s): the helper held 5.5-6.9% mean (9.8% at
most) through both arms alike, so every wait ran out. The idle rounds replace it:

| Arm | tok/s | ms of GPU a cycle | Tokens a cycle |
| --- | ---: | ---: | ---: |
| main | 105.0, 105.0, 104.9 | 22.05, 22.06, 22.07 | 2.34 |
| this branch | 107.3, 107.5, 107.7 | 21.58, 21.53, 21.50 | 2.34 |

GPU time a cycle 22.06 -> 21.53 ms (-2.4%), decode 105.0 -> 107.5 tok/s (+2.4%).

`NI_PROFILE=1` over 44 passes of the same prompt ([kernel/](kernel), isolated dispatches): `moe4_unique` 18.97 ->
8.05 us a call, 0.91 -> 0.39 ms a pass.

## Checks

- `fn_check kernels` with `FN_ROWS=1` ([kernel/](kernel)): the new kernel against the serial pass on the CPU for 400
  random top-k sets of 1-8 rows x 1-16 slots (experts distinct within a row, shared across rows), every list, mask,
  slot and count equal; and the verify MoE (`moe4_unique` + `moe4_gate_up` + `moe4_down`) against the decode kernels
  row by row, bitwise, 1-4 rows. The same with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` (static threadgroup
  memory 1 KB): "all kernels match".
- Output check ([output-check/](output-check)): `logit_dump --paths step,verify` of both builds: the prefill rows,
  192 step rows and 192 verify rows (the MTP cycle's chain of 3) **every row bitwise identical**. No floor file exists
  for this 4-bit pack; bitwise identity needs none. (Both dumps' `meta.json` name `d93297d`: the branch's build was of
  the uncommitted change on it.)

## Not run

Flint, whose verifies of 2-8 rows run the same kernel, and the other Macs: the same outputs, speed not measured. The
server (`kapitsa serve`, T 0.6).

## Files

`ab/`: the A/B script, command, output and every run's `ni_run` output (`run1/`, `ab_fn.json` with each run's ids and
GPU load); `ab/idle/`: the idle rounds the same way (the script with its 900 s wait, `run.sh`, `batch1/`, `batch2/`). `kernel/`: the `fn_check` and profile commands and outputs. `output-check/`: the dumps' commands and logs,
their `meta.json` and the comparison. Local paths are replaced by placeholders (`~`, `<repo>`: this branch's
checkout, `<ref>`: main's, `<scratch>`).
