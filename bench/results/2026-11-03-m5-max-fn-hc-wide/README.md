# Results, 2026-11-03: MacBook Pro M5 Max, 128 GB (Flash-Next's folded gated residual on wider grids)

Qwen3.8-Flash-Next's folded gated residual (the hyper-connections' low-rank mix, a down and an up projection of
rank 320 over 4 streams of 2,560, 6-bit, twice a layer: 96 pairs a decode or MTP-verify pass) for up to 4 rows on
wider grids, `NI_HC_WIDE` (on by default on the M5 Max, where it was measured; `0` the old grids):

- the down projection's per-chunk stats (each row's sums of squares and injection dots, and its injected row) in
  threadgroups of their own, one per chunk and row, beside the dots (`hc_down_sk4`), where `hc_down_sk3` ran them after
  the dots in its first row block's threadgroups, behind 3 threadgroup reductions a row;
- a verify's up projection on 2 blocks of 16 channels per threadgroup (`hc_up_mix4_c2`, 80 threadgroups at this shape)
  instead of 4 (`hc_up_mix4_c4`, 40: one a core on the M5 Max's 40-core GPU).

Every value is computed by the same code, so the outputs are the same to the bit. Qwen3.8-Flash-Next 4-bit (Nace AI's
MLX pack) at its default, its MTP head at depth 2 (3 verify rows), MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max,
40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode, `caffeinate -is`, 2026-11-03
([machine.json](machine.json)).

## Kernel

`fn_check hc-time` ([kernel/](kernel)): a down + up pair at Flash-Next's shape over 96 distinct weight sets (streamed
from DRAM, as in a pass), best of 3:

| Rows | Before (`hc_down_sk3` + `hc_up_mix3_c2` / `mix4_c4`) | `NI_HC_WIDE` (`hc_down_sk4` + `hc_up_mix3_c2` / `mix4_c2`) |
| ---: | ---: | ---: |
| 1 (decode, the MTP draft) | 22.4 us | 21.4 us |
| 2 | 32.6 us | 29.2 us |
| 3 (the default verify) | 41.2 us | 36.4 us |
| 4 | 49.8 us | 43.7 us |

The down projection alone, 3 rows: 19.8 -> 18.5 us; the up projection alone (`FN_HC_ONLY`), 3 rows: `mix4_c4` 21.6,
`mix4_c2` 18.1 us. Every variant's outputs (u, X, y, the injection weights, partials, injected rows, stats) are
bitwise the one-row kernels' (`hc-time.out`), and at small shapes for 4-, 6- and 8-bit weights, 1 to 4 rows, with and
without injection, under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` (`FN_ROWS=1 fn_check kernels`,
`fn-rows-validation.out`: the 24 new checks and the mode's others, "all kernels match"). Static threadgroup memory:
`hc_down_sk4` 512 B, `hc_up_mix4_c2` the 8,768 B `hc_up_mix4_c4` has (the layer runs both).

Tried and dropped: the up projection's prep (the rows' inverse RMS and h from the chunks' partials, which every
threadgroup of `hc_up_mix4` redoes) as one pass of its own, with the projection then on 160 threadgroups: 42.6 us at
3 rows, the extra dispatch costing what the wider grid saved; and the down projection two rows at a time (each weight
run unpacked once for both): bitwise the same, but 30.9 against 18.5 us (register pressure).

## Decode

`ni_run` ([ab/](ab)), one build, `NI_HC_WIDE=0` against the default, rounds alternating the arm order (old new, new
old, ...), the README's decode prompt ([ab/prompt-ids.txt](ab/prompt-ids.txt), 56 tokens), greedy, 512 tokens
(`--ignore-eos`), `--spec 2 --dflash none` (the server's default for this model). Every run waited for a quiet GPU
(no foreign process above 5% for 15 s: `bench/harness/gpu_hogs.py`) and sampled the foreign GPU load while it ran.
Every run wrote the same 511 tokens.

**Idle** ([ab/idle/](ab/idle): 5 rounds in two batches, this session idle, each quiet wait up to 900 s). The rounds in
which both arms found the GPU quiet and saw 0.0% foreign load while they ran (batch 1's second and third):

| Arm | tok/s | ms of GPU a cycle | Tokens a cycle |
| --- | ---: | ---: | ---: |
| `NI_HC_WIDE=0` | 105.9, 105.9 | 21.83, 21.84 | 2.34 |
| default | 107.9, 108.2 | 21.42, 21.37 | 2.34 |

Medians: GPU time a cycle 21.84 -> 21.40 ms (-2.0%), decode 105.9 -> 108.1 tok/s (+2.0%). In the other three
rounds every wait ran out at 900 s, the Claude desktop app's helper process (which this session runs in) at 2.8-4.8%
mean through both arms alike; they agree (`NI_HC_WIDE=0` 21.91-21.96 ms, the default 21.36-21.44).

**The first A/B** ([ab/run1/](ab/run1), 3 rounds, each quiet wait up to 60 s): the helper held 3.8-5.7% mean (9.6%
at most) through both arms alike, so every wait ran out. #921's title and the usage row quoted it until the idle rounds
replaced it:

| Arm | tok/s | ms of GPU a cycle | Tokens a cycle |
| --- | ---: | ---: | ---: |
| `NI_HC_WIDE=0` | 105.6, 105.6, 105.7 | 21.94, 21.92, 21.91 | 2.34 |
| default | 106.3, 108.0, 108.1 | 21.51, 21.42, 21.43 | 2.34 |

GPU time a cycle 21.92 -> 21.45 ms (-2.1%), decode 105.6 -> 107.5 tok/s (+1.7%; the first new run 106.3).

## Output check

`logit_dump --paths step,verify` of this build with `NI_HC_WIDE=0` and with the default ([output-check/](output-check)):
the prefill rows, the 192 step rows (one row a pass: the down projection's change) and the 192 verify rows (the MTP
cycle's chain of 3 rows: both changes) **every row bitwise identical**. No floor file exists for this 4-bit pack
(only `flash-next-q20.m5.json`); bitwise identity needs none.

## Not run

The other Macs: the default is the M5 Max's. `hc_up_mix4_c4` was chosen on another Mac against `hc_up_mix3` (65.0 ->
54.1 us at 3 rows, the source's note); `hc_up_mix4_c2` against it there was not measured. `NI_HC_WIDE=1` turns both
changes on there. The server (`kapitsa serve`, T 0.6): the change is
the same kernels in the same passes, and the poster cell was not re-measured.

## Files

`ab/`: the A/B script, command, output and every run's `ni_run` output (`run1/`, `ab_fn.json` with each run's ids and
GPU load); `ab/idle/`: the idle rounds the same way (the script with its 900 s wait, `run.sh`, `batch1/`, `batch2/`). `kernel/`: the `fn_check` commands and outputs. `output-check/`: the dumps' commands and logs, their
`meta.json` and the comparison. Local paths are replaced by placeholders (`~`, `<repo>`: this branch's checkout,
`<scratch>`).
