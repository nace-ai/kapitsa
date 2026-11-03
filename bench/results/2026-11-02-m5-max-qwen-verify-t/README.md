# Results, 2026-11-02: MacBook Pro M5 Max, 128 GB (the 35B's and Clover's tree verify attention on the matrix units)

The A/B and the output check of `attn8t_partial_zone_g8` (`src/kernels/tensor/v8attn_t.metal`): the 8-row KFlash tree
verify's history attention on the M5's matrix units, against `attn8m_partial_zone` (`NI_V8_ATTN_T=0`), on Qwen3.6 35B-A3B
and Clover 35B-A3B, one build, the arms interleaved (old, new, old, new), written by
[bench/harness](../../harness/README.md): `compare_engines.py configs/<model>.json sweep-configs/ab6.json --only <label>
--agentic --skip greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60`, each label the model's default with its
context at its limit (`--ctx`), the coding answer after a cold document of each size from the long corpus of
[2026-11-02-m5-max-long](../2026-11-02-m5-max-long/README.md), one request per size. MacBook Pro (16-inch, 2026),
`Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode, `caffeinate -is`, 20:06-20:44
local time; the build: main `b6345da` with the change ([machine.json](machine.json)).

## Decode at long context

Tok/s, the middle of a 512-token answer; each value one cold request:

| Model | Context | Old: decode (two runs) | New: decode (two runs) | Speedup | Prefill, old / new |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6 35B-A3B | 33K | 239.4, 238.9 | 302.2, 266.1 | 1.11-1.26x | 4,211 / 4,294 |
| Qwen3.6 35B-A3B | 131K | 135.8, 121.5 | 171.0, 186.9 | 1.26-1.54x | 1,935 / 1,930 |
| Qwen3.6 35B-A3B | 249K | 93.1, 100.3 | 127.6, 142.9 | 1.27-1.53x | 1,169 / 1,163 |
| Clover 35B-A3B | 33K | 227.5, 225.7 | 259.6, 273.7 | 1.14-1.21x | 4,112 / 4,291 |
| Clover 35B-A3B | 123K | 136.1, 119.2 | 155.7, 187.4 | 1.14-1.57x | 2,021 / 2,027 |

Prefill is the same (the change is the verify's). One layer of the 35B's heads, the kernel alone (8 rows, the engine's
zone chunks, best of 5): 0.48 / 0.86 / 1.61 ms after 32K / 128K / 256K positions against `attn8m_partial_zone`'s 1.71 /
1.68 / 3.99.

## Output check

`logit_dump` of main `b6345da` against this branch's, `tools/logit_compare.py` at twice each model's M5 floor
(`tools/logit_floors/<model>.m5.json`, measured on main, #891): [output-check/](output-check). Prefill, step and batch:
every row bitwise identical; verify: 0 of 192 rows bitwise, top-1 1.0000, 0 flips, KL mean / p99 7.71e-5 / 1.04e-3 (the
35B; gate 3.37e-4 / 1.03e-2) and 4.06e-4 / 5.40e-3 (Clover; 8.87e-4 / 1.63e-2): PASS on both.

## Contention

The Claude desktop app's GPU process, driving the session, held up to ~10% of the GPU (the harness's worst-phase mean)
while the agent worked, and the 60 s quiet wait timed out before most runs; both arms ran under it, interleaved. The
prefill of the companion change's output check (another checkout's `logit_dump`) took the GPU lock between some runs,
never during one.

## Files

`results/`, `logs/`: the runs (`*-v8t-old`: `NI_V8_ATTN_T=0`; `*-v8t-new`: the default) and their server logs.
`output-check/`: `logit_compare.py`'s report per model. `session/`: the runner's output and scripts.
`sweep-configs/ab6.json`: the labels. Local paths are replaced by placeholders: `~`, `<repo>` for this checkout, `<main>`
for main's, `<run-root>` for the harness's output folder, `<dumps>` for the logit dumps, `<host>` for the host name.
