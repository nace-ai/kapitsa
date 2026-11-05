# Results, 2026-11-05: the 27B's prefill in 4096-row chunks on the M5 Max (2048-row halves for the Neural Engine)

The Qwen3.8 27B splits each prefill FFN with the Neural Engine, and the default schedule is the two-half wavefront
(`Qwen35::enc_prefill_wave`). The Neural Engine runs one half's FFN while the GPU runs the other half's mixer. With
2048-row chunks, each half the Neural Engine sees is 1,024 rows.

**What changed:**
- **The split:** `FfnSplit::splits` / `begin_chunk` take a chunk up to two halves of 2048 rows in the wavefront.
  `Qwen35::ane_splits` requires the wavefront for a chunk past 2048 rows, and `ane_wave_rows` caps each half at 2048.
  The plain split (one lane) still takes at most 2048.
- **The engine:** for a single slot of the 27B on the 40-core M5 Max with the wavefront, the engine picks 4096-row
  chunks when the memory plan fits (`auto_chunk`, as for the 35B-A3B).

**Why it pays.** The Neural Engine runs more rows per evaluation faster. A probe of its matmul with runtime IOSurface
weights ([probes/ane_matmul_probe.cpp](probes/ane_matmul_probe.cpp), not built by the repo) measured these forms
(3,584 x 2,560 weights):

| Form | 1,024 rows | 2,048 rows |
| --- | ---: | ---: |
| int8 weights and inputs dequantized to fp16, as the split runs | 21.8 TFLOP/s | 25.5 TFLOP/s |
| fp16 inputs | 8.3 | 9.2 |
| fp16 weights, int8 inputs | 13.1 | 16.5 |

int8 straight into `matmul` does not compile. In the 27B's split its time per evaluation goes from 12.4 ms (1,024-row
halves) to 22.6 ms (2,048-row halves): ~10% less per row.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, under the GPU
lock, the session idle during the runs.

## Prefill, old against new

[run_ab.sh](run_ab.sh), [run_ab.log](run_ab.log).

- **Old:** main `d9d479d`, 2048-row chunks.
- **New:** the branch with `NI_CHUNK=4096`, the chunk the engine now picks.
- **The runs:** `ni_run qwen3.8-27b-4bit-mtp @<ids> 8 0 --pf-reps 2 --dflash none`. Rep 0 is the cold prefill, rep
  1 the next one back to back.
- **The order:** old new / new old per round, 45 s idle before each run.

| Prompt | Rounds | Old rep 0 (ms) | New rep 0 (ms) | Change | Old rep 1 | New rep 1 | Change |
| --- | ---: | --- | --- | ---: | --- | --- | ---: |
| 4,096 tokens ([prompt-ids-4k.txt](prompt-ids-4k.txt)) | 4 | 3,095.7, 3,080.6, 3,084.1, 3,081.2 | 2,953.1, 2,955.6, 2,954.2, 2,954.8 | **-4.2%** | 3,091.2, 3,082.4, 3,083.2, 3,087.2 | 2,995.8, 2,970.2, 2,989.8, 2,962.0 | -3.5% |
| 7,907 tokens | 2 | 6,195.6, 6,212.5 | 5,923.1, 5,923.4 | **-4.5%** | 6,655.9, 6,622.0 | 6,656.7, 6,628.2 | level |

The share stays 14 of 34 units: at 4096-row chunks, 12 and 13 units were slower ([probes/](probes)). So were 16 and 18,
where the Neural Engine's time per unit jumps (1.6 ms a unit up to 14, 2.1 at 16, 2.4 at 18). A third back-to-back 4K
prefill ran hotter and gave back the gain (+4% on that rep in the sweep). A cold prompt and a warm second one gain or
hold level.

## Output check

`logit_dump` of the old build at its 2048-row chunks, then the new on the reference's prefixes with `--chunk 4096`,
`tools/logit_compare.py` with the 27B's M5 floor ([output-check/](output-check)). The windows of 2,432-3,200 tokens
now prefill as one chunk instead of 2048 rows plus a tail.

| Path | Bitwise rows | KL mean | KL p99 | Result |
| --- | ---: | ---: | ---: | --- |
| prefill | 7 of 8 | 5.6e-7 | 4.2e-6 | |
| step | 168 of 192 | 3.1e-5 | 4.3e-5 | PASS |
| verify | 168 of 192 | 3.0e-5 | 2.2e-5 | PASS |
| batch | 168 of 192 | 3.1e-5 | 4.5e-5 | PASS |

There were no top-1 flips. One window's worth of rows differs (24 of 192). That fits the 2,432-token window: at 2048
rows its last 384 rows were under the split's 512-row minimum and ran on the GPU alone, and now they run inside a split
chunk. The 2,816- and 3,200-token windows also changed chunking and stayed bitwise identical, consistent with the
split's per-row arithmetic not depending on the chunk. Which window the 24 rows belong to was not checked.

## Not measured

- **Other Macs:** the M5 Pro (20 cores) and smaller Macs keep 2048-row chunks; the rule asks for 40 cores.
- **Other models:** the 4B and other split models are unchanged.
- **The server's time to first token:** the README cell's harness run follows in its own pull request.
