# Results, 2026-11-04: DFlash2's context commit on the tensor tile, M5 Max

After a prompt's prefill, `DFlash2::encode_prefill_commit` turns the target's captured hidden states into the drafter's
context. It does this in 512-row blocks, each running two GEMMs through `DFlash2::enc_qmm_rows`:

- **`fc_`:** N 2,048, K 16,384 on the 35B-A3B (8 captured layers).
- **`kv_all_`:** N 12,288 (6 layers' k and v), K 2,048.

That helper still ran the original simdgroup kernel (`qmm_b4_m64_n64`) on every Mac. The target's own prefill GEMMs
moved to the M5's matrix units long ago (`qmt_m64_n64`, `NI_QMT`). On an M5-class GPU (`qmt_on()`), the commit now
takes the same tensor tile for 4-bit group-64 weights at 64 rows or more. `NI_QMT=0` keeps the old kernel, as it does
for the target.

Only the drafter's context moves, because each output's K reduction runs in a different order. The target's outputs
are the verify's, so its tokens do not change.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power in High Power
mode, under the GPU lock, the session idle during the runs ([run_ab.sh](run_ab.sh), [run_ab.log](run_ab.log)).

- **Old:** main `d9e5430`.
- **New:** the branch: main plus this change.
- **The runs:** each arm's own Release build, `ni_run MODEL @prompt-ids.txt 64 0 --dflash auto --ignore-eos`. Clover
  ran with `--dflash` set to the 35B-A3B's drafter, as its harness config does.
- **What is timed:** the "prompt 4096 tok in X ms" line, one fresh process a run: the prefill plus the drafter's commit.
- **The order:** rounds alternate old new / new old, with 45 s idle before every run.

## Where the time went

`NI_PROFILE=2`, the 35B-A3B's 4K prompt ([profile-commit-old.txt](profile-commit-old.txt),
[profile-commit-new.txt](profile-commit-new.txt)):

| GEMM | Calls | Old: `qmm_b4_m64_n64` | New: `qmt_m64_n64` |
| --- | ---: | ---: | ---: |
| `fc_` (grid 32 x 8) | 4 | 2.97 ms a call (11.6 TFLOP/s) | 0.84 ms (40.8 TFLOP/s) |
| `kv_all_` (grid 192 x 8) | 4 | 2.04 ms a call (12.7 TFLOP/s) | 0.59 ms (44 TFLOP/s) |
| Commit total | | 20.0 ms | 5.7 ms |

## The 4K prompt, old against new

| Model | Rounds | Old (ms) | New (ms) | Change | Tokens per cycle (old = new) |
| --- | ---: | --- | --- | ---: | ---: |
| Qwen3.6 35B-A3B | 6 | 705, 705, 706, 708, 709, 706 | 692, 691, 690, 693, 693, 693 | **-1.9%** (706.0 -> 692.5) | 2.739 |
| Clover 35B-A3B | 2 | 705, 705 | 692, 692 | -1.8% | 4.846 |
| Qwen3.5 4B | 2 | 667, 666 | 648, 647 | -2.9% | 7.000 |
| Qwen3.5 9B | 2 | 1,387, 1,387 | 1,342, 1,343 | -3.2% | 7.000 |
| Qwen3.8 27B | 3 | 3,650, 3,644, 3,648 | 3,600, 3,603, 3,601 | -1.3% | 5.250 |

The old and new ranges are disjoint on every model.

- **Tokens:** every run of a model wrote the same 64 greedy token ids (`ids` md5 in the log).
- **Acceptance:** the drafter's tokens per cycle are the same to the third decimal on these prompts.
- **Prefill alone:** `--pf-reps` timed it at 659.0 against 658.3 ms on the 35B. The commit is the whole gain.

## Output check

`logit_dump` of main, then the branch on main's prefixes, compared by `tools/logit_compare.py`
([output-check/](output-check)):

- **Qwen3.6 35B-A3B:** every row bitwise identical on prefill, step, verify and batch; PASS on its M5 floor.
- **Qwen3.8 27B:** the same, on the same four paths; PASS on its M5 floor.

## Not measured

- **Other Macs:** only the M5 Max was measured, though the change covers every M5-class GPU on macOS 26 (`qmt_on()`).
  Older Macs keep the original kernel.
- **The server's time to first token** (`bench/harness`). The engine runs the same `encode_prefill_commit` after a
  prompt's last chunk.
- **Acceptance:** tokens per cycle were compared on one prompt per model, not across a prompt set.
