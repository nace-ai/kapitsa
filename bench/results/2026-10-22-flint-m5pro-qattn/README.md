# M5 Pro, 48 GB: Recursive Flint 40B-A2.5B on Kapitsa with #345, 2026-10-22 and 23

Kapitsa's Flint runs on the M5 Pro with [#345](https://github.com/nace-ai/kapitsa/pull/345), which moves Flint's
prefill attention onto the M5's matrix units (`NI_FLINT_QATTN`, on by default where the dense QMT path is on), on top
of [#340](https://github.com/nace-ai/kapitsa/pull/340)'s MoE prefill on the tensor units. rec and llama.cpp were not
rerun: their cells stay the stored 2026-10-22 runs of [`2026-10-22-flint-m5pro`](../2026-10-22-flint-m5pro/README.md).

All runs: MacBook Pro M5 Pro, 48 GB (`Mac17,8`), macOS 26.5.2, High Power mode, AC power, Recursive's 4-bit MLX export
([RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) @ 3b93f905). Times are this Mac's clock
(UTC+5).

## #345's serve ABBA: the poster's cell

[`pr345-ab/`](pr345-ab/): #345's head `631bd25`, built Release, run through
`bench/harness/compare_engines.py run/ab_pr2.json --only pr2-attn0,pr2-attn1 --sizes 4000,16000,32000 --skip
greedy,tools,agent` ([`pr345-ab/run/ab_pr2.json`](pr345-ab/run/ab_pr2.json)), once in order (off, on: the forward pass,
2026-10-22 22:49-23:21, driver log [`pr345-ab/run/driver-forward.log`](pr345-ab/run/driver-forward.log)) and once with
`--reverse` (on, off: 2026-10-23 00:18-00:32, [`pr345-ab/run/driver-reverse.log`](pr345-ab/run/driver-reverse.log)).
The two entries are the same binary and arguments (`kapitsa serve --ctx 72000 --slots 1`, `--temperature 1.0
--always-reasons`); `NI_FLINT_QATTN=0` on the off runs is the only difference. Each run held the GPU lock on its own.
Greedy decode was skipped.

| Run | Pass | `NI_FLINT_QATTN` | Decode T = 1.0 (min–max) | Prefill 4K / 16K / 32K (tok/s) | Follow-up TTFT 4K / 16K / 32K (s) | Peak (GB) | Flags |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| [`pr2-attn0-20261022-224904`](pr345-ab/results/pr2-attn0-20261022-224904.json) | forward | off | 149.60 (147.3–151.8) | 2,238 / 1,608 / 1,144 | 0.10 / 0.12 / 0.16 | 23.7 | contention, invalid |
| [`pr2-attn1-20261022-231822`](pr345-ab/results/pr2-attn1-20261022-231822.json) | forward | on | 149.21 (148.1–149.7) | 2,694 / 2,435 / 2,062 | 0.10 / 0.12 / 0.15 | 23.7 | contention, invalid |
| [`pr2-attn1-20261023-001849`](pr345-ab/results/pr2-attn1-20261023-001849.json) | reverse | on | 148.56 (146.9–149.7) | 2,688 / 2,479 / 2,110 | 0.11 / 0.12 / 0.15 | 23.7 | invalid; the third 32K follow-up ended at the token cap with no answer (below) |
| [`pr2-attn0-20261023-002212`](pr345-ab/results/pr2-attn0-20261023-002212.json) | reverse | off | 152.40 (151.7–153.2) | 2,244 / 1,634 / 1,171 | 0.10 / 0.13 / 0.16 | 23.7 | invalid |
| **on, mean of 2** | | on | **148.9** | **2,691 / 2,457 / 2,086** | 0.10 / 0.12 / 0.15 | 23.7 | |
| off, mean of 2 | | off | 151.0 | 2,241 / 1,621 / 1,158 | 0.10 / 0.12 / 0.16 | 23.7 | |

On / off, means: 1.20x prefill at 4K, 1.52x at 16K, 1.80x at 32K; decode 0.99x (the off runs read 149.6 and 152.4,
the on runs 149.2 and 148.6; #345 doesn't change the decode path).

**The poster's M5 Pro Flint Kapitsa cell is the mean of the two on runs**: 2,691 / 149 (prefill at 4K / decode at T
1.0; 148.9). Against the other engines' stored cells (prefill 4K / 16K, decode): rec 0.1.0 2,138 / 1,941 and 144.9,
llama.cpp b11804 2,072 / 1,773 and 92.4. Kapitsa reads 4K prompts 1.26x as fast as rec and 1.30x llama.cpp, 16K prompts
1.27x and 1.39x, and decodes 1.03x rec at T 1.0 (rec is ahead on greedy in its stored runs) and 1.61x llama.cpp.

**Flags.**
- *contention*, the forward pass's two runs: WebKit used the GPU in the 16K and 32K prefill (mean foreign load 8.4%
  over the off run, 6.5% over the on run). The reverse pass's runs are not flagged (mean foreign load 1.5% and 0%).
- *invalid*, all four runs: `run/ab_pr2.json` does not pass `configs/flint.json`'s `--ref-prompt-tokens
  4000:3874,16000:15066`, so the harness checked the prompts (3,873-3,876 and 15,064-15,066 tokens) against the default
  corpus's 4,133 and 16,286. Against `configs/flint.json`'s references they are within 1%, the same prompts as the
  poster's other Flint runs: rec's stored runs read 3,874 and 15,066 tokens, llama.cpp's 3,874 and 15,065, within 0.1%
  of the on runs' 3,873-3,875 and 15,064, so the ratios against them compare the same prompts. Neither has a stored
  32K cell, so no 32K ratio against them is quoted.
- *the third 32K follow-up*, reverse pass, on run: it stopped with `finish=length` and an empty answer (30,336-token
  prompt, 30,326 cached). With `--always-reasons` at T 1.0 a follow-up gets 1,024 tokens, and the reasoning used all of
  them before any answer text, so the harness ended the cached series there. The generation itself did not fail: the
  cold 32K answer and the first two follow-ups in the same run are coherent. Its TTFT (0.13 s) still counts, so the
  0.15 s 32K follow-up TTFT is the median of all three turns (0.150, 0.149, 0.133). It does not touch the 4K prefill
  or the decode that the cell uses.

Paths in the records and logs are anonymized: `<repo>` is #345's checkout, `<base-repo>` the checkout the harness ran
from, `<run-root>` the run's output folder.
