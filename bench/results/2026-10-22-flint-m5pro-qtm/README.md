# M5 Pro, 48 GB: Recursive Flint 40B-A2.5B on Kapitsa with #340, 2026-10-22

Kapitsa's Flint runs on the M5 Pro after [#340](https://github.com/nace-ai/kapitsa/pull/340) moved Flint's
routed-expert prefill GEMMs onto the M5's tensor units. rec and llama.cpp were not rerun: their cells stay the stored
2026-10-22 runs of [`2026-10-22-flint-m5pro`](../2026-10-22-flint-m5pro/README.md).

All runs: MacBook Pro M5 Pro, 48 GB (`Mac17,8`), macOS 26.5.2, High Power mode, AC power, Recursive's 4-bit MLX export
([RecursiveSI/flint-int4](https://huggingface.co/RecursiveSI/flint-int4) @ 3b93f905), `bench/harness/compare_engines.py`
with the bench arguments of `configs/flint.json` (`--temperature 1.0 --always-reasons --ref-prompt-tokens
4000:3874,16000:15066`). Times are this Mac's clock (UTC+5).

## #340's serve A/B: the poster's cell

[`pr340-ab/`](pr340-ab/): the A/B in #340's description, base / change / base, through
[`pr340-ab/run/ab_flint.json`](pr340-ab/run/ab_flint.json) (the poster's Kapitsa entry, once per
build). The base is main `1455ab4`; the change is `1455ab4` with the PR's change uncommitted in the tree, recorded as
`1455ab4+dirty`. Greedy decode was skipped (`--skip greedy,tools,agent`); otherwise this is the poster's protocol.

| Run | Build | Decode T = 1.0 (min–max) | Prefill 4K / 16K (tok/s) | Follow-up TTFT 4K / 16K (s) | Peak (GB) | Foreign GPU | Flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| [`flint-base-20261022-185958`](pr340-ab/results/flint-base-20261022-185958.json) | `1455ab4` | 152.65 (152.3–153.7) | 1,391.9 / 1,128.4 | 0.11 / 0.14 | 22.95 | 0% | none |
| [`flint-qtm-20261022-190135`](pr340-ab/results/flint-qtm-20261022-190135.json) | change | – | – | – | – | – | invalid: the server exited after loading, not counted |
| [`flint-qtm-20261022-190420`](pr340-ab/results/flint-qtm-20261022-190420.json) | change | **152.37** (151.5–153.4) | **2,261.5 / 1,634.5** | 0.10 / 0.13 | 22.95 | 0% | a cached 4K follow-up returned no content |
| [`flint-base-20261022-191410`](pr340-ab/results/flint-base-20261022-191410.json) | `1455ab4` | 153.19 (151.9–153.6) | 1,392.4 / 1,129.2 | 0.12 / 0.14 | 22.95 | 0% | none |

**The poster's M5 Pro Flint Kapitsa cell is `flint-qtm-20261022-190420`**: 2,262 / 152 (prefill at 4K / decode at T
1.0), one run, no foreign GPU load, chosen by request. It is the best of three runs of the change: the two on main
`91a249a` below read 1,743 and 2,098 at 4K (0.82x and 0.98x rec's), so the 4K lead rests on this run. Against the other
engines' stored cells (prefill 4K / 16K, decode): rec 0.1.0 2,138 / 1,941 and 144.9, llama.cpp b11804 2,072 / 1,773 and
92.4. Kapitsa reads 4K prompts 1.06x as fast as rec and 1.09x llama.cpp; rec and llama.cpp read 16K prompts 1.19x and
1.08x as fast; Kapitsa decodes 1.05x rec at T 1.0 (rec is ahead on greedy) and 1.65x llama.cpp. The runs were not
interleaved with rec's, which had Edge at 3-10% of the GPU; that can only favour Kapitsa.

## Two later Kapitsa-only runs on main `91a249a`

Kapitsa built Release from a clean checkout of main at `91a249a` (includes #340; binary sha256 `194ce9c3d518…`),
`compare_engines.py configs/flint.json --only flint-kapitsa --skip agent`, two legs, each its own
`lockf -k /tmp/kapitsa-gpu.lock` hold, each started with at least 40% of memory free (89% and 87%); scripts in
[`run/`](run/). **The Windows App was open during both.** They are kept as supplementary runs, not the poster's cell.

| Run | Decode T = 1.0 (min–max) | Greedy | Prefill 4K / 16K (tok/s) | Follow-up TTFT 4K / 16K (s) | Peak (GB) | Flags |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| [`flint-kapitsa-20261022-223254`](results/flint-kapitsa-20261022-223254.json) | 151.24 (149.2–152.4) | 145.73 | 1,743 / 1,350 | 0.10 / 0.13 | 23.25 | none (Windows App 0.6% mean) |
| [`flint-kapitsa-20261022-223805`](results/flint-kapitsa-20261022-223805.json) | 149.87 (149.0–151.9) | 151.14 | 2,098 / 1,423 | 0.11 / 0.14 | 23.25 | contention: Windows App in the 16K prefill (6.1% mean, 18% peak) |

Their 4K prefill differs by 20% and both read below #340's run (2,261.5); the cause was not found.

Paths in the records and logs are anonymized: `<repo>` is the Kapitsa checkout (for #340's runs, the PR's tree),
`<base-repo>` the base build's checkout, `<run-root>` a run's output folder, `<scratch>` the driver's folder.
