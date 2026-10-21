# Results, 2026-10-22: MacBook Pro M5 Pro, 48 GB (Dory, current main)

The Nemotron Dory1.0 4B row of [docs/benchmarks.md](../../../docs/benchmarks.md#nemotron-dory10-4b) is a 2026-10-19
session on 0.1.36, before the M5 prefill kernels. These two runs re-measure the two KV settings on current main
(`fd01fe9`, the docs branch over 180a650), written by
[bench/harness](../../harness/README.md): `compare_engines.py configs/dory.json --only dory-kapitsa[-perloop] --skip
tools,agent,sustain`, one config at a time under the GPU lock.

MacBook Pro (16-inch, 2026), `Mac17,8`, M5 Pro, 48 GB, macOS 26.5.2, on AC power, one Edge and one Slack process
open. Another app (Drata Agent Helper) used 7% of the GPU through both prefill phases of both runs, so the prefill
cells are lower bounds and both records are flagged; the decode phases ran with the GPU quiet.

| Result file | Setting | Decode T = 0.6 (tok/s) | Decode greedy (tok/s) | Prefill 4K / 16K (tok/s) | Cached follow-up 4K / 16K (s) | Peak footprint (GB) | Flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `dory-kapitsa-perloop-20261022-032657.json` | 2 loops, per-loop KV, speculative decoding | 72.4 (69.0-74.5) | 80.5 (79.8-82.3) | 618 / 543 | 0.10 / 0.15 | 5.1 + 0.65 mapped | contention (prefill) |
| `dory-kapitsa-20261022-031737.json` | 2 loops, last-loop KV (its default) | 71.7 (70.5-72.3)<sup>1</sup> | 72.5 (71.9-72.6) | 618 / 543 | 0.10 / 0.15 | 4.3 + 0.64 mapped | contention (prefill) |

1. One of the five sampled runs looped (distinct-4-gram 0.49; flagged) and was the fastest, 72.3 tok/s.
