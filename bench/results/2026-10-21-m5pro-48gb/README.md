# Results, 2026-10-21: MacBook Pro M5 Pro, 48 GB (prefill re-measure)

Kapitsa's prefill on the M5 Pro after the M5 prefill kernels landed ([#205](https://github.com/nace-ai/kapitsa/pull/205) dense 4-bit
GEMMs, [#215](https://github.com/nace-ai/kapitsa/pull/215) attention, the Nemotron-H kernels of 2026-10-20), written by
[bench/harness](../../harness/README.md): `compare_engines.py configs/kapitsa.json configs/lightning.json --skip agent`, the full
protocol minus the agent session, one config at a time under the GPU lock. MacBook Pro (16-inch, 2026), `Mac17,8`, M5 Pro, 48 GB, macOS 26.5.2, on AC power.
Kapitsa 0.2.2 (b50f1fc, main after #239). Times are local (UTC+5), 19:56-20:12.

The README's M5 Pro prefill cells (4K-token prompt) are these runs; its decode cells are still the 2026-10-19 runs, which were measured in the
same session as the other engines. The decode medians here are higher than those, but no other engine was re-run beside them, so they are not used.

| Result file | Model | Prefill 4K / 16K (tok/s) | Decode T = 0.6 / greedy (tok/s) | Flags |
| --- | --- | ---: | ---: | --- |
| `q35-4b-kapitsa-20261021-200057.json` | Qwen3.5 4B, DFlash | 3,032 / 2,736 | 217.6 / 231.5 | none |
| `q38-27b-kapitsa-20261021-195748.json` | Qwen3.8 27B, DFlash2 | 465 / 451 | 68.0 / 77.2 | none |
| `q36-35b-kapitsa-20261021-195622.json` | Qwen3.6 35B-A3B, DFlash2 | 1,823 / 1,468 | 208.1 / 233.9 | decode: repetition loop in one of five runs (prefill valid) |
| `q36-35b-kapitsa-20261021-200652.json` (second run) | Qwen3.6 35B-A3B, DFlash2 | 1,824 / 1,469 | 198.6 | none |
| `lightning-kapitsa-20261021-200211.json` | Lightning 30B, DSpark | 3,048 / 2,989 | 167.8 / 165.7 | none |
