# Results, 2026-10-18: Kapitsa with z-lab's DFlash drafter on Qwen3.5-4B

The result files behind [Kapitsa with z-lab's DFlash drafter](../../../../docs/benchmarks.md#kapitsa-with-z-labs-dflash-drafter)
in the Qwen3.5-4B section of docs/benchmarks.md: one hold of the GPU lock on 2026-10-18 between 22:49 and 23:09 on the
same MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1, High Power mode, on AC power), `bench/harness` with the `q35-4b-*`
Kapitsa configs. Kapitsa's previous build (main ecf8394, the MTP head at depth 1: the harness checkout of that commit,
whose 4B configs name no drafter) opened and closed the hold; the build with the drafter ran between, with its defaults
(`q35-4b-kapitsa` / `q35-4b-bf16-kapitsa`, which name the drafter with `--dflash`) and with its MTP head alone
(`q35-4b-mtp-kapitsa` / `q35-4b-bf16-mtp-kapitsa`, `--dflash none`). The drafter is
[z-lab/Qwen3.5-4B-DFlash](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) @9a1996c (Apache-2.0), the files
`kapitsa pull 4b` now fetches with the model.

| Result file | Started | Run | Weights | Decode, T = 0.6 / greedy (tok/s) | Flagged |
| --- | --- | --- | :---: | --- | --- |
| `q35-4b-kapitsa-20261018-224909.json` | 22:49 | previous build, opening | 4-bit | 120.5 / 124.2 | contention in the 16K prefill (Slack, 4.0%) |
| `q35-4b-bf16-kapitsa-20261018-225057.json` | 22:51 | previous build, opening | bf16 | 44.1 / 45.8 | contention in decode (Slack, 2.7%); 2 sampled answers looped |
| `q35-4b-kapitsa-20261018-225446.json` | 22:55 | DFlash drafter (default) | 4-bit | 178.7 / 197.7 | 1 sampled answer looped |
| `q35-4b-bf16-kapitsa-20261018-225612.json` | 22:56 | DFlash drafter (default) | bf16 | 89.1 / 87.6 | — |
| `q35-4b-mtp-kapitsa-20261018-230026.json` | 23:00 | MTP head, depth 4 | 4-bit | 159.0 / 168.6 | — |
| `q35-4b-bf16-mtp-kapitsa-20261018-230156.json` | 23:02 | MTP head, depth 4 | bf16 | 71.0 / 74.5 | — |
| `q35-4b-kapitsa-20261018-230411.json` | 23:04 | previous build, closing | 4-bit | 120.2 / 125.5 | — |
| `q35-4b-bf16-kapitsa-20261018-230558.json` | 23:06 | previous build, closing | bf16 | 45.9 / 45.4 | 1 sampled and 1 greedy answer looped |

Local paths in the files are replaced by placeholders: `<models>` the weights folder ($BENCH_MODELS, laid out as the
harness README describes), `<kapitsa-checkout>` this build's checkout, `<kapitsa-main>` the previous build's,
`<out>` the harness output folder, `<kapitsa-models>` Kapitsa's model cache the weights folder's links point into.
