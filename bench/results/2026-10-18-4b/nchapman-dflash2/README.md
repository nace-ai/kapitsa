# Results, 2026-10-18: oMLX with a third-party DFlash2 drafter on Qwen3.5-4B

The result files behind the note on [nchapman/Qwen3.5-4B-DFlash2](https://huggingface.co/nchapman/Qwen3.5-4B-DFlash2)
in the Qwen3.5-4B section of [docs/benchmarks.md](../../../../docs/benchmarks.md#qwen35-4b-2026-10-18): oMLX 0.7.1rc1
with that drafter on the 4-bit model, in one hold of the GPU lock on 2026-10-18 between 21:42 and 21:56, with Kapitsa
(68b7aee; b3e8fe9 in the files) first and last, on the same MacBook Pro (M4 Pro, 48 GB, macOS 27.0.1, High Power mode,
on AC power).

The drafter has no license. It was downloaded for this measurement only and deleted afterwards; neither its files nor
the config view oMLX read are in this repository. oMLX reads DFlash2 drafters in incoai's layout (a Qwen3 config with
a `dflash_config` block); this drafter, trained with vLLM's speculators library, has tensors of the same names but a
speculators config, so a local view rewrote that config into incoai's layout next to an APFS clone of the weights.
Its captured target layers (`aux_hidden_state_layer_ids` 2, 10, 17, 24, 29) were tried as they are (view A) and
shifted down by one (view B, the output-of-layer convention mlx-vlm captures); B was slightly faster and was kept.

| Result file | Started | Run | Decode, T = 0.6 / greedy (tok/s) | Flagged for contention |
| --- | --- | --- | --- | --- |
| `q35-4b-kapitsa-20261018-214207.json` | 21:42 | Kapitsa, opening | 129.7 / 126.8 | — |
| `q35-4b-omlx-dflash2-20261018-215202.json` | 21:52 | oMLX + the drafter (view B, block 5, verify adaptive, 4-bit at load) | 87.0 / 94.6 | — |
| `q35-4b-kapitsa-20261018-215443.json` | 21:55 | Kapitsa, closing | 125.6 / 126.7 | — |

Kapitsa's two runs agree within 5% on every rate and 0.02 s on follow-ups.

## Tuning

[tuning](tuning/) holds the short runs between Kapitsa's opening run and oMLX's run (three sampled and three greedy
decodes each, `--decode-runs 3 --skip prefill,tools`). "Tokens a pass" and "accepted" come from oMLX's per-answer
speculation lines in each file's `server_log`.

| Run | Setting | Decode, T = 0.6 (tok/s) | Decode, greedy (tok/s) | Tokens a pass | Drafts accepted |
| --- | --- | ---: | ---: | ---: | ---: |
| `smoke-q35-4b-omlx-dfl2-p1-A-b8` | view A, block 8, verify adaptive | 80.6 | 81.4 | 1.95 | 58% |
| `smoke-q35-4b-omlx-dfl2-p1-B-b8` | view B, block 8, verify adaptive | 82.9 | 79.9 | 2.12 | 60% |
| `smoke-q35-4b-omlx-dfl2-p2-B-b5` | view B, block 5, verify adaptive | 87.9 | 87.3 | 2.00 | 62% |
| `smoke-q35-4b-omlx-dfl2-p2-B-b16` | view B, block 16, verify adaptive | 72.9 | 71.4 | 2.02 | 59% |
| `smoke-q35-4b-omlx-dfl2-p2-B-b8-dflash` | view B, block 8, verify dflash | 65.2 | 75.5 | 2.34 | 58% |

For comparison, oMLX's MTP path wrote 101.4 / 103.8 tok/s in the short runs before bracket A and 102.4 / 104.5 in
bracket A. The drafter's card reports about 3.8 tokens a pass at temperature 0.7 under vLLM; oMLX got about 2 here,
so oMLX may not run this drafter as it was trained.

Files as in [the 4-bit folder](../README.md#files); `<models>/drafters/Qwen3.5-4B-DFlash2-A` and `-B` were the local
views.
