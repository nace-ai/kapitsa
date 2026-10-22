# Results, 2026-10-22: MacBook Pro M5 Max, 128 GB (Kapitsa and Splash re-measured)

The M5 Max came off the README poster on 2026-10-22 ([#276](https://github.com/nace-ai/kapitsa/pull/276)) because its
Kapitsa cells were 2026-10-17/18 runs, built before KFlash and the M5 kernels since. Every Kapitsa cell of that panel
and of its collapsed Nemotron table runs code that changed since (KFlash on every drafter target, the 16-row DFlash2 block, the M5 prefill and verify
kernels, Flash-Next's and Nemotron-H's model code), so each was run again on current main, and Splash 1.4.0 for the two
models it runs. The other engines' versions did not change, so their cells are still the
[2026-10-18 session](../2026-10-18-m5-max-session/README.md).

Written by [bench/harness](../../harness/README.md): `compare_engines.py configs/*.json --only <config> --skip tools,agent`,
one config at a time under the GPU lock, 00:22-01:20 local time. MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max,
40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power. Kapitsa 0.2.6, main `fd85910`; Splash 1.4.0, the release tarball
(`configs/splash.json`).

| Result file | Model | Engine | Prefill 4K / 16K (tok/s) | Decode T = 0.6, median (range) / greedy (tok/s) | Peak footprint (GB) | Flags |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `q38-27b-kapitsa-20261022-002219` | Qwen3.8 27B 4-bit | Kapitsa | 459 / 511 | 140.0 (125.6-158.2) / 145.0 | 20.3 | contention (6.5% mean, 15.4% max, com.apple.WebKit) |
| `q38-27b-splash-20261022-002916` | Qwen3.8 27B 4-bit | Splash 1.4.0 | 898 / 955 | 115.6 (113.8-130.5) / 124.6 | 22.7 | contention (6.9% mean, 15.2% max, com.apple.WebKit) |
| `q36-35b-kapitsa-20261022-003608` | Qwen3.6 35B-A3B 4-bit | Kapitsa | 4,702 / 3,442 | 325.4 (270.5-336.5) / 361.3 | 21.2 | contention (6.1% mean, 21.2% max, com.apple.WebKit) |
| `q36-35b-splash-20261022-004151` | Qwen3.6 35B-A3B 4-bit | Splash 1.4.0 | 6,640 / 5,866 | 290.0 (284.0-291.4) / 300.0 | 22.2 | contention (5.0% mean, 7.4% max, com.apple.WebKit) |
| `flash-kapitsa-20261022-004742` | Qwen3.8 Flash-Next 180B-A7B 4-bit | Kapitsa | 1,165 / 1,140 | 105.8 (98.2-108.1) / 106.0 | 70.6 | contention (5.9% mean, 16.4% max, com.apple.WebKit) |
| `bolt-kapitsa-20261022-005440` | Nemotron 3.5 Bolt 74B-A9B | Kapitsa | 1,016 / 1,062 | 153.2 (142.1-167.1) / 154.6 | 22.8 | contention (5.8% mean, 10.8% max, com.apple.WebKit); 2 decode run(s) in a repetition loop |
| `lightning-kapitsa-20261022-010104` | Nemotron 3.5 Lightning 30B-A3B NVFP4 | Kapitsa | 5,401 / 5,763 | 267.4 (250.2-278.9) / 277.9 | 19.0 | contention (5.8% mean, 10.2% max, com.apple.WebKit) |
| `super-kapitsa-20261022-010647` | Nemotron 3.5 Super-VL 120B-A12B NVFP4 | Kapitsa | 1,117 / 1,192 | 93.7 (91.1-97.8) / 87.2 | 70.3 | contention (5.6% mean, 7.0% max, com.apple.WebKit); repetition loop: decode, decode_greedy |
| `dory-kapitsa-perloop-20261022-011349` | Nemotron Dory1.0 4B NVFP4 (2 loops, per-loop KV) | Kapitsa | 1,077 / 1,021 | 119.5 (109.4-122.3) / 128.3 | 5.1 | contention (7.2% mean, 20.2% max, com.apple.WebKit); repetition loop: decode |

**Contention.** A system `com.apple.WebKit` GPU process stayed open the whole session (5-7% of the GPU on average), so
every run carries the harness's contention flag, as in the 2026-10-18 session (3-8%). Read the numbers as a lower bound.

**Dory predates #287.** The Dory row ran on `fd85910`, before [#287](https://github.com/nace-ai/kapitsa/pull/287) moved Dory's prefill GEMMs onto the matrix units (default on an M5 with macOS 26; 596 to 1,511 tok/s at 4K on the M5 Pro), so its prefill (1,077 / 1,021) is not main's now and its decode was not re-checked after that change. Dory's sampled decode and both of Super-VL's decode phases are repetition-loop flagged (valid runs), which flatters a speculative decoder; the earlier unflagged per-loop Dory run (117.4 tok/s, 2026-10-19) is within 2% of this one.

**Against the other engines.** Kapitsa has the fastest decode on the 27B, 35B-A3B, Bolt, Lightning, Super-VL and Dory (the last three are the README's collapsed Nemotron table). On Flash-Next mlx-serve
(108 tok/s) and oMLX (107) are ahead of Kapitsa's 106. Splash reads prompts faster than Kapitsa on both models it runs:
on the 27B 898 against 459 tok/s at 4K (its long prompts split between the GPU and the Neural Engine), on the 35B-A3B
6,640 against 4,702.

The server logs are in `logs/`. Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the
Kapitsa checkout, `<run-root>` for the run directory.
