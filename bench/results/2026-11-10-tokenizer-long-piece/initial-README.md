# Initial variant: CPU encoding of long BPE pieces

These are **redacted derived summaries**, not raw execution logs. Complete raw token arrays were compared literally
and are retained privately. The JSON files retain every measured case and arm, including slower samples.

A is `a283176326de221e9326e05b9f21e0759be75f99`; B production source and the shared C++ driver are
`801527c8cd0c2cf55413c38a5a9a2f7ba6315802`. Both arms use identical headers and driver sources, with only
`src/tokenizer/tokenizer.cpp` replaced in A. Binary, tokenizer and input SHA-256 identities are in each JSON.
Flags: `-std=c++20 -O3 -DNDEBUG`, CoreFoundation linked. This is the initial production variant; README.md records the later production change and its fresh measurements.

Runs: Apple M5 Pro, macOS 26.6.2, Apple clang 17.0.0 (clang-1700.0.13.5);
Apple M3 Ultra, macOS 26.4.1, Apple clang 17.0.0 (clang-1700.6.4.2). Both drivers are native arm64.
Qwen, Tekken and Flint name the three native tokenizer split families; their vocabulary/merge bytes are identified
by their complete tokenizer SHA-256 in the JSON. No neural weights were loaded.

## Measurement

Use [the driver and harness](../../tokenizer/README.md). Each family runs A1/B1/A2/B2 in separate processes.
`warm_ms` measures encode plus complete-array equality checks, after the first encode; tokenizer loading is excluded.
`first_ms`, repetitions, actual input bytes and token counts are retained in JSON. Corpus/source case labels name
prefix limits, not a promise that the file has exactly that many bytes. Random letters and unpunctuated CJK are stress
inputs. These are CPU encoding measurements, not model throughput or end-to-end request latency.

The M5 measurements below cover three tokenizer families. The M3 runs also compared all 228 output arrays successfully,
but background CPU contention was observed after that run. Its timings are preserved in `initial-m3-*.json` and excluded
from the headline performance claim; absolute times must not be compared across chips.

| Tokenizer | Case | A1 ms | B1 ms | A2 ms | B2 ms | mean A / mean B |
|---|---|---:|---:|---:|---:|---:|
| qwen | letters-8192 | 17.8716 | 0.7287 | 17.7654 | 0.7291 | 24.44x |
| qwen | letters-32768 | 196.8371 | 3.9387 | 195.9753 | 4.4579 | 46.78x |
| qwen | cjk-32768 | 15.9958 | 3.0747 | 13.3049 | 3.1102 | 4.74x |
| tekken | letters-8192 | 16.1910 | 0.6306 | 15.6736 | 0.7210 | 23.58x |
| tekken | letters-32768 | 178.8763 | 3.7595 | 196.2755 | 5.0342 | 42.66x |
| tekken | cjk-32768 | 10.8517 | 2.7496 | 10.4856 | 2.8268 | 3.83x |
| flint | letters-8192 | 17.5315 | 0.8562 | 19.9323 | 0.7239 | 23.71x |
| flint | letters-32768 | 208.6197 | 4.4839 | 203.3195 | 4.4277 | 46.22x |
| flint | cjk-32768 | 11.3291 | 2.7718 | 11.4272 | 2.9335 | 3.99x |

## Controls

| Tokenizer | Case | A1 ms | B1 ms | A2 ms | B2 ms | mean A / mean B |
|---|---|---:|---:|---:|---:|---:|
| qwen | corpus-64k | 0.8883 | 0.9161 | 0.8947 | 0.9119 | 0.98x |
| qwen | source-32k | 0.1511 | 0.1456 | 0.1437 | 0.1526 | 0.99x |
| qwen | chat-short | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.96x |
| qwen | repeat-32768 | 0.1690 | 0.1529 | 0.1454 | 0.1477 | 1.05x |
| qwen | base64-64k | 0.7518 | 0.7138 | 0.6705 | 0.7207 | 0.99x |
| tekken | corpus-64k | 0.8124 | 0.9011 | 0.8359 | 0.8649 | 0.93x |
| tekken | source-32k | 0.3009 | 0.3069 | 0.3082 | 0.3453 | 0.93x |
| tekken | chat-short | 0.0053 | 0.0053 | 0.0054 | 0.0064 | 0.91x |
| tekken | repeat-32768 | 0.2353 | 0.2216 | 0.2451 | 0.2372 | 1.05x |
| tekken | base64-64k | 1.3960 | 1.3934 | 1.4430 | 1.4173 | 1.01x |
| flint | corpus-64k | 0.6004 | 0.6190 | 0.6197 | 0.6087 | 0.99x |
| flint | source-32k | 0.1738 | 0.1828 | 0.1794 | 0.1867 | 0.96x |
| flint | chat-short | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.98x |
| flint | repeat-32768 | 0.1685 | 0.1570 | 0.1695 | 0.1691 | 1.04x |
| flint | base64-64k | 1.1263 | 1.1526 | 1.1318 | 1.1481 | 0.98x |

The first eight passes use the existing algorithm. A piece still longer than 64 symbols then uses local edge
updates. That path needs additional temporary O(n) storage; no chip/model switch was added.

## Follow-up ordinary-text controls

A separate, longer A1/B1/A2/B2 repeat investigated the slower control samples above. It uses the same binaries
and tokenizer files, with fixed longer iteration counts in all arms. Both runs are retained; no universal
no-regression or typical-chat speedup is claimed.

| Tokenizer | Case | A1 ms | B1 ms | A2 ms | B2 ms | mean A / mean B |
|---|---|---:|---:|---:|---:|---:|
| qwen | corpus-64k | 0.845122 | 0.954968 | 0.874884 | 0.886879 | 0.934x |
| qwen | source-32k | 0.146257 | 0.153380 | 0.151010 | 0.147281 | 0.989x |
| qwen | chat-short | 0.000732 | 0.000796 | 0.000767 | 0.000767 | 0.959x |
| qwen | repeat-32768 | 0.147749 | 0.180460 | 0.153970 | 0.153461 | 0.904x |
| qwen | base64-64k | 0.724513 | 0.817946 | 0.692122 | 0.745230 | 0.906x |
| tekken | corpus-64k | 0.877409 | 0.882693 | 0.840524 | 0.917168 | 0.954x |
| tekken | source-32k | 0.324582 | 0.349704 | 0.327275 | 0.349106 | 0.933x |
| tekken | chat-short | 0.005587 | 0.005691 | 0.005879 | 0.006513 | 0.940x |
| tekken | repeat-32768 | 0.223970 | 0.229934 | 0.231499 | 0.234746 | 0.980x |
| tekken | base64-64k | 1.388032 | 1.415917 | 1.454938 | 1.447947 | 0.993x |
| flint | corpus-64k | 0.653263 | 0.644792 | 0.681447 | 0.640957 | 1.038x |
| flint | source-32k | 0.190661 | 0.194932 | 0.200032 | 0.183655 | 1.032x |
| flint | chat-short | 0.001730 | 0.001641 | 0.001690 | 0.001621 | 1.048x |
| flint | repeat-32768 | 0.175688 | 0.165292 | 0.181054 | 0.159544 | 1.098x |
| flint | base64-64k | 1.195159 | 1.178474 | 1.227090 | 1.168691 | 1.032x |

## Correctness and limits

- Complete token arrays agree across all four arms in every 19-case run (three families, two Macs).
- 700 additional comparisons cover five actual tokenizer files, boundaries, mixed Unicode, combining marks,
  punctuation and added tokens. See `initial-correctness.json`.
- 138 unit checks pass on M5 and M3, including an independent array reference with ordered, reversed and shuffled
  merge ranks. M5 UBSan passes the same checks; coverage recorded 60 calls into the new heap path.
- ASan did not reach main on either Mac because its runtime hung during initialization; those attempts are retained
  privately and are not counted as passing tests.
- The `vk cpu-tests` run initially passed 20 commands and failed three. `test_bench` and `test_community`
  needed the existing `jsonschema` dependency added to the Python path; targeted reruns passed. The update
  fixture's concurrent binary probe observed one empty output; its independent repeat passed all 114 checks.
  The initial failures are retained. Optional reasoning-dataset rebuild and Drex companion update cases were skipped. Profile/comment gates and `prek run --all-files` pass.
- Qwen parser and structured checks pass; Flint parser passes. Tekken parser exits because its vocabulary lacks
  `<think>` in both versions. Tekken and Flint structured grammar checks fail in both versions; a separate
  Release reference build replaces only the tokenizer implementation, and direct grammar runs reproduce identical
  failure lines. See `integration-baseline.json`; these limits are not counted as passing integration checks.
- No GPU, neural-model output/logit, inference throughput, other operating system, or concurrency stress measurement.
