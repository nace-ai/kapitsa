# CPU encoding of long BPE pieces

These are **redacted derived summaries**, not raw logs. Complete token arrays were compared literally and retained
privately. Every measured arm and case, including slower samples, is retained in JSON.

Reference tokenizer: `a283176326de221e9326e05b9f21e0759be75f99`. Final production source: `d744df9db4a336530d2d6fadc90b5d2b8dcad72c`. The shared C++ driver and headers are
identical between arms; A uses the pre-change tokenizer implementation. All binary, tokenizer and case SHA-256
identities are in each summary. Flags: `-std=c++20 -O3 -DNDEBUG`, CoreFoundation linked, native arm64.

Runs: Apple M5 Pro / macOS 26.6.2 / Apple clang 17.0.0 (clang-1700.0.13.5), and Apple M3 Ultra /
macOS 26.4.1 / Apple clang 17.0.0 (clang-1700.6.4.2). Qwen, Tekken and Flint identify the three native tokenizer
split families; `tokenizer-identities.json` records exact tokenizer digests and model metadata. No neural weights
were loaded. The unchanged reference drivers were reused; all A1/B1/A2/B2 executions below are new runs.

## Method

Use [the driver and harness](../../tokenizer/README.md). Each family has four separate processes, A1/B1/A2/B2.
`warm_ms` includes encoding and a complete-array equality check, after the first encode. Tokenizer loading is excluded.
`first_ms` is the first encoding of a case, with cache entries from earlier cases retained. The JSON preserves actual
input bytes, repetition counts, token counts and first/warm times. Corpus/source names describe character-prefix limits;
they do not assert exact byte lengths. These measurements concern CPU tokenization, not model inference throughput
or end-to-end request latency. Continuous letters and unpunctuated CJK are stress inputs.

## Final M5 stress measurements

| Tokenizer | Case | A1 ms | B1 ms | A2 ms | B2 ms | mean A / mean B |
|---|---|---:|---:|---:|---:|---:|
| qwen | letters-8192 | 17.508500 | 0.662032 | 17.287222 | 0.720412 | 25.170x |
| qwen | letters-32768 | 195.763347 | 3.836194 | 194.020708 | 3.931139 | 50.182x |
| qwen | cjk-32768 | 13.488404 | 2.896651 | 14.129910 | 2.870955 | 4.789x |
| tekken | letters-8192 | 15.186810 | 0.630134 | 15.927176 | 0.630482 | 24.682x |
| tekken | letters-32768 | 200.669722 | 3.922736 | 192.217736 | 3.677625 | 51.693x |
| tekken | cjk-32768 | 10.561946 | 2.648679 | 10.556612 | 3.142606 | 3.647x |
| flint | letters-8192 | 17.891532 | 0.735958 | 17.261241 | 0.755972 | 23.562x |
| flint | letters-32768 | 210.258125 | 4.368139 | 212.423931 | 4.939042 | 45.415x |
| flint | cjk-32768 | 11.744830 | 2.692981 | 11.289330 | 2.697638 | 4.273x |

## Final M5 controls, initial run

| Tokenizer | Case | A1 ms | B1 ms | A2 ms | B2 ms | mean A / mean B |
|---|---|---:|---:|---:|---:|---:|
| qwen | corpus-64k | 0.839254 | 0.889677 | 0.850612 | 0.853152 | 0.970x |
| qwen | source-32k | 0.144539 | 0.139139 | 0.154647 | 0.144010 | 1.057x |
| qwen | chat-short | 0.000729 | 0.000710 | 0.000882 | 0.000751 | 1.103x |
| qwen | repeat-32768 | 0.140888 | 0.152702 | 0.141928 | 0.143458 | 0.955x |
| qwen | base64-64k | 0.698388 | 0.694289 | 0.685364 | 0.702368 | 0.991x |
| tekken | corpus-64k | 0.825814 | 0.849568 | 0.892531 | 0.833031 | 1.021x |
| tekken | source-32k | 0.317781 | 0.328330 | 0.330277 | 0.310852 | 1.014x |
| tekken | chat-short | 0.005551 | 0.005622 | 0.005825 | 0.005388 | 1.033x |
| tekken | repeat-32768 | 0.213607 | 0.232425 | 0.226152 | 0.232978 | 0.945x |
| tekken | base64-64k | 1.354830 | 1.398112 | 1.397358 | 1.414724 | 0.978x |
| flint | corpus-64k | 0.610101 | 0.668436 | 0.631706 | 0.607586 | 0.973x |
| flint | source-32k | 0.188261 | 0.209322 | 0.181488 | 0.181205 | 0.947x |
| flint | chat-short | 0.001480 | 0.001623 | 0.001470 | 0.001496 | 0.946x |
| flint | repeat-32768 | 0.164955 | 0.176635 | 0.164798 | 0.178036 | 0.930x |
| flint | base64-64k | 1.172400 | 1.131778 | 1.124084 | 1.111770 | 1.024x |

## Final M5 controls, longer independent repeat

The same five inputs use respectively 1000 corpus, 4000 source, 200000 chat, 5000 repeated-character and 1000 base64
iterations per arm. All counts are identical between A and B. The full 19-case run and this longer repeat are both
retained; neither supersedes the other.

| Tokenizer | Case | A1 ms | B1 ms | A2 ms | B2 ms | mean A / mean B |
|---|---|---:|---:|---:|---:|---:|
| qwen | corpus-64k | 0.844351 | 0.838858 | 0.857126 | 0.827805 | 1.021x |
| qwen | source-32k | 0.141532 | 0.142702 | 0.140488 | 0.141907 | 0.991x |
| qwen | chat-short | 0.000741 | 0.000799 | 0.000780 | 0.000752 | 0.980x |
| qwen | repeat-32768 | 0.136782 | 0.153572 | 0.140852 | 0.153109 | 0.905x |
| qwen | base64-64k | 0.771615 | 0.712102 | 0.682134 | 0.725938 | 1.011x |
| tekken | corpus-64k | 0.836149 | 0.828272 | 0.817496 | 0.819932 | 1.003x |
| tekken | source-32k | 0.309453 | 0.312535 | 0.303898 | 0.301460 | 0.999x |
| tekken | chat-short | 0.005780 | 0.005399 | 0.005299 | 0.006076 | 0.965x |
| tekken | repeat-32768 | 0.224658 | 0.234909 | 0.229194 | 0.225355 | 0.986x |
| tekken | base64-64k | 1.400359 | 1.431953 | 1.382524 | 1.378946 | 0.990x |
| flint | corpus-64k | 0.611848 | 0.602570 | 0.628698 | 0.592945 | 1.038x |
| flint | source-32k | 0.183626 | 0.177359 | 0.189549 | 0.179543 | 1.046x |
| flint | chat-short | 0.001506 | 0.001516 | 0.001547 | 0.001620 | 0.973x |
| flint | repeat-32768 | 0.165185 | 0.161540 | 0.167322 | 0.157841 | 1.041x |
| flint | base64-64k | 1.213387 | 1.132806 | 1.124139 | 1.119904 | 1.038x |

Qwen's 32 KiB repeated-character control remains about 10.5% slower in the longer repeat (about 0.139 to 0.153 ms).
There are also individual slower samples elsewhere. No universal no-regression or typical-chat speedup is claimed.
The heap adds temporary O(n) storage for a long piece.

M3's three complete ABAB runs also match all 228 output arrays. Background CPU contention was observed during this
investigation, so its timing files are preserved and excluded from the headline; do not compare absolute times across chips.

## Initial variant and regression investigation

`initial-*.json` and `initial-README.md` retain source `801527c8cd0c2cf55413c38a5a9a2f7ba6315802` and both initial
M5 control rounds. Slower Qwen/Tekken controls prompted disassembly inspection: inlining the heap helper introduced a
stack-protector prologue in `Tokenizer::bpe`, including cache-hit calls. The final source keeps that helper out of line;
the extra prologue is absent. The BPE algorithm, threshold and eight-pass budget are unchanged. A template-specialization
prototype was compiled but not measured or selected. No performance claim is made for it.

## Checked and remaining limitations

- Final source: 138 unit checks on each Mac, including an independent array reference with ordered, reversed and
  shuffled merge ranks. Final M5 UBSan: 138 checks pass. ASan attempts on the initial source hung before main on both
  Macs; no ASan pass is claimed and that unavailable runtime was not retried on the final source.
- Final source on M5: 700 new complete-array comparisons against newly executed reference runs across five actual tokenizer
  files, with mixed Unicode, combining marks, boundaries, punctuation and added tokens.
- Final source: Release configure/build, all 23 `vk cpu-tests` commands, Qwen parser and real-tokenizer structured
  checks, profile/comment gates, and `prek run --all-files` pass. Optional reasoning-dataset rebuild and Drex
  companion-update subcases were skipped. `cpu-checks.json` contains the actual command list and CLI binary identity.
- Initial source: two missing-`jsonschema` failures and one transient empty-output update fixture failure were retained;
  all three targeted repeats passed. These initial-source records are separate from final-source checks.
- Extra initial-source integration: Flint parser passed; Tekken parser lacked `<think>` in both versions. Tekken/Flint
  structured checks failed. Separately configured Release reference/candidate grammar runs reproduced identical
  160/125 failure lines (`integration-baseline.json`); these are baseline failures, not passing integration checks.
  Full grammar runs for these two vocabularies were not repeated after the no-inline annotation.
- No GPU, neural-model logit/output, inference throughput, other operating-system or concurrency stress validation.
