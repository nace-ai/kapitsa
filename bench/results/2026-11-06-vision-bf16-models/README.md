# Additional vision-reader model validation

Runs: MacBook Pro (16-inch, 2024), Apple M4 Pro (14 CPU / 20 GPU cores), 48 GiB,
macOS 15.7.1 (24G231), Apple Clang 17, CMake Release -O3, deployment target 14.
Reference d65a0afa, candidate implementation ae9ca905; binaries report d65a0afa / 3c12d22c.
Later commits add evidence only. Explicit CMake configure preceded both original builds.
identity.json records actual binary hashes; commands.json records the individual commands.
Earlier CPU, reader, 4B/9B tower, 4B logits and HTTP records remain frozen in the first
two evidence folders. No production code changed for these additional measurements.

## Full image encoders and model outputs

The identical vision-reload.cpp driver already published in the preceding folder runs
Qwen3.8 27B 4-bit and Qwen3.6 35B-A3B OQ4E-MTP image towers on the same fixed generated
64x64, 224x224 and 384x256 RGB inputs. It unloads/reloads each tower and repeats.
All six complete old/new outputs per model are byte equal, finite/nonzero, guarded,
and identical after reload. Per-arm reports and independent SHA256s are included.
These are 12 additional complete comparisons, beyond the 12 previously recorded on 4B/9B.

Both 27B and 35B language-model checks compare all 584 logits rows, every row bitwise
identical and finite: 8 prefill / 192 batch / 192 step / 192 verify. The standing comparator and
original metadata are retained verbatim. The 27B floor gate passes. The 35B OQ4E model
has no floor file: its report explicitly says reported-only; a separate assertion
requires all 584 rows byte equal and finite. No floor was invented, edited or overridden.
The 35B checkpoint has no local DFlash checkpoint and these runs are not drafter evidence.

Drivers take the GPU lock (the encoder driver was wrapped in an outer flock with
NI_LOCK_HELD=1); no other task's process or lock was changed. The logit commands use
candidate --from its model's reference dump. Raw distributions were removed only
after metadata, comparator output and byte-equality assertions had been saved.

## Native text benchmark records

kapitsa bench --model MODEL --quick --json --out FILE, default execution and native
GPU flock/quiet wait, fresh KAPITSA_HOME each invocation. Every native JSON retains
its three internal repetitions, power/thermal/activity records and content flags.
These text requests do not measure image TTFT. All runs below remain part of the record.

| Run | Sampled tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bench-4b-vision-repeat-01-base | 194.05 | 196.32 | 2.1344 | 0.0567 | 6.92 |
| bench-4b-vision-repeat-02-vision | 190.69 | 197.60 | 2.1002 | 0.0449 | 6.93 |
| bench-4b-vision-repeat-03-base | 203.28 | 213.77 | 2.0894 | 0.0508 | 6.93 |
| bench-4b-vision-repeat-04-vision | 194.54 | 207.65 | 2.0962 | 0.0435 | 6.94 |

| Run | Sampled tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bench-27b-vision-01-base | 60.85 | 65.36 | 10.5212 | 0.2261 | 19.17 |
| bench-27b-vision-02-vision | 62.69 | 66.13 | 10.5305 | 0.2270 | 19.18 |
| bench-27b-vision-03-base | 62.40 | 65.84 | 10.5330 | 0.2277 | 19.18 |
| bench-27b-vision-04-vision | 63.06 | 66.03 | 10.5208 | 0.2320 | 19.18 |

| Run | Sampled tok/s | Greedy tok/s | 4K TTFT s | Follow-up s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| bench-35b-vision-01-base | 112.18 | 112.39 | 3.7235 | 0.0655 | 20.57 |
| bench-35b-vision-02-vision | 110.56 | 111.57 | 3.7208 | 0.0654 | 20.57 |
| bench-35b-vision-03-base | 109.56 | 112.98 | 3.7163 | 0.0671 | 20.56 |
| bench-35b-vision-04-vision | 113.66 | 116.27 | 3.7153 | 0.0782 | 20.57 |

- bench-4b-vision-repeat-01-base: thermal nominal; other GPU mean/max 14.04/33.4%; flags ["CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy', 'prefill_4000', 'tools'] (mean 14.04%, top ['Google Chrome', 'ChatGPT'])"]; invalid [].
- bench-4b-vision-repeat-02-vision: thermal nominal; other GPU mean/max 11.71/35.5%; flags ["CONTENTION: sustained foreign GPU load > 5% in ['decode', 'decode_greedy', 'prefill_4000', 'tools'] (mean 11.71%, top ['ChatGPT', 'Google Chrome'])"]; invalid [].
- bench-4b-vision-repeat-03-base: thermal nominal; other GPU mean/max 3.49/27.33%; flags ["decode: repetition loop (distinct-4-gram 0.4659)", "CONTENTION: sustained foreign GPU load > 5% in ['decode'] (mean 3.49%, top ['Google Chrome'])"]; invalid [].
- bench-4b-vision-repeat-04-vision: thermal nominal; other GPU mean/max 0.54/1.52%; flags []; invalid [].
- bench-27b-vision-01-base: thermal fair; other GPU mean/max 0.05/1.2%; flags []; invalid [].
- bench-27b-vision-02-vision: thermal fair; other GPU mean/max 0.06/0.9%; flags []; invalid [].
- bench-27b-vision-03-base: thermal fair; other GPU mean/max 0.02/0.72%; flags []; invalid [].
- bench-27b-vision-04-vision: thermal fair; other GPU mean/max 0.09/1.2%; flags []; invalid [].
- bench-35b-vision-01-base: thermal fair; other GPU mean/max 0.0/0.0%; flags []; invalid [].
- bench-35b-vision-02-vision: thermal nominal; other GPU mean/max 0.03/0.58%; flags []; invalid [].
- bench-35b-vision-03-base: thermal nominal; other GPU mean/max 0.03/0.58%; flags []; invalid [].
- bench-35b-vision-04-vision: thermal nominal; other GPU mean/max 1.77/25.45%; flags []; invalid [].

The 4B repeat contains sustained foreign GPU load in its first two runs and a contention
flag in its third. Previous 4B measurements also observed desktop GPU activity. The
other model rows' exact conditions are listed above. None of those observations is
silently discarded or changed into an idle-GPU throughput claim. No stable full-model
speedup or universal no-slowdown result is asserted from the native text series.

The benefit is removal of the whole-tensor BF16 staging allocation/copy when k==ld.
The preceding complete-library 64 MiB reader probe recorded 206340096 bytes vs 139231232
bytes peak RSS in both A/B pairs (exactly 64 MiB lower). Its absolute peak is reader-level,
not whole-model memory, and its elapsed time was not controlled performance evidence.

Reproduce with complete Release builds and the previous folder's vision-reload.cpp,
then use the commands recorded here with your model paths. No image/model data or
floor is regenerated by this evidence addition. Build/CPU/supplemental/prek results
remain in ../2026-11-06-vision-bf16-followup/ and still cover the unchanged source.

Not verified: other image architectures/Macs, 9B full-language-model logits, controlled
image TTFT, and whole-model memory reduction. The native text measurements are not
substitutes for those claims. Ready for review with those explicit limits and the
complete reader/tower correctness checks above.
