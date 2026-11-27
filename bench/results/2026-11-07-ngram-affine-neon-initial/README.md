# Affine Q4 n-gram CPU decoding: initial evidence

Source: reference `01b4c6d69fbe69e862c00358f88c94a363129592`, candidate `8c0a7ec3`.
This folder records CPU checks only. No model or GPU result is claimed.

The existing mapped Q4 table row has eight codes per U32 word. The candidate
extracts two vectors of four nibbles and performs the same affine multiply/add,
with BF16 coefficients loaded once per group. It applies to 4-bit rows whose
row dimension and group size are multiples of eight. Other widths and group
boundaries retain the original loop. File admission, prefetching, worker count,
GGUF, BF16 and FP8 decoding are unchanged. This is independent of #1019's
admission checks and #992's gather worker-count change.

## Full-library regression

M4 Pro, 48 GiB, macOS 15.7.1 (24G231), Apple Clang 17, Release `-O3`.
The complete candidate library, CLI, ni_run and fn_store_test built successfully.
The build was configured at the base revision with the working-tree patch that
became `8c0a7ec3`; `identity.json` pins the library and test-binary hashes.
It must not be mistaken for a clean baseline build from its embedded version.

`m4-pro-fn-store.log` retains the complete test output. The new test compares
1,449,984 values through real mapped files with an independent bit-at-a-time
code reader: dimensions 40–256, group sizes 3–128, Q2/Q3/Q4/Q6/Q8, signed zero,
subnormal, finite extremes, infinities and quiet/signaling NaN BF16 coefficients,
four rounding modes, and output guards. Existing unequal-shard/gather and
GGUF checks also pass. This is not the complete CPU suite.

## Arithmetic-loop screen

The public helper extracts the exact affine case from both commits into
separate non-inlined translation units, built with `clang++ -O3 -std=c++20`,
without LTO. The helper itself was run successfully on the M4 Pro; its source
identities and raw output are retained. The same candidate affine body was
also run on the remote M3 Ultra (96 GiB, macOS 26.4.1, Apple Clang 17).

Each screen compares 1,572,864,000 output values, plus destination guards:
seven dimensions, five group sizes, all 65,536 BF16 scale patterns paired with
deterministic bias patterns and random codes, and four rounding modes. This
does not enumerate every possible scale/bias/code combination. Some standalone
screen geometries are partial groups that the production file loader rejects;
the full-library test separately uses valid file geometries.

For every timing cell the order is A/B/A/B, 200,000 calls per arm, using resident
synthetic data. Every arm is retained in the log. This excludes shard lookup,
madvise, page faults, threaded gather, model loading and GPU execution. On the
remote M3 Ultra, dimension 160/group 32 measured:

| Arm | Microseconds per arithmetic-loop call |
| --- | ---: |
| A1 | 0.110116 |
| B1 | 0.016709 |
| A2 | 0.110514 |
| B2 | 0.016797 |

The separate #1019 weight downloads were active during this CPU screen. CPU
frequency/power/thermal state was not controlled, and the resident loop timing
is preliminary. No inference-throughput or complete NgramTable row/gather
speedup follows from these numbers.

Reproduce on Apple silicon, with both commits present in the repository and a
new output directory:

```sh
python3 bench/results/2026-11-07-ngram-affine-neon-initial/reproduce.py /path/to/kapitsa /tmp/ngram-affine-screen
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target fn_store_test
build/fn_store_test
```

## Outstanding validation

The exact remote reference/candidate Release builds and complete candidate
CPU suite are being prepared separately. GPU integration, real trained table
row timings/gather costs, full-model logits, nonempty greedy continuations,
paired HTTP checks and native interleaved benchmarks remain outstanding.
The PR stays Draft. The running #1019 download may supply the same pinned
model data, but none of #1019's validation results count for this change.
All future results must be added as new evidence files/folders.
