# Q4 code-plane CPU decode evidence

Explicitly redacted derived evidence. Original execution commands and logs remain private.
The commands below are reproduction examples, not original execution transcripts.

`decode4` expands Q4_0/Q4_1 packed bytes to the existing low/high code planes with NEON on arm64.
The original scalar order handles overlapping source/code buffers. Scale and offset expressions,
Q4_K, other formats and Metal kernels are unchanged. This is independent of four-bit packing.

Production: 5bcbe7e2069f55485473ba45755ab4c65a3c44c5. Reference: 87c105238186beadb375805e47a89791a499319a. Both CPU arms use the same driver/header and `gguf.cpp`;
only `quant.cpp` comes from the designated source. The oracle's half conversion uses the unchanged
`from_f16` primitive; nibble extraction and ordered stores are checked independently.

The pinned real Q4_0 tensor supplies 1,024 rows with K=2,560. Each of eight alternating legs has
64 warmup and 256 timed passes. Each pass makes 1,024 separate decode calls. All 3,276,800 bytes
of codes/scales/offsets and unchanged input are checked outside the timer on every pass.

| Chip | Pair | Reference ns/row | Candidate ns/row | Latency change |
|---|---:|---:|---:|---:|
| M3 | 1 | 2857.397 | 353.388 | -87.63252% |
| M3 | 2 | 2854.433 | 365.486 | -87.19583% |
| M3 | 3 | 2872.668 | 352.608 | -87.72541% |
| M3 | 4 | 2985.927 | 354.687 | -88.12138% |
| M4 | 1 | 359.666 | 80.161 | -77.71242% |
| M4 | 2 | 359.301 | 110.335 | -69.29170% |
| M4 | 3 | 360.119 | 104.266 | -71.04682% |
| M4 | 4 | 354.810 | 104.268 | -70.61305% |

These are warm CPU decode component measurements, not whole import or inference speedups.
Every actual leg is retained in M3-CPU.json and M4-CPU.json. CHECKS.json records the completed
Release/CPU checks, sanitizers and deliberately rejected mutations. The initial CPU suite had
15 passes, an update-test fixture extraction failure, and six later checks that did not execute
after log redirection failed. Those seven checks passed in a new task temporary directory;
the 15 completed passes were not repeated. Original failures remain in the private record.
Full-model validation is
not established by this CPU checkpoint; Q4_1 real-model timing and non-arm64 execution were not run.

Example reproduction (the same driver and candidate test header in both CPU arms):

```sh
c++ -std=c++20 -O3 -I"$CAND_SRC/src" -I"$CAND_SRC/tools" -I"$CAND_SRC/third_party" \
  perrow-driver.cpp "$REF_SRC/src/gguf/quant.cpp" "$CAND_SRC/src/gguf/gguf.cpp" -o ref-row
c++ -std=c++20 -O3 -I"$CAND_SRC/src" -I"$CAND_SRC/tools" -I"$CAND_SRC/third_party" \
  perrow-driver.cpp "$CAND_SRC/src/gguf/quant.cpp" "$CAND_SRC/src/gguf/gguf.cpp" -o cand-row
./ref-row "$RAW_Q4_ROWS" A1.decoded
./cand-row "$RAW_Q4_ROWS" B1.decoded
```

Continue in A/B order to four pairs. `$RAW_Q4_ROWS` is the unchanged first 1,024 rows of the
pinned tensor, whose full identity and byte hash are in each CPU JSON. No model download is
part of this CPU command. Input preparation and full-model results must not be inferred from it.
