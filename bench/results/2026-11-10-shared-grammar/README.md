# Repeated-reference grammar validation

This folder contains explicitly redacted derived evidence and the exact CPU harness sources.
Original execution records and complete output files are retained privately. The JSON preserves
every A/B/A/B timing; the CPU microcase does not measure model inference.

`probe.cpp 9 4` constructs nine levels with four references to the same next definition at each
level. It times schema compilation, matcher initialization and accepting `"x"`. The reference
visits duplicate paths; the candidate keeps the first occurrence of each node in an alternative.

`mask_compare.cpp $TOKENIZER $OUTPUT` writes complete masks and acceptance flags for 216 states:
four reference depths, three leaf schemas, and 18 complete/partial/invalid input strings. Compare
the two output files with `cmp`. Each arm is a separately compiled executable, not a repeated call
to the same executable. These placeholder commands are reproduction examples, not raw transcripts.

Compile both harnesses with the same C++20 `-O3` flags, repository headers and native library.
For the M3 measurements, the existing baseline archive was at `903cdfce`; the grammar and tokenizer
source/header hashes were checked against `90078c7f` and matched before compiling. The candidate
linked its replacement `grammar.cpp` before the archive. All 17 collected M3 source/log/result
files passed digest checks, and the complete mask files also compared literally after collection.

The normal Qwen grammar suite passed 320 checks, including 5,471 states whose token masks agree
with the brute-force reference. No full-model GPU numerical or inference-performance claim is made.
