# Synthetic Flash-Next engine validation of n-gram admission

**GPU coverage here uses a small random synthetic Flash-Next model, not an actual trained checkpoint.**
It establishes integration of valid n-gram mappings with the complete engine. It does not establish
language quality, real-checkpoint compatibility/performance, or an inference speedup.

Runs: MacBook Pro (16-inch, 2024), M4 Pro (14 CPU / 20 GPU cores), 48 GiB,
macOS 15.7.1 (24G231), Apple Clang 17, CMake Release (-O3, gnu++20, arm64, deployment target 14).
Python 3.12.12 / NumPy 2.5.4. Production implementation 877e7b4d; candidate tree at dd8f9f2c.
The preceding full standalone build and complete CPU tests ran at 399dae9b; later commits add evidence.

## Exactly what was compared

Both drivers link the same complete candidate engine and the same ni_run.cpp.o. For the reference,
only the ngram.cpp.o archive member is replaced with the pre-change source from 934a6d72, compiled
with the same Release flags. This isolates n-gram admission within a complete engine; it is not a
full build of the 934a6d72 checkout. The actual old source, commands, library/object/driver hashes
and model file hashes are retained. No other engine object, kernel, compiler option or model input
differs between arms. The original standalone admission reproducer and its expected baseline failure
remain frozen in the first evidence folder.

The repository's existing tools/fn_synth.py --seed 1 generated its default 6-layer, hidden 512,
vocabulary 2048 model: 512 routed Q2_0 experts of width 64, four hyper-connection streams, Gated DeltaNet
and QSA attention, the PLE table before layer 2, plus the MTP head (loaded, not used for speculative
decoding here). It occupies 128 MB. The table has 8214 rows × 64 values; every engine log confirms it.

Three layouts were run:

- Original ple.safetensors with a nonzero 4-bit affine shard, group 32.
- Independent ngram_table.bin with exactly the same affine weight/scales/biases payloads and short
  tensor names. The model directory has no ple.safetensors, exercising the independently mapped path.
- Independent ngram_table.bin with a deterministic finite nonzero BF16 payload.

For each layout, two execution modes were compared in A/B/A/B order:

- A 16-token batched prefill followed by 64 greedy output tokens. The dump contains 64 x 2048 fp32
  rows (524288 bytes): the prefill's final logits followed by the decode logits.
- --tbt: the same 16-token prompt through token-by-token execution. Its dump contains 16 x 2048
  fp32 rows (131072 bytes); generation then still emits 64 output token IDs.

All 24 arms succeed. Within every mode/layout, complete logits are byte equal, finite and nonzero,
and all 64 generated token IDs are equal across both repetitions of both engine versions. The
full-output SHA256s are in results.json. This is a synthetic vocabulary; no human-language text or
quality claim is made. No EOS-ignoring option was used.

The comparison is not vacuous: preserving the affine payload while changing its file layout
preserves complete model logits; substituting the different BF16 payload changes model logits in
both execution modes. Both engine versions agree on those changed outputs. controls.json records
these independent cross-layout checks.

## Reproduction

1. Configure/build the candidate in Release and explicitly build ni_run. Copy libnaceinfer.a for
   the reference, compile reference-ngram.cpp using the flags in commands.json into ngram.cpp.o,
   replace that one archive member with ar -r and run ranlib. Assert exactly one ngram.cpp.o member.
   Link both drivers from the same build/CMakeFiles/ni_run.dir/tools/ni_run.cpp.o, with their respective
   libraries and the normal ni_run framework list from CMake's link.txt. No library in the checkout
   is modified. Actual command lines and hashes are included.
2. Run `python3 tools/fn_synth.py EMPTY_MODEL_DIR --seed 1`.
   Use the existing NumPy dependency of this tool, not a new engine dependency.
3. Run `python3 make_table_variants.py MODEL_DIR EMPTY_VARIANT_OUTPUT_DIR`. The helper is the same fixture
   conversion used by the recorded run. All common model files are symlinked unchanged.
4. Run each driver for each of the three directories with IDs 1,2,...,16, 64 output tokens,
   temperature 0 and --dump-all-logits NEW_FILE; repeat with --tbt, and use A/B/A/B order.
   ni_run takes no GPU lock itself: wrap each process in the project flock and set NI_LOCK_HELD=1.
   The recorded harness did that around every GPU process. Compare complete dumps and token IDs,
   require positive sizes and finite/nonzero floats. Never reuse an output label.

Exact commands, model file digests and all 24 logs are included. Elapsed times are diagnostic only:
the first reference run compiles Metal pipelines and takes about 12 seconds to load, while subsequent
runs use warm pipelines. They must not be read as a candidate speedup or controlled performance A/B.

## Other evidence and limits

The original malformed-header regression rejects 26 fixtures and admits 3 valid formats, and fails
on the pre-change implementation before accessing a malformed row. Full standalone Release/CPU,
supplemental tests and prek passed in ../2026-11-06-ngram-bounds-followup/. This addition changes no
production source. Existing frozen records are untouched.

Actual Flash-Next checkpoint loading, its full-model HTTP/logit/performance checks, speculative MTP
execution, long-context/streamed-expert configurations, other Macs and post-admission file truncation
were not tested here. No unrelated Qwen or vision PR's results are used as evidence for this fix.
