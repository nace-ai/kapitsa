# Reproduction method and derived examples

The following commands are derived examples with placeholders and relative outputs. They are not original execution logs. Each arm uses its own pinned production source, explicit Release configuration, actual target flags and source/binary digests.

```bash
cmake -S "$REF_SRC" -B "$REF_BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build "$REF_BUILD" -j 4 --target kapitsa logit_dump ni_run
cmake -S "$CAND_SRC" -B "$CAND_BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build "$CAND_BUILD" -j 4 --target kapitsa logit_dump ni_run
```

The included `iq4_nl_rows.cpp` has SHA256 `b1401721a0631600e67ec1fd46fd1407a7902d5829742f611f40b93d2e178a90`. It calls the official library once per row and does not copy either decoder. For each arm, a corresponding link example is:

```bash
clang++ -O3 -std=gnu++20 -arch arm64 -mmacosx-version-min=14.0 -I"$SRC/src" -I"$SRC/include" -I"$SRC/third_party" iq4_nl_rows.cpp "$BUILD/libnaceinfer.a" -framework Metal -framework Foundation -framework CoreFoundation -framework IOSurface -framework IOKit -framework ImageIO -framework CoreGraphics -framework Security -framework SystemConfiguration -framework Accelerate -o "$DRIVER"
"$DRIVER" --input "$MODEL_ROWS" --rows 1024 --cols 2560 --passes 16 --oracle "$REF_OUTPUT" --output "$LEG_OUTPUT"
```

`$MODEL_ROWS` is the first 1,024 rows of `blk.0.ffn_up.weight` in the pinned IQ4_NL GGUF, sample SHA256 `038e557ae654a48a0205501ca656b02a5d86c9da3e32f1dd1e8dd82f8542489b`. A reference oracle is generated outside timing. The measured order is A1/B1/A2/B2/A3/B3/A4/B4; each process loads input/oracle, allocates buffers and warms up once outside timing. Each timed pass includes only the 1,024 row calls. Complete output/input byte checks, finiteness and input/binary hashes occur outside elapsed time. Every pass and paired median time change is reported.

The full-model method imports the same complete pinned GGUF independently in fresh homes with each own-source CLI. All converted payloads and metadata are byte-compared; a manifest file-list ordering difference, if any, is reported separately only after full parsed-value equality. The selected trunk module's 5-bit/group64 affine format is checked explicitly; other per-module or MTP tensors are not assumed to have that format.

GPU/model checks use the shared exclusive lock. `$REF_MODEL` and `$CAND_MODEL` are the independent imported models. Corresponding numerical examples are:

```bash
lockf -k -t 600000 "$BENCH_GPU_LOCK" env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$REF_BUILD/logit_dump" "$REF_MODEL" output/reference --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus "$REF_SRC/bench/corpus.txt"
lockf -k -t 600000 "$BENCH_GPU_LOCK" env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$CAND_BUILD/logit_dump" "$CAND_MODEL" output/candidate --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from output/reference
python3 "$CAND_SRC/tools/logit_compare.py" output/reference output/candidate --no-gate --json output/report-only.json
python3 "$CAND_SRC/tools/logit_compare.py" output/reference output/candidate --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0 --json output/strict-zero.json
```

The independent literal audit checks all 584 rows/145,018,880 FP32 values per arm, complete logits and scored-input bytes, and finiteness; the 8 prefill rows have a separate literal gate. Report-only rc0 is a diagnostic, and explicit-zero scoring does not modify a noise floor. Actual verify metadata is reported: supplied-sequence tree verification is teacher-forced and does not qualify free MTP generation.

Default native/fixed checks use reference/candidate A/B/A/B under precise configuration with disk-cache 0. Fixed requests use sampler seeds 1/2/3, and complete content hashes apply only to the six decode cohorts. Prefill/cached text and random native transcripts have separate scope. Every original return code, failed output gate, warning and invalid result remains recorded; private machine-use details are represented by stable failed-gate IDs rather than converted into passing results.
