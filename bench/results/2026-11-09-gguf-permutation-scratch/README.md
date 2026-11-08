# Direct four-bit column permutation

This is an explicitly privacy-filtered derived export. Raw commands and records remain private; placeholder commands below are reproduction examples, not original execution logs. Drivers are copied byte-for-byte from the actual checks.

Production 9493a437/reference39446144 changes only the direct exact-four-bit branch of convert_rows. It decodes directly into the existing permutation scratch and writes permuted BF16 scale/offset outputs directly. This removes two per-row float-vector allocations and three temporary copies, without changing decoding, rounding or packing. The ordinary unpermuted loop remains linear. No hardware gate or opt-in is added.

The real Q4_0 tensor in INPUT.json exercises this branch. Each timed call converts1024rows ofK=4096 in one batch; the longer run uses2048passes/leg and eight alternating A/B legs. The M4 per-row nanoseconds are1536.51/1413.44,1542.68/1412.38,1550.83/1419.50,1553.01/1419.45: -8.009710%,-8.446340%,-8.468369%,-8.600073%. All complete output code/scale/offset bytes match the independent oracle before and after every timed leg. The shorter128-pass current-source screen is also retained, including its variability. These are CPU conversion component timings, not complete import/loading or inference measurements.

The boundary driver runs960cases per arm over Q4_0/Q4_1/Q4_K,K256/512/1024/4096,head widths0/32/64/128,key groups1/2,zero/partial row ranges and nonzero source offsets. All7334400 output bytes including canaries match. M4 ASan/UBSan passes the same cases; a deliberately wrong head permutation is rejected by the independent real-row oracle with rc 6. Exceptional source scales are included in the differential cases. Baseline differential equality is not an independent mathematical oracle for Q4_K.

M3 uses the same source and driver and passes the same 960 boundary cases. Its eight longer AB legs reduce normalized per-row latency by 10.223705%,6.615049%,6.277483%,6.830329%; all full output/oracle gates pass. These are within-chip comparisons; the absolute times across chips are not interpreted as a hardware ranking.

## Derived reproduction examples

Use the pinned tensor prefix from INPUT.json as `$ROW_PAYLOAD`. Build both arms with the same driver and support files. `$REF_IMPORT` denotes src/gguf/import.cpp extracted from the exact reference commit; `$IMPORT_SOURCE` is either that file or the candidate's src/gguf/import.cpp. Compile flags below reproduce the standalone method, not full engine qualification.

```sh
c++ -O3 -std=c++20 -arch arm64 -ffunction-sections -fdata-sections -Wl,-dead_strip -I. -Isrc -Iinclude -Ithird_party -DIMPORT_SOURCE=\"$IMPORT_SOURCE\" row-driver.cpp src/gguf/quant.cpp src/gguf/gguf.cpp -o row-driver
./row-driver "$ROW_PAYLOAD" output-A1.bin 2048
```

Repeat with the corresponding reference/candidate binaries in A1/B1/A2/B2/A3/B3/A4/B4 order, using new output labels and comparing full output bytes. The cases driver takes only an output filename. The real-row driver performs its independent oracle before and after timing.

## Remaining qualification

Full pinned Q4_0 model import, complete-vocabulary logit comparison and default runtime A/B/HTTP have not run. Complete these using separate exact-source Release builds under the exclusive GPU lock and caller-held markers; compare the actual imported out_proj code/scale/bias bytes and then full model output. The existing IQ4_NL checkpoint does not exercise this direct exact-four-bit branch. No GPU or whole-model correctness/speed claim is made. Additional model families and complete import/loading latency are unmeasured.

Concrete full-model follow-up examples (not yet executed for this candidate):

```sh
cmake -S "$REF_SRC" -B "$REF_BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$REF_BUILD" -j 2 --target kapitsa logit_dump ni_run
cmake -S "$CAND_SRC" -B "$CAND_BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$CAND_BUILD" -j 2 --target kapitsa logit_dump ni_run
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_HOME="$REF_HOME" "$REF_BUILD/kapitsa" import "$GGUF" --name qwen35-exact4-review --threads 4 --no-drafter
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_HOME="$CAND_HOME" "$CAND_BUILD/kapitsa" import "$GGUF" --name qwen35-exact4-review --threads 4 --no-drafter
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$REF_BUILD/logit_dump" "$REF_MODEL" output/reference --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus bench/corpus.txt
lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$CAND_BUILD/logit_dump" "$CAND_MODEL" output/candidate --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from output/reference
python3 tools/logit_compare.py output/reference output/candidate --no-gate
```

Follow with explicit strict-zero comparison where no applicable floor exists, native quick benchmark A/B/A/B and fixed-request/output-gated runtime checks on each setup. These commands are a plan, not results. Preserve every original failure and timing leg.
