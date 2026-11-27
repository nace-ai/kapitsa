# M3 follow-up for five-bit CPU packing

These are explicitly privacy-filtered derived summaries, not raw execution logs. Original records remain private and unchanged. Source and binary identities, all test outcomes and relevant timings are retained. Private runtime-condition details are omitted; an original failed environment gate is still reported as failed.

Production `90e10efce8462ede4c756ca77ccdb1510f4e9faf`; reference `66a5f3bc473df9af8b72ded9090c3a8acc16a43e`. Runs: M3 Ultra,60 GPU cores,96 GiB,macOS26.4.1,Apple clang17.0.0(clang-1700.6.4.2). Both Release builds use actual target `-O3`/arm64 without NDEBUG.

The pinned IQ4_NL model and source-backed affine5bit/group64 import checks are in M3-IMPORT.json. All nine fresh files match literally. M3-NUMERICAL.json records all 584 rows / 145018880 float32 values per arm finite and complete output/input bytes equal. Strict-zero covers step/verify/batch; prefill has independent finite/literal checks. Verify is a teacher-forced KFlash chain of eight, not free MTP generation.

Native A/B/A/B returned rc 0 with empty invalid arrays. A1 has no flags; B1/A2/B2 retain a greedy repetition flag each. Fixed decode completed24 × 512 tokens: six request hashes and complete content hashes match across four legs. Sampled seed 3 has the same repetition flag in every leg. Adjacent decode rate changes range -0.839917% to +1.423008%; no inference speedup or universal no-regression claim.

Original fixed B1 cold prefill failed a runtime environment gate, whose private detail is omitted. Its original timing is not qualified. A separately labeled cold-only A/B/A/B recovery passed all measured phase gates: TTFT 2.523527292/2.512586250/2.507483833/2.506780083seconds. The exact cold driver is included; it removes only the decode loop from the previously published fixed driver, keeps the same prefill routine and condition checks, and links to the same unchanged reference library. Prefill/cache text was not byte-compared, and generated histories are not matched-cache performance inputs.

Both original HTTP runs returned rc 1: 54 PASS / 1 FAIL. The same queued-chat fixture expected keepalives around 15 and 30 seconds, but only one preceded content at 23.67/23.54 seconds. No all-pass API claim is made. Existing M4 results and failures remain unchanged in the earlier evidence directory.

Portable reproduction examples below are derived examples, not original command transcripts. Use each exact source build and the pinned imported model, taking the exclusive GPU lock and passing caller-held markers. The published cold driver takes the port,model-id and server PID of the corresponding precise-profile server; the server uses one slot,context32768,disk-cache0. Run A1/B1/A2/B2 with fresh servers and retain each JSON under a new label.

```sh
NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$REF_BUILD/logit_dump" "$REF_MODEL" output/reference --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --corpus bench/corpus.txt
NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$CAND_BUILD/logit_dump" "$CAND_MODEL" output/candidate --paths step,verify,batch --windows 8 --rows 24 --chunk 2048 --batch 4 --batch-rows 2 --from output/reference
python3 tools/logit_compare.py output/reference output/candidate --min-top1 1 --max-mean-kl 0 --max-median-kl 0 --max-p99-kl 0 --near-tie 0 --max-far-flips 0 --max-nll-change 0
"$COLD_DRIVER" "$PORT" "$MODEL_ID" "$SERVER_PID" > cold-A1.json
```

The original worker and all six explicit servers exited. The recovery worker and its four explicit servers exited. Each GPU phase held the exclusive lock; neither worker left its lock held. No completed import, model-output, decode or HTTP check was replayed for the cold-only recovery.
