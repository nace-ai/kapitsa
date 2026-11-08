# Complete EXL3 model validation

Runs: Apple M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1 (25E253), Apple clang 17, CMake Release `-O3`.
Reference source: `f22effb9bdda7fa17be5e4ce614383a35e1cc62a`.
Candidate production source: `dc82cf0f39b942157e86b0cd5e518e5caed17e61`.
Both source trees were explicitly reconfigured and built before these runs. Source identities and binary SHA256s remain in the exported states; modified commands are labeled placeholder examples. Production files have not changed.

Model: `beamster/Qwen3.8-Flash-Next-Sushi-2bpw`, complete EXL3 mixed-format weights; `model-identity.json` records checkpoint file names and sizes; private resolved paths are omitted. The complete 32,000,153,976-byte n-gram table matches pinned revision `f37e7ba46f0f2993b6ca373221530c579f258712`, SHA256 `c8ab74bc343408cf3923d7d64b3698fbeb3e78c07ce7f85a650a8278731251d2`. Only that table's whole-file upstream SHA is asserted here. The existing unscaled-Yarn diagnostic is preserved; private host resource diagnostics are omitted from this export.

## Outputs

`logit_dump --paths step,verify` used eight corpus windows, with the candidate reading the reference prefixes. All 392 finite rows are bitwise identical: prefill 8, step 192 and verify 192. Verify is the MTP cycle's three-row chain and exercises the changed normalized batched router. The report has zero top-1 flips, KL and maximum absolute difference. No floor exists for this model: the tool reports rather than applying a floor, and the stronger observed byte equality is stated directly.

Verbatim comparison report:

```text
prefill: 8 records (model-sushi-complete)
  reference: 0.2.31 (f22effb9), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (dc82cf0f), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
step: 192 records (model-sushi-complete)
  reference: 0.2.31 (f22effb9), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (dc82cf0f), prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
verify: 192 records (model-sushi-complete)
  reference: 0.2.31 (f22effb9), verify "MTP cycle's chain of 3 rows", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  candidate: 0.2.31 (dc82cf0f), verify "MTP cycle's chain of 3 rows", prefill_chunk 2048, ane {"chunks":0,"evals":0,"reruns":0}
  every row bitwise identical
no noise floor for this model and no bounds given: reported only (tools/logit_floor.py measures one)
```

The original default-path attempt failed on the reference because Flash-Next does not implement batch decode. Its state and failure log remain in `initial/`. The supported-path follow-up used identical source and binary hashes, in a new output directory. Unsupported batch coverage is not claimed.

Four A/B/A/B `ni_run` continuations each emit 256 nonempty token IDs with no end tokens; all four ID files are identical. Flags are `--dflash none --spec 0`, so these are single-row continuation controls, not speculative-router evidence. The three-row verify check above supplies affected-path output evidence.

## HTTP

Candidate and reference `tests/test_server.py` logs are byte-identical: 48 PASS and one shared FAIL, `tool result: final answer mentions 3  ''`. Both exit 1. This is a baseline-shared failure, not an all-pass result. Both servers reached health and were stopped by their runner.

## Native A/B/A/B

`kapitsa bench --model MODEL --quick --json --out FILE` started its own server on each arm. The optimization is selected by source, not an environment switch. Each run uses MTP depth 2, three 512-token sampled runs and three 512-token greedy runs, plus 4K prefill. Every JSON has `flags=[]`, `invalid=[]`, passing tool calls. The exported JSON and server logs retain all individual performance measurements.

| Run | Source | Sampled tok/s | Greedy tok/s | 4K prompt tokens | Cold TTFT s | Follow-up s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A1 | f22effb9 | 57.67 | 57.17 | 4130 | 4.8528 | 0.1791 |
| B1 | dc82cf0f | 55.20 | 58.60 | 4133 | 4.9011 | 0.1651 |
| A2 | f22effb9 | 57.97 | 57.80 | 4133 | 4.8538 | 0.1678 |
| B2 | dc82cf0f | 58.07 | 58.75 | 4135 | 4.8512 | 0.1748 |

The first sampled pair is 4.28% slower and the second is 0.17% faster; sampled acceptance shares and prefill token counts differ between runs. These measurements establish completed native checks, not a stable overall speedup or a universal no-regression result. The earlier two-chip kernel measurements and their control limitations remain separate evidence.

## Coverage and remaining qualification

Actual EXL3 model output/HTTP/native A/B/A/B completed on M3. Two-chip small-kernel validation, output comparisons and interleaved kernel timings remain in the earlier evidence, as do this PR's own 4B/35B controls and complete CPU checks.

Not verified: the Q2_0 streamed next-layer prediction model path, a second applicable full-model setup, and generated ISA/register use. Overall performance qualification for the whole class remains unresolved. No inference speedup is claimed. The complete affected-model and two-chip kernel evidence makes the PR available for review; the unrun qualification work is disclosed rather than described as completed.

`evidence.tar.gz` contains privacy-filtered derived states, reports, placeholder command examples/runners, IDs, model identity and the original failure outcomes. `SHA256.json` hashes every member. Raw float dumps were hashed and removed only after comparison. The measurement data is unchanged; this declared privacy export replaces the archived environment records. No jobs were restarted to collect these results.

## Privacy export erratum

This archive and its checksum manifest replace the original public export to remove private machine paths, usernames and resource records. The replacement is explicitly a privacy-filtered derived export, not a verbatim raw log. Modified commands contain `$PRIVATE_PATH` or other explicit placeholders and are reproduction examples. Original archive bytes and their SHA256 remain privately preserved for audit. Timing values, output results, pass/fail records and tested source/binary identities are unchanged; no test was rerun for this export. Kernel static allocation measurements remain intact.
