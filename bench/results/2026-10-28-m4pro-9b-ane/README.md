# Experimental 9B ANE prefill: numerical evidence

This is an **opt-in, numerically different prefill path**, not a lossless optimization. It enables
Qwen3.5-9B's 4096-hidden FFN on the existing ANE split by using two 2048-channel inputs. The default
remains GPU-only for 9B. Set `NI_ANE_H4096=1` to opt in; `NI_ANE=0` overrides it.

**Provenance:** every GPU measurement in this directory ran the unmodified experimental commit
[`92e8b882b003971e6569fb58d1147b49e69c5d8a`](https://github.com/KeyingSong0123/kapitsa/commit/92e8b882b003971e6569fb58d1147b49e69c5d8a),
Kapitsa 0.2.21. These are historical measurements, not GPU verification of the submitted branch
rebased onto `909e282300704310a2be4878664103e1a9443cf9`. In particular, upstream changed ANE compiler-thread
lifetime in #706. The draft PR needs fresh GPU verification before merge.

Runs: MacBook Pro (14-inch, 2024), Apple M4 Pro, 20-core GPU, 48 GiB, macOS 26.4.1 (25E253),
2026-10-28; the same binary for both arms of every comparison. The experiment held the exclusive GPU
lock. No throughput benchmark ran in this three-model experiment.

## Models and controls

Each ANE arm is compared to **its own model's 4-bit, group-64 GPU arm**, not to another model's logits.

| Model | Hugging Face repository | Revision | Gate/up input geometry | ANE units |
| --- | --- | --- | --- | --- |
| 4B | lmstudio-community/Qwen3.5-4B-MLX-4bit | c43ee1d65576a5d98de1e8405cac93c371a655c1 | 1 x 2560 | 18/18 |
| 9B | lmstudio-community/Qwen3.5-9B-MLX-4bit | b455506b0f574c74616dbcd56879bde38fafcff3 | 2 x 2048 | 24/24 |
| 27B | mlx-community/Qwen3.8-27B-4bit | 3e6447f082e89cc7f0bc6e5441afd38dfce760ff | 2 x 2560 | 34/34 |

The full FFN shares match this M4 Pro's uncalibrated wavefront defaults. At these shares the GPU
still runs attention and the other model work; it keeps zero FFN intermediate channels.
All three tokenizer SHA256 values are
`06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`.
Model download checksums, token IDs and the corpus hash are preserved alongside the results.

Flags: `NI_ANE=0` (A) / `NI_ANE=1` (B), `NI_ANE_H4096=1`, `NI_ANE_WAVE=1`, `NI_ANE_ASYNC=0`,
`NI_ANE_MIN_ROWS=512`, `NI_ANE_VERIFY=1`, `NI_ANE_KEEP_S=0`, `NI_ANE_UNITS=18/24/34` as above,
`NI_BAL=0`, `NI_KFLASH=1`. No drafter. `logit_dump --paths step --batch 1 --chunk 2048`.
Exact commands, environment and timestamps for every leg are in `runs/*/*/command.json`.

- Primary: the first eight fixed code documents in `bench/corpus.txt`, prefixes 3200, 2816, 2432,
  2048, 1664, 1280, 896, 512; 96 teacher-forced step rows each, 768 total. A1, B1, A2, B2 per model.
- Long: two fixed corpus offsets at both 4096 and 16384 prompt tokens; 64 step rows per window,
  128 per context length, one A/B. The lengths share source text and are not independent corpora.
- PPL: each of the eight documents' first 1025 tokens, 1024 predictions per document, 8192 total;
  `ni_eval --chunk 1024 --skip 0`, no chat template, one A/B, reference top-k 256.
- Repeated runs are repeatability checks, not extra independent statistical samples.

## Why outputs change

With exact arithmetic, `W0*x0 + W1*x1 = W*x`. A 2048-wide segment does not need to be a multiple of
2560: the former fixed MIL/IOSurface width is now selected per supported hidden size. Both 2048 and
2560 fit the existing 128-channel input rotation blocks. Intermediate rotations remain 512-wide;
the down-projection segmentation remains at most 2560 channels. No channels are dropped.

The ANE path rotates and re-approximates the original quantized weights and quantizes activations,
with finite-precision intermediate operations. For orthogonal R and quantization errors ew, ex,
`delta = (Rw)^T*ex + ew^T*(Rx) + ew^T*ex + rounding`. Both existing 4B and 27B paths have this
approximation. The 9B change adds no unique quantization stage. Activations, outliers, nonlinearities,
layer sensitivity and accumulated KV/recurrent state differences determine final drift; widths alone
cannot establish a model ordering. Prefill state changes can alter subsequent decode tokens even
though the GPU decode algorithm is unchanged. The on/off comparison also includes wavefront's
internal chunking/reduction differences; it is not a causal ablation of 2048 versus 2560 alone.

## Metrics

For the same forced prefix, p is GPU and q is ANE. Full-vocabulary
`KL(p || q) = sum_v p(v) * ln(p(v)/q(v))`, in nats, measures distribution drift, not percent quality loss.
Top-1 flips count differing argmax tokens, with reference logit margins reported separately.
`NLL = -mean_i ln p(y_i | y_<i)` on observed next tokens; `PPL = exp(NLL)`.
`PPL_ANE / PPL_GPU = exp(NLL_ANE - NLL_GPU)`. Positive delta NLL worsens likelihood on this sample.
The PPL run's top-256-plus-rest KL combines the remaining vocabulary into one bucket and is a
**lower bound** on full-vocabulary KL, not interchangeable with the full-logit measurements.

## Primary: full vocabulary, 768 step rows

| Model | Mean KL | Median KL | p99 KL | Max KL | Top-1 flips | Mean delta NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | 0.000393959538 | 2.56204533e-05 | 0.00429257276 | 0.105012142 | 3/768 | -0.000962114149 |
| 9B | 0.000179526236 | 8.50607822e-06 | 0.00274450332 | 0.00969685713 | 3/768 | +0.00129706827 |
| 27B | 8.24867116e-05 | 1.83903198e-06 | 0.00134348345 | 0.00768846675 | 2/768 | -0.00106784973 |

The 9B/27B mean-KL ratio is **2.1764**, paired-document bootstrap 95% interval **[1.6852, 2.8201]**.
The 9B/4B ratio is **0.4557**, interval **[0.2422, 1.0208]**. The latter interval crosses 1.
These are 20,000 paired resamples of eight document clusters, seed 20261028. They describe uncertainty
within this small code sample, not task-wide generalization. Bootstrap PPL intervals use the rounded
per-document PPL values emitted by `ni_eval`.

Reference margins at the flips: 4B 0.327496 / 0.093308 / 0.001522; 9B 0.126949 / 0.045132 / 0.017811;
27B 0.007906 / 0.007515. The 9B flips are not all tiny ties. Exact positions, token IDs and margins
are in each `rows.json` and the unabridged `logit_compare.py` stdout reports.

For each model, A1/A2 and B1/B2 raw-logit SHA256 values match separately (prefill and step).
This establishes repeatability here, **not equality between A and B**: zero of the primary ANE
step rows are bitwise identical to their GPU references. The comparison used `--no-gate` for
descriptive measurements. No floor was edited, no `--reorder` exception used, and no gate pass claimed.

## Long contexts: full vocabulary

| Model | Context | Rows | Mean KL | p99 KL | Max KL | Flips | Mean delta NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | 4K | 128 | 0.00133884065 | 0.00822289174 | 0.136117142 | 2 | +0.0125689318 |
| 4B | 16K | 128 | 0.000701856906 | 0.00755601146 | 0.0374216247 | 0 | +0.00624139596 |
| 9B | 4K | 128 | 0.000371542561 | 0.00482780668 | 0.0213418379 | 0 | -0.00529949035 |
| 9B | 16K | 128 | 0.000223924687 | 0.00227292617 | 0.00282955822 | 0 | +0.00216910057 |
| 27B | 4K | 128 | 0.000127940267 | 0.00130998468 | 0.00442636045 | 0 | -0.000707691038 |
| 27B | 16K | 128 | 0.000109825812 | 0.00112277747 | 0.00126082946 | 0 | -0.000760583312 |

The 9B/27B mean-KL ratios are 2.904 at 4K and 2.039 at 16K. No 9B/27B top-1 flips occurred in
these 256 rows; equal argmax does not imply equal distributions. This is not a 16K speed result.

## PPL: 8192 observed-token positions per model

| Model | GPU PPL | ANE PPL | Relative change | Delta NLL | Relative change, document-bootstrap 95% |
| --- | ---: | ---: | ---: | ---: | --- |
| 4B | 2.26564 | 2.26824 | +0.11476% | +0.00114692064 | [+0.01978%, +0.20846%] |
| 9B | 2.01072 | 2.01052 | -0.00995% | -9.94718048e-05 | [-0.09754%, +0.07996%] |
| 27B | 1.44690 | 1.44765 | +0.05183% | +0.000518215278 | [-0.03062%, +0.14365%] |

The 9B interval crosses zero: no detected PPL deterioration on this sample, and no evidence of a
quality improvement. This can coexist with larger KL than 27B. These scoring positions differ from
the post-prefill step windows, so their delta NLL values must not be substituted for each other.
These references are the same 4-bit models, **not bf16 quality references**.

| Model | Top-256-plus-rest KL lower bound, mean | Top-1 agreement |
| --- | ---: | ---: |
| 4B | 0.00127376 | 99.182% |
| 9B | 0.000672844 | 99.475% |
| 27B | 0.00080445 | 99.585% |

Here 27B has larger aggregated KL than 9B; the ordering depends on positions and metric.

## Prefill endpoints (reported separately from step rows)

| Model | Sample | Rows | Mean KL | p99 KL | Max KL | Flips | Mean delta NLL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B | primary-1 | 8 | 0.000707470845 | 0.00396289045 | 0.00422816172 | 0 | -0.0474806134 |
| 4B | long-1 | 4 | 0.00199858912 | 0.00692267052 | 0.00712075327 | 0 | +0.0320938585 |
| 9B | primary-1 | 8 | 0.000478475173 | 0.00198935839 | 0.00206419645 | 0 | +7.50951293e-05 |
| 9B | long-1 | 4 | 0.000552963444 | 0.00193395452 | 0.00198841444 | 0 | -0.0144845692 |
| 27B | primary-1 | 8 | 0.000445988644 | 0.00246124246 | 0.00261033609 | 0 | +0.00924428571 |
| 27B | long-1 | 4 | 0.005537453 | 0.0210751522 | 0.0217134839 | 0 | -0.0254851727 |

## Validation and resource costs at the measured commit

- Small Metal geometry validation ran with **both** `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`:
  290 comparisons, 8,995,840 data/guard bytes, zero failures, widths 2560/4096/5120, offsets,
  tails and padding. This checks GPU packing/staging against full-width packing, not an entire ANE model.
- Every full-logit ANE leg recorded nonzero chunks/evals and zero reruns; GPU-only counts were zero.
  There were no nonfinite logits. All commands completed successfully. Complete comparison reports
  and ANE engagement metadata are preserved, including both A/B pairs.
- Startup's random FFN check: 4B about 2.11%, 9B 1.94%, 27B 0.75% relative L2. This local random-input
  check of the first two FFNs is not whole-model quality or a predictor of final KL ordering.
- The 9B first primary ANE leg logged **374.7 MB Metal allocation** and **31.7 s initialization**
  (31.511 s program compilation). This is added split storage, not a measured total peak footprint:
  ANE daemon/compiler memory, process peak, pressure and sustained memory effects were not measured.
  The GPU weight representation is retained. Compile-cache hits can change startup cost substantially.
- `api_test`, `fn_dflash_test`, CPU `ane_segment_check`, and `test_logit_compare.py` passed at 92e8b88.
  Current-branch CPU validation is recorded separately in the PR; it cannot validate runtime Metal/ANE.

## Earlier M4 mini experiment: summary-only provenance

The following is transcribed from the earlier experiment summary available when preparing this PR.
Its original per-leg timing records and telemetry were **not available in this checkout**, so it is
included for completeness, not as independently rechecked speed evidence or current-head validation.
It must be reproduced with every A/B leg before adopting a performance claim.

Runs: Mac mini (2024), M4, 10-core GPU, 24 GiB, macOS 26.5.2 (25F84); base `d40c852`, candidate `92e8b88`,
Qwen3.5-9B 4-bit. Custom native HTTP A/B, **not `kapitsa bench --quick`**. The source summary did not
preserve the exact model revision or complete flag set; those cannot be reconstructed from the figures.

| 4K request | Valid pairs | Loaded-model TTFT decrease | Paired geometric A/B ratio and reported 95% interval |
| --- | ---: | ---: | --- |
| Python scheduler | 3 | 63.51% | 2.740 [2.571, 2.920] |
| Chinese engineering explanation | 4 | 63.89% | 2.769 [2.634, 2.912] |

Approximate median TTFT: 17.3 s to 6.3 s after model/ANE preparation. First ANE compilation about
32.5 s was excluded. **Both 16K tasks had zero valid idle-GPU pairs**; the four-task overall
confirmation did not pass. No confirmed 16K speedup or steady-state decode-throughput improvement.

The same summary reports 2/192 top-1 flips on each full-logit decode path; all four Chinese answer
pairs differed. On a separate 4092-position quality sample, PPL was **6.41118 to 6.41682 (+0.088%)**.
This less favorable PPL result must not be discarded in favor of the newer code-only result. Its
reference provenance is insufficient here to claim a bf16 quality gate. The summary also reports
bitwise full-logit equality for 4B's default and 9B's opt-out comparisons; the raw reports are not
included, so fresh base-versus-head regressions are still needed. These statements are historical,
not proof of opt-out equivalence after rebase.

## Reproduction and evidence inventory

[`protocol.md`](protocol.md) was fixed before GPU measurements. [`inputs/manifest.json`](inputs/manifest.json)
identifies the frozen windows. [`summary.json`](summary.json) includes all aggregates, per-document
metrics and intervals. `runs/{4b,9b,27b}/` contains every command, stdout/stderr, metadata, sequences,
record keys, raw-logit SHA256, comparison JSON and per-row metrics; `*-comparison/stdout.log` preserves
the full `logit_compare.py` reports. `runs/geometry-validation/` contains the layered GPU check.

`publication.json` records original and published checksums. Only local paths were normalized to
`$CHECKOUT`, `$EVIDENCE`, `$MODEL_4B`, `$MODEL_9B`, `$MODEL_27B`, `$HOME`; the protocol's local-service
instruction was omitted. Numerical values and token IDs are unchanged. Huge raw logits were removed
after comparison; binary top-k references and private host/process inventories are not published.
The original runner is parameterized as `reproduce.py`; the aggregation algorithm is `summarize.py`.
These use NumPy (the measured analysis used 2.0.2), an analysis dependency, not an engine dependency.

To reproduce with the measured code, use a separate checkout at 92e8b88, build `kapitsa logit_dump
ane_segment_check ni_eval`, and provide model folders from the pinned revisions. Set `RESULTS` to
this evidence directory, `CHECKOUT` to that checkout, and `OUT` to a fresh, nonexistent output path.
A new-head run must use a distinct output directory and be labeled with its actual commit.

```sh
# Run from CHECKOUT; this takes the GPU lock once and sets NI_LOCK_HELD=1 for children.
.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 45 --label '9B ANE numerical comparison' -- \
  python3 "$RESULTS/reproduce.py" --checkout "$CHECKOUT" --output "$OUT" \
  --model-4b "$MODEL_4B" --model-9b "$MODEL_9B" --model-27b "$MODEL_27B"
python3 "$RESULTS/summarize.py" "$OUT" --output "$OUT/summary.json"
# CPU-only: recompute published aggregates from preserved row metrics, into a fresh file.
python3 "$RESULTS/summarize.py" "$RESULTS" --output /tmp/ane-summary-recomputed.json
```

The runner refuses to overwrite an output folder and does not stop existing services. All model
commands require the externally held lock. It validates ANE engagement and rejects fallback runs.
The parameterized wrapper itself was syntax-checked; the historical measurements used its original
fixed-path form, not a rerun of this public wrapper.

## Not verified / limits

Submitted rebased head on GPU; current base-versus-head logit gates (including default/opt-out paths);
new-head server/API, disk-cache isolation and ANE abort/fallback integration; partial FFN shares;
async startup and resource-pressure behavior; M5 or other chips; MoE and other hidden-4096 models;
`kapitsa bench --quick`; a valid 16K speed A/B; bf16 quality references and task accuracy/reasoning
benchmarks; broad language/task coverage; contexts beyond 16K. No model noise floor was changed.
These measurements support evaluating an explicit approximation, not changing a default or claiming
universal quality equivalence.
