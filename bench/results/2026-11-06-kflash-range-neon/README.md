# CPU affine range scan: initial evidence

## What changed and why

The existing CPU affine quantizer scans each 64-value BF16 group with a serial float min/max dependency chain. Widen four BF16 values and scan four lanes at a time with NEON, leaving scale/bias calculation, quantization, rounding and packing unchanged. Groups containing nonfinite values or a zero extremum use the original scalar scan to preserve NaN and signed-zero behavior.

This is a Draft while complete validation is pending. The first prototype disagreed on signaling NaN; the submitted fallback passes the expanded checks below.

## Default for

Not a new feature: replaces the range scan in the existing `dfl_quantize_cpu` for all its current 4/8-bit callers, on every supported Apple-silicon Mac. No new path selector, flag or device gate.

## Measured

Runs: local M4 Pro, 48 GiB, macOS 15.7.1, Apple Clang 17, C++20 `-O3 -mmacosx-version-min=14`; reference `d65a0afa`, implementation `279ce236`. Real BF16 drafter tensors for Qwen3.5 4B and 9B, four interleaved A/B pairs per tensor/row count/bit width/rounding mode.

Initial CPU screening, full-matrix default rounding:

| Drafter / tensor | 4-bit old/new ms | 8-bit old/new ms |
| --- | ---: | ---: |
| 4B fc | 5.146 / 2.687 | 5.634 / 2.906 |
| 4B down | 2.517 / 1.306 | 2.578 / 1.374 |
| 4B gate | 2.421 / 1.352 | 2.649 / 1.436 |
| 9B fc | 13.144 / 6.907 | 14.292 / 7.353 |
| 9B down | 4.944 / 2.604 | 5.666 / 2.979 |
| 9B gate | 5.141 / 2.826 | 5.541 / 2.993 |

All individual readings, other row counts and both rounding modes are retained in `bench/results/2026-11-06-kflash-range-neon/`. These are CPU-function times, with no thermal/quiet capture, so initial screening only. No model-load or inference speedup is claimed.

## Checked

- 1,600 shape/pattern cases: entire W/S/B outputs identical to the previous function.
- 2,161,664 edge comparisons: every BF16 bit pattern as a constant and in mixed groups, opposite signed zeros at every pair of lanes, all four CPU rounding modes, 4/8-bit and both quantizer rounding modes. Entire W/S/B outputs identical.
- 72 real-tensor cases across two drafters, including 1 row, 8 rows and complete matrices: entire W/S/B outputs identical; SHA256s retained.
- Standalone CPU checks compile the functions extracted from the actual revisions with identical flags. Local Release CLI build and initial `prek run --all-files` passed.

## Not verified

Not verified on a GPU. Full CPU regression and supplemental checks are running locally; they are not yet claimed as passed. The remote Mac is occupied by another test job and is left undisturbed.

Before Ready: finish full CPU/prek; rerun interleaved CPU measurements with recorded conditions on the remote Mac when available; compare actual 4/8-bit drafter buffers and full-model logits, HTTP regression and interleaved model benchmarks against the same configured reference. Keep new results in additional files. Other models and Macs remain unverified.

## Reproduce

```sh
python3 bench/results/2026-11-06-kflash-range-neon/reproduce.py . d65a0afa 279ce236 /path/to/dflash2/model.safetensors
```

Omit the final argument to run the shape and exhaustive edge checks only. These extracted CPU functions do not load a model or touch the GPU.
