# Dory shared split weights: validation snapshot

Source: `b3bece8931c78885e682bfb8da771598d721093a`; base: `bbd84e0d53be75866087aa615bdeb6f2411f393f`. M3 Ultra, 60 GPU cores, 96 GiB, macOS 26.4.1. Release build, explicit CMake source label.

The new kernel broadcasts split weights while each output dimension retains its original ordered sums. The host selects it for 16–64 splits. The original short reduction kernel remains unchanged.

`RESULTS.json` includes every one of 396 timing rows (33 cells, six A/B pairs per cell), not just winning cells. Synthetic partial buffers only: these are reduction timings, not whole attention or inference timings. Short cells still call the original production kernel; their timings against the separately compiled ordered reference are controls, not a claimed optimization.

The production-source checker passed 120 cases with both Metal validation layers, then 180 normal cases. It checks bitwise finite outputs, NaN classification for empty rows, FP64 error and output guards. Run `build/dory_reduce_check --gpu --small` with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, then `--gpu` and `--bench`; the tool takes the GPU lock.

This snapshot contains 11 completed current-source CPU checks; the full suite is still running. Model output gates, native HTTP/bench and a second device remain pending. Do not treat the prior iteration's CPU checks or component measurements as this source's qualification. No full-model speedup is claimed.

`validation.tar.gz` retains current transcripts/scripts plus the separately named prior-iteration evidence that motivated keeping two kernels. `MANIFEST.json` inside hashes each archived file. Future results will be added under a new path.
