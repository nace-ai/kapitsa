# Vision BF16 direct read: initial evidence

## What changed and why

`VitWeights::read_bf16` allocates a whole-tensor byte vector, reads into it, then copies it into the final buffer even when the tensor is already BF16 and its rows need no padding. For that case, read directly into the existing destination through `WeightStore::read_deferred`.

Five production lines remove the temporary whole-tensor allocation and copy. Padded BF16 rows and F16/F32 conversion keep the existing path. The loader's bounded uncached-read staging remains in use.

## Default for

Not a new feature: the same vision-weight reader on all supported Macs and image towers. The shortcut applies to BF16 with `k == ld`; no device/model gate or user-facing switch.

## Measured

Runs: local M4 Pro, 48 GiB, macOS 15.7.1, Apple Clang 17, C++20 `-O3 -mmacosx-version-min=14`; reference `d65a0afa`, implementation `ae9ca905`.

A separate process reads a 64 MiB BF16 test tensor into a pre-touched, guarded destination. Peak RSS in two A/B pairs:

| Run | Previous bytes | Direct-read bytes |
| --- | ---: | ---: |
| 1 | 206,356,480 | 139,247,616 |
| 2 | 206,356,480 | 139,214,848 |

This saves approximately the tensor's 64 MiB temporary allocation. It is a reader-level memory measurement, not full-model peak memory. All raw read times are retained but are not used to claim speed: cache state was not controlled, and the later real-weight checks overlapped local build work.

## Checked

- 42 actual `VitWeights::read_bf16` checks covering BF16/F16/F32, padded/unpadded rows, empty tensors, all 65,536 BF16 bit patterns, output guards and zero padding. Every output byte checked against independently constructed expected bytes.
- Six real BF16 matrices from the Qwen3.5 4B and 9B image towers, both direct and padded paths: 36 reads, complete output bytes equal to the original payload plus expected padding. Per-arm SHA256s and every run are retained.
- Initial local `prek run --all-files` passed.

Initial CPU probes compile each actual `vit_common.cpp` against the same existing support library (built at `96d2de68`; its loader code is unchanged through `d65a0afa`). They close the weight store and exit before `VitWeights::release`, which would initialize Metal even without owned buffers. No GPU or full image encoder was run by these probes. The reproduction instructions use full builds of both revisions; complete local builds/regressions are in progress.

## Not verified

Not verified on a GPU. Full CPU regression, complete image-encoder output comparison and model-level memory/performance measurements are pending. The remote Mac is occupied by another test job and is left undisturbed.

Before Ready: complete full Release builds/CPU/prek; repeat the probe against those libraries; run old/new complete image towers on fixed images and compare every output, including reloading the tower; then run the applicable HTTP/logit and interleaved model checks under the GPU lock. Add new evidence without modifying the initial records. Other image towers and Macs have not been checked.

## Reproduce a reader probe

Build each revision in a separate checkout with CMake Release. Then link `vision-probe.cpp` against each build’s `libnaceinfer.a` (set `TREE` to the old or new checkout and `PROBE` to this probe's absolute source path):

```sh
cmake -S "$TREE" -B "$TREE/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$TREE/build" -j4 --target naceinfer
clang++ -std=c++20 -O3 -mmacosx-version-min=14 \
  -I"$TREE/src" -I"$TREE/include" -I"$TREE/third_party" "$PROBE" \
  "$TREE/build/libnaceinfer.a" -framework Metal -framework Foundation \
  -framework CoreFoundation -framework IOSurface -framework IOKit \
  -framework ImageIO -framework CoreGraphics -framework Security \
  -framework SystemConfiguration -framework Accelerate -o "$TREE/build/vision-probe"
```

Run:

```sh
PROBE_OUT=old.bin /path/to/old-probe MODEL_DIR TENSOR_NAME ROWS K LD
PROBE_OUT=new.bin /path/to/new-probe MODEL_DIR TENSOR_NAME ROWS K LD
cmp old.bin new.bin
```

The probe prints peak RSS, time, dimensions, guard/padding status and an FNV checksum. `PROBE_OUT` supplies the complete bytes for comparison; the checksum alone is not the check. Use separate processes and retain every run. `vision-initial.json` describes synthetic fixtures; `vision-real.json` names the actual tensors and payload hashes. This CPU reader probe deliberately exits before initializing a Metal device in the otherwise-empty weights destructor.
