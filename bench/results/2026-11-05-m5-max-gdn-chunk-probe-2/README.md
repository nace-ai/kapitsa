# Results, 2026-11-05: the chunked DeltaNet scan, second round, and the 27B's GPU overlap (M5 Max)

Two prefill ideas, measured and not shipped, written down so the next attempt starts from the numbers.

MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), on AC power, under the GPU
lock.

## The chunked DeltaNet scan on staged tiles

The first probe ([2026-11-04-m5-max-gdn-chunk-probe](../2026-11-04-m5-max-gdn-chunk-probe/README.md)) wrote the
chunked gated delta rule as two kernels. It found the scan read each block's W, Q, K and P from device memory once for
each of a head's eight 16-row threadgroups. That came to about 1.2 GB a layer at 2,048 tokens.

`gdn_chunk_scan2_sg{4,8,16}` ([gdn-chunk2.diff](gdn-chunk2.diff), against main `b7dd4a9`, which includes the first
probe's kernels) removes those reads:
- one threadgroup holds 32, 64 or 128 rows of the head's state, transposed, in registers;
- each block's W, Q and K (32-column fp32 pieces) and P are staged once in threadgroup memory for all its
  simdgroups;
- each simdgroup computes its own 8 rows of U, O and the state update.

`bench_gdn T Hv Hk` with `GDN_CHUNK=1 GDN_SCAN2=1`, against `gdn_step_parr4_l16` from the same state:

| Shape (T 2048) | `gdn_chunk_scan2` sg4 / sg8 / sg16 | First probe's `gdn_chunk_scan` | `gdn_step_parr4_l16` |
| --- | ---: | ---: | ---: |
| 35B-A3B (Hv 32, Hk 16) | 4.130 / 3.146 / 2.772 ms | 1.445 ms | **1.208 ms** |
| 27B (Hv 48, Hk 16) | 7.038 / 5.773 / 5.759 ms | 1.642 ms | **1.677 ms** |

The new kernel is correct: the worst difference is 1.6e-4 of the largest reference value, a half step of `y`, the same
as the first probe ([bench_gdn-2048-32-16.out](bench_gdn-2048-32-16.out),
[bench_gdn-2048-48-16.out](bench_gdn-2048-48-16.out)). But it is 2.3x slower than the first probe.

A block's work for a head is about 7.3 MFLOP of fp32 matrix products: U = Uv - W·Sᵀ, O = Γ·Q·Sᵀ + P·U, and the state
update. With one or two threadgroups a head it all runs on one or two cores: 87 µs a block at sg16. The recurrence
walks the same 64 tokens in 38 µs (0.59 µs a token). In fp32 the chunked form does more arithmetic than the recurrence
(about 6·T·dk·dv against 4·T·dk·dv) on the same ALUs. Spreading that over more threadgroups brings back the re-reads
the first probe had.

## The same products on the tensor units

`zz_gdn_tprobe` (in the diff) times only the scan's matrix products on the M5's matrix units:
- U = W·Sᵀ and Q·Sᵀ, 64 x 64 x 128 each;
- S += Uᵀ·K, 64 x 128 x 64.

The setup: half inputs, fp32 accumulators, S as half in threadgroup memory, one threadgroup per (head, 64-row slice),
four simdgroups of 32 x 32 tiles, block after block. It has no decays, no P·U and no prep kernel, and its outputs are
not checked ([zz_gdn_tprobe.out](zz_gdn_tprobe.out)):

| T, Hv | Threadgroups x blocks | Matrix products alone | `gdn_step_parr4_l16` (the whole scan) |
| --- | --- | ---: | ---: |
| 2048, 32 (35B-A3B) | 64 x 32 | 0.544 ms | 1.208 ms |
| 2048, 48 (27B) | 96 x 32 | 0.583 ms | 1.677 ms |
| 4096, 32 | 64 x 64 | 1.045 ms | (2.42 ms) |

A full scan adds to that:
- the prep kernel: K·Kᵀ, Q·Kᵀ, the triangular solve, and W and Uv; the first probe's fp32 prep took 1.2 ms;
- P·U and the decays;
- probably a second, low half of S for each S product, to keep the state's precision.

The estimate is 1.0-1.2 ms in all:
- **35B-A3B:** about level with the recurrence. Its 4K prefill also already overlaps the scan with other GEMMs
  (`NI_PF_WAVE`).
- **27B:** 0.3-0.5 ms under the recurrence, 1-2% of a 4K prefill. It would need the output check for half-precision
  state products.

Not built further.

## The 27B: its GPU FFN part beside the next mixer

In the 27B's Neural Engine wavefront (`enc_prefill_wave`), half B's GPU FFN part of layer i and half A's mixer of layer
i + 1 sit between the same two Neural Engine joins. [ane-wave-pair.diff](ane-wave-pair/ane-wave-pair.diff) (against the
`NI_PF_WAVE` branch, `97cdfda`) plays them side by side with that branch's Encoder tapes, on its own `xh` / `hh` pair:
the mixer's DeltaNet scan or attention beside the FFN's gate/up GEMM.

`ni_run qwen3.8-27b-4bit-mtp @<4,129 ids> 8 0 --pf-reps 2 --dflash none`, `NI_CHUNK=4096`, alternating off / on, 40 s
idle before each run ([ane-wave-pair/](ane-wave-pair)):

| | Off (`NI_ANE_WAVE_PAIR=0`) | On |
| --- | --- | --- |
| Rep 0 (ms) | 3,045.1, 3,053.2 | 3,032.4, 3,037.8 |
| Rep 1 (ms) | 3,110.2, 3,077.1 | 3,071.4, 3,048.6 |

That is -0.4% and -1.3%, inside the runs' spread, with the same tokens. The 27B's prefill waits on the Neural Engine
(busy 90-97% of a split prefill), so shortening the GPU's half of the wavefront gives little. Not shipped.
