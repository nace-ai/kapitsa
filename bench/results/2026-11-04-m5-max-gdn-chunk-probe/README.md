# Results, 2026-11-04: a chunked Gated DeltaNet prefill scan that did not pay (M5 Max)

The Qwen prefill's DeltaNet recurrence, `gdn_step_parr4_l16` (`src/kernels/gdn.metal`), walks a chunk's tokens in
order, one row of the state at a time. It takes 10.4% of the 35B-A3B's 4K prefill on the M5 Max: 1.24 ms a call,
60 calls.

The established way past that chain is the chunked gated delta rule. It works on blocks of 64 tokens: a WY / UT form
within a block, and the state carried from block to block, as in the Gated DeltaNet paper and its
flash-linear-attention kernels. This probe wrote that form in Metal, as two kernels:

- **`gdn_chunk_prep`:** every block and head at once. It computes the masked Q·Kᵀ and K·Kᵀ with fp32 simdgroup
  matrices, then W and Uv by forward substitution, one column per thread.
- **`gdn_chunk_scan`:** one head's 16-row slice of the state across the blocks in order. Per block it computes
  U = Uv - W·Sᵀ, then O = diag(Γ)·Q·Sᵀ + P·U, then S ← Γ_last·S + Uᵀ·diag(Γ_last/Γ)·K.

It runs correctly but slower, so it is not in the engine. The patch is here for whoever takes the idea further
([gdn-chunk.diff](gdn-chunk.diff), against main `bcfc692`, Kapitsa 0.2.28).

## Correctness

[chunked_reference.py](chunked_reference.py) checks the block formulas against the sequential recurrence in fp64: the
outputs agree to 9e-16 of their largest value, the state to 2.8e-16 (absolute).

`bench_gdn` with `GDN_CHUNK=1` (the patch's tool mode) runs `gdn_prep_par`, then the two kernels, then
`gdn_gnorm`, from the same state as `gdn_step_parr4_l16`. On the 35B-A3B's shape (T = 2048, 32 value heads, 16 key
heads), against `gdn_step_parr4_l16`:

| Output | Max difference, relative to max value |
| --- | ---: |
| `out` | 3.6e-6 |
| state | 1.3e-6 |
| y (half) | 1.6e-4 (one half step) |

That is rounding level, from another summation order ([bench_gdn-2048-32-16.out](bench_gdn-2048-32-16.out)).

T = 100 (a partial last block), 4 value heads and 2 key heads ran clean under
`MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. The static threadgroup memory is 16,384 B (prep) and 14,592 B (scan).

## Speed

MacBook Pro M5 Max (`Mac17,6`, 40-core GPU), macOS 26.4, AC power, best of 5 isolated dispatches:

| Shape | `gdn_chunk_prep` | `gdn_chunk_scan` | Chunked total | `gdn_step_parr4_l16` |
| --- | ---: | ---: | ---: | ---: |
| 35B-A3B (Hv 32, Hk 16, T 2048) | 1.202 ms | 1.438 ms | 2.640 ms | **1.207 ms** |
| 27B (Hv 48, Hk 16, T 2048) | 1.700 ms | 1.641 ms | 3.342 ms | **1.679 ms** |

Each of the two kernels alone takes about as long as the whole sequential step. Both are latency-bound:

- **Device reads:** their fp32 8×8 simdgroup matrices read W, Q, K and P straight from device memory.
- **The serial chain:** the forward substitution is a 2,016-FMA chain per column.

The sequential kernel's own chain (per token: decay, dot, a 16-lane reduction, update, dot) is already short at
0.6 µs a token.

## What would have to change

The chunked form would need to be about 4x faster than written here to pay:

- tiles staged in threadgroup memory with cooperative loads;
- the block's inverse by blocks (8×8 diagonal inverses and matmuls) instead of a 64-step substitution;
- the prep folded into the scan;
- probably the half-precision tensor units for the matmuls, which is a precision question of its own. The fp32 form
  above differs from the current kernel at rounding level; a half one would need the output check (CONTRIBUTING.md,
  "Output check") before anything else.

Until then, `gdn_step_parr4_l16` stays.
