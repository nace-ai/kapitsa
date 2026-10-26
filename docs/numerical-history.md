# Historical numerical comparisons

These records were moved from AGENTS.md without new measurements. They describe the evidence and limits of
previous kernel changes, not permission to bypass the current output gate or change another default.
The contributing rules remain in [CONTRIBUTING.md](../CONTRIBUTING.md#output-check) and
[AGENTS.md](../AGENTS.md#rules-that-bite-here).

The 35B comparisons use an M4 Pro floor measured before the M5 tensor kernels. Flash-Next's sparse tensor
attention was reverted to opt-in after a 64K regression. This documentation cleanup changes no defaults or floors.

## 35B-A3B routed-expert prefill

The 35B-A3B MoE prefill GEMM on the tensor units became the M5 default on 2026-10-22. It keeps the same
half-domain products and reassociates their fp32 K reduction. The per-GEMM comparison against a double-precision
reference and the cold 4K and 16K temperature-0 continuations supported the change. Those continuations were
identical. The reassociation checks are described in AGENTS.md and CONTRIBUTING.md.

## 35B-A3B prefill attention

`qattn_t_g8s4k16` became the 35B's M5 default on 2026-10-22. `NI_QATTN_G8=0` keeps `attn_prefill11r`.
The comparison passes `--reorder`, with prefill mean KL 3.8e-4 and decode mean KL 5.6e-4, but reaches 1.10x
the floor's gate. The floor is an M4 Pro measurement of the kernels before these changes, not a measurement
of this M5 path. Re-measuring it with `tools/logit_floor.py` requires a separate PR.

## 35B-A3B split-head prefill attention

`qattn_t_g8d2s8` became the 35B's M5 default on 2026-10-23 in #365. `NI_QATTN_D2=0` keeps `qattn_t_g8s4k16`.
It splits the head dimension over two simdgroups and only reassociates the score sum over two halves of 128.
Against `qattn_t_g8s4k16`, it passes `--reorder` with prefill mean KL 1.1e-4, decode mean KL 6.2-6.5e-4,
and one near-tie flip in 192 rows. It reaches 1.27x the floor's gate.

The drift was dumped on the 4-bit pack only. The 35B's 5/6/8-bit and bf16 packs take the same attention path,
which does not read the weights, but were not dumped.

## Flash-Next 6-bit dense prefill GEMMs

`qmt6_m64_n64`, in `src/kernels/tensor/qmt6.metal`, became the M5 default for Flash-Next q20's 6-bit group-64
dense prefill GEMMs on 2026-10-24 in #510. `NI_QMT6=0` keeps `qrf_f6g64_m32_n128`.
Its f6g64 weight decode mirrors the simdgroup path's half-domain affine bitwise. The dequantized halfs and their
products are bitwise the shipped path's; only each output's K reduction is reassociated.

It fails the floor file's `reorder_gate` on `max_nll_change` by 1.09x. Prefill NLL change is 3.809e-2 against
0.0349. The other bounds pass, with 0 top-1 flips and agreement 1.0. A relaxed-bounds rerun with
`--max-nll-change 0.1` passes, but that is not a pass against the recorded gate. The exceedance is exactly 2x
the calibration source's recorded 8-record prefill drift.

Greedy continuations were bitwise identical on all four interleaved ABBA legs. The 4K ids-md5 `2ba67d51…`
holds on all four, and the 85-token decode dump `ef97e361…` holds on both gate arms. The floor file has not
been re-measured with this path enabled. A measurement of at least 192 prefill records would provide broader coverage.

The switch covers every 6-bit group-64 dense pack in Qwen3.5/3.6/3.8. The drift dumps and speed legs cover
only Flash-Next q20. The other packs share the half-domain decode and reassociation, but were not measured individually.

## Flash-Next sparse prefill attention

`qattn_t_sparse_g12d2`, in `src/kernels/tensor/qsattn_t.metal`, became default on an Apple10 Mac running
macOS 26 on 2026-10-25 in #570. `NI_QSA_TATTN=0` keeps the four-dispatch `attn_sparse_rows` chain that #561
added as opt-in. Unlike a reassociation, this path changes what is computed. It rounds Q, K and the
score-to-weight product into half rather than running the fp32 simdgroup chain. It therefore stays on the
floor's gate. The flash-next-q20 floor is an all-zero bitwise measurement, so the change fails that gate.

Recorded drift uses `models/flash-next-q20-full`, with the full PLE table, chunk 2048, against Kapitsa's own
reference with the option off:

| Comparison | Rows | Absolute NLL change | Mean KL | p99 | Top-1 flips |
| --- | --- | --- | --- | --- | --- |
| Tensor path against Kapitsa reference | 96 long rows, 9K to 24.6K | 1.340e-2 | 4.353e-3 | 6.312e-2 | 3 of 96 |
| Splash 1.4.0 against the same reference | Same rows | 4.4918e-1 | 7.196e-1 | 1.003e+1 | 13 |

Every measured cell is at or below the cross-engine reference's drift. This is evidence of cross-engine drift,
not a pass against Kapitsa's floor gate. The four dumps' `records.txt` and `sequences.txt` hashes match:
`b65170880903` and `c1761547e85a` for long rows, `0b5d61b94a22` and `c1bf15d26cd3` for short rows.
The prefixes are identical. The 192 short rows at or below 3.2K are bitwise identical between the two arms,
with `logits.f32` sha256 `a5d294b0…` on both.

The path requires `__builtin_available(macOS 26.0, *)`, Apple GPU family at least 10, `G == 12` and head dim
256. Engagement is logged once per process. A GPU below Apple family 10 or macOS before 26 does not engage
the path because `qmt_on()` is false. These measurements do not cover other GPUs. A floor measurement with
this path enabled requires a separate PR.

The default was turned off again on 2026-10-25 after [#585](https://github.com/nace-ai/kapitsa/issues/585).
On a 64K prompt, the path moved the first decode row from ` of`, probability 0.77 with the option off, to
`<|im_end|>`, probability 0.97, and the reply looped. The path is now opt-in with `NI_QSA_TATTN=1`.

## Qwen 4B and 9B prefill attention

`qattn_t_g4d2s8`, the G = 4 instantiation of `qattn_t_d2_impl` in `src/kernels/tensor/qattn_t.metal`, became
the dense 4B and 9B default on an Apple10 GPU running macOS 26 on 2026-10-25 in #581.
`NI_QATTN_G4D2=0` keeps `qattn_t_s8k16`. One simdgroup holds half of Q and half of the output accumulators;
the two halves' score sums meet through threadgroup memory. The head-dimension products are unchanged and
only their order changes, followed by the same softmax and P·V. This is the `--reorder` class. Chunks of
128 rows or fewer keep the short-prefill kernels.

The 4B passes its floor file's gate, but the file is an M4 Pro measurement from build 0.1.39 on 2026-10-21,
not a measurement of the M5 tensor kernels, which the M4 Pro cannot run. It is conservative rather than a bound
on the M5 Pro's noise. The 4B also
uses the Neural Engine split, whose int8 re-quantization of each chunk's attention output can turn a last-bit
change at a quantization boundary into a whole quantum of drift.

Recorded drift uses `qwen3.5-4b-4bit-mtp` on a MacBook Pro, 16-inch, 2026, M5 Pro with 20-core GPU, 48 GB,
macOS 26.5.2, on 2026-10-25. Both arms use binary `0.2.17 (ffd00dad)`, with `NI_QATTN_G4D2=0` against `=1`,
chunk 2048, and 192 rows per decode path:

| Path | Mean KL | p99 | Top-1 flips | Top-1 agreement |
| --- | --- | --- | --- | --- |
| Prefill | 5.134e-4 | 2.302e-3 | 0 of 8 | 1.0 |
| Step | 4.123e-4 | 7.457e-3 | 1 | 0.9948 |
| Verify | 4.330e-4 | 7.587e-3 | 1 | 0.9948 |
| Batch | 4.038e-4 | 7.322e-3 | 2 | 0.989583, 190 of 192 |

Prefill median KL is 2.230e-4 and maximum is 2.413e-3; no prefill row is bitwise identical. Every flip has a
top-2 margin below 0.0041. All cells are inside `tools/logit_floors/qwen3.5-4b-4bit.json`'s gate: mean KL at
most 1.87e-3, p99 at most 1.33e-2, top-1 agreement at least 0.989583 and no far flips. Batch agreement is
exactly at the minimum. The gate's `set_by` is the Neural Engine split's noise on the M4 Pro.

The decode rows use the prefill-scale gate, not the file's `decode_gate` of 1.02e-6 for a decode-only change.
The changed prefill produces the KV inherited by decode. With `NI_ANE=0`, the same switch moves the 4B's
decode paths at the 1e-7 scale.

The 9B has hidden size 4096, no ANE split and no floor file of its own. The PR measured it on an M5 Max:
prefill mean KL 1.4e-7, decode mean KL 2.6e-7 to 5.8e-7, 0 flips in 192 rows. Applied for scale, the 4B's
gate passes by four orders of magnitude. The M5 Pro pair shows the same order of magnitude with the split
off. This is not a measured 9B noise floor. Re-measuring the 4B floor with this path enabled requires a separate PR.

## Public change records

- 35B split-head attention: [#365](https://github.com/nace-ai/kapitsa/pull/365).
- Flash-Next dense GEMMs: [#510](https://github.com/nace-ai/kapitsa/pull/510),
  numerical record [#532](https://github.com/nace-ai/kapitsa/pull/532).
- Flash-Next sparse attention: [#570](https://github.com/nace-ai/kapitsa/pull/570),
  numerical record [#578](https://github.com/nace-ai/kapitsa/pull/578),
  regression [#585](https://github.com/nace-ai/kapitsa/issues/585),
  default-off revert [#586](https://github.com/nace-ai/kapitsa/pull/586).
- Qwen 4B and 9B attention: [#581](https://github.com/nace-ai/kapitsa/pull/581),
  numerical record [#583](https://github.com/nace-ai/kapitsa/pull/583).
