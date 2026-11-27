# Lightning attention projections in 8-bit (`NI_NH_ATTN_Q8`): speed A/B

MacBook Pro M5 Pro 48 GB, macOS 26.5.2, Nemotron 3.5 Lightning NVFP4, main 2fe7c3fa, 7 Nov 2026.

`nhq8_ab.sh`: 8 interleaved legs, default (P) and `NI_NH_ATTN_Q8=1` (Q), in the order P Q Q P Q P P Q. Each leg starts
its own `kapitsa serve` with the arm's environment (bench strips `NI_` variables from a server it starts; AGENTS.md)
and runs `kapitsa bench --quick --json` against it. The servers' load lines confirm the switch: P loads BF16
attention, Q loads Q8_0 attention (`legs/load-P.txt`, `legs/load-Q.txt`). No leg swapped; load stayed under 2.

| Median over 4 legs (range) | default | 8-bit attention | |
|---|---|---|---|
| Decode T 0.6, tok/s | 162.9 (160.8-165.0) | 164.9 (160.6-173.4) | about the same (+1%) |
| Decode T 0, tok/s | 171.1 (161.9-178.3) | 164.4 (160.2-173.4) | about the same (-4%, ranges overlap) |
| Prefill 4K, tok/s | 3,286 (3,280-3,297) | 3,011 (3,008-3,016) | 8% slower |

`legs/aa-*.json` is an earlier run of the same 8 legs through bench's own server, where both arms ran the default:
decode medians 160.5 against 158.6 (T 0.6) and 169.8 against 171.0 (T 0), with ranges about as wide as above.

So the 8-bit attention projections do not make decoding faster on this Mac, and they make 4K prefill 8% slower. They
are not a fast-profile member.
