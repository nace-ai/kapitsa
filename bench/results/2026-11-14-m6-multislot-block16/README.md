# The 16-row drafting block on multi-slot servers (matrix class), and what else was tried on a Mac mini M6

`DFlash2::plan_block_rows` gave a server of more than one slot the 8-row drafting block, so that requests decoding at
the same time can share a batched pass. `kapitsa serve` without `--slots` takes 2 slots for the 27B on a 32 GB Mac, so
the usual way to run it drafted 8 rows even with one conversation, which is what an agent client (here Hermes, one
session at a time) mostly sends. On the matrix class (the device profile's new `kflash_block16` row), a multi-slot
server now drafts 16 rows, as one slot already does; overlapping requests take turns (`Engine::batch_rows`).
`NI_KFLASH_BLOCK=8` keeps the old behavior.

Two guards come with it, for a drafter that can narrow (`NI_KFLASH_NARROW`, on by default on the 40-core M5 Max): a
request that resumes after waiting its turn sets the drafter back to its own block (another request may have narrowed
it to 8 meanwhile), and `batch_rows` keys on the block the drafter was loaded with, so a narrowed drafter does not pull
16-row requests into an 8-row batch. Before this change a multi-slot server never loaded a 16-row drafter, so neither
case could arise.

Runs: Mac mini (M6, 2026, `Mac18,5`), 12 CPU / 12 GPU cores, 32 GB, macOS 27.2.1 (26B131). Both binaries carry the
M6's Neural Engine share (nace-ai/kapitsa#1259, not yet on main): main alone reads the 27B's prompts on the GPU here,
which moves the prefill columns but not this change's decode effect. The Neural Engine
calibration `ane_calib --wave` wrote on this Mac (27B 31 of 34 units, 4B 18 of 18), read by every arm. Model:
`qwen3.8-27b:kquant` with its DFlash2 drafter. Each arm: `kapitsa serve --model M --port P --disk-cache 0` (plus the
arm's flags), the opencode agent session replayed against it (`bench/harness/agent_replay.py`, 19 requests, prompts up
to 22.8K tokens, `--gaps none`, temperature 0), then `kapitsa bench --quick` against the same server; each arm under
`lockf -k /tmp/kapitsa-gpu.lock`, one at a time. `arms.json` holds every arm's numbers. Thermal state nominal in every
arm; other GPU users 1.5-2.5% (the Claude app).

## The change: new default against `NI_KFLASH_BLOCK=8`, same binary, serve's default 2 slots

| Arm | replay | prefill | decode | writing | bench 4K prompt | bench greedy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NI_KFLASH_BLOCK=8` | 153.5 s | 66.8 s | 86.8 s | 37.93 tok/s | 462 tok/s | 42.7 tok/s |
| new default | 139.0 s | 65.5 s | 73.5 s | 45.76 tok/s | 450 tok/s | 51.1 tok/s |
| `NI_KFLASH_BLOCK=8` | 152.4 s | 66.1 s | 86.3 s | 38.14 tok/s | 460 tok/s | 42.1 tok/s |
| new default | 141.1 s | 67.6 s | 73.5 s | 45.76 tok/s | 460 tok/s | 54.7 tok/s |

The session -8.4% (153.0 -> 140.1 s), writing +20% (38.0 -> 45.8 tok/s), the bench's greedy writing +25%; prompt
reading unchanged. Earlier, on the parent binary (#1259's branch, `m6-ane-builtin-share`), the same comparison as an environment
switch: default 159.1 / 160.4 s and 35.9 / 36.5 tok/s against `NI_KFLASH_BLOCK=16` 147.3 / 148.0 s and 44.4 / 44.1
tok/s; one slot (`--slots 1`) 140.4 / 141.5 s and 45.3 / 45.5 tok/s, the speed the new default reaches with 2 slots.

**Two sessions at once** (two replays started together against one server, the case batching is for), parent binary:
default (batched 8-row) 295.1 / 293.5 s for both, `NI_KFLASH_BLOCK=16` (turns) 279.2 / 278.7 s (-5.2%). Taking turns
is less even: one session's median follow-up time to first token was 2.83 s and the other's 1.57 s, against 1.64-2.21 s
for both with batching.

**Memory:** the 2 slots' drafter rings take 0.46 GB instead of 0.25 GB; the slot count serve picks did not change (2).

**Output check** (`output-check.txt`): `logit_dump` with the 8-row verify against `--verify-rows 16` on the 27B: every
prefill and `step` row bitwise identical; `verify` 0 flips in 256 rows, KL mean 1.2e-7, max 2.5e-6; PASS against
`qwen3.8-27b-4bit.json` (no KQuant floor exists; same kernels). The replays' greedy answers: 17 of 19 turns identical in
every 27B arm; turns 13 and 18 differ between runs of the old default as well (near-ties).

**Other models:** the 4B's catalog drafter cannot draft 16 rows (not a DFlash2-v2 drafter with 4-bit group-64
projections), so it keeps 8: the new binary's 4B replay 39.6 s / 107.4 tok/s against the parent's 39.2 s / 108.3 tok/s.
A MoE target keeps 8 as before. The narrowing guards ran once, as a stress run: two sessions at once with
`NI_KFLASH_NARROW=1` (prompts above 16,384 tokens narrow to 8 rows, the others draft 16): both completed, every turn's
answer identical to the 16-row runs. That run is slower (303.3 s): narrowing is tuned for the 40-core GPU and is off on
the M6.

Not run: any other Mac of the class (M5, M5 Pro, M5 Max, M6 Pro/Max), so the default for the class rests on this one
Mac and one model; the M5 Max's narrowing with several slots beyond the stress run above; three or more sessions at
once.

## What else was tried on this Mac (no change in this branch)

Single runs unless noted; noise between identical arms is about +-2% on the session and +-1 tok/s on writing (four
default arms: 153.7-160.4 s, 35.8-38.1 tok/s).

| Knob, on serve's default (8-row) | session | writing | verdict |
| --- | ---: | ---: | --- |
| default (4 runs) | 153.7-160.4 s | 35.8-38.1 tok/s | |
| `NI_V8_TREE_G6=1` (the wide class's 8-row tree path) | 152.6 s | 39.1 | within noise |
| `NI_KFLASH_DFS`, `NI_V8_TREE_FUSE` + `GDNSPLIT=128`, `NI_GDN_LAZY`, `NI_KFLASH_LOWRANK=768`, `NI_KFLASH_ATTN_MINC=128`, `NI_KFLASH_ROWS=8,16,32` (the M3/M4 class's small-GPU knobs) | 156.9-158.6 s | 36.8-37.5 | within noise |
| `NI_KV_I8=1` | 157.5 s | 36.5 (bench greedy 36.9) | slower: it turns off the matrix-unit verify attention; only a memory win |
| `NI_KFLASH_HIST=3,8,4` (history lane) | 160.6 s | 35.3 | no gain |

On one slot (16 rows): the history lane 146.3 s / 43.2 tok/s, `NI_GDN_LAZY` 141.5 s / 45.5, `NI_KFLASH_LOWRANK=768`
141.9 s / 45.2, against 140.1-141.5 s / 45.1-45.5 without: no gain. `--prefill-chunk 4096`: the bench's 4K prompt
406 tok/s against ~455, so 2048 stays right here.

**The M6's two Neural Engines.** The I/O registry lists two 16-core devices, `H11ANE` (role ANE) and `H11ANE1` (role
ANE1), both `BondedModeSupported=1`. `tools/ane_dual_probe` ([a probe on the measuring branch](https://github.com/supervised-nl/kapitsa/blob/m6-research/tools/ane_dual_probe.cpp), not part of this change) times int8 matmul programs one at a time
and several at once: at the 27B's split shape (15,872 x 2,560 weights, 2,048 rows) one program does 40.8 TOPS and two
at once 42.5 (1.04x), three 41.0; at a 512-row tile 27.1 TOPS alone, 31.2 with two (1.15x, per-call overhead). One
program already gets the engine's throughput, so there is no idle second engine to give the split. During the 27B's
prefill the Neural Engine is busy 53-57% of a split chunk (the server's log): the GPU's part (attention and the
Gated DeltaNet layers, 3 of the FFN's 34 units) bounds it, not the Neural Engine.

**Leads from other engines** (oMLX, TensorFold, MLX, llama.cpp; read, not run here): a 4-bit KV cache (oMLX's Affine4;
Kapitsa has int8 only), offloading a cold prompt's prefill to a second machine (oMLX's prefill accelerator), macOS 27's
native quantized tensor-op formats in `matmul2d`, moving the Gated DeltaNet projections to the Neural Engine as well as
the FFN (oMLX's tuner), and keeping the GPU warm between agent turns (TensorFold's `--keep-warm`; Kapitsa's heater runs
for streamed experts only). Given the probe above, the Neural Engine route only pays if the GPU's part of a chunk
shrinks first.
