# Pre-registered decision procedure: Flash-Next 8-row tensor verify tiles (fn8t-1b)

This file fixes, before any measurement exists, the acceptance gates for the
ROUND 239 step-1b kernel work: the 8/16-row `tensor_ops` verify tile for Q2_0
experts (`qmt8v_*`) and the dense verify projections on the tensor units
(`qmm6t` on the f6g64 slab), engaged only for the Flash-Next q20 verify path
at R = 8, behind `NI_FN8T_D` (dense) and `NI_FN8T_E` (experts), default off.
It follows the qmt6/#510 style recorded in AGENTS.md ("Changing speed or
numerics"): thresholds fixed in advance, evidence traced to dumps, a stated
FAIL condition, and no post-hoc threshold movement.

## Class claim, and why the floor's gate fails by construction

Both kernel families keep every product and reassociate each output's K
reduction onto the tensor units' `matmul2d` of codes — the `--reorder` class —
with one departure: activations are reconstructed from a hi/lo half pair
(`x = hi + lo`, both products exact half×half in fp32), so the computation is
a reassociation plus a ~2^-22 relative representation residual, not a pure
reassociation. The tensor path does not run the simdgroup kernels' fp32
chain, so `tools/logit_floors/flash-next-q20.m5.json`'s all-zero bitwise
floor fails by construction with the flags on, exactly as recorded for #570.
The gates below therefore bound the drift against the scalar path instead of
requiring bitwise equality, and they reuse the floor file's own
`reorder_gate` — the user-governed bounds for order/representation-class
changes on this model — which is the #581 procedure ("accepted when every
measured cell is inside the model's own gate where a floor exists").

## Arms

- One binary per PR head; arm A = `NI_FN8T_D=0 NI_FN8T_E=0`, arm B =
  `NI_FN8T_D=1 NI_FN8T_E=1`. The flag toggle is the only difference.
- Machine: this Mac (MacBook Pro 16-inch 2026, M5 Pro, 48 GB, macOS 26.5.2).
  Model: `models/flash-next-q20-full`. Engagement guard: Apple GPU family
  ≥ 10 and macOS ≥ 26, logged once per process.
- Quiet legs only (`peak_pressure = 1`, `lockf` on `/tmp/kapitsa-gpu.lock`,
  at most one retry). An envelope abort (`peak_pressure > 1`) makes the leg
  VOID and rerunnable; it is not a FAIL.
- Arm B runs the hi/lo two-term tables. Single-term is a diagnostic build
  only (`NI_FN8T_LO=0`) and never produces gate evidence (G5).

## G1 — 192-row drift dump (logit_dump + logit_compare.py, B against A)

`tools/logit_dump` on the step and verify paths, 192 rows, chunk 2048, same
prefixes in both arms (`records.txt`/`sequences.txt` hashes equal); compared
with `tools/logit_compare.py`'s arithmetic. PASS requires every cell:

| Cell | Bound | Provenance |
| --- | --- | --- |
| top-1 agreement | ≥ 0.989583 | floor file `reorder_gate.min_top1` |
| far flips (top-2 margin ≥ 0.551) | 0 | `reorder_gate.max_far_flips` |
| every flip's top-2 margin | < 0.551 | `reorder_gate.near_tie` |
| mean KL | ≤ 0.00184 | `reorder_gate.max_mean_kl` |
| median KL | ≤ 0.000101 | `reorder_gate.max_median_kl` |
| p99 KL | ≤ 0.0283 | `reorder_gate.max_p99_kl` |
| \|mean NLL change\| | ≤ 0.0349 | `reorder_gate.max_nll_change` |
| max \|Δlogit\| | ≤ 7.48 | 2× the floor file's worst pure-reassociation source `max_abs_diff` (3.7395, `NI_QMT=0 NI_MATH_SAFE=1` step), following the file's own twice-the-worst-source construction |
| max \|Δlogit\| tripwire | ≤ 1e-3, else HOLD (protocol below) | the expected ~1e-6 band, 10³ below the tripwire; amendment, orchestrator ROUND 273 |

Expected band, recorded not gated: hi/lo drift should land ~1e-6 mean KL,
three orders inside these bounds. Measured values are booked in full either
way; a PASS at the 1e-3 scale would indicate the lo term is not engaged and
is caught by G5, not by widening G1.

Tripwire (amendment, orchestrator ROUND 273, made before any gate leg was
witnessed — the only point this procedure lets a threshold change): if max
|Δlogit| exceeds 1e-3 on any of the 192 rows, the G1 result is not a PASS —
it is a HOLD. A HOLD clears only by a written cause booked with the dump:
the named op that produced the drift, the source of the error, and why it is
not a bug; the orchestrator accepts or rejects the cause. Without an
accepted cause the HOLD counts as FAIL. The 7.48 hard ceiling is unchanged:
exceeding it is a FAIL with or without a cause. The tripwire exists because
7.48 is ~10⁶× the expected band and alone cannot catch a wrong kernel;
strict G3 md5 catches that for the flip PR, but a kernel PR (flags off)
relies on G2 alone.

## G2 — per-GEMM bench (bench_moe-style, both families)

For `qmm6t` against `qmvs_multi_f6g64` and for each `qmt8v` stage against
its `moe_q20v_*` scalar kernel, at the verify shapes (R = 8; dense
K = 2560, N ∈ {512, 640, 3584, 2560}; expert I = 640, D = 2560) plus the
M/N tail classes:

- every tensor output within one half step of the scalar path's output;
- both paths' max error against an fp64 reference within 2× of each other;
- the hi/lo build's max error against fp64 ≤ the single-term build's;
- small-shape dispatches layer-clean under `MTL_SHADER_VALIDATION=1
  MTL_DEBUG_LAYER=1` (static threadgroup memory ≤ 16,384 B per kernel, since
  the layer double-counts against the 32,768 B cap; a kernel above that says
  so in the PR with its size, per AGENTS.md).

## G3 — greedy continuations (ABBA, 256 tokens, w4k/w32k chat prompts)

`/tmp/df2-gate/ids/w4k.ids` and `w32k.ids`, temperature 0, no
`--ignore-eos`, 256 output tokens, arms interleaved A B B A, n = 2 legs per
arm per length, one binary.

- Kernel PRs (flags default off): PASS when the ids md5 is identical across
  all legs at each length. A mismatch is tolerated only when every divergent
  position's top-2 margin in the A arm is < 0.551 and at most 2 positions in
  256 diverge per length; otherwise FAIL.
- The default-on flip PR: the ids md5 must be identical across all legs at
  both lengths, strictly. Any mismatch blocks the flip; no conditional tier
  applies. (#585 reverted a tensor path because one continuation moved to
  `<|im_end|>` and looped; a flip that changes continuations is not this
  change's claim.)
- Every leg carries the ROUND 242 output check: 256 non-empty ids, zero
  248044/248046 among the first 256 output ids, ≥ 256 tokens before the
  first EOS special, decoded prefix in the ledger. A leg without it is VOID
  and is not booked.

## G4 — floor inviolability (flags off)

At every PR head, with both flags off, the flash-next-q20 floor leg passes
bitwise 192/192, and `flash-next-q20.m5.json` is unmodified in the diff. Any
bitwise failure with the flags off blocks the PR: it is a regression in a
shared path, not a gated departure. The floor file is never re-measured or
edited by this work; a re-measure with the path on is a PR of its own, per
AGENTS.md.

## G5 — hi/lo engagement evidence

Arm B's leg logs carry the once-per-process engagement line showing the hi/lo
two-term prep ran (lo tables written). Gate evidence from a build or leg
without that line is void. `NI_FN8T_LO=0` (single-term) exists for
diagnosis only and is never a gate arm.

## FAIL

Any cell of G1–G5 exceeded, missing, or untraceable to a dump is a FAIL. A
G1 tripwire HOLD without an accepted written cause counts as a FAIL under
this definition. On
FAIL: `NI_FN8T_D`/`NI_FN8T_E` stay off, no flip PR is opened, the dumps and
the comparison output go to the orchestrator, and the thresholds in this
file do not move. Thresholds are immutable once the first gate leg is
witnessed (results posted to the ledger); changing any threshold requires a
new orchestrator ROUND that says so explicitly, before new legs run. There
is no deadline auto-ratification branch: unlike #510's standing
"default on, don't ask me again" ruling, the flip here waits for the
orchestrator's explicit PASS ruling on the booked evidence. A late explicit
user decline supersedes everything in this file and is executed as a
one-line default-off revert (`NI_FN8T_D=0 NI_FN8T_E=0`) without further
discussion.

## Flip PR requirements (after a PASS ruling)

- Opened via the verify-kapitsa procedure with vk checks output for the head
  sha in the body.
- Body states plainly: not bitwise against the floor; accepted under this
  pre-registered procedure, citing this file's commit sha; every G1–G5 cell
  with its measured value and dump trace; the ABBA md5s; the Runs: line
  (Mac, macOS, commit, model).
- Performance evidence (booked, not a numeric acceptance gate): ABAB tok/s
  at 4K and 32K on quiet legs, and the fixed-clock `NI_CYCLE_TRACE`
  decomposition (draft / store-stall / verify-compute) riding the first dfl
  leg. First milestone: verify ms/cycle at or below the MTP tie line
  (~92 ms/cycle 4K, ~87 ms/cycle 32K); end goal ≤ 45 ms/cycle at 8 rows.
- Store-stall exposure clause (orchestrator ROUND 270): if verify-compute
  falls below ~45 ms/cycle and the expert-store stall (~24 ms/cycle on quiet
  legs) becomes the largest share of ms/cycle, that split is reported to the
  orchestrator with the flip evidence, and the Store S2/S3 work is ordered
  before any step-1 follow-on. The clause does not gate this PR; it re-orders
  what comes next.

## Provenance of this procedure

Ordered by the orchestrator (ROUND 270) before any fn8t GPU leg runs.
Threshold sources: `tools/logit_floors/flash-next-q20.m5.json`
(`reorder_gate`, `reorder_sources`), AGENTS.md acceptance entries #510
(qmt6, nll-cell departure and ABBA-bitwise evidence), #570/#585 (tensor path
reverted on a continuation change), #581 (per-cell inside-the-model's-gate
procedure). Design scope: `/tmp/df2-gate/1b-design.md` (subagent 128e063b).
Amendment (G1 tripwire cell): orchestrator ROUND 273, committed before any
gate leg was witnessed — the only point this procedure lets a threshold
change.

## Envelope amendment (orchestrator ROUND 359): r3b legs run the shared R280/R282 abort code

Ordered by the orchestrator (ROUND 359), committed 2026-10-26. The fn8t-w1
r3b retry legs replace the stricter abort envelope earlier `fn8t_leg.sh`
revisions carried (the per-10 s +500 MB swap-growth clause, pressure ≥ 4)
with the same abort code every other GPU job on this machine runs — the
shared R280/R282 envelope monitor plus ROUND 356's floor clause. A leg
aborts (VOID, rerunnable once; not a FAIL — Arms quiet-leg rule) on any
in-leg sample where:

- swap delta from leg start > 3072 MB, or
- swap used > 12288 MB, or
- `kern.memorystatus_vm_pressure_level` > 2, or
- `kern.memorystatus_level` < 10% (fixed floor, ROUND 356; the canonical
  monitor's 15%-with-others / 10%-alone switch is pinned to 10).

There is no load-phase exemption. Evidence motivating the amendment: r3's
g1-b aborts (+1147 MB / +1675 MB swap growth, pressure 2) sit well inside
the shared code's normal model-load behavior and would not have aborted
under these rules.

Implementation, live and orchestrator-verified against the files
(ROUND 363(c)) before r3b launch: `fn8t_leg.sh` sha256
`db75772ccb27b38946cf4e64da7ca98b43c598b5e067aaff0780217179f6841d`,
battery `fn8t.sh` sha256
`d44d5dab66195a2f235e266a669fc73dfc6fbc52ba95cc754a038c3229041057`,
waiter `fn8t_w1_r3b_waiter.sh` sha256
`a0bb81f035823b4376a62896d8fb898667289812e7815b7db4813feb3306d06b`.

Ordering disclosure: ROUND 359 reached the owner session after ROUND 363
confirmed r3b armed; the r3b gate opened (pf.yield cleared 04:39:48 with
STOREFIX's finish) and the battery launched 04:41:18, before this commit
landed. At commit time no r3b leg result was booked: g1-a did not re-run
(RESUME reuse, ROUND 356, booked `rc=skipped-resume-r356`), and g1-b was
mid-leg — attempt 1 envelope-aborted 04:43:18 on the floor clause alone
(`memorystatus_level` 9% < 10%, swap +376 MB, pressure 2 — the three swap
clauses all passed; the leg's own ~30 GB q20-full resident set holds the
level down for the leg's duration), with its single retry started 04:43:48.
The envelope the live legs implement is byte-identical to the spec above.

## fn8t-2 extension (A07): cross-row expert union + down retune

The orchestrator (ROUND 279) booked the port plan's Port 2 / A07 as fn8t-2
and ordered this one-section extension, committed before any fn8t-1b or
fn8t-2 leg was witnessed; every fn8t-1b threshold above is untouched.
Scope: the cross-row expert union walk in the Flash-Next q20 verify path —
per layer, read the union of the 8 verify rows' expert selections once and
apply each expert tile to every row that selected it — plus the down-slot
retune to the gate_up throughput class (slot baseline booked at 174 GB/s,
retune target 242 GB/s). Behind `NI_FN8T_U`, default off, same engagement
guard as fn8t-1b (Apple GPU family ≥ 10, macOS ≥ 26, Flash-Next q20 verify,
R = 8), logged once per process.

### Class claim: bitwise by construction

The union walk changes tile-to-row scheduling — which row's output consumes
which expert tile, and in what order tiles are fetched — not arithmetic:
each output element keeps the slot kernels' (`moe_q20v_gate_up`,
`moe_q20_down`) exact products and exact accumulation order. The down
retune is in class only if it likewise preserves each output's exact
reduction order (dispatch shape and threadgroup tiling change, accumulation
does not); a retune that reorders any output's K reduction is outside this
section and would need fn8t-1b's drift-gated class instead. The in-tree
precedent is the `moe4_*` 4-bit verify family, bitwise by the same
construction. Because the class admits no numerical drift, this section has
no G1 drift table and no tripwire/cause path: any departure from bitwise
identity is a bug, not a numerics class to adjudicate.
`tools/logit_floors/flash-next-q20.m5.json` stays untouched and is cited
only in the flags-off direction.

### Arms

- One binary per PR head; arm U0 = `NI_FN8T_U=0`, arm U1 = `NI_FN8T_U=1`,
  both with `NI_FN8T_D=0 NI_FN8T_E=0` so the reference is the shipped
  scalar path and A07's own effect is isolated — ROUND 169 R3: fn8t-2 books
  from one measured leg carrying both arms. One stacked diagnostic leg
  (`NI_FN8T_D=1 NI_FN8T_E=1 NI_FN8T_U=1`) may be booked, labeled
  diagnostic, non-gating, for cycle arithmetic only.
- Machine, model, quiet-leg rule (`peak_pressure = 1`), lock and one-retry
  rule: identical to fn8t-1b's Arms section. An envelope abort makes a leg
  VOID and rerunnable, not FAIL.

### U1 — bitwise identity gate (replaces G1 for fn8t-2)

`tools/logit_dump` on the step and verify paths, 192 rows, chunk 2048, the
same workload and prefixes in both arms (`records.txt`/`sequences.txt`
hashes equal). Three dumps, mirroring fn8t-1b's dump-a/dump-b/dump-a2
mechanics: dump-u0 (arm U0), dump-u1 (arm U1), dump-u0b (arm U0 again).
PASS requires bitwise identity of all 192/192 rows in both directions —
u0 against u1 and u1 against u0 — plus arm stability u0 against u0b:
max \|Δlogit\| = 0, top-1 agreement = 1.0, every KL/NLL cell = 0, equal
`logits.f32` sha256 per pair. Any non-zero deviation = FAIL-bug. There is
no HOLD state and no cause path in this section.

### U2 — per-kernel bench under the g2.json contract (replaces G2 coverage)

The bench writes `/tmp/fn8t-gate/g2u.json` with the same cell schema as
fn8t-1b's `g2.json` (`pair`, `shape`, plus the gate fields below), runs
under the GPU lock, and reports its binary name and CLI in the ledger. A
FAIL-miss on any cell stops the battery before any dump or ABBA leg.

| Cell | Bound | Provenance |
| --- | --- | --- |
| bitwise output identity, every cell | true (union vs slot at every benched shape) | the class claim above |
| union gate_up total kernel time ratio vs slot, verify shape (R = 8) | ≤ 0.605 | U/raw × 1.10 — U = 44.0 booked (store stats, dyn4k-1, ROUND 254(3): the handshake dedups per turn via `mark_`, so `lookups/turns` is the exact cross-row union per turn; turn = one (cycle, layer) row set — 48 layers, 2784 turns / 58 cycles), raw = 8 rows × top_k 10 = 80 (`config.json` `num_experts_per_tok = 10`); 10% scheduling slack |
| retuned down throughput | ≥ 220 GB/s | within 10% of the port plan's booked retune target 242 GB/s (gate_up class); slot down baseline booked at 174 GB/s |
| `static_tg_bytes` | ≤ 16384 | the validation layer's 2× accounting against the 32,768 B cap |
| `layer_clean` | true | small-size run under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1` |

Half-step and fp64-parity cells are not required here: the reference is the
slot kernel's own output, and bitwise identity is strictly stronger than
parity.

### U3 — engagement evidence (G5 class)

One log line per process when `NI_FN8T_U=1` engages the union path; its
pattern appended to `engagement-pattern.txt` before any arm-U1 leg runs.
Arm U0 prints no engagement line and is bitwise unchanged against base: the
flags-off floor leg stays bitwise 192/192 at every PR head, exactly as
fn8t-1b's G4.

### U4 — continuations (G3, unchanged)

256-token ABBA temperature-0 continuations on the `w4k` and `w32k` prompts,
strict ids-md5 identity across a1 b1 b2 a2 — and, because the class is
bitwise, identity with the arm-U0 md5 as well. Any divergence = FAIL-bug:
the 32K block does not run, the divergence analysis goes to the
orchestrator.

### FAIL definitions, evidence, economics

- FAIL-bug: any bitwise deviation in U1, in U3's flags-off identity, or in
  U4. No cause path; the fix is code, re-run from a new PR head.
- FAIL-perf: any U2 cell miss. The battery stops before the dumps; no
  dump/ABBA GPU is spent.
- FAIL-untraceable: a missing cell, a missing bitwise verdict, or a bench
  that does not write the json contract.
- VOID: envelope abort (`peak_pressure > 1`), at most one retry — not a
  FAIL.
- No fn8t-2 GPU leg runs on a build whose `fn_check` is not green
  (ROUND 279(3)).
- Threshold immutability: once any fn8t-2 leg (including the bench) has
  been witnessed, this section is frozen by the same rule as fn8t-1b.

### Terminal contract (ROUND 348)

The battery's EXIT trap writes exactly one terminal file on every exit
path, precondition failures and cap exits included:
`/tmp/fn8t2-gate/FN8T2-TERMINAL`, one line, one of

- `FN8T-2-DONE <ts> rc=0`
- `FN8T-2-FAIL <ts> rc=N <reason>`
- `FN8T-2-VOID <ts> rc=N <reason>`

Cap and tripwire-class stops fold into FAIL or VOID with the reason in the
text; the class carries no tripwire terminal of its own (bitwise class — a
U1/U4 divergence is FAIL-bug, VOID covers envelope aborts). The battery
`mkdir -p`s `/tmp/fn8t2-gate` and never writes terminals under
`/tmp/fn8t-gate` (fn8t-1b's gate dir). If fn8t-2 is dropped, the
orchestrator writes `FN8T-2-WITHDRAWN <ts> <reason>` to the same path. The
downstream waiter (cd6e8136's slot-8b) gates on
`^FN8T-2-(DONE|FAIL|VOID|WITHDRAWN)` in that file. Real runs only (R324):
dry runs and tests pin the terminal path to a scratch dir and never touch
the live file.

### Battery cap (ROUND 373/374)

The fn8t-2 battery runs under a binding wall-clock cap of 5400 s from the
first lock take (R308(b) semantics — the same stamp fn8t-1b's battery
uses, set before the first leg): at expiry the battery finishes the
current leg, books every unreached leg `NOT RUN (cap)` in the leg table,
and exits rc=9. Per the terminal contract above, the cap exit folds into
`FN8T-2-VOID <ts> rc=9 cap: wall-clock 5400s exceeded; unreached legs
booked NOT RUN (cap)` — VOID, not FAIL: a wall-clock stop is not a gate
verdict, and no leg reruns after a cap exit. Implementation:
`CAP_SECS=${FN8T2_CAP_SECS:-5400}` in the battery script, checked before
every leg (`cap_check` in `run_leg`); overriding the default via the env
at launch is a prereg deviation and needs orchestrator approval.

### Evidence base booked at commit time

U = 44.0 distinct experts per layer per turn (DFlash2 4K, dyn4k-1 store
stats, ROUND 254(3)); raw = 80 per layer (top_k 10 × 8 rows); U/raw = 0.55,
so the union walk removes 45% of expert weight reads. R=8-scaled estimate,
recorded not gated: ≈ −12.4 ms/cycle (union) and ≈ −3.2 ms/cycle (down
retune) against the 27.5 ms/cycle expert-verify estimate ported from the
MTP R=7 profile (MoE 24.065 ms/C). Prices and bars: port plan §3 Port 2.
Ordered by the orchestrator, ROUND 279: A07 booked as fn8t-2, extension
committed before any leg; GPU queue fnx-dec → 1b legs → fn8t-2 legs → 35B
DYN A/B; fn8t-2 implementation starts from the fn8t-1b commit in its own
worktree.
