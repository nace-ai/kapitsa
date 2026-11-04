# #325: Flash-Next on the M5 Pro, with and without #892-class verify attention (3 Nov 2026)

**Result: cannot measure a "with #892-class attention" arm, because no such Flash-Next path exists.** #892 moves the
35B-A3B's draft-verify attention onto the M5 matrix units (kernel `attn8t_partial_zone_g8`, G=8). Flash-Next's native
verify attention is a different, sparse kernel (`attn_rows_sparse`, 12 full-attention layers, at most 2048 selected keys
per row, G=12). It is 3.2% of a decode cycle at 32K and 3.0% at 128K (profile below), so a matrix-unit port would gain
about 2% at best and was not built. The README poster row runs `--engine splash` with KFlash, which none of #887, #890,
#892 or #897 touch.

## Old build against new build (native Flash-Next, interleaved A B A B per context)

Old `71826a86` (before #887/#890/#892), new `a2f42108` (with them), same model, same prompts, greedy, `--spec 2`,
`NI_EXPERT_STORE=24`, M5 Pro 48 GB, macOS 26.5.2, 20 s between runs. Power: the runs (12:17-13:31 local) ran before the 30-second power log started (18:19); the 4K and 32K pairs and the first 128K pair ran on battery at ~2% charge; AC from 13:08. Treat the speeds as drift only. Both builds take the same Flash-Next
path, so any difference is drift.

| Context | n | Decode old / new (tok/s, median) | Change (pairs) | Prefill change |
|---|---|---|---|---|
| 4K | 2 | 45.5 / 46.9 | +3.1% (+1.1 to +5.1%) | -2.0% |
| 32K | 3 | 34.7 / 32.7 | -5.8% (-12.3 to -3.2%) | -1.5% |
| 128K | 3 | 32.7 / 35.0 | +7.0% (+1.9 to +7.0%) | -5.2% |

4K has n=2: run 1 of the old build died with `kIOGPUCommandBufferCallbackErrorOutOfMemory` (Mac on 5-10 GB of swap).
The sign flips with context length and the diff does not touch this path: read it as "about the same", not a speedup
or a regression. Raw runs: `ab/`; script: `ab.sh`; summary: `ab-summary.txt` (`summarize_ab.py`).

## The profile behind the 3.2% / 3.0%

37 decode steps, isolated dispatches, `NI_PROFILE=2`, 131,000-token prompt at 128K. Scripts: `profile/prof32.sh`,
`profile/prof128.sh`; full output `profile/prof*.err`; decode blocks `profile/decode-32k-37steps.txt` (38.263 ms/step)
and `profile/decode-128k-37steps.txt` (39.798 ms/step). `attn_rows_sparse` = 1.215 ms/step (3.2%) at 32K and 1.213
ms/step (3.0%) at 128K; the experts' gate/up/down (`moe_q20*`) are 17-18% and 11% of a step.

Command (GPU lock held, one run each):

    NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 NI_EXPERT_STORE=24 NI_PROFILE=2 build/ni_run \
      $NACE_MODELS/flash-next-q20 @c128k.ids 96 0 --spec 2 --ignore-eos

If a reviewer wants an n>=3 interleaved A/B at 4K, the missing third pair is one `ab.sh` leg (a few minutes of GPU on AC with apps closed;
run-time not measured separately). No GPU runs are queued.
