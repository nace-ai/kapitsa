# Independent audit of task 758 local campaign

## Verdict

Raw arithmetic and generated-text gate: **PASS**. Timing admissibility: **NOT_ESTABLISHED**. The throughput values cannot clear the technical hold because the campaign did not record required isolation evidence. Missing observations do not establish that contention actually occurred.

This report covers the fixed local M5 Pro campaign only. It is not a whole-PR GO, a task-completion decision, or a CI verdict. The original goal authorizes this campaign, not an automatic optimization loop or another runtime attempt. No retry is recommended or authorized by this report.

## Scope and method

The reviewer did not implement the runner. The initial audit was one read-only snapshot while the campaign ran. After the parent supplied terminal data, the reviewer read the remaining response and receipts, independently recalculated all eight raw text hashes and decode rates, and aggregated the completed resource snapshots. These were offline JSON calculations, without importing the runner, loading a model, building, hashing weights, adding monitors, or controlling campaign processes. This report is the only evidence-directory file created by the reviewer.

The immutable [launch runner](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_ab.py>), [launch wrapper](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_locked.sh>) and [deadline watcher](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/wait_detached.py>) are the source references for the final assessment.

## Raw results

All eight responses report `done=true`, `done_reason=length`, and `eval_count=256`. Each arm's count preflight and raw `prompt_eval_count` equal its actual target, 4096 or 32768. The frozen requests contain temperature 0, top-k 1, top-p 1, seed 0, and 256 requested tokens. Fixture SHA256 values independently match the [frozen fixture manifest:1](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/ab-run-02/fixtures.json#L1>).

Decode rate was independently recalculated as `255e9 / eval_duration_ns`, not 256 divided by duration or a wall-clock rate. The resulting means and ratios agree with the [parent's independent raw recheck:1–76](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/parent-independent-raw-recheck.json#L1-L76>).

| Actual prompt tokens | A1 | B1 | A2 | B2 | Mean A | Mean B | Mean B / mean A |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 39.218271 | 39.353604 | 39.384922 | 39.603663 | 39.301597 | 39.478633 | 1.004504559786 |
| 32768 | 29.570740 | 30.268031 | 31.282600 | 31.219396 | 30.426670 | 30.743713 | 1.010419913938 |

Rates are backend decode tokens/s. Both arithmetic ratios exceed 0.98. These numbers do not establish an admissible no-slower result or a general speedup.

Canonical text is compact, sorted-key JSON containing both `content` and `thinking`, encoded as UTF-8 with `ensure_ascii=False`. Independent recomputation found one hash across all four arms at each length:

- 4096: `20f35afc847a08a7874d264070e7d5e75c65fb1db9643f857ea58e395d6d0818`.
- 32768: `bf9a56d068e036d11c317b76f2983303fafa9f5542314b26917a3c06d02aea5b`.

Content is empty in every response. Thinking is nonempty, with 1052 UTF-8 bytes at 4096 and 1190 bytes at 32768. Thus the combined-text gate passes; this is not evidence of a completed user-facing answer or token-ID/logit identity. The [last raw response:1](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/ab-run-02/32768-B2/response.json#L1>) and [last arm record:1](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/ab-run-02/32768-B2.json#L1>) agree.

## Findings

### F1. Required timing isolation was not established

[GPU policy:138–139](</Users/alisheramantay/.agents/skills/session-orchestration/GPU-JOBS.md#L138-L139>) requires start/end load per row, detection of foreign processes above 50% CPU, and refusal or invalidation during the first 15 minutes after the DSH host starts.

The [collector:99–109](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_ab.py#L99-L109>) records power, memory, thermal, storage, macOS and chip information, but no process CPU, load averages, or host age. The [prelaunch scan:30–41](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_locked.sh#L30-L41>) records PID, executable and arguments once to reject model processes. It cannot detect those missing conditions during a leg. The [timed-leg marker helper:9–17](</Users/alisheramantay/.agents/skills/session-orchestration/scripts/timed_leg.sh#L9-L17>) checks power mode and pauses declared CPU-heavy detached jobs; it supplies none of the missing observations.

This is an observed evidence omission. Similar rates and good resource snapshots cannot reconstruct foreign-process isolation. The runner's [summary:1](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/ab-run-02/summary.json#L1>) says `pass=true`, but its [implementation:315–330](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_ab.py#L315-L330>) gates only recorded arm success, hashes and arithmetic. Publication must retain this distinction.

### F2. Deadline shutdown can release the lock before cleanup

The [deadline branch:45–53](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/wait_detached.py#L45-L53>) invokes detached-job group stop, followed by forced group stop. The [stop implementation:785–796](</Users/alisheramantay/.agents/skills/session-orchestration/scripts/detach.py#L785-L796>) sends `killpg`, including the outer `lockf` process. [GPU policy:130](</Users/alisheramantay/.agents/skills/session-orchestration/GPU-JOBS.md#L130>) explicitly requires workload-only TERM so cleanup finishes under the lock.

The [normal server exit path:262–280](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_ab.py#L262-L280>) does terminate, wait, escalate and join its owned server. Group shutdown can bypass that ordering. Forced SIGKILL also cannot guarantee execution of the final arm-record preservation loop.

This is a source-only future lifecycle defect, not an observed failure of the completed campaign. The [terminal receipt:1–7](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/detached-terminal-03.json#L1-L7>) records exit 0 and `deadline_stop=false`.

### F3. Memory admission, abort and release checks are incomplete

[GPU policy:127–130](</Users/alisheramantay/.agents/skills/session-orchestration/GPU-JOBS.md#L127-L130>) requires free memory above 40% before starting, stopping below 10% during model use, and memory recovery to at least 40% before releasing the lock.

The [runner:119–121](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_ab.py#L119-L121>) instead uses a single below-3% threshold at admission and during execution. The [wrapper cleanup:49–64](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_locked.sh#L49-L64>) records listener and storage state, without a memory recovery check.

These are source-only guard defects. Recorded memory stayed above the stricter thresholds; no actual threshold crossing or unsafe handoff is established here.

## Observed resource state

Offline aggregation covered 92 saved snapshots, including preparation prelaunch and all timed-arm prelaunch, before, after and periodic guard snapshots.

- Minimum observed free memory was **44%**, including [32K B2 guard 0002:1](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/ab-run-02/32768-B2/guard-0002.json#L1>), and guards 0006, 0010, 0011 and 0012 in the same arm.
- Per-arm minima in A1, B1, A2, B2 order were 47%, 48%, 48%, 48% at 4096 and 45%, 45%, 45%, 44% at 32768.
- Swap was **0.2396875 GiB in every saved snapshot**. Maximum observed growth from each server's prelaunch baseline was **0 GiB**.
- All snapshots reported AC power and AC power mode 2. All memory-pressure levels were 1. All recorded invalid-reason lists were empty.
- Thermal commands returned the same messages throughout: no recorded thermal warning level, performance warning level, or CPU power status. This is the recorded diagnostic, not a direct temperature measurement.

These are sampled observations, not continuous proof. They do not replace the missing CPU/load/host checks. The [cleanup receipt:1–8](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/cleanup-03.json#L1-L8>) reports no listener on the private port and 124.928875 GiB available internally, but contains no post-stop memory sample.

## Identity and failure accounting

The [identity record:1](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/ab-run-02/identity.json#L1>) binds A to main `37f6f2fb1cd89541cc04019e63b5790332b87037` and B to PR head `8eb6db9b44940f26d693d913c92f09e89d340229`, with distinct binary hashes. The [build manifest:1–14](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/build-manifest.json#L1-L14>) records matched Release `-O3` controls. Launch records and server logs show the same native target, fast profile, 65536 capacity, single slot, `--dflash none`, `--spec 0`, disabled disk cache, and cached-zero measured requests. Capacity was not substituted for prompt length.

The [staging receipt:1–53](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/internal-stage-receipt.json#L1-L53>) contains matching source/staged file hashes and the weight digest. The [native configuration:118–121](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/native-model/config.json#L118-L121>) explicitly identifies NVFP4. No target mismatch was found. End-of-run success implies the [config/tokenizer digest comparison:415–419](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_ab.py#L415-L419>) passed. The runner does not perform an end-of-run full-weight digest; this audit did not either. No numerical-floor change was part of the campaign, and this report does not independently certify repository floor contents.

The first [launch failure:1–4](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/detached.log#L1-L4>) occurred at the immutable-lock write probe before model code. The second [launch log:3–13](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/detached-02.log#L3-L13>) and eight preserved error arm records disclose the TIME_WAIT probe failure before timed inference. Its [summary:1](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/ab-run-01/summary.json#L1>) is false, not silently discarded.

The [local socket reproduction:1–5](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/time-wait-repro.json#L1-L5>) records failure without reuse and success with reuse. The [13-test log:1–18](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/runner-cpu-tests-13.log#L1-L18>) records passing CPU tests, including TIME_WAIT and active-listener cases. Those tests were read, not rerun by this reviewer. Normal Python error handling preserves explicit arm records through the [finalization loop:420–428](<https://github.com/nace-ai/kapitsa/blob/evidence/pr1396-m5pro-20261123/bench/results/pr1396-m5pro-20261123/launch03-source/run_ab.py#L420-L428>); all eight current arm records are present and successful.

## Publication boundary

Publish the raw arithmetic/text PASS together with timing admissibility NOT_ESTABLISHED and F1. Keep F2 and F3 labeled as source-only defects, and retain the two observed launch failures. The completed campaign supplies valid local output observations but does not clear the timing hold. No broad chip coverage, paired-logit result, CI state, landing, or whole-task completion follows from this report. Any additional model-backed attempt requires separate explicit authorization and booking; it is not an implied next step of this fixed campaign.
