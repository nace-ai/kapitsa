## What

`Engine::wait_split` bounds a GPU + Neural Engine split chunk at a fixed 60 s: the split is retired, the GPU's
waits are released, and the chunk reruns on the GPU alone. Nothing had ever run that branch (issue #426): a job
that started and stalled is retired by its own 2 s watchdog first, and no test can wait out a real minute to
reach the bound behind it.

The bound is now `NI_ANE_SPLIT_WAIT_MS` (default `60000`, read per call), so the shipped behaviour and its log
line (`has not finished in 60 s`) are unchanged, and a shortened bound prints in ms.

`tests/test_ane_split_abort.py` (model folder, GPU and Neural Engine; it takes the GPU lock) runs it at 250 ms
with `NI_ANE_FAULT=0` on three fresh servers, each with its own `KAPITSA_HOME` and greedy replies:

0. `NI_ANE=0`: the two prompts' replies on the GPU alone, the control.
1. the split on: both prompts run split chunks that hold (the busy line counts evaluations), no chunk given up.
2. the fault: the bound runs out first, the split is given up with the abort's reason (not the watchdog's), the
   chunk reruns on the GPU, and both prompts give the control's replies again. Each line is logged once and the
   second prompt neither splits nor bounds again.

The legs hold `NI_ANE_MIN_ROWS=512` and use a 700-row prompt so the chunk takes the split's single-lane
schedule, the one where the GPU's command buffer waits on the job. On the wavefront schedule (two row halves,
from 256 rows) the GPU side does not wait at all and the chunk is given up by `Handoff::finish`'s own 2 s bound
instead, so the bound is not what a wavefront chunk reaches.

## What the run showed

M5 Pro, `qwen3.5-4b-4bit-mtp`, `NI_ANE_FAULT=0`, `NI_ANE_SPLIT_WAIT_MS=250`, 24-token replies — all 18 checks
of the test pass:

```
NI_ANE: a split prefill chunk has not finished in 250 ms; giving up the Neural Engine split        (once)
NI_ANE: Neural Engine FFN split stopped (a split chunk did not finish within 250 ms); this chunk
        reruns on the GPU, which runs the prefill FFN alone from now on                           (once)
prefill: Neural Engine busy 0% of ... ms over 1 split chunks (0 evaluations)
```

`e.wait()` returned 175 ms after the bound, so the abort's release reached the GPU; both prompts' replies were
identical to the `NI_ANE=0` run (24 tokens each); the faulted prompt took 2.5 s against 0.4 s for the control
and 0.4 s for the prompt after it — the 250 ms bound, then `Handoff::finish`'s 2 s bound for the job that never
reports, then the GPU rerun. The 60 s default is the same code path with a longer wait.

## Limits, stated

* The branch is exercised, but not by a stall the watchdog would have missed: the bound is what is shortened, and
  a hook that suppresses the 2 s watchdog for one job (issue #426's `NI_ANE_WEDGE` proposal) is not in this PR.
  With the default 60000 a real wedge still waits a minute.
* No numerics gate: the diff reads an environment variable and formats a log line; nothing else on the path
  changes and the default keeps the shipped text. #426's no-regression logit run belongs to #422, which shipped
  the override and is on main.
* Not in CI: CI has no GPU or Neural Engine. The test is for a Mac run by hand, and it is listed in AGENTS.md
  with the other model/GPU tests.

Closes #426.
