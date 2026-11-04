# Controlled-run source snapshots

These are the one-off measurement helpers used for this experiment, retained for audit. They are not installed
tools and should not be run from this snapshot directory: their paths describe the experiment's original layout.

In a fresh candidate checkout, recreate this layout:

```text
build/                              candidate executables
build-m4pro-baseline/                baseline executables, built from e87d0f87 in a separate source checkout
build-m4pro-validation-20261104/
  scripts/                          the Python and Objective-C files from this snapshot
  thermal                           compiled thermal.m helper
  prompts/                          the sibling prompts/ directory
  models/qwen3.5-4b-4bit/            model directory or symlink, with the pinned dflash2/ companion
  evidence/                         initially empty
```

Compile the read-only thermal helper before running the GPU tests:

```sh
clang -Os -fobjc-arc -framework Foundation \
  build-m4pro-validation-20261104/scripts/thermal.m \
  -o build-m4pro-validation-20261104/thermal
```

Run from the checkout root. `record.py` takes the exclusive project GPU lock and sets both caller-held markers;
the server helper refuses to start without one. Each label and results directory must be new. Servers are owned
by the foreground runner and stopped when that runner finishes.

```sh
python3 build-m4pro-validation-20261104/scripts/record.py --label fixed-greedy -- \
  python3 build-m4pro-validation-20261104/scripts/fixed_ab.py
python3 build-m4pro-validation-20261104/scripts/record.py --label fixed-sampled -- \
  python3 build-m4pro-validation-20261104/scripts/fixed_ab.py --sampled
python3 build-m4pro-validation-20261104/scripts/record.py --label drained-keepalive -- \
  python3 build-m4pro-validation-20261104/scripts/keepalive_drained.py
```

The helpers import the checkout's unchanged `bench/harness/bench.py` and `gpu_hogs.py`. After collection, apply the
same harness's `distinct4` check at its 0.5 threshold to every saved response, as recorded in `../analysis.json`.
Do not treat a raw record's empty flags as overriding this offline quality check. The standard native series is
`python3 build-m4pro-validation-20261104/scripts/standard_ab.py`; it takes the lock for each arm through `record.py`.
The supplemental `token_traces.py` is also run directly: it takes the lock for each CLI invocation through
`record.py`. Its timing fields are diagnostic and do not supply the throughput verdict.
