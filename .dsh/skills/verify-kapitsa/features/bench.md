# kapitsa bench

`kapitsa bench` measures Kapitsa on this Mac with the protocol of the published benchmarks. It starts the model's
server (1 slot, 32K context) and measures writing speed: 512-token coding answers at temperature 0.6 (Flint 1.0) and
greedy, the median of 5. It also measures reading 4K and 16K prompts cold, follow-up time to first token, peak memory
and tool calls. It records what could skew the numbers (other GPU apps, power, thermal state, swap), saves JSON, and
can submit the result as a community pull request. A run takes 3 to 15 minutes and holds the GPU lock throughout.

## Sub-features

- BN1: `--help` (CPU)
- BN2: a run, `--quick` (3 runs per test, 4K reading only) or full (GPU)
- BN3: `--json` and `--out FILE`
- BN4: `--submit --dry-run`: prints the payload, the gh commands and the PR text; sends nothing
- BN5: `bench compare`: Kapitsa against the other engines installed here
- BN6: the protocol matches bench/harness (`tests/test_bench.py`, CPU)

## How to get to it (user POV)

`kapitsa bench --model 4b`, then `kapitsa bench submit` to add the result to bench/community.

## Driving it with vk

Preconditions: BN1 and BN6 need only `$V build`. BN2–BN5 need the task's GPU permission, an idle GPU (`$V doctor`
shows the lock free) and a model folder. Runs take 3 to 15 minutes. Run them in a persistent terminal.
The [speed A/B example](../references/output-check.md#2-speed-ab) keeps the script and log without external job tools.

| ID | Command | Observable result |
| --- | --- | --- |
| BN1 | `$V cli bench --help` | exit 0; usage with `--quick`, `--json`, `--submit`, `--dry-run` |
| BN6 | `python3 tests/test_bench.py build` | exit 0 (needs `jsonschema`; real Mac only) |
| BN2, BN3 | `KAPITSA_NO_AUTO_UPDATE=1 build/kapitsa bench --model "$NACE_MODELS/qwen3.5-4b-4bit-mtp" --quick --json --out /tmp/verify-kapitsa/evidence/<run>/bench-new.json` | progress on stderr, JSON on stdout and in the file, with decode tok/s, prefill and the Mac's state |
| BN4 | `build/kapitsa bench submit /tmp/verify-kapitsa/evidence/<run>/bench-new.json --dry-run` | the PR body and gh commands printed; exit 0; nothing on GitHub |

## Gotchas

- `kapitsa bench` takes the GPU lock itself. Do not wrap it in `vk serve` or another `lockf`: it would wait on
  itself.
- It writes to the real `~/.cache/kapitsa/bench/` unless `--out` is given. Pass `--out` into the evidence dir.
- Never `--submit` without `--dry-run` from a verification run. Submissions come from the user's fork, as a PR with
  the `benchmark` label.
- For a speed A/B use this command on both builds, interleaved (references/output-check.md). A single run is not a
  comparison.
- A result flagged contention or invalid stays in the results but is not in the README numbers.
