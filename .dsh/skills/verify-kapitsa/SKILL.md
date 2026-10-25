---
name: verify-kapitsa
description: Drive this checkout's Kapitsa the way a user does (the kapitsa CLI, a running `kapitsa serve` and its OpenAI/Anthropic APIs, structured output, bench, update) and decide what a change must show before it is called no slower and the same outputs. Use to verify a change or PR in this repo, to answer "does this PR need a GPU run, on which models", or to reproduce a user-visible behavior with evidence.
---

# verify-kapitsa

Kapitsa's users meet it through one binary, `kapitsa`: model commands (`pull`, `list`, `rm`), `serve` with its
HTTP APIs and chat page, `bench` and `update`. This skill drives that binary from the checkout you are in, keeps
every command's output as evidence, and tells you which checks a diff requires (`references/pr-checks.md`).

Everything goes through `scripts/vk` (run it from anywhere in the checkout; `vk` with no arguments prints its usage).
It never touches `~/.cache/kapitsa`: each checkout gets its own scratch home, so worktrees and other sessions on this
Mac do not see each other's state.

## Ground rules

- **No GPU unless the task allows it.** `vk build`, `vk cli`, `vk cpu-tests` and `vk checks` stay on the CPU. Only
  `vk serve` and the steps of `references/output-check.md` load a model, and each holds `/tmp/kapitsa-gpu.lock` while
  it does. `vk serve` fails at once if another process holds the lock. `logit_dump` and `kapitsa bench` queue for it.
  Never kill another session's lock holder.
- **Work in a worktree of your own** for anything you will build or commit: `git worktree add -b <branch>
  ../kapitsa-<topic> origin/main` (the `create_worktree` tool fails here: git error 129). The main checkout and the
  `/private/tmp/*` worktrees belong to other sessions.
- **Report what ran.** A check you did not run is listed under "Not verified", and a PR not run on a GPU says
  "Not verified on a GPU", in its title too (AGENTS.md).

## Launch

`vk` and `required_checks.py` act on the caller's checkout: `$VK_ROOT` if set, else the Kapitsa checkout (one with
`src/engine`) around the current directory, else the checkout the skill lives in. One copy of the skill (for example
symlinked at `~/.agents/skills/verify-kapitsa`) serves every worktree: `cd` into yours and run it by its full path.
`vk doctor` prints which checkout it picked. The `V=` line below assumes you are in a checkout that has the skill.
A branch cut before the skill was merged (#590) has no `.dsh/skills/verify-kapitsa`: fetch the PR's head, then run
the installed skill with `VK_ROOT` set to that checkout. `vk checks` reads its diff against `origin/main`, so no rebase is
needed.

```bash
V=.dsh/skills/verify-kapitsa/scripts/vk
$V build                 # configure build/ (Release) once, then the CLI and every CPU test target; log in the evidence dir
```

For a GPU check, with the task's permission:

```bash
$V serve "$NACE_MODELS/qwen3.5-4b-4bit-mtp"      # a model FOLDER: a catalog name (4b) would download it into the scratch home
$V serve "$NACE_MODELS/qwen3.5-4b-4bit-mtp" --slots 1 --context 32k   # extra args go to kapitsa serve
```

`vk serve` runs `lockf -k -t 0 /tmp/kapitsa-gpu.lock kapitsa serve --model M --host 127.0.0.1 --private --port
$VK_PORT` (default 8090). It returns once `/health` and `/v1/models` both answer 200, the readiness check
`kapitsa bench` uses. The server log is `<evidence>/serve.log`.

## Which GPU

Each Mac has one GPU lock, and the queue for it is whoever is blocked in `lockf`. Before a GPU leg, see who holds
each Mac's lock and which frees up first:

```bash
$V gpu                                                    # every Mac: holder, waiters, ETAs, detach jobs, memory, swap; ends with "pick:"
$V gpu --model qwen3.8-27b-4bit --chip M5 --min-macos 26  # only the Macs that can run this leg
$V gpu run --eta 25 --label "#590 4B logit gate" -- bash leg.sh       # queue behind this Mac's lock, declaring 25 min
```

- **Advice only.** The pick never outranks the orchestrator's ROUND priority. `gpu run` queues on this Mac alone, and
  never reorders, kills or preempts another holder or waiter. Nothing launches on another Mac. A leg there is started
  by the session that owns it, under that Mac's own `lockf`, after checking `ps` and free memory.
- **Macs** come from `~/.config/verify-kapitsa/hosts` (`$VK_HOSTS`), one per line: `name user@host [socks=HOST:PORT]`.
  On a Mac where Tailscale runs in userspace mode, ssh goes through its SOCKS proxy: `socks=localhost:1065`. `vk gpu`
  lists tailnet Macs that are missing from the file. Add one only with the user's OK. The probe is read-only:
  `sysctl`, `vm_stat`, `lsof`, `ps`, the `detach.py` job files and the model folders.
- **Each lock user** shows its start time, its declared ETA, the `detach.py` job it runs under and that job log's last
  line. **The estimate** adds the holder's remaining ETA to every waiter's ETA. A job without a declared ETA counts as
  20 min (`$VK_GPU_DEFAULT_ETA`). A job past its ETA counts as 5 more min. A model process loaded without the lock
  makes the Mac busy. `flock` does not promise FIFO, so the order of the waiters is a guess.
- **`gpu run`** writes `start ETA label` to `/tmp/kapitsa-gpu.queue/<pid>`, then runs `lockf -k /tmp/kapitsa-gpu.lock
  CMD` with `NI_LOCK_HELD=1`, so the job takes the lock exactly once. Run preflight waits (free memory, a build)
  before it, outside the lock. Do not put `lockf` or `vk serve` inside CMD: a nested `lockf` waits forever on its own
  parent's hold, and `vk serve` fails at once (exit 75). Start `kapitsa serve` directly in the leg script. A job over
  10 min goes under `detach.py`, with the lock inside: `detach.py run --name <job> -- $V gpu run --eta N -- CMD`.
- **Choosing the Mac.** Both arms of an A/B, and both dumps of an output check, run on one Mac. A floor file is a
  measurement on the Mac named in its `measured.mac` field. Tensor-unit kernels (`src/kernels/tensor/`, the M5
  defaults in AGENTS.md) need `--chip M5 --min-macos 26`. Another Mac runs a build of your commit, never its own old
  checkouts. Label every number with the Mac, chip, RAM and macOS.

## Doctor

```bash
$V doctor
```

Read-only. It prints the checkout and HEAD, and whether `build/kapitsa --version` is HEAD's build (STALE otherwise:
`vk build`). It also prints the Mac and macOS (the PR's `Runs:` line), who holds the GPU lock, and whether this run
has a server. It exits 1 when the binary is missing or stale. Uncommitted `src/` changes cannot be seen in the
version string; rebuild after editing.

## Drive

| What | How |
| --- | --- |
| Any CLI command, offline | `$V cli <args>`: e.g. `$V cli pull --dry-run --json 4b`, `$V cli list --json`, `$V cli update --check` |
| The CPU test list of AGENTS.md | `$V cpu-tests` (one PASS/FAIL line per test), then `prek run --all-files` |
| What a branch must show | `$V checks [BASE]`: tier, required checks, models and the floor file the gate uses on this Mac |
| The APIs, on the GPU | `$V serve DIR`, then `python3 tests/test_server.py http://127.0.0.1:8090` or `curl` (features/serve-api.md) |
| The base build to compare against | `$V ref` (origin/main, built once per commit under `/tmp/verify-kapitsa/ref/`; prints its build dir) |
| Outputs and speed, on the GPU | `references/output-check.md` |

`vk cli` points the Hugging Face endpoint and the update server at a closed port (`127.0.0.1:9`). A command that
would download or check for a release fails loudly instead of fetching. The feature files in `features/` say, per
feature, what to run and what you should see; start at `features/README.md`.

## Evidence

Everything lands in `/tmp/verify-kapitsa/evidence/<run>/<HEAD>/` (`$VK_EVIDENCE` moves it), one folder per commit.
`<run>` is `<checkout dir>-<hash of its path>`, or `$VK_RUN`. It does not change when you commit or rebase, so
`vk stop` still finds a server started before the commit:

- `cli/NNN.{cmd,out,err,rc}`: each `vk cli` call: the exact command with its environment, stdout, stderr, exit code
- `cpu-tests/<test>.log`, `build.log`, `cmake.log`
- `required-checks.txt`: the last `vk checks`
- `serve.log`: the server's own log

Quote the files in the PR, not from memory. Logit dumps are about 190 MB per path on Qwen: keep only
`logit_compare.py`'s report and delete the dumps.

## Cleanup

```bash
$V stop        # the server this run started (lockf's child, by its recorded pid), never another
$V cleanup     # stop, then delete this run's scratch home; the evidence directory stays
```

Run `vk cleanup` before you finish. A server left running holds the GPU lock and blocks every other session's
measurements.

## Helpers

- `scripts/vk`: build, doctor, offline CLI runner, CPU tests, the base build (`ref`), serve and stop under the GPU
  lock, cleanup
- `scripts/required_checks.py`: changed paths to tier, checks and models (`--files a b` for a list, `--json`)
- `scripts/test_required_checks.py`: pins sample paths' tiers and models; run it after changing a rule
- `scripts/gpu.py` (`vk gpu`): which Mac's GPU frees up first, and `run` to queue a job with an ETA;
  `scripts/test_gpu.py` pins how it reads holders, waiters and ETAs
- `references/pr-checks.md`: the tiers, which models and datasets, and what a PR description must carry
- `references/output-check.md`: the logit gate and the interleaved speed A/B, step by step
- `features/`: the feature map
