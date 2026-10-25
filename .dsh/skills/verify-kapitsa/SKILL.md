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
- `references/pr-checks.md`: the tiers, which models and datasets, and what a PR description must carry
- `references/output-check.md`: the logit gate and the interleaved speed A/B, step by step
- `features/`: the feature map
