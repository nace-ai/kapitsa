# Kapitsa feature map

What a user can do with `kapitsa`, and how to drive each feature from this checkout with `scripts/vk` (below `V`,
i.e. `V=.dsh/skills/verify-kapitsa/scripts/vk`).

## Baseline preconditions

- `$V doctor` exits 0: `build/kapitsa` is HEAD's build.
- CPU features need nothing else. GPU features need the task's permission to load a model, a free GPU lock
  (`$V doctor` shows the holder) and a model folder (this Mac: `$NACE_MODELS`, e.g. `qwen3.5-4b-4bit-mtp`).

## Driving conventions

- CLI: `$V cli ARGS`, never `build/kapitsa` against the real `~/.cache/kapitsa`. Each call prints its exit code and
  where its transcript went.
- Server: `$V serve DIR` on port 8090 (`VK_PORT` changes it), then `curl -s http://127.0.0.1:8090/...`. `$V stop`
  when done.
- An action and its observable result always go together: the command, then what you should see. A different
  result is a finding.

## Reporting proof and skips

Report each sub-feature by its ID as `proven` (with the evidence file), `failed` (expected vs observed), or `skipped`
(why: no GPU permission, model not on this Mac, needs the network). GPU skips also go under "Not verified" in the PR.

## Entry contract

Each feature file has an H1, one paragraph of user-visible behavior, and four H2s in order: `Sub-features` (short
IDs), `How to get to it (user POV)`, `Driving it with vk` (starts with `Preconditions:`), `Gotchas`. Behavior only;
how the code does it belongs in the code.

## Features

| File | Feature | GPU |
| --- | --- | --- |
| [model-store-cli.md](model-store-cli.md) | `pull`, `list`, `rm`, model names and exit codes | no |
| [serve-api.md](serve-api.md) | `kapitsa serve`: health, models, Chat Completions, Responses, Messages, chat page | yes |
| [structured-output.md](structured-output.md) | JSON schema replies and forced tool calls on all three APIs | offline part no |
| [bench.md](bench.md) | `kapitsa bench`: the published benchmark protocol on this Mac | yes |
| [update.md](update.md) | `kapitsa update`, `--check` and automatic updates | no |
