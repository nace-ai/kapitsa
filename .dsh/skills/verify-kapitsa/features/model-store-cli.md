# Model store CLI

A user names a model the short way (`4b`, `27b:8bit`, `35b`), a full catalog name (`qwen3.5-4b:4bit`), a Hugging Face
repo or a folder. `kapitsa pull` downloads it with its MTP head and DFlash2 drafter into `~/.cache/kapitsa/models`
(`KAPITSA_HOME` moves it). `--dry-run` shows what a name resolves to without downloading. `list` shows what is
downloaded and `rm` deletes it. Exit codes are 0 for success, 1 for a failure (download, port, model) and 2 for a
usage error such as an unknown model.

## Sub-features

- MS1: `pull --dry-run --json NAME` resolves a short name to a pinned repo, revision, variant and companions
- MS2: `pull --dry-run NAME:VARIANT` prints the human summary (folder, weights, MTP head, drafter, API name)
- MS3: an unknown model exits 2 and lists the valid names
- MS4: `list` / `list --json` on an empty store
- MS5: `rm` of a model that is not downloaded says so and exits 0
- MS6: a real `pull` (network, gigabytes): out of scope for the offline runner

## How to get to it (user POV)

In a terminal: `kapitsa pull 4b --dry-run`, `kapitsa list`, `kapitsa rm 4b`. `kapitsa --help` lists every model and
its short names.

## Driving it with vk

Preconditions: `$V doctor` exits 0. Nothing else: the scratch home starts empty and the Hub is a closed port.

| ID | Command | Observable result |
| --- | --- | --- |
| MS1 | `$V cli pull --dry-run --json 4b` | exit 0; one JSON object with `"model":"qwen3.5-4b:4bit"`, `"downloaded":false`, a 40-hex `revision`, `companions` of kind `mtp-sidecar` and `dflash`; `dir` is under the scratch home |
| MS2 | `$V cli pull --dry-run 27b:8bit` | exit 0; lines `folder:` (with "not downloaded yet"), `weights:`, `variant:    8bit (MLX affine 8-bit, group 64)`, `MTP head:`, `drafter:`, `API name:   qwen3.8-27b:8bit` |
| MS3 | `$V cli pull --dry-run nosuchmodel` | exit 2; stderr `kapitsa: Unknown model "nosuchmodel". Choose qwen3.5-4b (4b), ...` |
| MS4 | `$V cli list --json`, then `$V cli list` | `[]`; then "No models downloaded yet (<scratch>/models)." and an `Available:` list with sizes |
| MS5 | `$V cli rm 4b` | exit 0; "qwen3.5-4b:4bit is not downloaded." |
| all | `python3 tests/test_server.py --cli build/kapitsa` | the repo's own offline CLI checks pass (names, aliases, dry runs, list, rm, early port errors) |

## Gotchas

- `pull --dry-run` never touches the network, so it passes offline. A real `pull` against the closed port fails
  with exit 1, which is the runner working as intended.
- The JSON field order and the `Available:` sizes come from the catalog. A catalog change moves them, and AGENTS.md
  ("Docs follow numbers") wants every copy of a changed number updated.
- A folder path is a valid MODEL everywhere and is never copied into the store.
