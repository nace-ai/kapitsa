# Update

`kapitsa update` replaces an install.sh install with the newest release, after checking its sha256 and signature, and
keeps the old binary for `--rollback`. `--check` only asks: it exits 0 when up to date, 100 when a newer release exists
and 1 when it cannot tell. Once a day a command also starts a quiet automatic update in the background. It is off
with `KAPITSA_NO_AUTO_UPDATE=1`, `kapitsa update --auto off`, when `CI` is set (and not 0/false), and for root.
Homebrew, app bundle and source builds are detected and never replaced.

## Sub-features

- UP1: `update --check` when the release server cannot be reached: exit 1
- UP2: `update --check` up to date / newer / older / prerelease (against a local release server)
- UP3: `update` on an install.sh install: one-rename switch, rollback copy, refusals of bad releases
- UP4: a source build (this checkout's `build/kapitsa`) is never replaced
- UP5: automatic updates: once a day, detached, the off switches

## How to get to it (user POV)

`kapitsa update --check`, `kapitsa update`, `kapitsa update --rollback`, `kapitsa update --auto off`.

## Driving it with vk

Preconditions: `$V build` with `update_test` (the default `vk build` builds it). Nothing touches GitHub: `vk cli`
points `KAPITSA_UPDATE_URL` at a closed port, and the test serves its own releases.

| ID | Command | Observable result |
| --- | --- | --- |
| UP1 | `$V cli update --check` | exit 1; stderr `kapitsa: Can't check for a new release: can't reach 127.0.0.1 (...)` |
| UP2–UP5 | `python3 tests/test_update.py build` | exit 0; sections 1–7 pass against a release server in-process, with temporary installs only |

## Gotchas

- Never run `build/kapitsa update` without `vk cli` or the test. On a real install it would replace the user's
  `kapitsa`.
- `vk` exports `KAPITSA_NO_AUTO_UPDATE=1` for everything it runs, so a verification run never starts a background
  update.
- `KAPITSA_UPDATE_URL` must be https, or http to this Mac. Anything else is refused with an error naming the
  variable.
