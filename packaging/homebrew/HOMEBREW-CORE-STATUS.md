# Homebrew-core submission status

## Current: submitting on 2026-11-15

The repo turned 30 days old on 2026-11-15, the last homebrew-core requirement it lacked.

## Acceptance checklist

| Requirement | Status |
| --- | --- |
| Notability (225 stars for self-submission) | 797 stars, 105 forks (2026-11-15) |
| Repo age (30 days) | 30 days on 2026-11-15 (created 2026-10-16) |
| Open-source license (DFSG-compatible) | Apache-2.0 |
| Builds from source | CMake, Apple Clang |
| Stable releases | v0.2.36 (2026-11-15) |
| Active development | Pushed 2026-11-15 |
| livecheck | GitHub latest release strategy |
| Platform restriction documented | arm64 macOS only (Metal kernels) |
| No .app bundle | CLI + server |
| No self-update | Uses Homebrew's upgrade mechanism |

## Formula

`kapitsa-core.rb` builds from the source tarball with CMake. It installs:

- `kapitsa` (CLI and server)
- `kapitsa-systemone` (Drex decision models)
- `kapitsa-drex-pointer` (pointer head tool)
- Bench harness in pkgshare

## SHA256

The `sha256` in the formula is the hash of the GitHub archive tarball at
`https://github.com/nace-ai/kapitsa/archive/refs/tags/v<VERSION>.tar.gz`.
To recompute it for a new release:

```bash
curl -sL -o kapitsa-src.tar.gz \
  "https://github.com/nace-ai/kapitsa/archive/refs/tags/v<VERSION>.tar.gz"
shasum -a 256 kapitsa-src.tar.gz
```

This is NOT the same as `gh api repos/nace-ai/kapitsa/tarball/v<VERSION>`, which returns
a different archive with a different hash.

## To submit

1. Wait until 2026-11-15 (30-day repo age)
2. Fork `Homebrew/homebrew-core`
3. Create branch, copy `kapitsa-core.rb` as `Formula/k/kapitsa.rb` without its header comments (homebrew-core formulas carry none)
4. Update URL and SHA to the latest release (see SHA256 section above)
5. Open PR with the notability evidence (stars, forks, community benchmark results)

## Tap/core name clash

Both the tap (`nace-ai/tap/kapitsa`) and the homebrew-core formula install a `kapitsa` binary.
Users must `brew uninstall nace-ai/tap/kapitsa` before installing from homebrew-core, or Homebrew
will refuse. The formula header notes this.

## `kapitsa update`

`kapitsa update` (src/update/update.mm) reads the Cellar's INSTALL_RECEIPT.json for the
source tap. A homebrew-core install (`"tap": "homebrew/core"`) runs `brew upgrade kapitsa`;
a tap install or a missing receipt runs `brew upgrade nace-ai/tap/kapitsa` as before.
The `--HEAD` variant adds `--fetch-HEAD` in both cases. Tests in tools/update_test.cpp
cover core receipt, core HEAD, tap receipt, and no receipt (404 checks).

## Differences from the tap formula

The tap formula (`kapitsa.rb` in `nace-ai/homebrew-tap`) installs a pre-built binary.
The homebrew-core formula (`kapitsa-core.rb`) builds from source, as homebrew-core requires.
The `head` block, bottles, and binary install path are removed.
