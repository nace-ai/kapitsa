# Homebrew-core submission status

## Current: ready to submit after 2026-11-15

Kapitsa meets every homebrew-core requirement except the 30-day repo age guideline.

## Acceptance checklist

| Requirement | Status |
| --- | --- |
| Notability (225 stars for self-submission) | 698 stars, 102 forks |
| Repo age (30 days) | 23 days (created 2026-10-16), eligible ~Nov 15 |
| Open-source license (DFSG-compatible) | Apache-2.0 |
| Builds from source | CMake, Apple Clang |
| Stable releases | v0.2.31 (2026-11-05) |
| Active development | Pushed 2026-11-08 |
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

## To submit

1. Wait until 2026-11-15 (30-day repo age)
2. Fork `Homebrew/homebrew-core`
3. Create branch, copy `kapitsa-core.rb` as `Formula/k/kapitsa.rb`
4. Update URL and SHA to the latest release
5. Open PR with the notability evidence (698 stars, 102 forks, 9 models benchmarked on 5 Macs)

## Differences from the tap formula

The tap formula (`kapitsa.rb` in `nace-ai/homebrew-tap`) installs a pre-built binary.
The homebrew-core formula (`kapitsa-core.rb`) builds from source, as homebrew-core requires.
The `head` block, bottles, and binary install path are removed.
