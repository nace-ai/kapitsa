# Results, 2026-10-28: Splash 1.4.1 and Kapitsa 0.2.23 on an EC2 M4 Pro, macOS 26.7

Written by [bench/harness](../../harness/README.md) on a second AWS `mac-m4pro.metal` instance: a Mac mini (Mac16,11),
Apple M4 Pro, 14 CPU cores, 20 GPU cores, 48 GB, macOS 26.7 (25G229), `powermode 2`, sleep off, on AC power. Splash
needs macOS 26.4 or later, so this Mac runs it; the [first Mac](../2026-10-28-m4pro-ec2/README.md) is on macOS 15.8 and
has Kapitsa's other five models. Both engines ran back to back on this one instance, one config at a time, on the same
weights (`kapitsa pull 35b 27b`), the same drafters and the harness's frozen corpus, and the harness held its GPU lock for
each run. Times are UTC.

- **Splash** is the 1.4.1 release tarball (`splash-1.4.1-arm64-macos26.tar.gz`, sha256 `5746068d…0d5c`), run through the
  harness's `splash.json` with its version relabelled 1.4.1, `--max-context 32K`, `--expert-memory off`, offline, the
  DFlash2 drafter. The relabelled config is not committed; it was made on the Mac with
  `sed 's/"version": "1.4.0"/"version": "1.4.1"/; s/Splash 1.4.0/Splash 1.4.1/' configs/splash.json > configs/splash-141.json`
  and its other fields are `splash.json`'s. That `sed` changed only the first mention in the config's `note`, so the
  Splash records' `note` still names `splash-1.4.0-arm64-macos26.tar.gz`: that is the config's unedited wording, and the
  tarball that ran is the 1.4.1 one above (`logs/setup.log`).
- **Kapitsa** is the v0.2.23 release binary (`628abd76…`, the same file as on the first Mac) with its own DFlash2
  drafter, `serve --ctx 32768 --slots 1`.
- The harness is a source tarball of 909e2823 (no git checkout).

The run order was Kapitsa 35B (11:45), Kapitsa 27B (11:48), then, after a first Splash attempt that found no engine
(the `.sha256` file from the release is not in `shasum -c` format, so the tarball was never unpacked; the checksum
was compared by hand and matched), Splash 35B (12:15) and Splash 27B (12:19).

| Result file | Started | Decode, T = 0.6 / greedy (tok/s) | Prefill, 4K / 16K (tok/s) | First token, 4K / 16K (s) | Peak footprint (GB) | Agent session, writing (tok/s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `q36-35b-kapitsa-20261028-114533.json` | 11:45 | 168.9 / 180.0 | 1171 / 979 | 3.53 / 16.64 | 21.7 | 142.2 |
| `q36-35b-splash-20261028-121551.json` | 12:15 | 139.9 / 140.6 | 1056 / 915 | 3.92 / 17.80 | 23.6 | 123.6 |
| `q38-27b-kapitsa-20261028-114819.json` | 11:48 | 65.5 / 68.4 | 392 / 343 | 10.55 / 47.51 | 21.0 | 60.7 |
| `q38-27b-splash-20261028-121936.json` | 12:19 | 49.5 / 56.9 | 251 / 236 | 16.48 / 69.11 | 27.1 | 54.6 |

Kapitsa against Splash 1.4.1 on this Mac: the 35B-A3B reads a 4K prompt 1.11x faster and decodes 1.21x faster at
T = 0.6; the 27B reads 1.56x faster and decodes 1.32x faster.

Flags, which the records keep and `bench/README.md` shows:

- `q36-35b-splash`: one of the five T = 0.6 decodes (136.7 tok/s) tripped the repetition check (distinct 4-grams
  0.47); all five count towards the median, which is 139.9 with it and 146.1 without (five valid runs, 132.7 to 174.1), so the
  35B Splash decode cell is the conservative one. The greedy decodes are clean.
- `q38-27b-kapitsa`: the agent replay wrote 3,209 of the recorded 3,577 tokens (4 turns stopped early), which does not touch the decode or prefill cells.

Prompt token counts match the harness's references to within 2 tokens (the two engines' tokenizers, 4,132 to 4,135
for the 4K prompt). Foreign GPU use averaged at most 0.05 % (peak 3.1 %) in each run, and none was flagged as contention.

## Record corrections

The Kapitsa records had the same `"version": "08e85c4"` as the [first Mac](../2026-10-28-m4pro-ec2/README.md#record-corrections):
Homebrew's checkout, not Kapitsa's. They are corrected the same way (`version` 0.2.23, the recorded value kept as
`version_recorded`, `provenance.kapitsa_git` removed). The Splash records are as written.

[logs](logs/) holds each run's harness and server log, `setup.log` (the bootstrap, with the `system_profiler` hardware
lines) and `splash.log` (the second Splash attempt).
