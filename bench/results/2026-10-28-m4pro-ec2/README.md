# Results, 2026-10-28: Kapitsa 0.2.23 on an EC2 M4 Pro

Written by [bench/harness](../../harness/README.md) on an AWS `mac-m4pro.metal` instance: a Mac mini (Mac16,11),
Apple M4 Pro, 48 GB, macOS 15.8 (24H23), `powermode 2`, sleep off, on AC power. Nothing else ran on the instance: every
run has 0.0 % foreign and ambient GPU; the flags that remain are the records' own and are listed below. Times are UTC.

Kapitsa is the v0.2.23 release binary, installed by `install.sh --version 0.2.23` into `/opt/homebrew/bin`. Its
sha256 (`628abd76…`, in each record's `provenance.binary`) equals the binary in
`kapitsa-0.2.23-darwin-arm64.tar.gz`. The harness is a source tarball of 909e2823 (no git checkout, so
`harness.git_head` is null). Weights come from `kapitsa pull 4b 35b 27b dory lightning bolt flint` with the catalog's
default variants and companions; the 4B's drafter is z-lab's DFlash v1 (the catalog's `dflash4`, in the model's
`dflash2/` folder), as the server log says.

## Kapitsa (configs/kapitsa.json, dory.json, lightning.json, bolt.json, flint.json)

One config at a time, 10:53–11:18, in the harness's protocol (five sampled and five greedy decodes of 512 tokens,
prefill at 4,000 and 16,000 tokens).

| Result file | Started | Decode, T = 0.6 / greedy (tok/s) | Prefill, 4K / 16K (tok/s) | Peak footprint (GB) | Foreign GPU, mean / max (%) | Flagged |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `q35-4b-kapitsa-20261028-110404.json` | 11:04 | 187.2 / 202.2 | 1982 / 1569 | 4.7 | 0.0 / 0.0 | — |
| `q38-27b-kapitsa-20261028-105701.json` | 10:58 | 65.1 / 68.7 | 390 / 343 | 20.4 | 0.0 / 0.0 | agent replay wrote 3,209 of 3,577 recorded tokens |
| `q36-35b-kapitsa-20261028-105346.json` | 10:54 | 164.3 / 184.2 | 1172 / 981 | 21.1 | 0.0 / 0.0 | 1 of 5 T = 0.6 decodes: repetition loop |
| `dory-kapitsa-perloop-20261028-111203.json` | 11:12 | 61.2 / 68.1 | 576 / 502 | 4.5 | 0.0 / 0.0 | 1 of 5 T = 0.6 decodes: repetition loop |
| `lightning-kapitsa-20261028-110538.json` | 11:06 | 110.0 / 119.4 | 995 / 882 | 18.6 | 0.0 / 0.0 | — |
| `bolt-kapitsa-20261028-110818.json` | 11:09 | 70.0 / 74.6 | 543 / 537 | 22.7 | 0.0 / 0.0 | — |
| `flint-kapitsa-20261028-111516.json` | 11:16 | 125.6 / 128.5 | 1018 / 863 | 22.3 | 0.0 / 0.0 | cached 16K follow-up returned no content |

Against the MacBook Pro M4 Pro's Kapitsa cells in the top-level README (main with KFlash trees, macOS 27.0.1), prefill
at 4K and T = 0.6 decode:

| Model | MacBook Pro, README | EC2 Mac mini, 0.2.23 |
| --- | ---: | ---: |
| Qwen3.5-4B | 1930 / 201 | 1982 / 187 |
| Qwen3.8-27B | 396 / 63 | 390 / 65 |
| Qwen3.6-35B-A3B | 1168 / 161 | 1172 / 164 |
| Nemotron Dory1.0 4B | – / 64 | 576 / 61 |
| Nemotron 3.5 Lightning | 945 / 112 | 995 / 110 |
| Nemotron 3.5 Bolt | 519 / 69 | 543 / 70 |
| Recursive Flint | 968 / 128 | 1018 / 126 |

## Record corrections

The harness recorded `"version": "08e85c4"` and a `provenance.kapitsa_git` block. Both describe Homebrew's own git
checkout at `/opt/homebrew`, not Kapitsa: `checkout_of()` took the repository around the binary, whatever it was.
In these files `version` is set to `0.2.23` (what `kapitsa --version` prints), the recorded value is kept as
`version_recorded`, and `kapitsa_git` is removed. The same change fixes `checkout_of()` to accept only a Kapitsa
checkout.

Flags in these records, none of which moves a cell's median by more than 0.7 tok/s:

- `q36-35b`: one T = 0.6 decode tripped the repetition check (distinct 4-grams 0.46); the median is 164.4 with it and 164.9 without.
- `dory-perloop`: the same check on one T = 0.6 decode (0.48); 61.2 with it, 61.1 without.
- `q38-27b`: the agent replay wrote 3,209 of the recorded 3,577 tokens (4 turns stopped early); decode and prefill are untouched.
- `flint`: the 16K cached-prefix follow-up returned no content, so the harness stopped that cached series; the other
  cells are intact.

[logs](logs/) holds each run's harness and server log.
