# Results, 2026-11-01: Clover-35B-A3B on the MacBook Pro M5 Max, 128 GB (main `7f4e709`)

DiffuSpace's Clover-35B-A3B re-measured on main `7f4e709` (Kapitsa 0.2.25, Release), the same binary as Kapitsa's other
M5 Max cells of that day ([2026-11-01-m5-max-kapitsa](../2026-11-01-m5-max-kapitsa/README.md)), so the poster's whole
Kapitsa column is one build. Written by [bench/harness](../../harness/README.md), 11:17-11:25 local time:
`compare_engines.py configs/kapitsa.json clover.json --only clover-kapitsa --agentic --skip tools,agent,cache,sustain
--quiet-timeout 60`, three runs under the GPU lock, with the context sweep (cold prefill and decode from 1K to 32K).
MacBook Pro (16-inch, 2026), `Mac17,6`, M5 Max, 40-core GPU, 128 GB, macOS 26.4 (25E246), AC power, High Power mode
([machine.json](machine.json)).

**The setting.** Kapitsa's default for Clover (`NI_CLOVER_DECODE=auto`): Clover's causal pathway has the 35B-A3B's
shape, so the 35B's DFlash2 drafter drafts and KFlash verifies, the setting the 2026-10-28 cell found fastest (`auto`
319 against Clover's own self-speculative `spec` 214, [that session](../2026-10-28-m5-max-clover/README.md)). The label
is [`clover.json`](clover.json), passed after `configs/kapitsa.json`; the repository's
[`configs/clover.json`](../../harness/configs/clover.json) is the same label written out in full. The weights are
`kapitsa pull clover` (Nace AI's 4-bit conversion of `Diffuspace/Clover-35B-A3B-Preview` @ `4b0402b`).

| | Prefill 4K / 16K | Decode T 0.6, median (range) | Greedy | Cached follow-up 4K / 16K | Peak to 16K / with the 32K sweep |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2026-10-28, `80740c0`, `kapitsa bench` (one run) | 6,002 / 5,313 | 319.2 (249.7-334.1) | 332.4 | 0.04 / 0.05 s | 21.1 GB |
| **2026-11-01, `7f4e709`, three runs** | **5,981 / 5,227** | **304.0** (303.5-319.8) | 332.6 | 0.04 / 0.04 s | 21.5 / 22.0 GB |
| Change | -0.3% / -1.6% | -4.8% | +0.1% | | |

The three runs: 319.8, 304.0 and 303.5 tok/s decode, 5,973 / 5,211, 5,997 / 5,227 and 5,981 / 5,268 prefill. The old
cell (one run, 319.2) is at the top of the new runs' range; greedy is level. The build is Release on `7f4e709`, every run
on AC power in High Power mode, and the third run has no contention at all (0.0% in every phase), so the lower median is
not the desktop app's load. Two of the three runs carry a repetition flag on one decode answer, as the 2026-10-28 cell
did: Clover's `# ------` separator lines count as repeated 4-grams (that session's README explains it). The first two
runs are contention-flagged in their prefill or sweep phases by the desktop app driving the session (6.9% worst-phase
mean in prefill, 13.7% in the first run's sweep); no decode phase is flagged (3.2% or less).

**From 1K to 32K context** (the median of the three runs; two cold requests per size per run, each answered with 192
tokens, so the decode column is that answer's rate):

| | 1K | 2K | 4K | 8K | 16K | 32K (31,549 tokens) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill tok/s | 4,572 | 5,546 | 5,688 | 5,280 | 4,707 | 3,788 |
| Decode tok/s | 230.0 | 229.6 | 231.9 | 203.1 | 167.3 | 143.3 |

## Files

- `results/`: the three runs. `logs/`: their server logs. `session/block7.log`: the runner's console output.
- `clover.json`: the label as run. [machine.json](machine.json): the Mac, macOS, power settings and the binary's sha256.

Local paths are replaced by placeholders: `~` for the home folder, `<repo>` for the Kapitsa checkout, `<run-root>` for
the harness's output folder, `<results>` for this folder, `<host>` for the host name.
