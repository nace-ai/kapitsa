# Community results

Kapitsa's speed on the Macs people actually have, measured with `kapitsa bench` and collected in
[RESULTS.md](RESULTS.md). Every run uses the protocol of the published benchmarks ([bench/harness](../harness/README.md)),
so a result from your Mac compares with the README's numbers and with everyone else's.

## Add yours

```bash
kapitsa bench --submit            # the best model for this Mac (in a terminal it asks which)
kapitsa bench --model 35b --submit
kapitsa bench --quick --submit    # a shorter run, marked quick
```

A full run takes 3 to 15 minutes, depending on the model and the Mac. For numbers that reflect the Mac and not what
else it was doing: plug it in, close other apps (browsers, video calls, anything that uses the GPU), and leave it alone
until the run ends. `kapitsa bench` waits for a quiet GPU before it measures and flags anything that disturbed a
measurement anyway.

At the end it asks once whether to send it: Enter is yes, `n` keeps it on your Mac only. Then:

- with [gh](https://cli.github.com) installed and logged in, it opens a pull request that adds the file under
  [results/](results/), from your fork (made if you have none). It uses your gh login and nothing else.
- without gh it sends nothing and says how to get gh (`brew install gh`, then `gh auth login`). Results are added only
  as pull requests from your fork, never as issues: to do it by hand, fork the repo, add the file under
  [results/](results/) and open a pull request.

`kapitsa bench submit` sends a result you saved earlier (they are in `~/.cache/kapitsa/bench/`), and
`--submit --dry-run` prints exactly what it would send, with the gh commands and the pull request, without sending
anything.

## What is in a result

The JSON follows [schema.json](schema.json) and is about 2 KB:

- **The Mac:** its model identifier (`sysctl hw.model`, e.g. `Mac16,7`) with Apple's name for it and the year
  ([mac-models.json](mac-models.json)), the chip, CPU and GPU cores, memory, macOS version and build.
- **Kapitsa:** its version and commit; the model, its precision, its pinned weights' revision, and how it decoded (the
  DFlash2, LiLiCorr or DSpark drafter, the MTP head, or plain).
- **The numbers:** writing speed (decode) at temperature 0.6 (Flint at 1.0, its model card's) and greedy, each run's and
  the median; reading a 4K- and a 16K-token prompt (prefill: tokens per second to the first token); the time to the first
  token of three follow-ups in each conversation; the server's peak memory; whether tool calls work.
- **The conditions:** power source, Low Power Mode and the energy mode, the worst thermal state, pages swapped out,
  other apps' share of the GPU and which apps, and the phases they disturbed; the harness's flags, and any measurement
  that failed its checks.

No serial number, host name, user name or file path.

## How runs are measured

`kapitsa bench` runs the harness's `bench.py` protocol, request for request (the same prompts, cut from the same frozen
corpus and embedded in the binary; the same sampling and thinking settings; the same checks and rates), as a client of
the local API, the way the harness measures every engine. It starts the model's server the way `compare_engines.py`
does: one conversation slot, a 32K context, `NI_*` variables left out; warms it up with a 2.9K-token prompt and a tool
call; waits until other apps leave the GPU alone for 15 seconds (5 minutes at most, 2 for a quick run); then measures
while a monitor samples every process's GPU time, the thermal state and the power source every 2 seconds.

| Measurement | Full run | Quick run |
| --- | --- | --- |
| Writing, T = 0.6 (Flint: 1.0) and greedy: a 512-token coding answer, the rate over the middle of the stream | 5 runs each, median | 3 runs each |
| Reading: a cold ~4K- and ~16K-token prompt (4,133 and 16,286 tokens with the Qwen templates, 4,183 and 16,514 with Nemotron's, 3,874 and 15,066 with Flint's) | both | the 4K one |
| Follow-up: three new messages in that conversation, 1 s apart, time to the first token | median of 3 | median of 3 |
| Peak memory: the server's `phys_footprint_peak` (what `footprint` shows), GB of 2^30 bytes | yes | yes |
| Tool calls: streamed and whole replies with two tools, thinking off and on, and a round trip | yes | yes |

A measurement is marked ⚠ when other apps used more than 5% of the GPU for three samples in a row, or on average, while
it ran (the harness's contention flag). macOS's photo and media analysis and the terminal showing the run are ambient:
recorded, never flagged (`BENCH_GPU_AMBIENT` adds names, as for the harness).

## How results are checked

Every pull request that adds files under `results/` runs [the check](../../.github/workflows/community-bench.yml),
`python3 tools/community_check.py`. A result passes when it validates against the schema and is plausible:

- a Kapitsa version that was released, or the current one;
- a model and precision of the catalog ([src/hub/catalog.cpp](../../src/hub/catalog.cpp)), at their pinned weights;
- Apple's name and year for its model identifier, as [mac-models.json](mac-models.json) has them (an identifier it
  does not have yet passes, with a warning);
- the protocol of its mode, the frozen prompt corpus, and no invalid measurement;
- rates, times and memory within physical bounds and consistent with each other (each median is its runs' median;
  prompt tokens within 1% of the reference; tokens per second equal to tokens over the time to the first token);
- not a copy of a result already in the table.

The pull request is labeled `benchmark`, plus `quick`, `contention`, `flagged` or `invalid` as they apply, and the
job's summary lists what passed and why anything failed. Results with flags are accepted and shown with them. Once
results reach `main`, the workflow rewrites RESULTS.md (`python3 tools/community_table.py`).

## Files

| File | What it is |
| --- | --- |
| [RESULTS.md](RESULTS.md) | The table: grouped by Mac, then model and precision, a row per run |
| [results/](results/) | One JSON file per run |
| [schema.json](schema.json) | The result's format (version 1) |
| [mac-models.json](mac-models.json) | Apple silicon Macs by model identifier, from Apple's "Identify your Mac model" pages |

For maintainers: `python3 tools/community_check.py FILE...` checks result files, and CI runs it on every pull request
that touches `results/`. `python3 tools/mac_models.py`
updates mac-models.json from Apple's pages when new Macs come out (`--check` only compares).
