# Math reasoning accuracy

Whether a precision (4-bit, a 4/8-bit mix, 6-bit, 8-bit) solves competition math as well as another, with thinking
on. [bench/quality](../quality/README.md) measures how far each precision's next-token distributions are from bf16 on
code, prose and a coding-agent chat; this measures the answers themselves, over chains of thought thousands of tokens
long, where small errors can add up. A claim that 4-bit "falls off a cliff" on such problems rests on one MATH-500
problem decoded greedily, where a single flipped token changes the answer; this needs enough problems and samples to
tell a real difference from sampling noise.

Standard library only (macOS's own `python3`), any OpenAI-compatible server. It reuses the harness's streaming client
and GPU lock (`bench/harness/bench.py`, `gpu_hogs.py`).

## The problems

| Set | Problems | Samples each | `max_tokens` | What |
| --- | ---: | ---: | ---: | --- |
| `math500-quick` | 50 | 1 | 32768 | MATH-500, level 5 |
| `math500-hard` | 150 | 1 | 32768 | MATH-500, levels 4 and 5 |
| `math500` | 500 | 1 | 32768 | all of MATH-500 |
| `aime2025` | 30 | 4 | 38912 | AIME 2025 I and II (integer answers) |

The quick and hard sets are fixed lists in [sets/](sets/): the problems of those levels, spread over MATH-500's seven
subjects in proportion to their share, the first of each subject in sha256(id) order. `reasoning.py sets --rebuild`
redraws them and checks that they match.

The problems are not stored here. `reasoning.py fetch` (or the first `run`) downloads them to
`~/.cache/kapitsa/reasoning` (`--cache` or `$KAPITSA_REASONING_CACHE` to move it) from pinned Hugging Face revisions
and checks each file's sha256: [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
(OpenAI's PRM800K subset of the MATH dataset, MIT) and
[opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025) (the problems are the Mathematical
Association of America's). math-ai/aime25 has the same answers but three problems whose LaTeX lost a backslash
(`\frac` stored as a form feed), so it is not used.

## A run

Each problem goes out as Qwen's math prompt (the problem, then "Please reason step by step, and put your final answer
within \boxed{}."), thinking on (`chat_template_kwargs.enable_thinking`), Qwen's sampling for thinking (temperature
0.6, top_p 0.95, top_k 20, no penalties) and a seed of its own per problem and sample, so a rerun asks the same
questions. Greedy decoding is not used: Qwen advises against it with thinking (it loops). The reply streams back;
its last `\boxed{}` is graded against the reference by [grader.py](grader.py). A reply cut short at `max_tokens` is
graded on what it answered, never on a box its unfinished thinking wrote.

`max_tokens` covers the thinking too. The server must hold the prompt and the reply in one conversation: `run` reads
its context from `/v1/models` and lowers `max_tokens` to fit, with a note. Start Kapitsa with `--ctx 40960` for the
sets' full budgets.

Every sample is one line of the results file (default `runs/<label>-<set>-<time>.jsonl`, git-ignored): the reply's
reasoning and content, the answer read, the verdict, the finish reason, tokens, time to first token and decode rate,
and a distinct-4-gram ratio (below 0.5: a loop). A header line per run records the server's `/v1/models` entry, the
request's settings, the machine, this checkout's commit and the dataset revision. A results file resumes: rerunning
the same command skips the samples already graded and retries the ones that failed (HTTP errors, timeouts). The GPU
lock (`/tmp/kapitsa-gpu.lock`) is held for the run unless `--no-gpu-lock` or `NACE_GPU_LOCK_HELD=1`.

```bash
cd bench/reasoning
kapitsa serve --model 27b --ctx 40960 &
python3 reasoning.py run --set math500-hard --label 27b-4bit
python3 reasoning.py run --set aime2025 --label 27b-4bit
kill %1
kapitsa serve --model 27b:8bit --ctx 40960 &
python3 reasoning.py run --set math500-hard --label 27b-8bit
python3 reasoning.py run --set aime2025 --label 27b-8bit
kill %1
python3 reasoning.py report runs/27b-*.jsonl --baseline 27b-4bit
```

A run takes about problems x samples x reply tokens / decode speed: with replies of 4,000 tokens at 60 tok/s, the
150 problems of `math500-hard` take under 3 hours, and twice that at half the speed. `--limit 3` makes a smoke test.
`--parallel N` sends N requests at once; Kapitsa batches their decoding, which can round differently, so use the same
`--parallel` for every label you compare.

## The report

`reasoning.py report FILE...` prints, per set and label: accuracy (the mean over problems of each problem's share of
right samples) with a 95% bootstrap interval over problems; the replies cut short, without an answer and looping; the
median reply length and decode rate; accuracy by MATH level. Each other label is then compared with the baseline on
the problems both ran: the difference of per-problem accuracy with its 95% interval, the problems each did better on
and, with one sample per problem, McNemar's exact p. `--json OUT` writes the same numbers; `--rename OLD=NEW` relabels;
`--regrade` grades the saved replies again with the current grader (after a grader fix, no rerun needed).

With 150 problems a single accuracy's interval is about plus or minus 7 points, and a paired difference of a few
points is within noise: a difference counts when its interval excludes zero. AIME's 30 problems need its 4 samples
each.

## The grader

[grader.py](grader.py) reads the last `\boxed{}` and compares it with the reference after removing presentation
(`\left`/`\right`, `\dfrac`, spacing, degrees, percent, dollar signs, thousands separators, unit words): as text
(`\text{(C)}`, names), base-n numerals, equations, tuples and intervals (brackets must match), unions, matrices,
unordered lists (`\pm` gives two), and expressions evaluated as complex numbers, with variables at seeded random
points. A mixed number like `1\frac{4}{5}` is a sum. What it cannot parse is compared as text, so unusual notation can
be marked wrong: `reasoning.py check` grades every MATH-500 reference solution against its answer (500 of 500) and
reads every AIME answer as an integer.

Tests: `python3 tests/test_reasoning.py` (no server, no GPU).
