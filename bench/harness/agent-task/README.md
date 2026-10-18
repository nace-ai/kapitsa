# Agent task: ledger

The coding task behind the agent-session benchmark ([agent-traces/opencode-ledger.json](../agent-traces/opencode-ledger.json),
replayed by [agent_replay.py](../agent_replay.py)). Original code written for this benchmark, licensed like the rest of
this repository (Apache 2.0).

`project/` is what the agent gets as its working directory (`/work/task` in the trace): `ledger`, a small Python
library and command line for personal ledgers kept as CSV files (money amounts, a CSV parser, monthly and per-category
reports), with a `unittest` suite of 35 tests and a sample ledger. Python 3.9 or later, standard library only. At the
start 15 tests fail (10 errors, 5 failures), from five problems a reviewer would find by reading the failing tests:

1. `Money.parse` rounds through binary floats and Python's half-to-even `round`, so `"1.005"` becomes 1.00 instead of
   1.01 (the docstring promises halves away from zero).
2. `Money.parse` does not accept accounting negatives such as `"(12.50)"`, which its docstring lists.
3. The parser splits lines on every comma, so a quoted description with a comma (`"Rent, January"`) or doubled quotes
   breaks the line; the sample ledger has both.
4. `monthly_totals` sorts months by month number only, so December 2025 comes after January 2026.
5. The command line's `monthly` command lacks the `--since` and `--until` options its usage text documents.

Everything else (the sample ledger's numbers, the table format, the error paths) already works, so a fix is a few small
edits in four files. The agent got [prompt.txt](prompt.txt) as its only message:

> The tests in this project are failing. Run them with `python3 -m unittest`, find the bugs and fix them in the code
> so that every test passes. Do not change the tests.

To check a working copy: `cd <copy> && python3 -m unittest` (35 tests, OK) and `diff -r project/tests <copy>/tests`
(no changes).

In the recorded session (opencode 2.0.28 on Qwen3.6-35B-A3B, 4-bit, greedy, thinking off) the agent ran the tests,
listed the folders, read the four modules and the four test files (four reads per request) and the sample ledger,
fixed the parser, `Money.parse`, the month order and the command line in five edits, ran the tests, needed five more
requests for the date options (the bad-date test wants argparse's exit code 2, and `cli.py` lacked an import), ran
the tests again and summed up: 19 requests, all 35 tests passing, the tests unchanged.
