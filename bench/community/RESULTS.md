<!-- Written by tools/community_table.py from bench/community/results/*.json: do not edit by hand. -->

# Community results

Kapitsa measured on people's own Macs with `kapitsa bench`: the protocol of the published benchmarks ([bench/harness](../harness/README.md)), as a client of the local server. Add yours with `kapitsa bench --submit`; [README.md](README.md) says how runs are checked.

Writing: tokens per second of a 512-token coding answer, median of the runs (5, or 3 in a quick run), at temperature 0.6 and greedy. Reading: a cold prompt of about 4K and 16K tokens, prompt tokens per second to its first token. Follow-up: the time to the first token of a new message in that conversation. Memory: the server's peak, GB of 2^30 bytes (a quick run reads no 16K prompt, so its peak is lower: about 0.4 GB on the 4B). ⚠: other apps used the GPU during that measurement (contention flagged), so it may be low. Different Macs, macOS versions and Kapitsa versions give different numbers; these are what each run measured.

No results yet. Add the first with `kapitsa bench --submit`.
