# M2 Max, 64 GB: Qwen3.8-Flash-Next Q2_0 resident, 43 and 46 tok/s decode, 2026-10-26

The Q2_0 route in [docs/usage.md](../../../docs/usage.md) describes a Flash-Next pack whose routed experts
are 2-bit, sized so that a 48–64 GB Mac streams them. This records what it measures where the **whole**
store fits: on 64 GB every one of its 24,576 experts stayed resident, and nothing was read from the SSD
after the load.

Two full `kapitsa bench` runs on one MacBook Pro (14-inch, 2023), `Mac14,5`, Apple M2 Max, 38-core GPU,
64 GB, macOS 27.0.1 (26A434), AC power, power mode automatic, thermal state nominal, 0 swapouts.
Kapitsa `0.2.17` at `62c9ee0`. One run was flagged for contention and one was quiet; both are below, and
the result files are [run1-with-contention.json](run1-with-contention.json) and
[run2-quiet.json](run2-quiet.json).

## What was run

The published protocol, through `kapitsa bench` with no flags: five 512-token coding answers at T = 0.6 and
five greedy, a cold 4K- and a 16K-token prompt, three follow-ups in each, the tool-call check, one slot, a
32K context, `NI_*` variables at their defaults. Decoding used the model's **MTP head at depth 2**
("KFlash (MTP)"), which is the default for this pack.

The model is the pack `docs/usage.md` describes, built with `tools/qwen4_exp_q20.py` from ISTA-DASLab's
GSQ-RCO Q2_0 GGUF (experts) and Nace AI's MLX 4-bit pack (everything else, the MTP head and the 32 GB
n-gram table). Its load line here:

```
read 4.99 GB of weights in 584 ms (8.5 GB/s)
MTP head loaded in 0 ms
expert store: all 34.2 GB of experts fit the 51.3 GB the budget leaves: resident, no SSD reads
expert store: 24576 slots of 1.39 MB (34.23 GB, 100.0% of the 24576 experts) ... read at load in 9226 ms (3.91 GB/s)
loaded ...: Qwen3.8-Flash-Next, 48 layers (12 attention), 5.0 GB of weights in 14497 ms;
  4-bit group 64 4%, 6-bit group 64 3%, 8-bit group 64 2%, Q2_0 91%, bf16 0%;
  n-gram table 320001536 x 160, 4-bit group 32, 32.0 GB mapped; routed experts: 24576 of 24576 in memory
```

So the 2-bit experts are 91% of the weight bytes, and the config's top-level `"bits": 4` names only the
dense part. `kapitsa bench` labelled the folder "4-bit" before #649 (this run's `model.precision` was
recorded under that older build); the format is 2-bit in the experts and 4–8-bit in the rest.

## Results

| | T = 0.6 decode | greedy decode | prefill 4K | prefill 16K | follow-up TTFT | peak memory |
|---|---:|---:|---:|---:|---:|---:|
| run 1 (flagged) | 46.07 tok/s | 43.24 tok/s | 433.7 tok/s | 386.5 tok/s | 0.16 s | 39.1 GB |
| run 2 (quiet) | 42.75 tok/s | 42.95 tok/s | 413.7 tok/s | 372.3 tok/s | 0.15–0.17 s | 39.1 GB |

Each decode figure is the median of five runs, all `finish=length` (the 512-token limit, so no answer
ended early and inflated a rate). The runs, in order:

| run | T = 0.6, five runs | median | greedy, five runs | median |
|---|---|---:|---|---:|
| 1 | 53.37, 45.79, 48.29, 42.06, 46.07 | 46.07 | 42.60, 43.43, 41.60, 45.35, 43.24 | 43.24 |
| 2 | 49.92, 44.12, 41.72, 42.75, 41.54 | 42.75 | 42.95, 43.29, 42.03, 40.91, 44.32 | 42.95 |

Run 1 carried `CONTENTION: sustained foreign GPU load > 5% in ['prefill_4000', 'prefill_16000'] (mean
5.18%, top ['ZCode'])` — the editor this was driven from. Its decode numbers are the *higher* of the two,
so the contention did not flatter them, but it is why run 2 is here as the cleaner reading. The two runs
differ by 7% at T = 0.6 and 0.7% greedy, which is this Mac's spread rather than a difference in settings;
nothing was changed between them.

Load time is about 25 s (5 s of weights, 9.2 s to read the 34.2 GB of experts, the rest buffers and
pipelines), and it happens once per server start.

## What is not claimed

- **Not a comparison.** No other engine was run on this Mac, so this says nothing about Kapitsa against
  Strata, llama.cpp or MLX here. It is one engine's numbers under the published protocol.
- **Not the 4-bit pack's numbers.** Those need a 128 GB Mac; the whole point of this route on 64 GB is
  that the experts are 2-bit, which is a quality trade the pack's own card discusses.
- **One Mac, two runs.** The spread above is 5 runs within a run and 2 runs across; that is not a
  population.
- The n-gram table is complete (32 GB, `--table full`), not sparse, so the follow-up TTFTs are not
  flattered by a partly filled table.
