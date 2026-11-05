## What changed and why

<!-- One change per PR. -->

## Default for

<!-- A new feature (kernel path, decode/draft/verify strategy, cache or memory policy, flag or env variable) is on by default
     for at least one model on at least one Mac: name them, e.g. "Qwen3.6-35B-A3B on M5 Pro". Off on every model and Mac
     does not merge. Not a feature (kill switch, debug switch, measuring tool, docs): write "not a feature". -->

## Measured

<!-- Before/after numbers, interleaved A/B on an idle GPU (A B A B), with a `Runs:` line: Mac, macOS, model, commit.
     Output check for anything that touches numerics: greedy tokens and logits against the old path, differences only at near-ties.
     New or changed Metal kernels: say you ran them at small sizes with MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1. -->

## Checked

<!-- What you ran: the CPU tests, prek. CI covers only the CPU side. -->

## Not verified

<!-- Everything you did not run, above all anything that needs a GPU. "Nothing" only if true. -->
