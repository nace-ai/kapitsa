# Low-rank cache-key validation

Reference `39f20c8d`, candidate `15e494d1`. Subsequent evidence commits change no production source.

The CPU regression in `tools/dflash_lowrank_cache_test.cpp` holds scales/biases fixed, changes the final Q4 word and computes both heads' factors. The former key only saw identical dimensions/coefficients; the new key and the actual factors differ. Coefficients, rank, dimensions and restoration are checked separately.

On M3 Ultra (60 GPU cores, 96 GiB, macOS 26.4.1 / 25E253), `evidence.json` records the complete 584-row bitwise logit report, 270-case drafter intermediate comparisons, and reference/candidate cold/warm cache logs. Set `NI_DFLASH_LOWRANK=768` to exercise this path on M3; it is a default on the base 10-core M4, which was not available. Run `qmm_draft_check MODEL PROMPT_IDS OUT --from REFERENCE_OUT` under the repository GPU lock. Commands and binary hashes are in the evidence.

The rank-768 generation smoke uses a chat-templated counting prompt and `ni_run MODEL @IDS 1024 0 --dflash auto`. Four ABAB legs produce nonempty decoded text, 1,024 identical tokens with no EOS, 129 cycles and 7.930 tokens/cycle. Its 807.7/806.4/813.2/812.3 tok/s figures are a narrow no-regression check, not a throughput improvement claim. Prompt: `<|im_start|>user\nCount from 1 to 2000, one number per line, nothing else.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.

The separate default `kapitsa bench --quick` ABAB records are also retained. That command clears NI_* for its server, so these do not exercise rank 768 on M3. Each reports AC power, nominal thermals, no swapouts or contention. Neither campaign establishes a speedup.

Full HTTP tests report 47 passes and the same two timing assertions failing on both versions: a 693-token reply ends before the delayed side request is sent, and a holder ends before two 15-second keepalives can elapse. Increasing context/max_tokens retained the same baseline failures. These records are not counted as passes. No arithmetic, dispatch or Metal sources change. Other models and base-M4 hardware were not tested.

The complete `vk cpu-tests` suite, supplemental AGENTS.md CPU tools and `prek run --all-files` passed on the remote candidate. Their logs are included in `evidence.json`.
