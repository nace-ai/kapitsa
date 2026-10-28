# Single requests and `clover_bench` runs behind the write-up's other numbers

Same Mac, macOS, power and pack as [the README](README.md), 2026-10-28, #776's code (`b0e7160` / `80740c0`; the
decoding is the same). The server lines are `kapitsa serve`'s own log line at the end of each request (`chat: ...`),
cut to the prompt, the tokens and the cycles; one server per block, `--ctx 8192 --slots 1 --disk-cache 0` unless said.

**The bench's coding prompt** (`src/bench/protocol.cpp`'s decode request: its system prompt, `[run <nonce>] Write a
complete, well-commented Python module that implements an LRU cache with TTL expiry, ...`, 512 tokens, thinking off):

```
auto (DFlash2), T 0.6, three nonces:
chat: prompt 89 (cached 0) in 84 ms, 512 tokens in 1684 ms (303.4 tok/s), length, 3.15 tok/cycle, KFlash (DFlash2): 162 cycles
chat: prompt 91 (cached 0) in 63 ms, 512 tokens in 1808 ms (282.7 tok/s), length, 2.82 tok/cycle, KFlash (DFlash2): 181 cycles
chat: prompt 86 (cached 0) in 62 ms, 512 tokens in 1780 ms (287.1 tok/s), length, 2.94 tok/cycle, KFlash (DFlash2): 174 cycles
spec (self-speculative), greedy, four nonces, then without one:
chat: prompt 93 (cached 0) in 84 ms, 512 tokens in 2553 ms (200.2 tok/s), length, 4.22 tok/cycle (8-row draft and verify), KFlash (none): 121 cycles
chat: prompt 93 (cached 0) in 55 ms, 512 tokens in 2592 ms (197.1 tok/s), length, 4.12 tok/cycle (8-row draft and verify), KFlash (none): 124 cycles
chat: prompt 92 (cached 0) in 56 ms, 512 tokens in 2320 ms (220.3 tok/s), length, 4.65 tok/cycle (8-row draft and verify), KFlash (none): 110 cycles
chat: prompt 93 (cached 0) in 56 ms, 512 tokens in 2728 ms (187.3 tok/s), length, 3.96 tok/cycle (8-row draft and verify), KFlash (none): 129 cycles
chat: prompt 73 (cached 0) in 75 ms, 512 tokens in 2286 ms (223.5 tok/s), length, 4.73 tok/cycle (8-row draft and verify), KFlash (none): 108 cycles
diffusion (block diffusion, 32-row blocks, threshold 0.95), greedy:
chat: prompt 93 (cached 0) in 84 ms, 512 tokens in 11200 ms (45.6 tok/s), length, 16 blocks, 411 denoising passes (1.25 tok/pass), KFlash (none): 16 cycles
```

**Other prompts:** "Explain in detail how a hash map works, then implement one in Python with open addressing." (512
tokens, T 0.6): `auto` 278.6 tok/s, 3.01 tokens a cycle; the same prompt on the causal pathway with no drafter (the pack
presented as a plain `qwen3_5_moe` folder) 212.4 tok/s, one token a cycle. "A train leaves at 3:40 pm and the trip takes
2 hours 35 minutes. When does it arrive?" (thinking on, both answered `6:15 pm`): `auto` 298.1 tok/s at 3.41 tokens a
cycle, `spec` 207.6 at 4.41.

```
chat: prompt 31 (cached 0) in 67 ms, 512 tokens in 1834 ms (278.6 tok/s), length, 3.01 tok/cycle, KFlash (DFlash2): 170 cycles
chat: prompt 40 (cached 0) in 76 ms, 76 tokens in 252 ms (298.1 tok/s), stop, 3.41 tok/cycle, KFlash (DFlash2): 22 cycles
chat: prompt 40 (cached 0) in 66 ms, 76 tokens in 361 ms (207.6 tok/s), stop, 4.41 tok/cycle (8-row draft and verify), KFlash (none): 17 cycles
```

**`build/clover_bench <pack> N [prompt]`** (self-speculative cycles without the server, greedy, the first four cycles
left out of the timing):

```
short C++ answer (its default prompt), 300 tokens: 5.98 tokens/cycle; draft 9.86 ms + verify 10.84 ms GPU, 21.02 ms wall a cycle; 276.1 tok/s
  the same with NI_CLOVER_ACCEPT=exact: 5.75 tokens/cycle; 21.06 ms a cycle; 264.1 tok/s
the bench's coding prompt, 512 tokens: 5.01 tokens/cycle; draft 10.06 ms + verify 10.77 ms GPU, 21.14 ms a cycle; 237.0 tok/s
the hash-map prompt above, 600 tokens: 4.19 tokens/cycle; draft 10.16 ms + verify 10.82 ms GPU, 21.48 ms a cycle; 193.9 tok/s
```

**Two slots** (`--slots 2`, `spec`, #776's review): two greedy requests alone, then both at once; the concurrent
replies are identical to the solo ones, and the two decodes took turns 23 and 22 times:

```
chat: prompt 30 (cached 23) in 12 ms, 351 tokens in 2066 ms (169.4 tok/s), stop, 6.73 tok/cycle (8-row draft and verify), ...; turns given to other requests: 0 in the prefill, 23 while decoding
chat: prompt 24 (cached 17) in 12 ms, 153 tokens in 1882 ms (80.8 tok/s), stop, 3.45 tok/cycle (8-row draft and verify), ...; turns given to other requests: 0 in the prefill, 22 while decoding
```
