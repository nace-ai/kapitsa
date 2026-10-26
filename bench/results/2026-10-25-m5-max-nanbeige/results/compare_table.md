
### Nanbeige4.3-33B (Apple M5 Max, 128 GB, macOS 26.4)

| engine | decode tok/s (median, T=0.6) | decode greedy tok/s (T=0) | prefill 4k tok/s | prefill 16k tok/s | cached TTFT 4k s | cached TTFT 16k s | peak memory GB | tools | flags |
|---|---|---|---|---|---|---|---|---|---|
| Kapitsa (ours) | **28.8** (28.5-29.0) | **28.7** (28.5-28.9) | 532 | 483 | 0.26 | 0.43 | **25.3** | pass | - |
| oMLX | 28.0 (27.3-28.2) | 28.4 (28.3-28.5) | **766** | **634** | 0.50 | 1.56 | 35.2 | pass | cached_16000: turn 1 finish_reason=length (engines may not cache a 'length' turn); cached_16000: follow-up reused only 0 of 20522 prefix tokens |
| mlx-vlm server | 27.6 (27.6-27.7) | 28.0 (28.0-28.0) | 705 | 589 | 0.66 | 1.52 | 34.6 | pass | decode: repetition loop (distinct-4-gram 0.4424); cached_4000: turn 1 finish_reason=length (engines may not cache a 'length' turn); cached_16000: turn 1 finish_ |
| mlx-lm | 27.0 (26.9-27.0) | 27.4 (27.4-27.4) | 668 | 573 | 0.40 | 0.65 | 40.6 | pass | - |
| llama.cpp | 25.9 (25.8-26.1) | 25.4 (25.3-26.0) | 496 | 394 | **0.25** | **0.33** | 33.3 | pass | decode: repetition loop (distinct-4-gram 0.3841); decode: repetition loop (distinct-4-gram 0.3963); decode: repetition loop (distinct-4-gram 0.4066); decode: re |
| Ollama | 25.6 (25.5-25.8) | 24.8 (24.8-24.9) | 503 | 394 | 0.25 | 0.39 | 14.5 (+18.4 mmap) | pass | decode: repetition loop (distinct-4-gram 0.4462); cached_4000: turn 1 finish_reason=length (engines may not cache a 'length' turn); cached_16000: turn 1 finish_ |

Runs: Kapitsa = `nb43-33b-kapitsa` 5bee92c (2026-10-25 19:37:22); oMLX = `nb43-33b-omlx` omlx 0.7.3.dev2 mlx_lm 0.32.3.dev8+g0d4d3d04f mlx 0.32.4 (2026-10-25 19:56:10); mlx-vlm server = `nb43-33b-mlxvlm` unknown (2026-10-25 19:49:42); mlx-lm = `nb43-33b-mlxlm` mlx_lm 0.32.2 mlx 0.32.4 (2026-10-25 19:43:37); llama.cpp = `nb43-33b-llamacpp` 0.7.0-dev (b11871, b3fa69955) (2026-10-25 20:03:03); Ollama = `nb43-33b-ollama` 0.40.11 (2026-10-25 20:09:27)

`!` = that measurement is INVALID (see flags); bold = best valid value (memory: only rows whose weights count in phys_footprint; `+N mmap` = resident mmap'd weight files on top); ranges are min-max over runs.
