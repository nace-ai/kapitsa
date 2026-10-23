# M3 Pro rival cells: fresh or reused (fixed before Phase B, 2026-10-22 23:20 M3 time; approved by the orchestrator)

The user's rule (21:02): a competitor runs once per machine, and only for a model this M3 Pro has no number for. Kapitsa
runs fresh on every model, on main e86dea7 (0.2.9), built on this Mac. Every rival cell carries the date it was measured.

The only M3 Pro rival numbers already merged are Flint's (`2026-10-22-flint-m3pro`, #306, battery, and
`2026-10-22-flint-m3pro-ac`, #341, AC). The community M3 Max runs are another chip and are not reused.

## Reused

- Flint × llama.cpp b11804 (Q4_1), measured 2026-10-22 on AC (#341, clean pass): 50.9 tok/s decode, 772 / 645 tok/s prefill at 4K / 16K.
- Flint × llama.cpp b11804, measured 2026-10-22 on AC (#341): the context sweep, 1K to 64K at -c 70000, which covers 1K to 32K.
- Flint × rec 0.1.0, 2026-10-22: refuses to load on 36 GB (#306 and #341).

## Fresh (the first M3 Pro number for that model × engine)

- Qwen3.5 4B: MTPLX (sustained, depth 1), mlx-vlm + MTP, llama.cpp + MTP (n-max 2), Ollama MLX runner.
- Qwen3.8 27B: MTPLX (turbo, depth 3), mlx-vlm + MTP, Ollama MLX runner; llama.cpp + MTP (n-max 3) in Phase B2.
- Qwen3.6 35B-A3B: MTPLX (sustained, depth 1), mlx-vlm + DFlash2, Ollama MLX runner; llama.cpp + MTP (n-max 1) in Phase B2.
- Qwen3.5 9B: llama.cpp (unsloth Q4_K_M), mlx-lm.
- Nemotron-Labs Dory 4B: mlx-lm (PR #1988).
- Nemotron 3 Nano 4B: llama.cpp (NVIDIA's Q4_K_M).
- Nemotron Nano 9B v2: mlx-lm (mlx-community 4-bit).
- Nemotron 3.5 Lightning: mlx-vlm + DSpark drafter.

Phase B2 runs after the sweeps: the two llama.cpp cells read Ollama's MTP GGUFs (about 25 and 20 GB), each pulled
only when no measurement runs, skipped if it would leave under 30 GiB free (the M3's floor), and deleted once its cell
is recorded. The other rival-only downloads (the Ollama MLX runner's models, the Qwen GGUFs, the 4B MTP drafter) are
deleted after Phase C, their last use. The Nemotron rivals' models are deleted after their sweeps (below).

## Not measured

- × (needs a newer macOS than 14.1): mlx-serve (macOS 26.2+), oMLX (macOS 15+), Splash.
- —: LM Studio (not installed); Bolt × mlx-vlm (the catalog bundle lacks the files it loads); Dory × llama.cpp (needs
  NVIDIA's fork); Lightning × llama.cpp (its GGUF is not downloaded).

## Phase C, 1K to 32K

Kapitsa is swept fresh on the 4B, 27B, 35B-A3B, 9B and Flint, each against the fastest of its Phase B rivals by decode
(T = 0.6 median); Flint × llama.cpp reuses #341's sweep. The rival sweeps ran after the Kapitsa ones, not beside them: the
queue's picker read the wrong field and chose no rival for any of the four, so they run in the Nemotron phase (below),
each the rival with the fastest valid decode: Ollama's MLX runner on the 4B, 27B and 35B-A3B, and the faster of the two
9B reruns on the 9B. Same machine, display asleep, and the same harness checks.

## Nemotron-H, after #351

On e86dea7 (and 0.2.8) Kapitsa could not run Dory, Nano 4B, Nano 9B, Lightning or Bolt on macOS 14.1: its Nemotron-H
kernels used device-scope fences (Metal 3.2, macOS 15+), and the server stopped at start with a Metal compile error.
#351 (9fc7ff3) split those merges on Metal 3.1. These five run on main c490f81 (the first main with #351): the protocol,
then each one's 1K-to-32K sweep against its rival above, swept once (Bolt has no rival cell). The two 9B rival cells
run again in this phase with the display asleep: the M3's display was woken during their first runs (01:11:53 to
01:19:22, the lock screen, video decoder idle). Their first records are kept in `results/display-woken/`.

The Nemotron-H cells are checked against the Nemotron prompt reference (4184 / 16517 tokens at 4K / 16K, Bolt 4183 /
16514, the harness configs' own for Dory, Lightning and Bolt), not the Qwen one (4133 / 16286). Nano 4B × llama.cpp
first ran against the Qwen reference and was marked INVALID for its prompt length only (4187 / 16519, within 1% of the
Nemotron reference). It runs again in this phase, and its first record is kept in `results/qwen-ref/`.

Nano 9B × mlx-lm stays INVALID for its prefill and follow-up phases. Its decode, 24.95 tok/s over 5 of 5 valid runs,
stands. The model's chat template turns thinking off only for a `/no_think` in a message and ignores the
`enable_thinking: false` that the harness sends every engine, so mlx-lm thinks there (`</think>` in the answer). The
prefill rates (250 / 244 tok/s at 4K / 16K) are times to the first token of that thinking answer. They are reported with
that flag and not rerun: the harness sends every engine the same request.

## Each rival's version and flags

The harness configs on main (`bench/harness/configs/*.json`, the M4 Pro's fastest setup per engine, not re-tuned on the
M3) plus `m3pro.json` here. Commands as the harness resolves them on the M3 (`compare_engines.py --dry-run`), `{models}`
being the model folder:

| Cell | Engine and version | Server command (flags) |
|---|---|---|
| `q35-4b-mlxvlm-mtp` | mlx-vlm server + MTP; mlx-vlm 0.7.8, mlx 0.32.4 | `HF_HUB_OFFLINE=1 APC_ENABLED=1 APC_DISK_ENABLED=0 ~/engines/mlxvlm/bin/python -m mlx_vlm.server --model {models}/qwen3.5-4b-4bit-mtp --host 127.0.0.1 --port 8282 --prefill-step-size 2048 --max-num-seqs 1 --draft-model {models}/drafters/Qwen3.5-4B-MTP-4bit --draft-kind mtp --draft-block-size 3` |
| `q38-27b-mlxvlm-mtp` | mlx-vlm server + MTP; mlx-vlm 0.7.8, mlx 0.32.4 | `HF_HUB_OFFLINE=1 APC_ENABLED=1 APC_DISK_ENABLED=0 ~/engines/mlxvlm/bin/python -m mlx_vlm.server --model {models}/qwen3.8-27b-4bit-mtp --host 127.0.0.1 --port 8285 --prefill-step-size 2048 --max-num-seqs 1 --draft-model {models}/qwen3.8-27b-4bit-mtp --draft-kind mtp --draft-block-size 3` |
| `q36-35b-mlxvlm-dflash2` | mlx-vlm server + DFlash2; mlx-vlm 0.7.8, mlx 0.32.4 | `HF_HUB_OFFLINE=1 APC_ENABLED=1 APC_DISK_ENABLED=0 ~/engines/mlxvlm/bin/python -m mlx_vlm.server --model {models}/qwen3.6-35b-a3b-4bit --host 127.0.0.1 --port 8289 --prefill-step-size 2048 --max-num-seqs 1 --draft-model {models}/drafters/Qwen3.6-35B-A3B-DFlash2 --draft-kind dflash` |
| `q36-35b-llamacpp-mtp1` (Phase B2: its GGUF is pulled after Phase C) | llama.cpp + MTP (n-max 1); 0.7.0-dev (b11804, c7fc50941) | `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600   (inherited LLAMA_ARG_* cleared) ~/flint-m3/engines/llama/llama-b11804/llama-server -m '<unresolved {env:QWEN36_35B_GGUF={ollama_gguf:qwen3.6:35b-a3b-mtp-q4_K_M|qwen3.6-35b-a3b-nomtp}}>' --host 127.0.0.1 --port 8083 -a qwen3.6-35b-a3b -c 32768 -np 1 -ngl 999 -fa on -b 2048 -ub 2048 -lm none --jinja --chat-template-file {models}/qwen3.6-35b-a3b-4bit/chat_template.jinja --reasoning-format deepseek --cache-ram 8192 --ctx-checkpoints 32 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --no-webui --spec-type draft-mtp --spec-draft-n-max 1` |
| `q38-27b-llamacpp-mtp3` (Phase B2: its GGUF is pulled after Phase C) | llama.cpp + MTP (n-max 3); 0.7.0-dev (b11804, c7fc50941) | `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600   (inherited LLAMA_ARG_* cleared) ~/flint-m3/engines/llama/llama-b11804/llama-server -m '<unresolved {env:QWEN38_27B_GGUF={ollama_gguf:qwen3.8:27b-mtp-q4_K_M|qwen3.8-27b-nomtp}}>' --host 127.0.0.1 --port 8083 -a qwen3.8-27b -c 32768 -np 1 -ngl 999 -fa on -b 2048 -ub 2048 -lm none --jinja --chat-template-file {models}/qwen3.8-27b-4bit-mtp/chat_template.jinja --reasoning-format deepseek --cache-ram 8192 --ctx-checkpoints 32 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --no-webui --spec-type draft-mtp --spec-draft-n-max 3` |
| `q35-4b-llamacpp-mtp2` | llama.cpp + MTP (n-max 2); 0.7.0-dev (b11804, c7fc50941) | `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600   (inherited LLAMA_ARG_* cleared) ~/flint-m3/engines/llama/llama-b11804/llama-server -m ~/.ollama/models/blobs/sha256-a9d0a3e1d8c91732a3812becd31f8d4a8ff57414903f226306069f0437e2ac50 --host 127.0.0.1 --port 8083 -a qwen3.5-4b -c 32768 -np 1 -ngl 999 -fa on -b 2048 -ub 2048 -lm none --jinja --chat-template-file {models}/qwen3.5-4b-4bit-mtp/chat_template.jinja --reasoning-format deepseek --cache-ram 8192 --ctx-checkpoints 32 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --no-webui --spec-type draft-mtp --spec-draft-n-max 2 -md ~/.ollama/models/blobs/sha256-83ba6dc1b62e441fd679cbfdb1df0fac0e14231a34ac890e4295d6e792e5dea9` |
| `q36-35b-ollama-mlx-q4` | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth); 0.40.8 | `OLLAMA_HOST=127.0.0.1:8193 OLLAMA_MODELS={models}/ollama-mlx OLLAMA_NOPRUNE=1 OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600   (inherited OLLAMA_*, LLAMA_ARG_* cleared) ~/engines/ollama/ollama serve` |
| `q38-27b-ollama-mlx-q4` | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth); 0.40.8 | `OLLAMA_HOST=127.0.0.1:8193 OLLAMA_MODELS={models}/ollama-mlx OLLAMA_NOPRUNE=1 OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600   (inherited OLLAMA_*, LLAMA_ARG_* cleared) ~/engines/ollama/ollama serve` |
| `q35-4b-ollama-mlx-q4` | Ollama MLX runner (MLX 4-bit + MTP head, adaptive depth); 0.40.8 | `OLLAMA_HOST=127.0.0.1:8193 OLLAMA_MODELS={models}/ollama-mlx OLLAMA_NOPRUNE=1 OLLAMA_CONTEXT_LENGTH=32768 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1 OLLAMA_FLASH_ATTENTION=1 GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600   (inherited OLLAMA_*, LLAMA_ARG_* cleared) ~/engines/ollama/ollama serve` |
| `q36-35b-mtplx` | MTPLX (sustained profile, depth 1); mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | `MTPLX_SESSION_POSTCOMMIT_MODE=inline HF_HUB_OFFLINE=1 ~/engines/mtplx/bin/mtplx serve --model {models}/qwen3.6-35b-a3b-4bit --host 127.0.0.1 --port 8082 --no-auth --yes --context-window 32768 --no-stats-footer --mtp --profile sustained --depth 1` |
| `q38-27b-mtplx` | MTPLX (turbo profile, depth 3); mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | `MTPLX_SESSION_POSTCOMMIT_MODE=inline HF_HUB_OFFLINE=1 ~/engines/mtplx/bin/mtplx serve --model {models}/qwen3.8-27b-4bit-mtp --model-id qwen3.8-27b --host 127.0.0.1 --port 8082 --no-auth --yes --context-window 32768 --no-stats-footer --mtp --profile turbo --depth 3` |
| `q35-4b-mtplx` | MTPLX (sustained profile, depth 1); mtplx 2.12.2 mlx_lm 0.31.3 mlx 0.32.2 | `MTPLX_SESSION_POSTCOMMIT_MODE=inline HF_HUB_OFFLINE=1 ~/engines/mtplx/bin/mtplx serve --model {models}/qwen3.5-4b-4bit-mtplx --model-id qwen3.5-4b --host 127.0.0.1 --port 8082 --no-auth --yes --context-window 32768 --no-stats-footer --mtp --profile sustained --depth 1` |
| `dory-mlxlm-4bit` | mlx-lm (PR #1988, 4-bit, 2 loops, per-loop KV); mlx_lm 0.32.2.dev0 mlx 0.32.4 | `HF_HUB_OFFLINE=1 ~/engines/mlxlm-dory/bin/mlx_lm.server --model ~/kq/m3/out/state/dory-mlxlm-4bit --host 127.0.0.1 --port 8321 --prompt-cache-size 4 --prompt-cache-bytes 8589934592 --max-tokens 16384 --prefill-step-size 2048` |
| `lightning-mlxvlm-dspark` | mlx-vlm server + DSpark drafter (block 6); mlx-vlm 0.7.8, mlx 0.32.4 | `HF_HUB_OFFLINE=1 APC_ENABLED=1 APC_DISK_ENABLED=0 ~/engines/mlxvlm/bin/python -m mlx_vlm.server --model {models}/nemotron-3.5-lightning-nvfp4 --host 127.0.0.1 --port 8273 --prefill-step-size 2048 --max-num-seqs 1 --draft-model {models}/nemotron-3.5-lightning-nvfp4/dflash --draft-kind dflash --draft-block-size 6` |
| `bolt-mlxvlm` (not run: the bundle lacks the files mlx-vlm loads) | mlx-vlm server (the bundle's model file, no drafter); unknown | `HF_HUB_OFFLINE=1 ~/engines/mlxvlm/bin/python -m mlx_vlm.server --model {models}/nemotron-3.5-bolt-24gb --trust-remote-code --host 127.0.0.1 --port 8243` |
| `q35-9b-llamacpp` | llama.cpp (unsloth Q4_K_M GGUF); 0.7.0-dev (b11804, c7fc50941) | `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600   (inherited LLAMA_ARG_* cleared) ~/flint-m3/engines/llama/llama-b11804/llama-server -m {models}/gguf/Qwen3.5-9B-Q4_K_M.gguf -a qwen3.5-9b --host 127.0.0.1 --port 8083 -c 32768 -np 1 -ngl 999 -fa on -b 2048 -ub 2048 -lm none --jinja --reasoning-format deepseek --cache-ram 8192 --chat-template-file {models}/qwen3.5-9b-4bit/chat_template.jinja --ctx-checkpoints 32 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --no-webui` |
| `q35-9b-mlxlm` | mlx-lm; mlx_lm 0.32.2 mlx 0.32.4 | `HF_HUB_OFFLINE=1 ~/engines/mlxlm/bin/mlx_lm.server --model {models}/qwen3.5-9b-4bit --host 127.0.0.1 --port 8081 --prompt-cache-size 4 --prompt-cache-bytes 8589934592 --max-tokens 16384 --prefill-step-size 2048` |
| `nano-4b-llamacpp` | llama.cpp (NVIDIA's Q4_K_M GGUF); 0.7.0-dev (b11804, c7fc50941) | `GGML_METAL_RESIDENCY_KEEP_ALIVE_S=3600   (inherited LLAMA_ARG_* cleared) ~/flint-m3/engines/llama/llama-b11804/llama-server -m {models}/gguf/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf -a nano-4b --host 127.0.0.1 --port 8083 -c 32768 -np 1 -ngl 999 -fa on -b 2048 -ub 2048 -lm none --jinja --reasoning-format deepseek --cache-ram 8192 --no-webui` |
| `nano-9b-mlxlm-4bit` | mlx-lm (mlx-community 4-bit); mlx_lm 0.32.2 mlx 0.32.4 | `HF_HUB_OFFLINE=1 ~/engines/mlxlm/bin/mlx_lm.server --model {models}/mlx/NVIDIA-Nemotron-Nano-9B-v2-4bits --host 127.0.0.1 --port 8081 --prompt-cache-size 4 --prompt-cache-bytes 8589934592 --max-tokens 16384 --prefill-step-size 2048` |
