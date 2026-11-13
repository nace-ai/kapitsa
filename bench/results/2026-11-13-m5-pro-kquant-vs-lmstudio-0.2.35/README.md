# Results, 2026-11-13: the KQuant 9B and 35B-A3B against lmstudio-community's 4-bit on Kapitsa 0.2.35

One A B B A per model on an M5 Pro, with the same drafter and MTP head on both packs, so the weights are the only
difference. No leg was re-run.

- **35B-A3B: KQuant writes faster**, 236.2 against 221.3 tok/s pooled over both decode modes (+6.7%): +7.6% at T 0.6
  and +5.9% at T 0. 4K prefill is the same (-0.4%).
- **9B: within 5% pooled** (146.5 against 144.8 tok/s, +1.2%), but the two modes go opposite ways: KQuant is
  **slower at T 0.6** (131.5 against 140.3, -6.3%) and **faster at T 0** (161.5 against 149.3, +8.2%). 4K prefill is
  2.7% slower. Three of the eight 9B answers carry bench's repetition-loop flag, on both packs (table below).

| Pack | Repo and revision | sha256 of the files the legs read |
| --- | --- | --- |
| 9B A, lmstudio | [`lmstudio-community/Qwen3.5-9B-MLX-4bit`](https://huggingface.co/lmstudio-community/Qwen3.5-9B-MLX-4bit) @ `b455506` | shards 1-2: `973cc1efdedb4d32…`, `597dae0ed72b60ac…` |
| 9B B, KQuant | [`nace-ai/Qwen3.5-9B-MLX-Q4-KQuant`](https://huggingface.co/nace-ai/Qwen3.5-9B-MLX-Q4-KQuant) @ `ff7b61b` (`9b` in 0.2.35) | language shards 1-2: `f9716dd6b67d40ba…`, `b00e4b65fa8d2fd7…`; vision: `61dd0161c76bea9c…` |
| 9B drafter, both | [`z-lab/Qwen3.5-9B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash) @ `5fc3b3d` (the catalog's) | `model.safetensors` `0a42274b32554f48…` |
| 9B MTP head, both | `Qwen/Qwen3.5-9B` @ `c202236` sidecar (the catalog's) | `mtp.safetensors` `5f7c656b3930220c…` |
| 35B A, lmstudio | [`lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit`](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit) @ `0c4a20a` | shards 1-4: `09f3e6ecb0b7af6e…`, `31dcdb1c49eebdb1…`, `3e66de06a1f03dad…`, `a5d0cf03519c26f8…` |
| 35B B, KQuant | [`nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant`](https://huggingface.co/nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant) @ `37e804f` (`35b` in 0.2.35) | shards 1-4: `44b58d84b22116de…`, `0d87665c02cc4722…`, `b1d5c3c38d4acfc3…`, `71feb1db60a9f131…` |
| 35B drafter, both | [`incoai/Qwen3.6-35B-A3B-DFlash2`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) | `model.safetensors` `d1f16c0ff85afbe6…` |
| 35B MTP head, both | `mtp.safetensors` and `mtplx_runtime.json` (the head's contract) from the KQuant repo | `6661d105178c3d7c…`, `fe2c406ebf56fd76…` |

Each shard and drafter hash equals the LFS sha256 the Hub publishes at that revision. The lmstudio 35B-A3B shards are
byte for byte the shards of `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed`, the "MTPLX" pack of the
[2026-11-12 record](../2026-11-12-m5-pro-kquant-35b-0.2.35/README.md), so that record also timed KQuant against
lmstudio's 35B-A3B weights.

Setup: Kapitsa 0.2.35 (`30b18f9`, binary sha256 `19737e9cd311…`), MacBook Pro M5 Pro 48 GB, macOS 26.5.2 (25F84), on
AC, power mode 2 (not Low Power), the DSH window hidden, every model folder on the internal disk. 9B legs 05:18 to 05:22
Almaty time, 35B-A3B legs 05:23 to 05:27, each model under one GPU lock hold. The harness is [legs.sh](legs.sh), run with
`TOPUPS=0` (no extra legs): each leg runs `kapitsa bench --model <folder> --quick --json`, so bench starts and measures
its own server (512-token answers, three at T 0.6 and three at T 0; a leg's figure is the median of its three). Before a
leg it waits for AC and a 1-minute load average under 4; a leg is void if it ends on battery, at a load of 4.5 or more,
or if a Kapitsa process outside the leg's own bench ran during it (a census every 5 s). No leg was void, the census
found no other Kapitsa process, and no leg reports contention.

| Model | Leg | Pack | T 0.6, tok/s | T 0, tok/s | Accepted share, T 0.6 / T 0 | 4K prefill, tok/s | Loop flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 9B | 1-A | lmstudio | 139.8 | 153.9 | 0.717 / 0.738 | 1782 | none |
| 9B | 4-A | lmstudio | 140.8 | 144.7 | 0.734 / 0.725 | 1782 | T 0.6 |
| 9B | 2-B | KQuant | 132.7 | 159.7 | 0.727 / 0.764 | 1738 | T 0 |
| 9B | 3-B | KQuant | 130.2 | 163.3 | 0.717 / 0.769 | 1731 | T 0.6 |
| 35B-A3B | 1-A | lmstudio | 210.2 | 231.1 | 0.727 / 0.748 | 3225 | none |
| 35B-A3B | 4-A | lmstudio | 212.1 | 231.8 | 0.717 / 0.748 | 3205 | none |
| 35B-A3B | 2-B | KQuant | 224.0 | 241.6 | 0.754 / 0.766 | 3180 | none |
| 35B-A3B | 3-B | KQuant | 230.4 | 248.7 | 0.746 / 0.769 | 3228 | none |

Why the 9B splits by mode: at T 0.6 both packs accept about the same share of drafts (0.72-0.73), and the KQuant 9B
reads 5.6% more bytes per token (its `down_proj` in all 32 layers and `in_proj_z` in 24 are 5-bit), so it is slower. At
T 0 KQuant's share is higher (0.765 against 0.73), which outweighs the bytes. The 35B-A3B packs read the same bytes per
token, and KQuant's higher share in both modes makes it faster in both.

The loop flag is bench's distinct-4-gram < 0.5 rule on a 512-token coding answer, which also fires on legitimate
repeated code. Without the flagged answers the 9B figures are -5.1% at T 0.6 (one leg each) and +9.4% at T 0. A
separate plain-decode check on an M4 Pro (40 answers per pack at 2048 tokens) found 1 runaway loop for KQuant and 0 for
lmstudio, Fisher p = 1.0.

`python3 rows.py` prints the leg tables and the per-mode, pooled and prefill figures above from the leg files in
[9b/legs](9b/legs) and [35b/legs](35b/legs); `python3 rows.py --check` checks that every leg ran on AC with bench's own
server and no contention.
