# Qwen3.8 27B: the KQuant pack against the plain 4-bit pack, Kapitsa 0.2.35, M5 Pro and M4 Pro

The evidence behind `27b` defaulting to the KQuant pack (#495): decode on two Macs of different chip classes, with
the same drafter and MTP head on both packs, so the weights are the only difference.

| | M5 Pro | M4 Pro |
| --- | --- | --- |
| Mac | MacBook Pro M5 Pro 48 GB, macOS 26.5.2 (25F84), on AC | AWS EC2 Mac mini M4 Pro 48 GB (Mac16,11), macOS 26.7.2 (25G332) |
| Kapitsa | 0.2.35 (`30b18f9`) | 0.2.35 (`30b18f98`), binary sha256 `bb3dae9d80e59f0c…` |
| Legs | A B B A, then one A top-up (4-A void) | A B B A, all valid |
| T 0.6, KQuant against plain | 82.9 against 80.6 tok/s, +2.8% | 65.4 against 63.5, +3.0% |
| T 0, KQuant against plain | 93.0 against 88.4, +5.2% | 68.9 against 70.0, -1.5% |
| Pooled | **87.9 against 84.5, +4.0%** | **67.2 against 66.8, +0.6%** |

Both Macs meet the rule for this default, "KQuant not slower": no mode is more than 3% slower. The M4 Pro's T 0 is
1.5% slower and its T 0.6 3.0% faster. On the M5 Pro the plain pack's own T 0.6 legs span 75.6 to 85.7 tok/s, so its
+4.0% is within that spread, not a resolved speed-up.

## Packs

The same files on both Macs: the weight shards' sha256 match the Hugging Face LFS hashes at these revisions (checked
on the M5 Pro's folders; the M4 Pro pulled them with `kapitsa pull 27b:4bit` and `kapitsa pull 27b:kquant` from
0.2.35's catalog, which pins the same revisions), and the MTP head and drafter hash the same in both folders on both
Macs.

| | Repository @ revision | sha256 |
| --- | --- | --- |
| A, plain 4-bit (`27b:4bit`) | `nace-ai/Qwen3.8-27B-MLX-4bit` @ `d76e1f92` | shards `471a9a81…`, `e7474b8f…`, `1038989b…` |
| B, KQuant (`27b:kquant`, now `27b`) | `nace-ai/Qwen3.8-27B-MLX-Q4-KQuant` @ `8e62e9e0` | shards `c107f478…`, `5ab53dd6…`, `d4fabd52…` |
| MTP head, both (`mtp.safetensors`) | in each pack's folder | `c58feddc584f3797…` |
| Drafter, both (`dflash2/model.safetensors`) | `nace-ai/Qwen3.8-27B-DFlash2-Kapitsa` @ `4f950469` | `d76cee298a985b9a…` |

Every leg's JSON says the drafter was "found in the model's folder".

## How each leg ran

`kapitsa bench --model <folder> --quick --json` with bench starting its own server (no `KAPITSA_BENCH_SERVER`),
under the GPU lock (`NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1` inside `lockf -k /tmp/kapitsa-gpu.lock`). Before each leg the
script waits for a load average under 4 (three readings on the M4 Pro). On the M5 Pro a leg is void when it ends on battery
or at a load of 4.5 or more, or when a Kapitsa process outside the leg's bench ran during it (a census every 5 s);
on both Macs a leg bench marks `invalid` is void. A pack left with fewer than two valid
legs gets alternating top-ups, at most two. Decode is bench's 512-token coding answer, T 0.6 (`decode`) and T 0
(`decode_greedy`), each the median of three runs; the pooled figure is the mean of the two modes' valid-leg means.

The M5 Pro's 4-A is void: the load rose from 2.67 to 7.46 during it, and bench flagged a browser's GPU process
(CONTENTION in `decode`, `prefill_4000` and `tools`). Its 1-A (`prefill_4000`, `tools`) and 3-B (`prefill_4000`) carry
the same flag outside decode only, and count. The M4 Pro's four legs carry no flag; the most other GPU use bench saw
in any of them was 0.56%.

## Files

- `m5-pro/legs/` and `m4-pro/legs/`: every leg's bench JSON, including the voided one, and `legs.log`.
- `m5-pro/run.sh`, `m5-pro/legs2.sh`, `m4-pro/legs.sh`: the scripts that ran the legs, with the paths of the Macs they ran on.
- `python3 rows.py` prints both leg tables and the figures above from the leg files; `--check` fails unless they
  equal the figures docs/benchmarks.md states.
