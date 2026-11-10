# Results, 2026-11-10: KQuant packs for Qwen3.5-4B and 9B against the packs the catalog served

The catalog's 4-bit Qwen3.5-4B and 9B packs were lmstudio-community's (`c43ee1d`, `b455506`): MLX affine 4-bit, group 64,
plain round-to-nearest. The KQuant packs keep the content of every other file of those packs (config, tokenizer, norms, the
vision tower; `config.json` and the safetensors index are re-serialized, so their bytes differ) and rebuild only the
quantized weights, scales and biases, so the file list and the size are the same:
3.034 GB of weights for the 4B, 5.950 GB for the 9B.

The Qwen3.6-35B-A3B KQuant pack has its own record: [2026-11-10-m5-pro-kquant-35b](../2026-11-10-m5-pro-kquant-35b/README.md).

## How a KQuant pack is built

Plain MLX quantization sets each group of 64 weights' scale from the group's min and max and rounds to the nearest
step. KQuant (importance-weighted scale search, #409) first runs mlx-lm's 128 default calibration texts through the bf16
model and records how strongly each input channel of every linear layer fires. Then, per group, it tries 11 clip ranges,
refits the scale and offset by least squares weighted by those channel strengths, and keeps the candidate with the
smallest weighted error. The format is unchanged, so any MLX affine reader loads it.

## Quality

| Pack | KL from bf16 | Top-1 agreement | GSM8K-100 | GSM8K, 1,319 |
| --- | ---: | ---: | ---: | ---: |
| 4B 4-bit, lmstudio-community | 0.0699 | 0.881 | 0.95 | 0.914 |
| 4B 4-bit, KQuant | **0.0479** | **0.899** | 0.92 | **0.923** |
| 4B 3-bit, KQuant (not shipped) | 0.1921 | 0.807 | 0.92 | 0.879 |
| 9B 4-bit, lmstudio-community | 0.0806 | 0.885 | 0.96 | 0.927 |
| 9B 4-bit, KQuant | **0.0366** | **0.924** | 0.96 | **0.944** |

KL and top-1: next-token distributions over 32,768 tokens (64 chunks of 512: WikiText-2 test, GSM8K training questions and answers as chats,
HumanEval prompts with their solutions), against `Qwen/Qwen3.5-4B` @ `851bf6e8` and `Qwen/Qwen3.5-9B` @ `c2022362` in
bf16. GSM8K: zero-shot, greedy, thinking off, the last number in the answer. Scored with mlx-lm 0.32.2 on Modal L40S GPUs
from the exact files in the repos. On the full 4B set the two 4-bit packs disagree on 82 questions: 47 only KQuant gets
right, 35 only the old pack (McNemar z = 1.33, not significant): no worse. On the 9B they disagree on 70: 46 only KQuant gets right, 24 only the old pack
(z = 2.63): better. The 3-bit pack is worse (0.879) and stays out
of the catalog.

## Speed

Kapitsa 0.2.33 (`7c511224`), `kapitsa bench --quick` against a `kapitsa serve` started per leg with its default drafting
(KFlash with the catalog's DFlash2 drafter): decode a 512-token answer at T 0.6 and at T 0, three answers
per leg, prefill a 4K prompt. MacBook Pro M5 Pro 48 GB, macOS 26.5.2 (25F84), on AC, legs interleaved, each leg waits for
1-minute load under 4 ([ab_legs.sh](ab_legs.sh), [legs_4b.sh](legs_4b.sh), [legs_9b.sh](legs_9b.sh)). Bench marks each
JSON `INVALID: server` because it did not start the server itself; that is expected for this setup.

| Pack | Decode T 0.6, tok/s | Greedy, tok/s | Draft share accepted | Prefill 4K, tok/s |
| --- | --- | --- | --- | --- |
| 4B lmstudio-community | 199.1, 198.1 | 218.8, 218.5 | 0.695, 0.686 | 4278, 3386 |
| 4B KQuant 4-bit | **219.2, 236.6** | **222.5, 230.5** | **0.729, 0.738** | 4136, 4025 |
| 4B KQuant 3-bit | 211.7, 206.8 | 233.3, 236.2 | 0.719, 0.721 | 2263, 2271 |
| 9B lmstudio-community | 130.9, 143.9 | 122.0, 131.1 | 0.729, 0.744 | 1455, 1569 |
| 9B KQuant 4-bit | **149.4, 140.1** | **148.7, 147.1** | 0.754, 0.744 | 1505, 1502 |

Pooled over both decode modes (four runs per pack), the 4B goes from 208.6 to 227.2 tok/s (9% faster) and the 9B from
132.0 to 146.3 (11% faster). At T 0.6 alone the 9B's runs overlap (130.9 and 143.9 against 149.4 and 140.1), so the 9B
is faster only on the pooled figure; the legs inside one arm differ by 5-9%.

Order: 4B `A B C C B A`, 9B `A B B A`; per-leg files in [4b/](4b) and [9b/](9b). The 4B's first attempt was voided by
the leg guard (Spotlight indexing the new pack swapped 3.3 GB) and rerun whole. The KQuant 4B writes faster because its
drafts are accepted more often (0.73 against 0.69 of drafted tokens).

What the leg files flag: every JSON carries `CONTENTION` (mean foreign GPU load 84-91%, the only user `kapitsa`); that
is the leg's own `kapitsa serve`, which bench counts as foreign because it did not start it. Leg `4b/4-C` (the 3-bit
pack, not shipped) swapped 16,598 pages during its 4K prefill and flags a repetition loop in one T 0.6 answer. Legs
`9b/3-B` and `9b/4-A`, one per 9B arm, flag repetition loops in T 0.6 answers (one and two). None of these changes the
comparison.
