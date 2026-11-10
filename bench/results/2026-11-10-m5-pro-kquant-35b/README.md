# Results, 2026-11-10: a KQuant pack for Qwen3.6-35B-A3B against the pack the catalog serves

The catalog's 4-bit Qwen3.6-35B-A3B is `Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed` at `95460516`: the four weight files of
`mlx-community/Qwen3.6-35B-A3B-4bit` plus an `mtp.safetensors` head. The KQuant pack keeps the layout and rebuilds only the quantized
weights, scales and biases of the language model, so its 19.51 GB of language-model weights are the same size. It is
`nace-ai/k436-test-qwen3.6-35b-a3b-mlx-4bit-kquant` at `ead4c792` (private; it moves to `nace-ai/Qwen3.6-35B-A3B-MLX-4bit-KQuant` when published).
The repo is 20.43 GB: the language model plus the vision tower, with no `mtp.safetensors`.

## Two standards

| Number | Measured against | Where |
| --- | --- | --- |
| KL, top-1, GSM8K | LM Studio's 4-bit build of the model, language-model weights only (19.51 GB), the KQuant pack's text weights (`kq-4bit-text`) | Modal H200, mlx-lm 0.32.2 on mlx with CUDA |
| Decode, prefill | the catalog pack above, against the KQuant pack with the catalog's MTP head and DFlash2 drafter | MacBook Pro M5 Pro 48 GB, Kapitsa 0.2.32 |

The Hub's `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit`, `mlx-community/Qwen3.6-35B-A3B-4bit` and the catalog pack list the same
SHA-256 for all four weight files (checked on the Hub on 10 November). The Modal copy of the LM Studio build was not hashed against
the Hub's files. The quality runs use the text weights, not the vision tower the repo also carries.

## Quality

| Pack | KL from bf16 | KL p99 | Top-1 agreement | GSM8K, 1,319 |
| --- | ---: | ---: | ---: | ---: |
| LM Studio 4-bit | 0.0961 | 1.310 | 0.890 | 0.9484 |
| KQuant 4-bit | **0.0586** | **0.784** | **0.912** | 0.9522 |

KL by source (WikiText-2, GSM8K chats, HumanEval code): 0.0861, 0.1546, 0.0712 for LM Studio; 0.0518, 0.0926, 0.0472 for KQuant.
KL and top-1: next-token distributions over 32,704 tokens against `Qwen/Qwen3.6-35B-A3B` @ `345d467b` in bf16. GSM8K: zero-shot, greedy,
thinking off, the last number in the answer. The two packs disagree on 43 of 1,319 questions: 24 only KQuant gets right, 19 only the
LM Studio build (McNemar z = 0.76, not significant): no worse. Files: [cat35_4.json](cat35_4.json), [kq35_4.json](kq35_4.json),
[rtn8_35.json](rtn8_35.json) (the 8-bit round-to-nearest check of the reference, KL about 0.01).

The bf16 reference and the importance pass run the experts one matmul per expert: `mx.gather_mm` on mlx with CUDA gave wrong outputs
for the bf16 experts (NLL 4.94 against 1.52 for the 4-bit packs; 147 experts saw no tokens). The reference NLL used here is 1.487.

## Speed

Kapitsa 0.2.32 (`e728524`), `kapitsa serve` started per leg with its default drafting (MTP head and DFlash2 drafter on both arms), then
`kapitsa bench --quick`: decode a 512-token answer at T 0.6 and at T 0 (median of three answers per leg), prefill a 4K prompt (4,131 to 4,134 tokens).
MacBook Pro M5 Pro 48 GB, macOS 26.5.2 (25F84), on AC, power mode 2, 1-minute load 2.6 to 3.7 at leg start and end
([ab_legs.sh](ab_legs.sh), [legs_35b.sh](legs_35b.sh)). Bench marks each JSON `INVALID: server` because it did not start the server itself;
that is expected for this setup.

| Leg | Pack | Decode T 0.6, tok/s | Greedy, tok/s | Draft share accepted (T 0.6, T 0) | Prefill 4K, tok/s | Peak RSS |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 1-B | KQuant | 213.4 | 233.3 | 0.738, 0.768 | 3171 | see below |
| 2-A | catalog | 193.0 | 219.0 | 0.701, 0.736 | 3220 | 20.29 GB |
| 3-B | KQuant | 221.8 | 236.6 | 0.742, 0.770 | 3258 | 20.30 GB |
| 4-A | catalog | 206.1 | 228.2 | 0.734, 0.754 | 3249 | 20.30 GB |
| 5-B | KQuant | 222.2 | 232.1 | 0.752, 0.770 | 3146 | 20.30 GB |
| 6-A | catalog | void | void | | | |

Means over the valid legs: decode at T 0.6 219.1 against 199.5 (10% faster), greedy 234.0 against 223.6 (5% faster), pooled over both
modes 226.5 against 211.5 (7% faster). Prefill 3192 against 3234 tok/s, 1.3% slower, inside the 4% the KQuant legs differ by: reading
a prompt is about the same (first token after 1.27 to 1.31 s on both). Peak RSS is the same on both, 20.3 GB.

Leg 6-A is void: a process that did not exist when the leg started, `rg` at 51% CPU, ran during it (its numbers are in
[legs/6-A.json](legs/6-A.json) and are not used). That leaves three KQuant legs and two for the catalog pack. Leg 1-B's RSS sample read
2.1 GB, cause not found; its speed is in line with the other KQuant legs, so it is used for speed and not for peak memory.
Every JSON carries `CONTENTION` (mean foreign GPU load above 80%, the only user `kapitsa`): that is the leg's own `kapitsa serve`,
which bench counts as foreign. Order `B A B A B A` (B KQuant, A catalog) in one lock hold; per-leg files and `legs.log` are in [legs/](legs).
The faster writing comes with more drafts accepted (0.74 to 0.75 against 0.70 to 0.73 of drafted tokens at T 0.6).

The card draws its numbers from [results.json](results.json).
