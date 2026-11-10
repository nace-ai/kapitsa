# Results, 2026-11-10: a KQuant pack for Qwen3.8-27B against the catalog's 4-bit pack (speed held)

The catalog's 4-bit Qwen3.8-27B is `lmstudio-community/Qwen3.8-27B-MLX-4bit` at `6067b15c`. The KQuant pack keeps its layout and rebuilds only
the quantized weights, scales and biases of the language model, so the 15.13 GB of language-model weights are the same size. It is
`nace-ai/Qwen3.8-27B-MLX-4bit-KQuant` (private, weights and MTP head as of `cf91c446`; the model card is added in a later commit); the public `nace-ai/Qwen3.8-27B-MLX-4bit` is not touched. Compared on the Hub
(10 November): its three weight shards have the same SHA-256 as those of the test repo `nace-ai/k436-test-qwen3.8-27b-mlx-4bit-kquant` at `566dc4e8` that
the Mac legs were built from, and `mtp.safetensors` and `mtplx_runtime.json` are Youssofal's `Qwen3.8-27B-MTPLX-Optimized-Speed` at `1d5087d2`, unchanged.
The repo is 16.32 GB with the vision tower and the MTP head.

The tokenizer files and chat template are those of `Qwen/Qwen3.8-27B` at `37561816` (`tokenizer.json` SHA-256 starts `0997f410c5`), not the catalog pack's. The
catalog pack's `tokenizer.json` differs from the base's on Devanagari and Thai ids (it lost `\p{M}`); English, code and math ids are identical, so the
quality and speed numbers below stand for those (every run here used the catalog pack's tokenizer; English, code, math, CJK, emoji and RTL text were compared id by id, Devanagari and Thai were not measured), and nothing here claims Indic or Thai quality. The chat template renders the same text as the catalog's and
differs only by the `{% generation %}` markers. This record
carries the quality result. The speed result is held, see below.

## Quality

| Pack | KL from bf16 | KL p99 | Top-1 agreement | GSM8K, 1,319 |
| --- | ---: | ---: | ---: | ---: |
| Catalog 4-bit | 0.0544 | 0.678 | 0.912 | 0.9500 |
| KQuant 4-bit | **0.0349** | **0.471** | **0.929** | 0.9530 |

KL by source (WikiText-2, GSM8K chats, HumanEval code): 0.0472, 0.0645, 0.0683 for the catalog pack; 0.0296, 0.0410, 0.0468 for KQuant.
KL and top-1: next-token distributions over 32,704 tokens against `Qwen/Qwen3.8-27B` @ `37561816` in bf16. GSM8K: zero-shot, greedy, thinking
off, the last number in the answer; on the first 100 questions both packs score 0.94. The two packs disagree on 30 of 1,319 questions: 17 only
KQuant gets right, 13 only the catalog pack (McNemar z = 0.73, not significant): no worse. Files: [cat27_4.json](cat27_4.json),
[kq27_4.json](kq27_4.json), [cat27_4.gsm_full.json](cat27_4.gsm_full.json), [kq27_4.gsm_full.json](kq27_4.gsm_full.json). Both packs were scored on
Modal (H100) from their text weights, not the vision tower the repos also carry.

## Speed: held

No speed claim is made yet. B's speed legs ran the KQuant arm from a pack directory that lacked `mtplx_runtime.json` (the shipped repo has it),
so the KQuant arm did not run the same way users will. The legs are kept in [legs/](legs) for the record and are not used by the card; one more
KQuant leg will decide what the card says. What the held legs showed, for the record only (Kapitsa 0.2.32 `e728524`, MacBook Pro M5 Pro 48 GB,
macOS 26.5.2, AC power mode 2; order `B A B A B A`, B KQuant, A catalog; `verdict27.py` recomputes this from the leg files):

| | KQuant | Catalog | Change |
| --- | ---: | ---: | ---: |
| Decode T 0.6, tok/s | 74.4 | 80.0 | 6.9% slower |
| Decode greedy, tok/s | 91.2 | 82.8 | 10.2% faster |
| Decode pooled, tok/s | 82.8 | 81.4 | 1.8% faster |
| Prefill 4K, tok/s | 738.9 | 719.3 | 2.7% faster |

That is two valid KQuant legs (1-B, 3-B) against three catalog legs (2-A, 4-A, 6-A); leg 5-B is void (a foreign `Application` process at 51% CPU).
Leg 6-A is valid but reads high at T 0.6 (83.8) and low at prefill (677 tok/s); without it the catalog means are 78.0 and 740.6 tok/s, and KQuant is 4.6%
slower at T 0.6 and 0.2% slower at prefill. A top-up KQuant leg ([legs-void-topup/](legs-void-topup)) is void (a `Python` process at 67% CPU that was not
there at the start). The pooled and prefill differences are inside the spread of the legs, so they do not support a speed claim either way.

The harness is [ab_legs_k436.sh](ab_legs_k436.sh), started by [legs_27b_six4.sh](legs_27b_six4.sh); every leg logs a `LOCKPROOF` line (the `lockf` holder is in the harness's process ancestry) in [legs/legs.log](legs/legs.log).

The card draws its numbers from [results.json](results.json). Its `speed_held` key keeps the speed row, the speed headline and the Speed section off the
card; removing it, with the new legs in the file, brings them back.
