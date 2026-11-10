# Results, 2026-11-10: a KQuant pack for Qwen3.8-27B against the catalog's 4-bit pack

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
carries the quality and the speed result.

## Quality

| Pack | KL from bf16 | KL p99 | Top-1 agreement | NLL on the same tokens (bf16: 1.6069) | GSM8K, 1,319 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Catalog 4-bit | 0.0544 | 0.678 | 0.912 | **1.6282** | 0.9500 |
| KQuant 4-bit | **0.0349** | **0.471** | **0.929** | 1.6347 | 0.9530 |

One metric goes the other way: the cross-entropy of the tokens themselves is 0.0278 nats above bf16 for KQuant and 0.0213 for the catalog pack (KQuant has less
distance to bf16's predictions, the catalog pack slightly lower loss on this text). KL and top-1 are this record's stated metrics; the NLL is shown so the
table has all three.

KL by source (WikiText-2, GSM8K chats, HumanEval code): 0.0472, 0.0645, 0.0683 for the catalog pack; 0.0296, 0.0410, 0.0468 for KQuant.
KL and top-1: next-token distributions over 32,704 tokens against `Qwen/Qwen3.8-27B` @ `37561816` in bf16. GSM8K: zero-shot, greedy, thinking
off, the last number in the answer; on the first 100 questions both packs score 0.94. The two packs disagree on 30 of 1,319 questions: 17 only
KQuant gets right, 13 only the catalog pack (McNemar z = 0.73, not significant): no worse. Files: [cat27_4.json](cat27_4.json),
[kq27_4.json](kq27_4.json), [cat27_4.gsm_full.json](cat27_4.gsm_full.json), [kq27_4.gsm_full.json](kq27_4.gsm_full.json). Both packs were scored on
Modal (H100) from their text weights, not the vision tower the repos also carry.

## Speed

The measured rows follow; the card makes no speed claim. Three KQuant legs against three catalog legs
(Kapitsa 0.2.32 `e728524`, MacBook Pro M5 Pro 48 GB, macOS 26.5.2, AC power mode 2; B is KQuant, A is the catalog pack; order `B A B A B A` and then one more B leg;
`verdict27.py` recomputes every figure here from the leg files). Every leg ran the pack directory with the DFlash2 drafter `incoai/Qwen3.8-27B-DFlash2` at `015e7956` (the one the 0.2.32 catalog uses, the same files in both arms' directories, SHA-256 of `model.safetensors` starts `67fc76d68dc5`) and the MTP head; it is not the `nace-ai` retrain a later catalog attaches, so absolute speeds will differ on that drafter, and the comparison is between the two packs on the same drafter.

| | KQuant | Catalog | Change |
| --- | ---: | ---: | ---: |
| Decode T 0.6, tok/s | 72.5 | 80.0 | -9.4% |
| Decode greedy, tok/s | 91.0 | 82.8 | +9.9% |
| Decode pooled, tok/s | 81.7 | 81.4 | +0.4% |
| Prefill 4K, tok/s | 735.0 | 719.3 | +2.2% |

The pooled figure averages the two modes and hides both halves; the card does not use it. Sampled legs vary widely on this Mac: KQuant 76.7, 72.2 and 68.5 tok/s
(legs 1-B, 3-B and the shipping-shape leg), catalog 77.6, 78.5 and 83.8 (2-A, 4-A, 6-A). Leg 6-A is valid but reads high at T 0.6 (83.8) and low at prefill
(677 tok/s).

Two legs are void and in no mean: leg 5-B (a foreign `Application` process at 51% CPU) and a top-up KQuant leg ([legs-void-topup/](legs-void-topup); a `Python` process at 67% CPU
that was not there at the start).

The first two KQuant legs (1-B, 3-B) ran the pack directory without `mtplx_runtime.json`; the shipped repo has it. The third KQuant leg, the shipping-shape leg
([legs-shipping-shape/](legs-shipping-shape)), ran the pack as it ships (the staged repo at `cf91c446`, every file equal by SHA-256, see
[pack-sha256.txt](legs-shipping-shape/pack-sha256.txt)): the server log shows the MTP head loaded with the trunk hidden state post-norm and the base chat template loaded,
`LOCKPROOF` ok, AC power mode 2, peak 16.53 GB. Its numbers: decode T 0.6 68.5, greedy 90.5, prefill 4K 727.3 tok/s (time to first token 5,685 ms). It has the lowest
sampled figure of the six legs of either pack, and the two earlier KQuant legs are in the means as run, so the record does not say how much of the sampled gap the missing file explains.
Every leg's JSON carries a `CONTENTION` flag whose top contributor is `kapitsa`, and the `INVALID: server ... did not start` line for `KAPITSA_BENCH_SERVER`; both are expected, because the harness measures its own `kapitsa serve` (the bench prints "Another Kapitsa is running" in every leg of both arms for the same reason). The shipping-shape leg, and the void top-up leg, also list `com.apple.WebKit.GPU` in `gpu_users`, which none of the six earlier legs does (they list `kapitsa` and apps such as Slack, Edge or Claude). The process is DSH's own window: the job that ran this leg logged "user active (110s idle): not hiding DSH" at its start (18:44, the leg ran at 19:11; see [legs-shipping-shape/job-start.log](legs-shipping-shape/job-start.log)), so DSH stayed visible, and the six earlier legs' `gpu_users` list no WebKit process. DSH's GPU process sharing the GPU is a plausible cause of this leg's low sampled number, so 68.5 is not read as the pack alone. The leg's three greedy runs are uneven too (91.2, 73.3 and 90.5 tok/s; the median, 90.5, is what the means use), which also points to something else on the GPU. The leg stays in the means and the sampled figures read as a range: 68.5 to 76.7 tok/s for KQuant against 77.6 to 83.8 for the catalog pack.

A further shipping-shape KQuant leg with DSH hidden ([legs-dsh-hidden-flawed/](legs-dsh-hidden-flawed), 20:21 to 20:22) is not usable and in no mean: sampled 77.3, greedy 85.8, prefill 642 tok/s (6,436 ms). DSH was hidden (its job logged "DSH hidden" before the leg, [job-start.log](legs-dsh-hidden-flawed/job-start.log)), but `gpu_users` is `kapitsa` and `Claude`, and `1-B.err` carries two "Another Kapitsa is running" warnings (pids 98738 and 94201) where the shipping-shape leg has one. A test run on the same Mac overlapped the leg, so it fails the rule that only `kapitsa` may use the GPU. The figures above stay as they are.

The harness is [ab_legs_k436.sh](ab_legs_k436.sh), started by [legs_27b_six4.sh](legs_27b_six4.sh); every leg logs a `LOCKPROOF` line (the `lockf` holder is in the harness's process ancestry) in [legs/legs.log](legs/legs.log).

The card draws its numbers from [results.json](results.json). Its headline, top table and chart carry no speed (`speed_headline` false): this KQuant pack is opt-in, and the
Speed section prints the measured rows and the spread of the sampled runs, and makes no speed claim. The card's run line, `kapitsa serve --model 27b:kquant`, needs an opt-in catalog variant that is not in the catalog yet
(`27b` stays the plain pack), so `catalog_pinned` is false here and the published card is generated with true. The speed legs ran the catalog's DFlash2 drafter, which a variant attaches and a bare repo name does not.
