# Online drafter adaptation for Kapitsa: survey, fit, gaps, top 3, plan

Status: research, 2 Nov 2026, task #316; the baselines (§6.5) and the live runs of the top two ideas (§6.2, §6.3) ran on the M5 Pro on 4-5 Nov 2026. The rest was not run on a GPU. **Measured** = a number from a
logged Kapitsa/Splash run or from a replay of such logs (source given). **Paper-reported** = from the cited source.
**Inference** = my reading; it is marked. Hardware reference: M5 Pro 48 GB.

## 1. Summary

- Kapitsa has no learning code today. The drafter is quantized once at load and its bf16 source dropped
  (`src/model/dflash2.cpp:297`, `:421`). The only online-learned state is `KfBudget`: 2 logistic parameters per depth plus a
  cost table (`src/model/kflash_budget.cpp:153`, `:311`), saved per model and chip, not per user.
- "Weaver" is, as far as I can find, the tree builder of "Trees from Marginals" on top of DFlash ([arXiv 2607.06763](https://arxiv.org/abs/2607.06763), Oda, Mathieu, Knyazhitskiy and Chakhvadze; the paper has four authors). Mirai Labs publishes the checkpoint (`trymirai/weaver`) and the M5 port in `uzu`; its blog numbers are a vendor claim (§3 row 30).
  It is trained offline. It is a competitor and a design idea (§2.1), not an online learner. Questions for Alish are in §2.1.
  Measured 3 Nov 2026 in `uzu` on the 27B (M5 Pro, 108 legs): about 2.0x over no speculator and about 15% over the plain chain, no clear gain over the chain in Kazakh (§5.7).
- 87 methods surveyed (§3). Every published method that updates drafter weights from live traffic ran on CUDA servers.
  No source reports online drafter training on a Mac or phone.
- The room for a better **policy** on logged cycles is small and already mostly banked: a depth/row policy is worth ≤ +1.2 %
  (measured replay, `DESIGN-draftlen.md` (in the `kapitsa-ksx-tree` worktree)). More acceptance needs new **information**: the user's own text
  (history lane), the user's own vocabulary (draft head rows), or weights trained on it.
- **Scope (Alish, 4 Nov 2026): the workload is English coding prompts (the headline number), then English non-coding as the second class.** Other languages are an optional extra check, never a gate. The ranking below is judged on English code first. The draft-head idea (§6.1) is a non-English idea: English has 0.0 % of text outside the default head, the ranked head measured 1.00x and the full head 0.945x, so it leaves the top 3 and its work (old phase 2, asks F1 and F1b) is optional.
- Top 3 for English code (§6): (1) a persistent history lane in the lattice (§6.2; the one recorded English coding-agent session gains +11 to +13 % tokens per cycle in replay and **+7 % live on the GPU (two runs)**, single-request prompts 0 to +1.2 %), (2) idle-time CPU updates of the selector codebooks from verify-pass data (§6.3; +8.6 % on that session in replay when state carries across requests and **+6 % live (two runs)**, -1.2 % on English chat in replay, so it needs an English gate), (3) a draft model fine-tuned offline on English code and agent traces: **cannot measure because it needs a training run** (the notes' F72 estimate for offline-trained heads is -6 to +32 %, central +10 to +15 %; nothing on this Mac trains the drafter, and no CPU proxy for the gain exists). A wider verify tree was the other candidate and is now measured: more tokens per cycle on the English agent session, but it loses on cost for the 35B (§6.4). (1) needs no training.
- Prototypes on this branch: `tools/online_drafter_sim.py` (calibration and break-even replay, 16 recorded sessions).
  A second stacked PR adds `tools/history_lane_replay.py`. Both are CPU, stdlib only, with tests.

## 2. Context

### 2.1 What "weaver" is

| Rank | Candidate | Chance it is what Alish meant | Note |
|---|---|---|---|
| 1 | Weaver in DFlash-TfM, "Trees from Marginals" (Oda, Mathieu, Knyazhitskiy, Chakhvadze; [arXiv 2607.06763](https://arxiv.org/abs/2607.06763), v3 21 Oct 2026) | ~65 % the name, ~10 % also online (inference) | 56.7 M-param autoregressive adapter that turns DFlash top-512 marginals into a proposal tree; rollback-free Gated-DeltaNet tree verify. Offline-trained. Mirai Labs advertises it on Apple M5 in `uzu` (blog, 3 Sep 2026, vendor claim: "almost 2x" over MTPLX, "over 3x" over llama.cpp, Qwen3.6 27B, M5-series; read 2 Nov 2026, not reproduced). Code: `trymirai/sglang` (CUDA fork of SGLang, Apache-2.0, checked on the GitHub API). |
| 2 | OmniDraft ([arXiv 2507.02659](https://arxiv.org/abs/2507.02659)) | ~35 % the description | Online distillation plus LoRA, framed for on-device. No Mac numbers, no code. |
| 3 | Aurora, ATLAS, TTS, TIDE, DVI, OnlineSPEC | ~20 % each, description only | Online, server-side GPU. |

Questions for Alish: (a) Is "weaver" the Trees-from-Marginals name (shipped by Mirai on M5), or a description of a drafter that adapts while you use it?
(b) Should adaptation change drafter weights (OSD-style) or only what the drafter proposes (draft tree, n-gram store, head rows)?
(c) Is Weaver-style tree building in scope as the base drafter? (Kapitsa's selector already scores 16x16 transitions per
position, `src/kernels/spec_dflash.metal:1498`; that is the same job, conditional dependencies between proposed tokens.)

### 2.2 Kapitsa baseline numbers (measured, M5 Pro 48 GB unless stated)

| Quantity | Value | Source |
|---|---|---|
| 35B-A3B decode, DFlash2 + KFlash trees | 201.1 tok/s T0.6, 222.3 greedy (median of 3) | `bench/results/2026-10-23-m5-pro-35b-365/README.md` |
| Mean tokens per verify cycle (tau), code, 85-token prompt | 3.38-3.87, median ~3.6; chat/tool-call 3.0-3.25; agent follow-ups 4.9 (4K), 4.6 (16K) | same run, `logs/q36-35b-kapitsa-20261023-084520.log` |
| Cycle GPU time by live rows, 1 → 8 | 14.26 → 21.48 ms (1.51x), ~1.03 ms per row; mean 5.71 rows per cycle | same log, file line 59 = the 19th of its 27 `KFlash:` lines (`1459 cycles`); the last `KFlash:` summary (line 80) reads 18.20 ms at 1 row |
| Draft cost per cycle (Flash-Next DFlash2 head) | 0.51-1.14 ms, 0.7-1.5 % of the cycle | `kapitsa-f72-acceptance-sweep.md` §2 |
| Value of one more accepted token | one token ≈ one cycle (~18 ms at 201 tok/s and tau 3.6); +0.1 tok/cycle ≈ +2.8 % tok/s (derived) | derived |
| Calibration the 35B has learned (M5 Pro cache) | alpha -0.4…-0.9, beta 0.45-0.83 by depth: the drafter is overconfident | `~/.cache/kapitsa/kflash/qwen3.6-35b-a3b-4bit-dflash2-b4-Apple_M5_Pro.json` |
| Drafter size, 35B | 1.05 GB bf16 file, 0.63 GB resident at 4-bit; +0.151 GB rings per slot | catalog, local log |
| Plain (no drafter, no speculation) 35B-A3B on M5 Pro, English coding-agent session | **98.6 tok/s** (`--dflash none --spec 0`, 4 Nov 2026, AC, no swap, one run; §6.5) | measured on that one session only |
| Past replays: better row/depth policy | +0.3…+1.2 % (bar was +2 %); oracle length +16…+18 % unreachable | `DESIGN-draftlen.md` (in the `kapitsa-ksx-tree` worktree) |
| Standalone small AR drafter | 0.32-0.52x of plain speed (dead) | `kapitsa-f72-acceptance-sweep.md` §1.3 |
| Flint prompt lookup (replaces the model's own path) | 203.6 vs 263.3 tok/s, -22.7 % (M5 Max) | `bench/results/2026-11-01-m5-max-kapitsa/README.md:122-123` at `09d49ef0` |

Reading: verify cost per row and acceptance matter about equally; the draft pass is cheap. Anything that adds a verify row
must earn more than ~1 ms of ~18, i.e. ≥ 6 % of a cycle, per row (the F72 break-even is p ≈ 10-30 %).

### 2.3 Where adaptation would hook in (file:line, checked in this checkout, origin/main `09d49ef0`)

| Component | Where | What it does today | Online hook |
|---|---|---|---|
| Drafter object, weights | `src/model/dflash2.h:82`, ctor `dflash2.cpp:414`; `quantize` `:297`; bf16 source released `:421` | 5-6 layer block drafter; projections 4-bit g64; norms, selector codebooks, conv kernels kept bf16 (`:433-438`) | The two `[V,256]` codebooks are `own()`ed bf16, not quantized: trainable in place without requantizing. |
| Selector / lattice | `dfl_lattice` `spec_dflash.metal:1498`; host `dflash2.cpp:1823`, `:1830` `dfl_kf_table`, `:1849` `kf_build`; builder `src/kernels/kflash.h:230` | top-16 candidates per slot, 16x16 transition scores `E = A[pred]·s·B[cand]`, best-first tree | Bias or add candidate rows without touching verify; output text cannot change, only proposals. |
| Reduced draft head | `dflash2.cpp:540-545`, `build_head` `:765`; default 98,304 rows in token-id order, 65,536 with a ranked id file (`dflash2.h:93`); `NI_DFLASH_HEAD_IDS` takes the ranked id file | Drafts only tokens inside the head | Rank rows by the user's own tokens. |
| Verify, capture | `qwen35_spec.cpp:1061` `encode_verify`, `qwen35_v8.cpp:1074` `encode_verify8`; capture hook `qwen35.h:125`, `enc_capture` called `qwen35_v8.cpp:1179` | Target residual rows at 8 taps (35B) copied to bf16 for the drafter | Only the accepted path's rows survive as ring K/V; rejected rows are overwritten next cycle. |
| Acceptance | GPU `kf_accept` `kflash.h:270`, `dfl_accept` `spec_dflash.metal:1724`; host read `src/engine/engine.cpp:3677`, tokens `:3690`, `commit_rows` `:3701` | Per-node (depth, log q, hit) records, `KfBudget::observe` `kflash_budget.cpp:207` | The host holds finished-cycle data between `e2->wait()` (`:3671`) and the next encode: place for a CPU-side update or a trace append. |
| Per-cycle trace | none in the engine; recorders: `NI_DFLASH_CONF`, `NI_DFLASH_LAT` (`tools/ni_run.cpp:421-440`), `tools/tree_sim.py` replays any tree policy | Aggregates only (`tok/cycle`, `usage.kapitsa.kflash.cycles`) | Add a per-cycle ring (a, rows, conf[7], ms) at `src/engine/engine.cpp:3677-3691`. |
| History drafting | `ngram_drafts` `src/engine/engine.cpp:88` (nmax 4, nmin 2); callers Dory/Nemotron/Flint only; `KfLookup` `kflash_wrap.h:57` | Not wired into the Qwen DFlash2 path (PR #109 copy lane closed, `docs/kflash-map.md`) | A history branch in the tree: verify keeps per-branch recurrent state (`docs/kflash.md:113-143`). |
| Idle GPU time | `TurnLock` `src/engine/engine.h:312`, `try_lock` `src/engine/engine.cpp:160`; history prefill after 250 ms quiet `src/engine/engine.cpp:4248`; read-ahead `POST /v1/draft` `server/main.cpp:1601` | One GPU turn, 30 ms quantum (`NI_TURN_MS`) | Adaptation must follow the same `try_lock`, one chunk, re-check pattern, and yield to read-ahead. |
| Persistence | `KfBudget` cache `~/.cache/kapitsa/kflash/<model-drafter-bits-chip>.json`, `Engine::save_kflash` `src/engine/engine.cpp:654`; SSD state cache `src/engine/engine_disk.cpp`, `kStateFormat = 1` `:43` | Per model and chip | Adapted weights must live **outside** the drafter folder: the cache namespace hashes its file sizes and mtimes (`src/engine/engine_disk.cpp:375`). Changing `fc`/`hidden_norm`/`kv_all`/`k_norm` invalidates every stored ring; the selector, head and calibrator do not. |
| Splash backend | `src/splash/plan.cpp:137`, `protocol.h:175` | Splash's drafter is inside its binary; only `drafted/accepted/draft_cycles` are parsed, not forwarded | Not hookable. Everything below is for Kapitsa's own engine. |

There is no user id anywhere in `src/` (no session header). "Per user" means per install, or per slot.
| Systemone pointer head | `src/systemone/systemone.cpp:647` `PointerHead::load`, `:711` `logits`, `:749` `final_norm_row`; encoder `:300` `encode`, `:390` `encode_max`; server loop `tools/systemone.cpp:280` `score_max`, `:311` `score` | Scores the backbone's hidden rows against a decide row: `logits = (k(h_option) @ q(h_decide)) / sqrt(dp) / T`; no token is generated (`src/systemone/systemone.h:6-7`) | **None.** No decode loop and no drafter: no method in §3 has a place to attach. Online adaptation of this head is out of scope here. |

## 3. Survey: 87 methods

Row numbers are stable across the research notes. "Paper-reported" unless flagged vendor claim / blog / projection /
(inference) / unverified; "conflict" = two sources disagree and both are kept. Licences marked unknown were not checked
(GitHub API was rate-limited for some repos).

### 3.1 Methods

| # | Method (year, id) | Family | Adapts what, how often, from what signal | Reported result (baseline, hardware; flag vendor/blog claims) | Train cost | Code, license | Link |
|---|---|---|---|---|---|---|---|
| 1 | Online Speculative Decoding, OSD (2023, arXiv 2310.07177, ICML 2024) | OnlineWeights | Full draft (LLaMA-160M, T5-small); batch distillation every I iters (I=8 per D, "dynamic" per B); target logits at error positions; FKL/RKL/JSD. | Latency 1.42x-2.17x (abstract; Vicuna-7B, Flan-T5-XL; baseline not stated) vs README "at most 3.06x" (conflict); 2.63x uses simulated alpha. A100-80G (A, D); not reported (B). | Analytic only: FLOPs ratio 18.75; draft mem ~2% of target (~10% routed); wall clock not reported | repo LiuXiaoxuanPKU/OSD; none declared | https://arxiv.org/abs/2310.07177 |
| 2 | Test-Time Speculation, TTS (2026-05, arXiv 2605.09329) | OnlineWeights | Full draft (DFlash 5-layer, EAGLE-3 head, PARD); one Adam step on position-weighted KL(target,draft) plus proximal KL every round (S=1) or S=5; target distribution from verify pass. | AL up to +72% (+41% avg) over vanilla DFlash; 1.71x throughput at S=5 (Qwen3.5-122B AIME24 28.2 -> 46.8 tok/s); 8x RTX PRO 6000 Blackwell, SGLang. | One backward+Adam step ~ a few spec rounds; memory not reported | no TTS repo (builds on z-lab/dflash, MIT); TTS license unknown | https://arxiv.org/abs/2605.09329 |
| 3 | OmniDraft (2025-07, arXiv 2507.02659, NeurIPS 2025) | OnlineWeights | One drafter for many targets; draft (full or LoRA r32), n-gram vocab-translation cache, acceptance head; online, <1 epoch, batch 8; target tokens/logits on accepted+corrected tokens. | "Up to 1.5-2x"; A: Llama-68M+Vicuna-7B GSM8K 1.44x -> 2.20x distill-only; D: Llama3-8B GSM8K 1.70x, Alpaca 1.30x (A 1.61x, conflict). Hardware not stated (A) / A100 (D). | Not reported as numbers; n-gram cache 0.5-4.6 MB; LoRA low memory (claim) | none found; unknown | https://arxiv.org/abs/2507.02659 |
| 4 | Aurora = ATLAS-2 open framework (2026-02, arXiv 2602.06932, ICML 2026; Together AI) | OnlineWeights | Whole EAGLE-3 draft; async trainer on server-logged accepted/rejected branches + hidden states; CE + forward-KL on rejected; weights hot-swapped every 48-1600 requests (~80 best). | From scratch 1.45x on MiniMax-M2.1 229B (abstract 1.5x); Qwen3-Coder-Next 1.45x (A) vs 1.23x (B) conflict; +1.25x over static speculators; 4x H200 TP=4 FP8. | One extra learner GPU for 229B/80B targets; buffer memory (App. A.2 not read) | repo togethercomputer/aurora; MIT | https://arxiv.org/abs/2602.06932 |
| 5 | TIDE, Temporal Incremental Draft Engine (2026, arXiv 2602.05145; Moreh) | OnlineWeights | EAGLE-3 draft (1 layer + LM head); target hidden states dumped during SGLang serving; separate batch retrain when enough signal; deploy only if AL improves. | Up to 1.15x throughput over static SD (ShareGPT 1.08x, Science 1.22x); draft training 1.67x faster; H100 inference, AMD MI250 training. | Storage 0.19 TB vs 4.66 TB SpecForge offline; totals not reported | none found; unknown | https://arxiv.org/abs/2602.05145 |
| 6 | OnlineSPEC "When Drafts Evolve" (2026-03, arXiv 2603.12617, ICML 2026) | OnlineWeights | Online-learning view: online GD (Online-LR), optimistic gradient reuse (Opt-Hydra), online ensemble (Ens-Eagle); verification feedback; granularity and loss not extracted. | "Up to 24% over prior SOTA" (7 benchmarks, 5 targets; baseline not stated); Vicuna-7B OSD 1.23x/1.12x/1.09x/1.08x vs vanilla SD 1.00x; 4x A800 (A), not reported (B). | Not reported; ensemble memory grows (inference) | repos ZinYY (A) vs miyaio (B), conflict; none declared | https://arxiv.org/abs/2603.12617 |
| 7 | chimere `online_finetune.py` for DFlash (2026, GitHub, no paper) | OnlineWeights | Full DFlash drafter finetuned from buffer of on-policy features in nightly/manual job; warm start lr 1e-4; rollback unless tau >= 0.95x old. | not reported (CUDA, RTX 5060 Ti per repo description) | not reported | repo AIdevsmartdata/chimere; Apache-2.0 | https://github.com/AIdevsmartdata/chimere |
| 8 | TLT "Taming the Long-Tail" (2025-11, arXiv 2511.16665, ASPLOS 2026) | OnlineWeights | 1-layer drafter updated every RL step by spot trainer on GPUs idled by long tail; cached target hidden states and tokens; BEG-MAB picks SD strategy. | >1.7x end-to-end RL training speedup; 1.7-2.1x over VeRL (Qwen2.5-7B/32B etc.), accuracy kept; 8 DGX-H100 nodes (64 GPUs). | Designed for zero extra cost (idle GPUs) | repo mit-han-lab/fastrl; Apache-2.0 | https://arxiv.org/abs/2511.16665 |
| 9 | FastGRPO (2025-09, arXiv 2509.21792, ICLR 2026) | OnlineWeights | Draft model updated continuously during GRPO RL so it tracks the evolving policy; feedback from the target; concurrency-aware SD. | End-to-end 2.35x-2.72x (Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct; baseline not stated); GPU type not extracted. | not reported | repo yedaotian9/GRPO_speculative; license not checked | https://arxiv.org/abs/2509.21792 |
| 10 | System-integrated SD for RL rollouts, NeMo-RL (2026, arXiv 2604.26779; NVIDIA) | OnlineWeights | EAGLE-3 head or MTP heads in NeMo-RL+vLLM; optional per-RL-step adaptation reusing gradient-detached hidden states and verifier log-probs from the GRPO forward pass. | 1.8x rollout throughput at 8B sync RL (8 GB200 NVL72 nodes); up to 2.5x end-to-end at 235B async RL is a projection (simulator), not measured. | Near zero extra (reuses forward pass, as described) | NeMo-RL (repo not checked); unknown | https://arxiv.org/abs/2604.26779 |
| 11 | DistillSpec (2023-10, arXiv 2310.08461, ICLR 2024) | OnlineWeights | Not online: full draft, offline white-box KD once per target/task on draft-generated sequences with target logits; divergence chosen per task. | 10-45% speedup over standard SD; +26% avg GSM8K draft on 23 BigBenchHard tasks (1.93x/1.78x -> 2.21x/2.02x); TPUv4. | not reported | none found; unknown | https://arxiv.org/abs/2310.08461 |
| 12 | Draft-OPD, On-Policy Distillation for Draft Models (2026, EMNLP 2026; arXiv id not verified) | OnlineWeights | Offline/iterative: EAGLE-3/DFlash draft learns from replayed verification-exposed error positions after target-assisted rollouts; frequency and loss not read. | "Over 5x lossless acceleration for thinking models"; +23% over EAGLE-3, +13% over DFlash (benchmarks, baselines, hardware not read; abstract only). | not reported | none found; unknown | https://doi.org/10.18653/2026.emnlp-main.1139 |
| 13 | EDA, Efficient Draft Adaptation (2026-03, arXiv 2603.09527) | OnlineWeights | Offline: only private part of shared+private draft updated per finetuned target, on target-regenerated data with sample selection. | Qwen2.5-7B -> Qwen2.5-Math-7B: AL 4.79 vs 4.37 baseline adaptation, at 60.8% of full-retraining cost; 4x H200. | Full retrain updates 462 MB, 5.1 h on 4x H200 | repo Lyn-Lucy/Efficient-Draft-Adaptation; license not checked | https://arxiv.org/abs/2603.09527 |
| 14 | EvoSpec (2026, arXiv 2605.27390) | OnlineLoRA | LoRA r32 on W_q,W_v of draft, async step per 32 admitted samples, lr 1e-5; target top-k logits KD; plus dynamic vocab buffer. Base frozen. | B (Qwen3-8B, EAGLE-2, 2x RTX 4090): 2.18x over vanilla, 1.20x over EAGLE-2, AL 3.72; D (EAGLE-3): only 1.13x vs FR-Spec; draft/hardware conflict. | Aux memory 2,936 vs 4,128 MiB OSD full (B); peak 9.0 vs 12.3 GB (D); conflict | none found; unknown | https://arxiv.org/abs/2605.27390 |
| 15 | Draft, Verify & Improve, DVI (2025-10, arXiv 2510.05421) | OnlineLoRA | Self-speculative: shallow layers + LoRA draft head; online update of head LoRA only from replay buffer using verifier logits (KL) + accept/reject reward. | 2.16x wall-time on Spec-Bench, Vicuna-7B, on par with EAGLE-2; single H100; baseline not stated (AR assumed, inference). | 2,000 prompts / 2,000 steps; no figure, claims no extra model/KV | not yet released; none found; unknown | https://arxiv.org/abs/2510.05421 |
| 16 | SpecLoRA (2025-12, GitHub repo, no paper) | OnlineLoRA | MLX: Qwen2.5-0.5B drafts 4 tokens for Qwen2.5-3B; after 30 failures draft is LoRA-trained on saved disagreement cases and fused back. | README claims "1.5-2x out of the box"; no adaptation gain quantified; unverified. Apple silicon M1-M4, 16 GB+, 4-bit. | not reported | repo Armd04/SpecLoRA; MIT; 0 stars | https://github.com/Armd04/SpecLoRA |
| 17 | vLLM RFC: LoRA adapters for DFlash drafters (2026-08, vllm issue 52038) | OnlineLoRA | RFC only: LoRA r64 per domain on DFlash drafter, swapped per request; offline-trained, not online. | R64 adapter ~28x smaller than 0.8B drafter, within ~2% of full per-domain drafter in AL (RFC author's claim; GPT-OSS-120B target; hardware not stated). | Adapter ~28x smaller than drafter | RFC only; vLLM Apache-2.0 (not rechecked) | https://github.com/vllm-project/vllm/issues/52038 |
| 18 | EAGLE (2024-01, arXiv 2401.15077, ICML 2024) | HeadFamily | Single-layer head autoregressing on second-to-top hidden state; offline on fixed data (hidden-state regression + token loss); never updated online. | LLaMA2-Chat 70B 2.7x-3.5x latency speedup, lossless (abstract; baseline vanilla decoding, inference; GPUs not reported). | Head ~2-5% of target params; training cost not reported | SafeAILab/EAGLE; NOASSERTION (B) / unknown (C), conflict | https://arxiv.org/abs/2401.15077 |
| 19 | EAGLE-3 (2025-03, arXiv 2503.01840) | HeadFamily | Single-layer draft, offline; fuses low/mid/high target hidden states, trained with training-time test on regenerated data. | Up to 6.5x, ~1.4x over EAGLE-2; SGLang 1.38x at batch 64; Baseten 1.5-2.5x production (blog); DFlash paper: 1.8-2.2x on Qwen3. Hardware not extracted. | One day on a single host (Vertex AI report, LMSYS blog 2025-12-01) | SafeAILab/EAGLE; license unknown | https://arxiv.org/abs/2503.01840 |
| 20 | HASS Harmonized Speculative Sampling (2024-08, arXiv 2408.15766, ICLR 2025) | HeadFamily | EAGLE-2 head, offline; top-K harmonized objective distillation on target hidden states/logits plus draft's own earlier-step features. No inference overhead. | 2.81x-4.05x wall-clock avg over 3 datasets, 4 LLaMA models; 8%-20% over EAGLE-2; hardware not reported. | not reported | repo HArmonizedSS/HASS; none declared | https://arxiv.org/abs/2408.15766 |
| 21 | GRIFFIN (2025-02, arXiv 2502.11018) | HeadFamily | EAGLE-style head, offline; masks highly misaligned tokens in loss; token-alignable draft model feeds input tokens. | Avg AL +8%, speedup +7% over best prior methods (LLaMA, Vicuna, Qwen, Mixtral); hardware not reported. | not reported | repo hsj576/GRIFFIN; Apache-2.0 | https://arxiv.org/abs/2502.11018 |
| 22 | Medusa / Medusa-2 (2024-01, arXiv 2401.10774, ICML 2024) | HeadFamily | Parallel decoding heads on last hidden state, tree-verified; Medusa-1 frozen backbone, Medusa-2 joint; offline CE on model's own outputs. | Medusa-1 over 2.2x; Medusa-2 2.3-3.6x (baseline vanilla decoding, inference); recycling paper table 1.65x at 7B (C). Hardware not reported. | Medusa-1 "cheap"; numbers not reported | repo FasterDecoding/Medusa; Apache-2.0 | https://arxiv.org/abs/2401.10774 |
| 23 | Hydra / Hydra++ (2024-02, arXiv 2402.05109) | HeadFamily | Medusa-style heads conditioned on earlier draft tokens, frozen base, offline; OnlineSPEC made an online Opt-Hydra. Signal not extracted. | Hydra++ up to 1.31x vs Medusa and 2.70x vs AR decoding; hardware not reported. | not reported | repo zankner/Hydra; Apache-2.0 | https://arxiv.org/abs/2402.05109 |
| 24 | Apple "Your LLM Knows the Future" gated-LoRA MTP (2025-07, arXiv 2507.11851) | HeadFamily | Gated LoRA (r128) + sampler head on mask tokens predict k future tokens, base frozen, bit-identical NTP; offline SFT 50,000 iterations, not verify-pass data. | ~1.5x to ~5.2x accepted tokens per step (not wall-clock; no baseline comparison per third-party review); ~5x code/math, ~2.5x chat (Tulu3-8B, k=8; 8x A100). | Rank 1 still helps; ranks >128 hurt; LoRA memory "negligible" | no first-party code; third-party siihwanpark/MTP-GLoRA; license unknown | https://arxiv.org/abs/2507.11851 |
| 25 | FastMTP (2025-09, arXiv 2509.18362) | HeadFamily | Single MTP head (<3% params), position-shared weights, offline CE on self-distilled data (389.4K samples, 3 epochs, K=3). | Avg 2.03x over standard next-token decoding on 7 benchmarks (7B); 82% over vanilla MTP at 1.21x; single H20 server. | Under 1 day on a single H20 server | repo Tencent-BAC/FastMTP; none declared; ICLR page "withdrawn" | https://arxiv.org/abs/2509.18362 |
| 26 | DFlash block-diffusion drafter (2026-02, arXiv 2602.06036, ICML 2026) | HeadFamily | 5-layer drafter emits 16-token block in one pass from target features of 5 layers; offline (~800K target-regenerated samples), CE; no verify-pass data. | Avg 4.9x, AL 6.54 (Qwen3-4B, temp 0, vs AR, H200); ~2.4x over EAGLE-3 (Table 1) vs "up to 2.5x" intro (conflict); AL decays with length (per TTS). | GPU-hours not reported; cached hidden states grow with tapped layers | repo z-lab/dflash; MIT | https://arxiv.org/abs/2602.06036 |
| 27 | P-EAGLE parallel drafting (2026-02, arXiv 2602.01469) | HeadFamily | ~4-layer one-pass EAGLE head, offline; learnable shared hidden state + mask embedding for later positions; target hidden states + tokens. | 1.10x-1.36x over AR EAGLE-3 (GPT-OSS 120B/20B, Qwen3-Coder 30B, vLLM); vLLM blog B200 1.05x-1.69x; 0.98x (slower) at C=2 Qwen 30B. | Memory scales with seq len x parallel positions; not extracted | vLLM integration Apache-2.0; trainers in speculators, SpecForge | https://arxiv.org/abs/2602.01469 |
| 28 | DSpark, DeepSeek-AI (2026-07, arXiv 2607.05147) | HeadFamily | DFlash-style backbone + Markov head + confidence head scheduling verify length; offline (DeepSpec), TV-dominant loss on target hidden states. | DeepSeek-V4 live traffic: 60%-85% faster per-user generation than MTP-1 production baseline at matched throughput (paper-reported, static drafter); hardware unspecified. | not reported | DeepSpec (link not verified); vLLM Apache-2.0 | https://arxiv.org/abs/2607.05147 |
| 29 | Domino (2026-05, arXiv 2605.29707, EMNLP 2026) | HeadFamily | DFlash backbone + light GRU head with rank-256 residual correction; offline, block 16; CE vs target-regenerated tokens. | Up to 5.49x (Transformers), 5.8x (SGLang, Qwen3); vs DFlash AL +16.6%, speedup +12.3%, +56M params; A100-SXM4-80GB. | Not reported; head latency 2.64 -> 1.20 ms (fused) | repo jianuo-huang/Domino; none declared | https://arxiv.org/abs/2605.29707 |
| 30 | Weaver / DFlash-TfM "Trees from Marginals" (2026-07, arXiv 2607.06763; Oda, Mathieu, Knyazhitskiy, Chakhvadze; checkpoint by Mirai Labs) | HeadFamily | 56.7M AR transformer over DFlash top-512 marginals builds proposal tree; offline per target, frozen DFlash; not an online learner; signal not reported (distillation, inference). | arXiv abstract (v3): 4.08x over AR, +16.3% over DFlash; HF card `trymirai/weaver` and the sglang README: 392.8 vs 315.0 tok/s/seq = 4.37x, +24.7% over tuned DFlash (both Qwen3.6-27B, B200, batch 1, T 1.0; the card gives no paper version, so which version it matches is unverified: conflict, both read 2 Nov 2026); Mirai blog (vendor claim, read 2 Nov 2026), Apple M5: "almost 2x" over MTPLX, "over 3x" over llama.cpp; its draft budgets are 16-32 tokens. The blog and the paper are not shown to use the same checkpoint or tree budget. | not reported | repo trymirai/sglang; Apache-2.0; uzu Metal source unchecked | https://arxiv.org/abs/2607.06763 |
| 31 | SuffixDecoding (2024-11, arXiv 2411.04975, NeurIPS 2025 spotlight) | TrainFree | No weights; global suffix tree of past outputs (extended per request) + per-request tree per token; CPU ~20 us/token; scored by frequency. | AgenticSQL 5.3x over vanilla, 2.8x over EAGLE-2/3; SWE-Bench 2.5x; Llama-3.1-8B batch 1; 1 H100 (C, A) vs 8x H100 (D) conflict; AL 6.3 (D) vs 7.82 (C) conflict. | None; O(n) host RAM; ~1 month cached output on large server | ArcticInference; Apache-2.0 (B, C); not checked (A) | https://arxiv.org/abs/2411.04975 |
| 32 | SAM Decoding (2024-11, arXiv 2411.10666) | TrainFree | Suffix automaton over static corpus + current sequence, O(1) amortised update every token; falls back to secondary drafter if match short; no training. | +18% or more over other retrieval-based SD on Spec-Bench; up to 1.3x on code; +3.28% to +11.13% with EAGLE-2; 1x RTX A6000 48 GB, batch 1. | None for training; corpus memory not reported | repo hyx1999/SAM-Decoding; unknown | https://arxiv.org/abs/2411.10666 |
| 33 | Token Recycling (2024-08, arXiv 2408.08696, ACL 2025) | TrainFree | Top-k successors (k=8) of every verified position stored in a vocab x k matrix, overwritten by verify results; BFS draft tree; no params. | ~2x across 7B-33B Vicuna/Code Llama; +30% over prior train-free, +25% over Medusa; Table 1 7B 2.03x SpecBench (reading uncertain); 1x A100-80GB. | <2 MB extra storage; no training | URL not captured; unknown | https://arxiv.org/abs/2408.08696 |
| 34 | Cacheback (2025-11, arXiv 2511.21699, EMNLP 2025) | TrainFree | LRU leader/follower n-gram tables; frozen offline table + dynamic table updated from accepted tokens each step; no training. | Best speedup/acceptance among training-free model-agnostic methods on SpecBench (Vicuna 7B/13B/33B); exact multipliers not captured; Ryzen 5965WX + 4x RTX 4090. | Negligible; drafting in microseconds | URL not captured (D: open-sourced); unknown | https://arxiv.org/abs/2511.21699 |
| 35 | CopySpec (2025-02, arXiv 2502.08923) | TrainFree | No training; rolling hash finds earlier repeat of last gamma=3 tokens in context/chat history; speculates next 10; stackable on draft-model SD. | Up to 2.35x CNN/DM, 3.08x second-turn MT-Redundant; +49% avg on top of SD (7 LLMs, 5 datasets); 4x A100, batch 1, temp 0. | None; no extra GPU memory | repo RazvanDu/CopySpec; MIT | https://arxiv.org/abs/2502.08923 |
| 36 | LogitSpec (2025-07, arXiv 2507.01449) | TrainFree | No training; last-token logits guess next-next token; references for both retrieved from prompt/context. | Up to 2.61x speedup, 3.28 mean accepted tokens per step (baseline not stated); A100. | None | repo smart-lty/LogitSpec; unknown | https://arxiv.org/abs/2507.01449 |
| 37 | REST retrieval-based SD (2023-11, arXiv 2311.08252, NAACL 2024) | TrainFree | Static (not self-updating) datastore built once; context match retrieves continuations as draft tree; no training. | 1.62x-2.36x on code/text, 7B and 13B, batch 1; 1x A6000, 96 CPU cores. | Datastore 27 GB (~2.7M Python samples) | repo FasterDecoding/REST; Apache-2.0 | https://arxiv.org/abs/2311.08252 |
| 38 | Lookahead Decoding (2024-02, arXiv 2402.02057) | TrainFree | No draft model; exact parallel Jacobi iteration; discovered n-grams pooled online and verified in same pass. | Up to 1.8x MT-bench; up to 4x multi-GPU on code (A100 S1, RTX 3090 S2); lags PLD on Spec-Bench on H100 per CAS-Spec (C). | None; extra per-step FLOPs | repo hao-ai-lab/LookaheadDecoding; Apache-2.0 | https://arxiv.org/abs/2402.02057 |
| 39 | PIA / Lookahead, Alipay (2023-12, arXiv 2312.12728) | TrainFree | No training; global token trie with frequency decay and pruning updated before/after every step; hierarchical multi-branch drafts. | Deployed at Alipay since Apr 2023: 2.66x-6.26x; AntGLM-10B AntRAG 52.4 -> 165.4 tok/s with LLMA (3.16x), 5.36x Lookahead; A100/A10/V100. | None | repo alipay/PainlessInferenceAcceleration; MIT | https://arxiv.org/abs/2312.12728 |
| 40 | Prompt Lookup Decoding (2023-11, GitHub) / LLMA (2023-04, arXiv 2304.04487) | TrainFree | No training; match last n tokens against prompt/reference, copy next k, verify in one pass. | PLD README 2x-4x input-grounded (Mistral-7B, A100 40GB; not peer reviewed); LLMA 2.48x (7B). Mac blog (third party, Gemma 4 31B 4-bit, M3 Ultra): json_rag 28.42 -> 43.65 tok/s; code fails parity. | Zero | in transformers and vLLM; PLD license unknown; LLMA not checked | https://github.com/apoorvumang/prompt-lookup-decoding |
| 41 | llama.cpp n-gram drafters `ngram-simple`/`ngram-map-k`/`ngram-cache`/`ngram-mod` | TrainFree | No training; rolling-hash pool (4M entries) updated every accepted token; resets on low acceptance; shared by server slots. | Not reported in docs; docs: small n not recommended, MoEs need long drafts; CPU-side table, any backend incl. Metal. | ~16 MB, microseconds/step (inference) | llama.cpp; MIT (not re-checked) | https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md |
| 42 | ANPD Adaptive N-gram Parallel Decoding (2024-04, arXiv 2404.08698, NAACL industry) | TrainFree | No gradients; n-gram LM from prompt updated with each accepted token, multi-level N. | Up to 3.67x on LLaMA and fine-tuned variants (baseline not stated; hardware not captured). | No extra GPU memory, no retraining | not captured; unknown | https://arxiv.org/html/2404.08698 |
| 43 | ADED Adaptive Draft-Verification (2024-07, arXiv 2407.12021, AAAI 2025) | TrainFree | No fine-tuning; tri-gram "representative" updated online from recent outputs and verification results; MCTS-inspired draft maker. | not reported (not extracted); hardware not reported. | not reported | anonymous.4open.science/r/ADED-C7D5; license not checked | https://arxiv.org/abs/2407.12021 |
| 44 | DAS Distribution-Aware SD for RL (2025-11, arXiv 2511.13841) | TrainFree | Nothing trained; suffix tree over sliding window of recent rollouts updated after each verification; per-problem trees, length-aware budget. | Rollout time down up to 50% vs VeRL, identical training curves (1.5B-8B, math/code); up to six 8xH100 nodes. | CPU suffix-tree maintenance; no gradients | not found; unknown | https://arxiv.org/abs/2511.13841 |
| 45 | EfficientRollout (2026-06, arXiv 2606.18967) | TrainFree | Quantised copy of current RL policy as self-speculative drafter, re-induced each step; SD only when memory-bound; acceptance-aware draft length. | Rollout latency down up to 19.6%, end-to-end up to 12.7% vs accelerated AR baseline; hardware not extracted. | None | repo furiosa-ai/EfficientRollout; Apache-2.0 | https://arxiv.org/abs/2606.18967 |
| 46 | AgSpec (2026-10, arXiv 2610.01108; KAIST) | TrainFree | No params; session/workspace/global corpora over SAM/Suffix engines updated online; per-agent draft-length cap learned online from verification feedback. | 2.27-4.37x over AR at batch 1 (SWE-bench Verified, TeamBench); +18.0% avg over fastest prior; 1x RTX PRO 6000 Blackwell, vLLM 0.17.1, 4-bit targets. | Corpus 2.5-4.3 GB host RAM; no GPU memory | none found; unknown | https://arxiv.org/abs/2610.01108 |
| 47 | Hybrid suffix + model drafting: Snowflake patent US 12,572,745 and SGLang PR #37237 (2026) | TrainFree | No training; router picks suffix/n-gram or EAGLE3 per request from match length and draft quality score (fixed thresholds); both updated with accepted tokens. | Speedup not reported; GSM8K 86.0% hybrid vs 87.0% AR (Qwen3.5-35B-A3B, H20-141GB), maybe noise (inference). | Two drafters' CUDA graphs held at init | PR open, license not checked; patent: check freedom-to-operate (inference) | https://github.com/sgl-project/sglang/pull/37237 |
| 48 | BanditSpec (2025-05, arXiv 2505.15141, ICML 2025) | TrainFree | Nothing trained; bandit (UCBSpec, EXP3Spec) picks drafter or length per round from accepted tokens; arms PLD, REST, Suffix Tree, EAGLE-2. | LLaMA3-8B batch 1 A100 Spec Bench: vanilla 35.73, EAGLE-2 98.15, UCBSpec 105.72 tok/s; Code Editor EAGLE-2 128.76 -> UCBSpec 138.27. | None for training; all drafters resident | repo sail-sg/BanditSpec; none declared | https://arxiv.org/abs/2505.15141 |
| 49 | TapOut (2025-11, arXiv 2511.02017) | TrainFree | Meta-bandit (UCB1 over UCB-Tuned) picks among dynamic stopping policies (SVIP, AdaEDL, margin) per draft; only arm values learned, blended accept reward. | Competitive or better than tuned baselines; multipliers not captured; ablation coding 0.78 -> 1.18 (unit/baseline not verified); hardware not captured. | None beyond draft model | not found; unknown | https://arxiv.org/abs/2511.02017 |
| 50 | FlashSpec (2026-06, Zenodo preprint, not peer reviewed) | TrainFree | Triton accept/reject kernel + online bandit (UCB1/Thompson) selecting draft model from acceptance; weak credibility. | not reported; GPU/Triton (CUDA-specific). | not reported | "open-source" per abstract, URL not captured | https://doi.org/10.5281/zenodo.20766119 |
| 51 | "Not-a-Bandit" / HedgeSpec online drafter selection (2026, ICLR 2026; arXiv id not seen) | TrainFree | Nothing trained; selects among N drafters; verified trajectory prefilled into all others for full-information acceptance feedback. | Beats EAGLE3 and BanditSpec where domain-expert drafters exist (paper claim; numbers not extracted); hardware not extracted. | Prefilling chunk through extra drafters | "available" per abstract; URL not captured; unknown | https://proceedings.iclr.cc/paper_files/paper/2026/file/20563b8508ba42e1b688d922e926ee26-Paper-Conference.pdf |
| 52 | TABED Test-time Adaptive Batched Ensemble Drafting (2026, EACL Findings 2026) | TrainFree | LVLM drafts ensembled in a batch; ensemble weights adapted at test time from deviations vs ground truth seen during SD. | Avg 1.74x wall-time over AR, +5% over single drafting (11 datasets); hardware not checked. | Small (parameter sharing) | repo furiosa-ai/TABED; license not checked | https://aclanthology.org/2026.findings-eacl.205/ |
| 53 | Ouroboros (2024-02, arXiv 2402.13720, EMNLP 2024) | TrainFree | No fine-tuning; generates draft phrases to parallelise drafting and lengthen drafts; phrase-pool detail not captured. | Up to 2.8x over speculative decoding, 3.9x over vanilla; 2x A800 80GB. | None | repo thunlp/Ouroboros; Apache-2.0 | https://arxiv.org/abs/2402.13720 |
| 54 | Empirical study of SD on software-engineering tasks (2026-04, arXiv 2604.26469, ISSTA 2026) (study, no new method) | TrainFree | Not applicable; measures PLD, Suffix Decoding, MLP speculators, EAGLE-3 on code generation/repair/editing. | Code 5-gram repetition 9.62% vs NL 6.85%; agent loops (226 tasks L31-8B) inflate model-free speedups; EAGLE-3 tau 0.43-0.64 vs 1.87 Qwen3-32B; hardware not captured. | not applicable | not captured; unknown | https://arxiv.org/abs/2604.26469 |
| 55 | EAGLE-2 (2024-06, arXiv 2406.16858, EMNLP 2024) | ShapeCtrl | No new training (EAGLE head offline); per step expand highest-confidence nodes of draft tree, rerank to fixed budget; draft confidence as acceptance proxy. | 3.05x-4.26x speedup, 20%-40% faster than EAGLE-1 (3 LLM series, 6 tasks), lossless; hardware not reported. | none new | SafeAILab/EAGLE; NOASSERTION / unknown | https://arxiv.org/abs/2406.16858 |
| 56 | Sequoia (2024-02, arXiv 2402.12374) | ShapeCtrl | Offline DP finds optimal static tree for budget; hardware-aware optimiser picks size/depth from measured acceptance by depth and forward latency. | Up to 4.04x, 3.73x, 2.27x (Llama2-7B, 13B, Vicuna-33B, A100); 70B offloaded L40 0.56 s/token (abstract) vs 0.60 (HTML), conflict. | Offline search only; not reported | repo Infini-AI-Lab/Sequoia; unknown | https://arxiv.org/abs/2402.12374 |
| 57 | OPT-Tree (2024-06, arXiv 2406.17276, TACL) | ShapeCtrl | No training; per step searches tree maximising expected acceptance length under node budget from draft probabilities. | Up to 3.2x over AR; >10 tokens/step with strong drafter; MT-Bench case AL 7.74 (LLaMA-2 7B); RTX 4090 (7B), L20 (13B), 4x A100-40GB (70B). | Per-step search cost not captured | repo Jikai0Wang/OPT-Tree; unknown | https://arxiv.org/abs/2406.17276 |
| 58 | SpecInfer (2023-05, arXiv 2305.09781, ASPLOS 2024) | ShapeCtrl | Small models produce token tree; LLM verifies whole tree in parallel; none trained for tree method (draft tuning not verified). | 1.5-2.8x distributed, 2.6-3.5x offloading vs existing serving systems; 2 AWS g5.12xlarge (4x A10 24GB each). | not reported | repo flexflow/FlexFlow; unknown | https://arxiv.org/abs/2305.09781 |
| 59 | CAS-Spec with Dynamic Tree Cascade, DyTC (2025-10, arXiv 2510.26843, NeurIPS 2025) | ShapeCtrl | No training; drafts from target itself (layer sparsity, activation quantisation); DyTC routes levels and sets length/tree from online acceptance + latency predictor. | Avg 1.1x-2.3x over AR; DyTC +47% over cascade baseline, +48% over tree baseline (Spec-Bench); H100. | None; no extra weights | not found; unknown (PDF CC BY-NC-SA 4.0) | https://arxiv.org/abs/2510.26843 |
| 60 | TALON (2026-01, arXiv 2601.07353) | ShapeCtrl | Training-free per-step adaptive tree from draft confidence: deep-narrow for deterministic, shallow-wide for uncertain, per-layer node budget. | Up to 5.16x end-to-end over AR; beats EAGLE-3 consistently (5 models, 6 datasets); 1x H200 140 GB, batch 1. | None | not found; unknown | https://arxiv.org/abs/2601.07353 |
| 61 | AdaptiveSpec "Margins, Not Windows" (2026-07, arXiv 2609.02897) | ShapeCtrl | No training; per step lossy margin rule + tree policy from draft confidence and rolling acceptance window. | Up to +56% throughput over EAGLE-3 at 93%-100%+ of lossless accuracy; Qwen3-8B 1.68x -> 1.95x (1x A100 80GB, SGLang, batch 1, greedy). | None | not found; unknown | https://arxiv.org/abs/2609.02897 |
| 62 | ASDT Adaptive Speculative Decoding Tree (2025-12, IEEE AIBDF 2025) | ShapeCtrl | No training; runtime controller raises/lowers tree depth and width from token acceptance feedback. | 2.0-4.3% speedup over static tree baselines (up to 4.3% on 3B; abstract); hardware not reported. | not reported | none found | https://doi.org/10.1109/aibdf67964.2025.11440682 |
| 63 | AdaSpark (2026-10, arXiv 2610.05774; Zeraix) | ShapeCtrl | Online-learned scheduler statistics (verify widths, verify-time law, per-candidate acceptance) from verify outcomes and timing; drafter weights unchanged; n-gram rows compete. | 1.5-3.1x faster than llama.cpp DSpark, same drafters; 1.17-1.52x vs imparo 3-token chain; hardware not extracted (Metal verify-time steps cited). | None (statistical fits) | repo zeraix/imparo; license not checked | https://arxiv.org/abs/2610.05774 |
| 64 | SpecDec++ (2024-05, arXiv 2405.19715, COLM 2025) | ShapeCtrl | Acceptance-prediction head on draft trained offline once (near-training-free); stop when rejection probability passes threshold. | 2.04x Alpaca (+7.2% over baseline SD), 2.26x GSM8K (+9.4%), 2.23x HumanEval (+11.1%); llama-2-chat 7B/70B, 2x A100-80G. | not reported | repo Kaffaljidhmah2/SpecDec_pp; MIT | https://arxiv.org/abs/2405.19715 |
| 65 | AdaEDL (2024-10, arXiv 2410.18351, NeurIPS 2024 ENLP workshop) | ShapeCtrl | No training; stop drafting when entropy bound on acceptance falls below online-updated lambda; signal draft entropy + acceptance history. | 10%-57% over static draft length; up to 10% over other training-free stopping methods; 1x A100 80GB, FP32. | None | not found; unknown | https://arxiv.org/abs/2410.18351 |
| 66 | SVIP Self-Verification Length Policy (2024-11, arXiv 2411.18462, EMNLP 2025) | ShapeCtrl | Stop drafting when sqrt(draft entropy) > one threshold h; no training. | Up to +17% over fixed draft length on MT-Bench at 8K; +22% QwQ long-form reasoning; hardware not reported. | None | not found; unknown | https://arxiv.org/abs/2411.18462 |
| 67 | TETRIS (2025-02, arXiv 2502.15197) | ShapeCtrl | No training; in batch SD, picks most promising draft tokens across requests to verify, from draft-probability acceptance estimates. | Up to 6.13% lower end-to-end latency vs best baseline, 9.32% vs standard SD; 4xL40 to 8xH100, vLLM. | None | repo ZhaoxuanWu/Tetris; license not checked | https://arxiv.org/abs/2502.15197 |
| 68 | Together ATLAS (2025-10, blog + NeurIPS 2025 talk; Together AI) | Production | Heavyweight static speculator + lightweight adaptive speculator updated "continuously" from live traffic + confidence-aware controller; architecture, size, frequency, signal not reported. | Vendor claim: DeepSeek-V3.1 105 -> 501 TPS vs FP8 no-spec baseline (batch 1, 4xB200); 2.65x (blog) vs 3.18x (NeurIPS abstract), conflict; RL acceptance 10% -> 80% (vendor-reported). | not reported | none (proprietary); n/a | https://www.together.ai/blog/adaptive-learning-speculator-system-atlas |
| 69 | Baseten live draft-model training (2026-06-25, blog) | Production | EAGLE-3/DFlash draft trained "continuously" on live target hidden states streamed over RDMA; frequency, loss, rejected-token use not reported. | Blog: median accept-rate +20% where rolled out; 100%+ AL on some traffic; no baseline/benchmark; offline EAGLE-3 post 1.5-2.5x on Qwen3-4B (blog); NVIDIA clusters. | Extra GPU/pinned memory; one Kimi K2 sample >2 GB; totals not reported | none (proprietary); n/a | https://www.baseten.co/blog/live-draft-model-training-for-speculative-decoding/ |
| 70 | Fireworks FireOptimizer adaptive speculative execution (2024-08; docs current) | Production | Custom draft trained offline on customer-traced production data; re-aligned manually or on trigger when distribution shifts. Not continuous. | Blog "up to 3x latency improvements" (no benchmark detail); poorly matched drafter can be slower; hardware not reported. | not reported | none (proprietary); n/a | https://fireworks.ai/blog/fireoptimizer-adaptive-speculative-decoding |
| 71 | Snowflake Arctic Inference / ArcticSpeculator / ArcticTraining (2025-01 to 2026-09) | Production | MLP/LSTM speculator trained offline (KD on target-regenerated data) + SuffixDecoding cache updated per request (no gradient). | Blog: Llama 3.3 70B up to 4x agentic, 2.8x ShareGPT/HumanEval (vLLM 0.8.4); MLP acceptance 13.7% -> 42.7%; gpt-oss 1.6-1.8x; SuffixDecoding up to 5.3x. | 3 h (gpt-oss-20B), 21 h (120B) on 16 H200; LSTM 1.76B | ArcticInference, ArcticTraining; Apache-2.0 | https://www.snowflake.com/en/blog/engineering/fast-speculative-decoding-vllm-arctic/ |
| 72 | NVIDIA TensorRT Model Optimizer speculative-decoding training | Production | Per run: EAGLE-3/DFlash/MTP draft, base frozen, on fixed dataset (online, offline dumps, or live vLLM stream); lr 1e-4, 1 epoch, TTT 3. Not live adaptation. | not reported; NVIDIA GPUs. | Offline mode "several to tens of terabytes" disk; flex-attention 2x+ speedup, 1.5x memory on Llama-3.2-1B; GPU-hours not reported | repo NVIDIA/Model-Optimizer; Apache-2.0 | https://github.com/NVIDIA/Model-Optimizer |
| 73 | SpecForge (SGLang) v0.1-v0.3 (2025-07 to 2026-08) | Production | Draft training framework per run on prepared prompts (online hidden-state generation or ~12 TB offline); EAGLE3, DFlash, Domino, DSpark, MTP; not live-traffic. | Blog: Llama 4 Maverick draft 2.18x, Scout 2.0x MT-Bench; v0.3 ~10% better training throughput than colocated (8xH20); also ROCm, Ascend. | Not in GPU-hours; storage blog-reported | repo sgl-project/SpecForge; MIT | https://github.com/sgl-project/SpecForge |
| 74 | vLLM speculators library, Red Hat (v0.3-v0.5+, 2025-2026) | Production | Draft/MTP head (100M-400M) trained per run from verifier hidden states on regenerated responses; v0.5.0 online from on-demand hidden states; not live-traffic. | DSpark vs DFlash Qwen3-8B 20k ShareGPT full-block acceptance 0.415 vs 0.309; tutorial online EAGLE-3 ~17 min on 4xH100; task speedups not read. | MTP-finetune lightest; ~17 min on 4xH100 | repo vllm-project/speculators; Apache-2.0 | https://github.com/vllm-project/speculators |
| 75 | mlx-lm draft-model speculative decoding (2024-2025, PR #85 merged 2025-04-22) | Apple | Nothing trained; classic two-model draft/verify on one GPU with trimmable KV rewind; default 2 draft tokens; any smaller same-tokenizer model. | Community README (M4 Pro): Qwen2.5-0.5B draft for 3B 40 tok/s, 28% acceptance, 2.4 tokens/step; no non-spec baseline so no speedup readable; "2-3x" headline generic. | None; draft shares bandwidth with target | mlx-lm; MIT | https://github.com/ml-explore/mlx-lm/pull/85 |
| 76 | EAGLE-3 prototype on MLX (2026-02, mlx-lm discussion #890) | Apple | Nothing trained; community port of released EAGLE3-LLaMA3.1-8B head (32k reduced vocab, 4-bit draft), no tree verify. | Author-reported, M3 Ultra 96 GB, Llama-3.1-8B 4-bit: 118.9 vs 113.7 tok/s = 1.05x; fp16 draft 0.94x; acceptance 34%. | not applicable | scripts in discussion; license unknown | https://github.com/ml-explore/mlx-lm/discussions/890 |
| 77 | Native MTP SD in mlx-lm (2026-03, PR #990, open) | Apple | Nothing trained; checkpoint's own 1-layer MTP head (Qwen3.5/3.6) drafts; GatedDeltaNet state snapshots for rollback. | PR-reported, Qwen3.6-27B 4-bit, M4 Pro: baseline 15.7 tok/s; MTP temp 0 24.6 (1.57x, accept 88.3%); temp 1.0 1.50x; greedy output identical (author's check). | not applicable | PR branch AirRunner/mlx-lm; mlx-lm MIT | https://github.com/ml-explore/mlx-lm/pull/990 |
| 78 | llama.cpp speculative framework: draft-simple, EAGLE-3, DFlash, DSpark, MTP (2023-2026) | Apple | One `--spec-type` list; model-based types offline-trained, n-gram types learn online; `--spec-default` enables `ngram-mod`. | Third-party, M2 Ultra 192 GB, Qwen3.6-35B-A3B Q8, single RAG prompt x3: baseline 68.07 -> 77.68 tok/s (1.14x, n=4); other source (single README claim) Q4 1.35x tuned (conflict); drift suspected. | not applicable | llama.cpp; MIT | https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md |
| 79 | mlx-lm PR #1297: ngram-simple + ngram-mod port (2026) | Apple | Nothing trained; pool updated online; `DraftStrategy` protocol with optional adaptive 3-gram repetition gate (threshold 0.02), cross-request shared memory. | PR-reported, Llama-3.2-3B 4-bit, long multi-turn editing: baseline 54.09 -> 91.57 tok/s (1.69x, ngram-simple nd=4); ngram-mod nd=6 1.57x; cold start can regress; reviewer M5 32 GB. | ~16 MB for ngram-mod | PR; merge status not verified | https://github.com/ml-explore/mlx-lm/pull/1297 |
| 80 | ReDrafter, Apple Recurrent Drafter (2024-03, arXiv 2403.09919) | Apple | RNN draft head on last hidden state, base frozen, offline KD on target logits; beam search + dynamic tree; MLX port. | MT-Bench Vicuna up to 2.8x on H100, up to 2.3x on Apple silicon; MLX fp16 M1 Max 7B 1.32x, M2 Ultra 7B 1.52x, 33B 2.28x. | ~1.5 h on 8 H100; no Mac training cost reported | repo apple/ml-recurrent-drafter; Apache-2.0 | https://arxiv.org/abs/2403.09919 |
| 81 | Mirror Speculative Decoding (2025, arXiv 2510.13161; Apple) | Apple | Draft (0.6B for Qwen) trained offline; target early-exit signals launch parallel draft rollouts; target on GPUs, draft on NPUs (ANE); only token ids cross devices. | SpecBench, 14B-66B targets: 2.8x-5.8x wall-time, +30% avg over EAGLE3; 8x Apple M2 Ultra, Thunderbolt 5. | not reported | none found | https://arxiv.org/abs/2510.13161 |
| 82 | Apple Intelligence on-device SD + draft-model toolkit (2025-07, arXiv 2507.13575) | Apple | OS daemon runs speculative decoding for ~3B on-device model; developer toolkit trains r32 LoRA adapters and optional draft per base-model version, offline; signal not described. | not reported for the draft; Apple silicon, 2-bit QAT, KV-cache sharing. | not reported | toolkit separate download, not examined; unknown | https://arxiv.org/abs/2507.13575 |
| 83 | ANE draft + GPU verify prototype (harsha-gouru/ane-gpu-speculative, 2026-04, unreviewed) | Apple | Nothing trained (distillation only a plan); Core ML/ANE Qwen 0.6B draft (~2 ms/token claimed), MLX-GPU 7B 4-bit target verifies 5 tokens. | "Expected" projection, unverified: MLX GPU-only 66 tok/s at 30 W; ANE draft + GPU verify 160-200 tok/s at 33 W (2.5-3x); README baseline ~100 tok/s also estimate; Apple silicon 24 GB. | not reported | repo harsha-gouru/ane-gpu-speculative; no license, 1 star | https://github.com/harsha-gouru/ane-gpu-speculative |
| 84 | EdgeLLM (2024, IEEE TMC, DOI 10.1109/tmc.2024.3513457) | Apple | No gradient; online scalar fallback threshold from verified-accuracy history; resident draft builds width-adaptive tree by branch confidence. | Per-token time 2.9-9.3x vs single-LLM baselines on Jetson TX2/Orin NX; 3.47-4.67x on Xiaomi 10/11 phones; LLaMA2-13B >1 tok/s on phone; gains from memory wall (inference). | no training | none found | https://doi.org/10.1109/tmc.2024.3513457 |
| 85 | SpecEdge (2025-05, arXiv 2505.17052; KAIST) | Apple | Nothing trained; consumer GPUs draft, server verifies, only tokens cross network; proactive edge drafting overlaps verification. | 1.91x cost efficiency, 2.22x server throughput, 11.24% lower inter-token latency vs server-only; consumer GPUs + server. | not reported | repo kaist-ina/specedge; NOASSERTION | https://arxiv.org/abs/2505.17052 |
| 86 | sd.npu (2025-10, arXiv 2510.15312) | Apple | Retrieval SD on phone NPU (Qualcomm QNN), suffix automaton drafts; "lightweight online calibration" of drafting distribution to current task (details not read); no neural training evident. | 1.06-3.81x end-to-end vs vanilla; 1.09-2.53x vs existing SD frameworks; energy 1.11-4.18x lower vs v4 abstract 1.07-4.71x (conflict); Redmi K60 Pro/K70 Pro, OnePlus 13. | not reported | built on open-source mllm; own repo not seen; unknown | https://arxiv.org/html/2510.15312v4 |
| 87 | Apple Speculative Streaming / SpecStream (2024; mentioned only, abstract not read) | Apple | Stub: no auxiliary model (multi-stream attention, target drafts n-grams); fine-tunes target with extra streams (unverified understanding); signal not read. | not read | not read | none found | https://machinelearning.apple.com/research/llm-inference |

### 3.2 Mac platform facts

| Fact | Number | Link | Caveat |
|---|---|---|---|
| mlx-lm LoRA throughput, Mistral-7B, batch 1, 4 LoRA layers, M1 Max 32 GB | ~250 tok/s (doc); third-party log (7B QLoRA, 0.426M trainable) 256-299 tok/s, 2.6-3.3 it/s, chip not stated | https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md | Memory not reported. Drafter scale-down (~2,500-14,000 tok/s for 0.1-0.5B) is (inference), a linear guess, not measured. |
| 4-bit weight memory vs full precision | ~3.5x lower | https://developer.apple.com/videos/play/wwdc2025/298/ | WWDC25 talk statement. |
| LoRA on M3 Ultra 96 GB: 9B rank 32, 16 layers | 297-306 tok/s, 12.0 GB active, 17.5 GB peak | https://leanzero.hashnode.dev/fine-tune-qwen38-27b-with-mlx-lora-on-a-mac | Blog, single machine, author-measured. 30-step smoke run under-reads peak by up to 20 GB. |
| LoRA on M3 Ultra: 27B rank 32, 16/64 layers | 112.8 tok/s, 31.7 GB active, 57.3 GB peak; 27B rank 128 32 layers ~80 tok/s, 37.4 GB active (54.5-86.7 GB segment peaks); all 64 layers rank 128 ~87 GB, 5.5 GB swap, killed | https://leanzero.hashnode.dev/fine-tune-qwen38-27b-with-mlx-lora-on-a-mac | Metal working-set ceiling 77.8 GiB on the 96 GB machine; 2,500 steps took 11.35 h. Blog. |
| 9B rank 128, 32 layers on M3 Ultra | ~156-165 tok/s, 16.2 GB active | https://leanzero.hashnode.dev/fine-tune-qwen38-27b-with-mlx-lora-on-a-mac | Blog. |
| mlx-train-perf (M1 Max 32 GB, macOS 26.5): Qwen3-8B-4bit LoRA r8, seq 8192 | Peak 25.68 GiB stock vs 12.75 GiB flash; longest trainable seq 7,936 -> 23,040; packing 2-3x (3.0x Alpaca-shaped); fused loss ~3900x less loss-layer memory | https://github.com/IonDen/mlx-train-perf | Author-measured community library (MIT); mlx 0.32.4 added fused SDPA VJP. |
| LoRA training on iPhone 17 Pro (SmolLM3-3B 4-bit, rank 8 q,v, 36 blocks, batch 1) | 166 tok/s cold, 83-89 tok/s after ~6 min (half the loss) to 32-38 min (within 5% of plateau); ~4.0-4.2 W; ~0.01 s and 0.04 J/token; 4.7 GB at 256 tokens no checkpointing, 4.1 GB at 1024 with checkpointing (process limit ~6 GB of 12 GB) | https://arxiv.org/abs/2610.06325 | Phone; thermal throttling; pausing/burst schedules did not recover throughput. Laptop thermals not measured. |
| Quantised-backward kernel fix in MLX (A19 Pro) | Backward 1.65-2.13x faster, step 1.45-1.68x, 405-record run 2.55 h -> 1.74 h, 39.3 kJ -> 26.2 kJ; backward 1.8-2.2x forward vs 3.4-4.4x before | https://arxiv.org/abs/2610.06325 | Bug found on A19 Pro; whether M5 Macs hit the matrix-unit path is open. Rank 8 -> 32 costs 8-10%; all 7 projections vs q,v costs 17-27%; last 18 of 36 blocks 0.60x time. |
| SADAIR on-device continual LoRA (1.8B on 16 GB Mac; 3B on 32 GB Mac) | 1.8B: 5.6 min training, 17.4 min full update, 7.31 GiB peak; 3B: 7.5 min, 22.2 min, 10.20 GiB; pass rate 75.7% -> 96.0% (1.8B), 84.7% -> 97.7% (3B) | https://sadair.ai/research/continual-learning-on-personal-devices | Company post (vendor claim), not peer reviewed; chip not stated. |
| Foreground cost of background training on a Mac | p95 foreground latency +29-49% while training scheduled | https://sadair.ai/research/continual-learning-on-personal-devices | In-flight kernels cannot be interrupted; authors prefer short sessions between active use. Company post. |
| maderix/ANE training on M4 ANE | Stories110M 91 ms/step (main README) or ~115 ms/step (training README), conflict; Qwen3-0.6B ~412 ms/step; static pipeline 106.7 ms/step + 7.6 s compile per 10-step restart; ANE 0.87-1.15 TFLOPS sustained (5-9% of ~15.8-19 peak) | https://github.com/maderix/ANE | Private APIs; ~119 programs per process compile limit; dW/Adam/loss on CPU; full-model training, not LoRA; author says not a path to large models. MIT. 600-2,500 train tok/s is (inference). |
| Orion ANE training/inference compiler (M4 Max) | Stories110M step 1,345 ms (849 compute + 494 reload) vs 5,108 ms; 1,000 steps 22.4 min vs ~85 min (3.8x); delta reload 4,200 -> 494 ms; 0.656 TFLOPS; GPT-2 124M ANE 170+ tok/s vs CPU 283 tok/s | https://arxiv.org/abs/2603.06728 | One chip; private APIs; LoRA-as-input path not wired into full training loop; ANE peak ~19 TFLOPS fp16, 32 MB SRAM, ~30% drop beyond SRAM. |
| Core ML updatable models | Trainable layers: conv and fully connected only; losses MSE/CE; SGD/Adam | https://apple.github.io/coremltools/docs-guides/source/updatable-model-examples.html | Cannot train a transformer or LoRA (inference). 2019 blog constraints may be outdated; Apple page body not retrieved; whether updates run on ANE not stated. |
| Apple-silicon training profile (arXiv 2501.14925) | GPT-2-large pass: M2 Ultra MLX 2.710 s vs MPS 3.239 s (batch 8); M2 Max 2.924 vs 3.645 s; M2 Pro 8.692 vs 10.704 s; RTX A6000 2.771 s. GPU energy Whisper-medium iter: M2 Ultra 114.86 J, M2 Max 129.30 J, A6000 212.39 J, RTX 4090 108.58 J | https://arxiv.org/abs/2501.14925 | Whisper/GPT-2, not drafters. Near-capacity training on M2 Max unstable; RSS grows lazily. |
| MLX vs MPS conv backward | PyTorch-MPS 3-6x faster than MLX on conv fwd+bwd on M1 Pro/Max (scatter-add atomics); fixed in MLX PR #1541 (M1 Max 4.615 ms MLX vs 11.93 ms PyTorch) | https://github.com/ml-explore/mlx/issues/1313 | Op-specific; check embedding/lm-head gradient (inference). IDAP 2026 MLX-vs-MPS LoRA paper (doi 10.1109/idap71869.2026.11728070): title only, results not read. |
| M5 / A19 Neural Accelerators, Metal TensorOps | no numbers | https://developer.apple.com/videos/play/tech-talks/111432/ | Aimed at compute-bound work (prefill); decode (bandwidth-bound) gains least (inference). |
| Metal interactivity watchdog kills long GPU command buffers when display active | 4/4 repro on macOS 26.2 and 26.3.1; crash with ~256-token seqs (2.75 GB, ~0.8 it/s, ops ~1.2 s), not ~40-token (1.75 GB, ~3 it/s); `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` | https://github.com/ml-explore/mlx/issues/3267 | `MLX_MAX_OPS_PER_BUFFER=1` and `MLX_MAX_MB_PER_BUFFER=10` did not prevent it. Exact time limit not documented. M5 Max detail from an aggregator (not primary). |
| No public per-queue priority on Metal compute | n/a; WWDC22 concurrent dispatch 30% faster (2 interleaved images) and 70% (3) on M1 Ultra | https://upd.dev/ml-explore/mlx/issues/4716 | "No per-queue priority" is the MLX reporter's claim; MTLCommandQueue page body not read. Priority exists only for Metal IO queue. |
| ANE vs GPU power, M2 Ultra, Qwen3.5-0.8B FP16 | MLX GPU prefill 62.05 W GPU + 2.34 W CPU; MLX decode 14.17 W GPU; ANE prefill 0.22 W GPU + 1.58 W ANE + 3.77 W CPU (5.57 W); ANE pure decode 7.05 W total; batched ANE 268 tok/s (0.8B), 173 tok/s (2B), 11.3x/7.3x over sequential (~42 ms/token) | https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf | Author-reported project paper, not on arXiv; licence not checked. |
| Core ML `compute_units=ALL` routing and ANE compile hazard | ANE power <=0.024 W (ran on GPU); forced `CPU_AND_NE` caused 43-min ANE compile, then `kIOGPUCommandBuffer...` crashes in later MLX decode | https://atomgradient.github.io/hybrid-ane-mlx-bench/paper.pdf | macOS 26.3; author-reported. |
| laya-apple concurrent GPU+ANE serving (M4 Max, macOS 26.6.2) | Short request 9.9 ms ANE vs 12.2 ms MLX; mixed throughput 2.92-4.57x vs GPU-only; short-request P99 1592.9 -> 79.5 ms | https://github.com/ml-explore/mlx/discussions/4551 | Gain from concurrency/queue isolation, not engine speed; laya-apple not examined. |
| Unified memory: CPU/GPU/ANE share one pool; MLX arrays usable by CPU and GPU without copies | no number; decode throughput scales ~linearly with memory bandwidth (ACM characterisation) | https://dl.acm.org/doi/10.1145/3771563 | Decode slowdown from concurrent backward not measured by any source (open question). |
| Cost of one extra verified token, 4-bit Llama-3.1-8B, M3 Ultra (EAGLE-3 MLX port) | ~0.5-1 ms (6-11% of ~8.8 ms step); sync ~0.5 ms | https://github.com/ml-explore/mlx-lm/discussions/890 | Author-reported single prototype; (inference) that wide trees are not free on Apple GPUs. |
| fp16 vs bf16 on Apple GPUs | MLX: float16 consistently faster than bfloat16 (ReDrafter App. A.2.2); M1/M2 lack native bf16 on GPU, MTP can slow badly unless `--dtype float16` (mlx-lm PR 990) | https://arxiv.org/abs/2403.09919 | Matters for on-device training dtype. |
| Hypothesis: ANE as free co-processor for drafter backward (GPU forward/verify, ANE backward via LoRA-as-IOSurface-input, CPU dW + Adam) | ANE step 91-412 ms for 0.1-0.6B models | https://github.com/maderix/ANE | UNVERIFIED design synthesis; nobody has combined these. Private APIs, 119-program compile limit, ANE fp16 overflow, only M4/M4 Max data. |

## 4. Fit map: where each family would hook into Kapitsa, and what blocks it on a Mac

Rows refer to the survey table (§3). File:line is in this checkout (`09d49ef0`).

| Family (survey rows) | Hook in Kapitsa | Mac-specific blockers |
|---|---|---|
| **Weaver / tree builders** (30, 55-62: EAGLE-2, Sequoia, OPT-Tree, DyTC, TALON) | The tree is already built from the selector's 16x16 transition scores: `dfl_lattice` `spec_dflash.metal:1498`, `dfl_kf_table` `dflash2.cpp:1830`, `kf_build` `kflash.h:230`. A Weaver-style AR adapter would sit between `dfl_lattice` and `kf_build`. | Weaver is offline-trained per target (Qwen3.6-27B); its published verify is CUDA. Kapitsa already verifies trees with per-branch recurrent state (`docs/kflash.md:113-143`), the piece Weaver had to invent. A 56.7 M-param extra pass per cycle must stay near the ~1 ms draft cost (F72). |
| **Online weight updates, full or LoRA** (1-17: OSD, TTS, OmniDraft, Aurora, TIDE, OnlineSPEC, EvoSpec, DVI, SpecLoRA) | Data: capture rows `qwen35_v8.cpp:1179` (bf16 `[rows, 8 taps x 2048]` on the 35B), target logits `qwen35.cpp:1150`, `kf_accept` records `kflash.h:270`; host read point `src/engine/engine.cpp:3677`. Update site: drafter weights, which exist only as 4-bit packed buffers (`dflash2.cpp:297`) with the bf16 source dropped (`:421`). | **No backward kernels, no optimizer, no master weights.** Training needs new Metal kernels or a CPU path. The Metal watchdog kills long command buffers when the display is on (mlx issue 3267, 4 of 4 repros); no queue priority (mlx 4716); background LoRA raised foreground p95 latency 29-49 % on Macs (SADAIR, a company post). Changing `fc`/`hidden_norm`/`kv_all`/`k_norm` invalidates every stored drafter ring, including the SSD prefix cache (`src/engine/engine_disk.cpp:375`, `kStateFormat` `:43`); the selector, head, codebooks and calibrator do not. TTS: a naive per-round step can erase the gain; its best stride was 5, on a separate stream. |
| **Head/feature drafters, offline** (18-30: EAGLE-3, HASS, Medusa, ReDrafter, MTP heads, DFlash, DSpark) | MTP head: `qwen35_spec.cpp:213` `load_mtp`, `:954` `encode_mtp`, accept loop `src/engine/engine.cpp:3956`; used only when there is no DFlash2 drafter. | Offline training is out of scope. Mac evidence is thin: EAGLE-3 on MLX 1.05x (about 1 ms per extra verified token on M3 Ultra, row 76); ReDrafter's best beam on Apple GPUs is 1-3 (row 80). |
| **History / suffix / n-gram / retrieval** (31-47: SuffixDecoding, SAM, Token Recycling, Cacheback, CopySpec, REST, PLD, llama.cpp ngram-mod, AgSpec) | `ngram_drafts` `src/engine/engine.cpp:88` serves only Dory, Nemotron and Flint (`:3777`, `:3833`, `:3889`); it is not in the Qwen DFlash2 loop (copy lane PR #109 closed). Candidate rows can enter through the lattice tables (`dflash2.cpp:1830-1849`). | The DFlash2 loop is double-buffered and the next anchor is chosen on the GPU, so the host cannot inject proposals per cycle (`kapitsa-m1-ngram-lookback-note.md`, "Seam analysis"). A history lane must run on the GPU or at park boundaries. mlx-lm PR 1297 fails on non-trimmable recurrent caches (Qwen3.5/3.6); Kapitsa's tree verify handles them exactly. Snowflake holds US 12,572,745 on a suffix-plus-model router (row 47): check before copying that design. |
| **Draft-length, tree-shape, bandits** (48-67: BanditSpec, TapOut, AdaSpark, SpecDec++, AdaEDL, SVIP, TETRIS) | Shipping already: `KfBudget` (`kflash_budget.cpp:153` fit, `:207` observe, `:311` refit; `kflash_budget.h:65`), the tree-size rule `E(n) - rate*cost[n]` (`kflash.h:230`), opt-in `DepthGate` (`dflash2.h:251`, inert under trees `dflash2.cpp:621`). | None; replays say a better policy is worth ≤ +1.2 % (`DESIGN-draftlen.md`). §6 reproduces the calibration side. |
| **Reduced draft vocabulary** (14: EvoSpec's dynamic vocab buffer; the frequency-ranked head idea) | `build_head` `dflash2.cpp:765`; default 98,304 rows in token-id order, 65,536 rows when `NI_DFLASH_HEAD_IDS` gives a ranked id list (`dflash2.h:93`, `dflash2.cpp:784`), plus every id from the first end-of-sequence id on; `NI_DFLASH_HEAD=n` sets the row count, 0 the full vocabulary (`dflash2.cpp:540-545`). | None: rows are a `memcpy` of the target's LM head (`:845-851`). |
| **Production systems** (68-74: ATLAS, Baseten, Fireworks, Snowflake, Model Optimizer, SpecForge, speculators) | Ideas only. ATLAS: a static floor, a small adaptive speculator and a confidence controller that falls back on drift. That is a frozen drafter plus `KfBudget`. | All server GPU; internals unpublished (ATLAS, Baseten). |
| **Systemone / Drex pointer-head requests** (every family above) | **No hook, by design.** `POST /v1/systemone` generates no token: it prefills one row per question and the pointer head scores the backbone's post-final-norm hidden rows (`PointerHead::logits` `src/systemone/systemone.cpp:711`, `final_norm_row` `:749`; scoring loop `tools/systemone.cpp:367-424`, `score_max` `:280`). There is no decode loop, so there is nothing to draft or verify, and `src/systemone/` never mentions `dflash`, `draft`, `kflash` or `lattice` (`rg -n` over `src/systemone/` finds none). The drafter hooks above live on the chat decode loop (`src/engine/engine.cpp:3677-3691`) and are not reached by a systemone request. | Only a shared one: if a backbone gets a ranked head or a different `NI_DFLASH_HEAD` setting, systemone is unaffected (it never builds the draft head). The pointer head itself is a small f32 head loaded from `pointer.bin` next to the weights (`PointerHead::load` `src/systemone/systemone.cpp:647`, layout `src/systemone/systemone.h:108-110`); an online update of it would be a separate, unstudied topic. |
| **Apple / on-device** (75-87) | ANE bridge `src/ane/ane.h` (private framework, IOSurface weights as inputs, shared with Metal through one `MTLSharedEvent`); only the dense prefill FFN split uses it (`NI_ANE=1`). | ANE training uses private APIs, was tested on M4-class only, with 5-9 % utilisation (maderix/ANE) and 91-412 ms per step for 0.1-0.6 B models (Table 2). Core ML `compute_units=ALL` ran on the GPU on macOS 26.3. Nothing combines ANE backward with a GPU verify (unverified synthesis). |

**Unified memory with a 35B resident.** The drafter is 0.63 GB resident plus 0.151 GB of rings per slot (§2.2). fp32 master
weights plus Adam for the two bf16 `[248320,256]` selector codebooks would be about 1.5 GB (derived); updating only the rows of
tokens the user produced is megabytes. LoRA rank 32 on the drafter's q and v is a few megabytes (inference). Memory is not the
blocker; scheduling is.

**Idle time and read-ahead.** Kapitan's read-ahead sends the draft after a 350 ms typing pause, and Kapitsa prefills it only while
idle (`useReadAheadDraft.ts`, `POST /v1/draft` `server/main.cpp:1601`). The engine already has three `try_lock` idle jobs
(history prefill after 250 ms quiet, read-ahead, SSD write-behind). Read-ahead checkpoints the drafter's context at the user turn's
start, so a weight change that touches the ring path between the checkpoint and the send would orphan it. Adaptation therefore
(a) takes the turn only through `try_lock`, one chunk at a time, and gives it up when a request or read-ahead starts, (b) never
runs while the composer holds a non-empty draft, and (c) swaps weights only at a request boundary. How long typing, reading and
between-reply gaps last is logged nowhere.

## 5. Gaps: where published work is CUDA-only, slow, or absent on Apple silicon

1. **The training signal is shipped to another GPU; on a Mac it is already in memory.** TIDE stores hidden states to disk (0.19 TB for
   gpt-oss-120b, row 5); Aurora and TTS use a second GPU or stream (rows 2, 4). Kapitsa's capture hook already writes the 8-tap bf16 rows
   of every verify pass (`qwen35.h:125`, `qwen35_v8.cpp:1179`); only the accepted path survives as ring K/V, the rest is overwritten
   by the next verify (`docs/kflash.md:143`). Keeping them costs one GPU copy behind the verify (about 262 KB per cycle on the 35B,
   derived) into a ring the CPU reads in place: no PCIe, no extra forward. **Proposal:** a per-cycle capture ring at
   `src/engine/engine.cpp:3677-3691`.
2. **Every online trainer assumes a backward pass on a free GPU stream; Metal has neither a preemptible queue nor a long
   command-buffer budget.** TTS updates every round or every 5th at "a few speculation rounds" per step. **Proposal:** skip the
   dense backward. Update only the selector codebook rows of tokens the user produced (row-sparse, CPU/Accelerate, megabytes), or a
   small LoRA in idle slices of at most ~5 ms under `try_lock`, the way Kapitsa already preempts idle prefill. The sparse update
   leaves rings valid, so no `kStateFormat` bump.
3. **History drafters are CUDA-plus-host-RAM designs, and the Mac port fails on hybrid models.** SuffixDecoding keeps a CPU suffix
   tree (about 20 µs per token); mlx-lm PR 1297's n-gram path breaks on Qwen3.5/3.6 recurrent caches (row 79). **Proposal:** a
   rolling-hash pool in unified memory, looked up in a Metal kernel inside the lattice build, feeding candidate rows to the tree
   that Kapitsa already verifies with per-branch recurrent state; `KfCosts` charges the extra rows honestly.
4. **Vocabulary trimming is English-centric and static.** The reduced draft head keeps the first 98,304 token ids, "a frequency
   proxy" (`dflash2.cpp:768-773`), and its comment verifies coverage on English and code only. §6.1 shows what it does to Russian,
   Kazakh, Hindi, Arabic, Korean and Japanese. EvoSpec's answer is an HNSW vocabulary buffer on CUDA; the Mac answer is a counter
   and a `memcpy`.
5. **Nobody has measured decode speed while training shares an Apple GPU** (Table 2). Kapitsa can: frozen ask F4.
6. **Weaver's Apple M5 port is closed and vendor-benchmarked.** Mirai's `uzu` blog numbers (about 2x over MTPLX, over 3x over llama.cpp; read 2 Nov 2026, vendor claim, not reproduced)
   are a blog claim and the paper's kernels are CUDA. KSX/KFlash trees are the open Metal counterpart; a head-to-head needs
   Qwen3.6-27B in both engines (not run). What was measured, inside `uzu` only, is Weaver against that engine's own plain DFlash
   chain on the same weights (§5.7).

### 5.7 Weaver against its own chain on the M5 Pro (measured 3 Nov 2026, 108 legs)

Mirai `uzu` 0.6.15 on an Apple M5 Pro (48 GB), `trymirai/Qwen3.6-27B-M`, greedy, 256 output tokens, one engine and one set of
weights. Three arms: Weaver tree builder, the same DFlash drafter with plain argmax chain construction (Weaver switched off in
`shapes.json`; both arms use tree budget and depth 16 on this chip), and no speculator. Four languages (ru, kk, zh, en) x 3 prompts
x 3 repetitions x 3 arms = 108 legs, arm order rotated per repetition, all on AC power. All 108 are valid (rc 0, 256 output tokens,
no void). The figure is `generate_tokens_per_second` (decode only), the mean of the three repetitions per prompt, then the mean of
the per-prompt ratios. The interval is a bootstrap (4,000 resamples) over prompts: 12 for "all", 3 per language, so the per-language
intervals are wide.

| Class | Weaver tok/s | Argmax chain | No speculator | Weaver / no speculator | Weaver / chain |
|---|---|---|---|---|---|
| ru | 35.1 | 28.8 | 17.7 | 1.99x [1.74, 2.32] | 1.21x [1.15, 1.27] |
| kk | 21.4 | 20.8 | 17.7 | 1.21x [1.19, 1.24] | 1.03x [0.99, 1.06] |
| zh | 36.5 | 32.2 | 17.8 | 2.05x [1.86, 2.27] | 1.13x [1.06, 1.19] |
| en | 50.6 | 41.5 | 17.6 | 2.88x [2.62, 3.10] | 1.22x [1.20, 1.23] |
| **all (12 prompts)** | 35.9 | 30.9 | 17.7 | **2.03x [1.68, 2.38]** | **1.15x [1.09, 1.19]** |

- Weaver is about 2x faster than no speculator and about 15% faster than the plain chain overall. The chain gain is positive in ru, zh
  and en. **In Kazakh there is no clear gain over the chain**: 1.03x, interval 0.99x to 1.06x, which includes 1.0. Kazakh also
  gains least over no speculator (1.21x).
- **Scope.** The model files were read from the external SSD for all 108 legs. Loading is outside every reported number
  (time to first token equals input tokens divided by the prompt rate) and the 16 GB of weights stay resident during decode (decode
  runs about 170 times faster than reading 16 GB per token over USB4 would allow). The figures are decode speed only. There is no
  cold-start, time-to-first-token or disk-streaming claim.
- **Limits.** `uzu`, not Kapitsa: this shows what Weaver does for the 27B on this chip, not what it would do inside Kapitsa's DFlash2
  loop, whose selector already builds trees from 16x16 transitions. Weaver is offline-trained for Qwen3.6-27B only, so it does not
  transfer to the 35B-A3B. One chip, one engine version, 256 tokens, three prompts per language. A first run of 71 legs on battery
  power is void and not used; it agreed (1.99x, 1.14x).
- Rerun: the legs are `mirai bench <arm dir> <task>.json` for 12 prompts; the scripts and raw JSON stay with the task outside the repo.

## 6. Top 3, ranked (for English code first, English non-code second)

Scope (Alish, 4 Nov 2026): English coding prompts are the headline number, English non-coding the second class, and a multilingual set is an optional extra check, not a gate. The sections keep their numbers from the first version, so §6.1 is now an optional multilingual idea and §6.2 comes first.

Feasibility is my score, 0-1, for "built and shown to win a default in about two weeks of engineering". Expected gains are
inferences from the measured numbers cited. GPU results: F1 (§6.1, "M3 Pro, battery 99→91%, 3 prompts/class", INCONCLUSIVE), F1b (§6.1), the 108 Weaver legs in `uzu` (§5.7), and F2 (§6.5, the 35B on the English coding-agent ledger, M5 Pro, AC power, two runs).

### 6.1 A draft head that follows the user's tokens (feasibility 0.8; optional, non-English only)

**Scope note (4 Nov 2026).** This idea does nothing for English code or English prose: English and C++ have 0.0 % of text outside the default head, the ranked head measured 1.00x on English (F1b, M3 Pro) and the full head 0.945x (F1). It is out of the English-first top 3. Everything below is kept as the optional multilingual check, and the F1 and F1b results are direction for non-English users only.

**Read this first.** Most of the gain that this idea appears to offer non-English users is what `NI_DFLASH_HEAD=0` (the full 248k
vocabulary, already in the engine) gives today with no new code. A user-ranked head is worth building only if it beats the **full
head on tok/s**, not the 98,304-row head on tokens per cycle. The full head reads about 173 MB more per draft pass, about +0.58 ms
(derived at ~300 GB/s; 4.1 % of a 14.3 ms one-row cycle, 2.7 % of a 21.5 ms eight-row cycle) and roughly doubles the drafter's
cost. A ranked head matches the full head on tok/s only if it keeps the in-head share of generated tokens at about 98-99 % (geometric
acceptance model with the 35B's tau 3.6 and 7 drafts: ≥ 98.1 % at a 14.3 ms cycle, ≥ 98.7 % at 21.5 ms). The measured in-head
shares below are 47-96 % for the same-size head built from half an article of history, so with that little history it would not
get there; with months of a user's text it might.

**What.** Count the tokens the user's own conversations produce (CPU, a 248k-entry vector, no text stored) and fill the head's rows
with those first, then id order, at the same row count; compare against `NI_DFLASH_HEAD=0`. Output text cannot change: only
proposals move.

**Engine facts the numbers depend on** (`dflash2.cpp:764-805`, `dflash2.h:93`): with no ranked list the head is the first 98,304
ids in id order; with `NI_DFLASH_HEAD_IDS` it is **65,536** ranked rows unless `NI_DFLASH_HEAD=n` sets another count; in both cases
every id from the first end-of-sequence id on (248,044 for this tokenizer) is added. The comment at `dflash2.cpp:768-773` that the
id-order head "covers 99.8-100 % of generated tokens" was measured on English (and code) only. The Splash/Flash-Next rowlogs in the
evidence below may have used an id-order head too; I did not verify which head they ran.

**Evidence (measured on CPU).**
- Out-of-head share of text under the 35B's tokenizer (one article of up to 30 kB per language; small samples): English 0.0 %, C++ 0.0 %,
  Spanish 14.1 %, Turkish 24.7 %, Chinese 36.9 % (314 tokens), Russian 51.0 %, Kazakh 51.0 %, Korean 51.9 %, Japanese 53.1 %,
  Arabic 69.0 %, Hindi 73.3 %. (Ids ≥ 98,304 only; the 248,044+ ids that the engine adds to the head are rare in text.)
- On 26 recorded Flash-Next prompts (T0, `kapitsa-ksx-tree/.auto/runs/rowlog-all-20261029`) 25 % (Japanese), 28 % (Korean), 11 % (mixed) and
  35 % (Chinese) of generated tokens have id ≥ 98,304; 55-75 % of those were accepted drafts. Cutting each logged accepted chain at
  its first out-of-head token (an estimate) lowers tokens per cycle by 31.9 % (ja), 34.5 % (ko), 22.0 % (mixed), 35.2 % (zh), and 0 % on
  code and count. This is the cost of the id-order head against the **full** head on those runs, i.e. the part of the gain that
  `NI_DFLASH_HEAD=0` already recovers. A head ranked by the other 4 sessions of the same language recovers only 3-4 points of the loss.
- A perfect drafter limited to the head (`tools/draft_head_coverage.py`, stacked PR), half of one article as the user's history, tokens per
  cycle: id-order head / ranked 65,536 rows / user-first 98,304 rows / full vocabulary. Kazakh 1.91 / 6.63 / 6.69 / 7.99, Hindi 1.38 / 6.98 /
  6.99 / 8.00, Korean 1.90 / 4.70 / 4.78 / 8.00, Turkish 3.43 / 5.17 / 5.44 / 7.99, Arabic 1.43 / 3.31 / 3.32 / 7.99, Russian 2.04 / 2.62 /
  2.86 / 7.92, Spanish 4.86 / 5.30 / 5.83 / 7.99, Japanese 1.96 / 2.60 / 3.01 / 7.92, Chinese 3.14 / 1.89 / 3.20 / 7.85, English 7.94 /
  6.95 / 7.94 / 7.94. A ceiling is not a speed-up, and even with a perfect drafter no ranked head here reaches the full-vocabulary ceiling.

**Expected gain on M5 Pro, 35B-A3B (inference).** English and code: 0 %. For text with 25-75 % of tokens out of head, switching from
the default head to **the full head** is worth about +20 to +45 % tokens per cycle, minus its +2.7 to +4.1 % cost: roughly **+15 to +40 %
tok/s, which `NI_DFLASH_HEAD=0` gives with no new code.** The extra gain from a user-ranked head over the full head is at most the 2.7-4.1 %
the full head costs, and only if the ranked head keeps nearly all tokens (≥ 98 %): **0 to +4 % over the full head**. Alish's own traffic
(Kazakh, Russian, English) sets the weight. The replay uses Flash-Next traces from Splash; Kapitsa's own head on the 35B is unmeasured.
The more useful product change is a **language check that turns on the full head** (phase 2 below), which needs no ranking at all.

**Test that would reject it.**
- CPU, now: tokenise 100 real conversations of the target users (Kapitan's `statev2.sqlite`, with their consent) and run
  `draft_head_coverage.py`. **Reject the head work if the id-order head's ceiling is within 3 % of the full-vocabulary ceiling**
  (out-of-head share under about 2 %).
- GPU, frozen ask F1 (re-gated; order A B F F B A per prompt, 2 legs per arm): the default head against `NI_DFLASH_HEAD=0` (full) and against a ranked head, on 3
  Russian, 3 Kazakh, 3 Chinese and 3 English prompts. Two questions, in order: (1) does the full head beat the default on tok/s for
  non-English (expected yes, reject the whole of §6.1 if it gains < 3 % on every non-English class); (2) does the ranked head beat the
  **full head on tok/s** (needs ≥ +2 % on some class, else ship the full head behind a language check and drop the ranking).

**F1 result (run, not inferred).** "M3 Pro, battery 99→91%, 3 prompts/class" (start 99 % in `run-env.txt`; end 91 % in the last leg's `env.txt`, `en2_6_A`: "leg end 2026-11-02T17:23:09Z battery 91%"; every leg's `env.txt` has its own start and end battery), 72 legs, Qwen3.6-35B-A3B 4-bit, greedy, 256 tokens, A B F F B A per prompt; greedy ids are identical on all 12 prompts and every leg exited 0. **Verdict: INCONCLUSIVE.** A is the default head, B the ranked head, F the full head; each cell is the mean over 3 prompts.

| Class | F/A | B/A | B/F |
|---|---|---|---|
| ru | 1.116 | 1.101 | 0.987 |
| kk | 1.047 | 1.107 | 1.058 |
| zh | 1.136 | 1.095 | 0.965 |
| en | 0.945 | 1.001 | 1.059 |

- The full head **fails the English gate**: en F/A is 0.945, below the 0.97 bar (the full head costs English 5.5 %, more than the 3 % limit).
- kk F/A 1.047 is below the 1.10 bar, so only ru and zh clear it.
- The ranked head gains about 10 % over the default on non-English (B/A 1.095-1.107) with English flat (B/A 1.001), and ties or loses to the full head on ru and zh (B/F 0.987, 0.965) but beats it on kk (1.058).
- The result shows direction, not an M5 Pro gain: one machine, on battery, 3 prompts per class, ranking lists from 3-4 Wikipedia articles. The gates were fixed before the run and are not changed. Per-prompt output is in the research notes (`f1-results-m3pro/analyse-output.txt`, kept outside the repo to stay under the PR size cap).
- The follow-up run "F1b" (below; not the frozen ask F2 in §7) compares the ranked head with a language-gated full head on 12 prompts per class.

**F1b result (run, not inferred; the follow-up to F1, not the frozen ask F2 in §7).** "M3 Pro, battery 90→64%, 12 prompts/class" (start and end in `run-env.txt`: "battery at start: 90%", "battery at end: 64%"), 264 legs, same model, commit and gates as F1; 0 failed legs, greedy ids identical on every prompt. F1b asked whether a language-gated full head G (full head on ru/kk/zh, default head on English) beats the ranked head B; English legs are A and B only, so G/B there is A/B. **Verdict: PER-LANGUAGE (gated better: zh; ranked better: kk).** Mean over 12 prompts ± 95 % half-width, as printed by `analyse.py`:

| Class | G/B | B/A | F/A |
|---|---|---|---|
| ru | 1.013 ±0.013 | 1.134 ±0.047 | 1.149 ±0.046 |
| kk | 0.949 ±0.009 | 1.105 ±0.018 | 1.049 ±0.021 |
| zh | 1.056 ±0.032 | 1.055 ±0.026 | 1.114 ±0.037 |
| en | 0.997 ±0.006 | 1.003 ±0.006 | n/a |

- zh: the gated full head wins (G/B 1.056, lower end 1.024 ≥ 1.00). kk: the ranked head is clearly ahead (G/B 0.949, upper end 0.958 ≤ 1.00), and kk does not qualify for the full head (F/A 1.049 < 1.10). ru ties: G/B 1.013 with its lower end at 1.000, short of the 1.02 bar.
- The ranked head gains 5.5-13.4 % over the default on every non-English class and costs English nothing (B/A 1.003). The full head's English cost (0.945 in F1) is why a gate is needed to use it on zh.
- Direction, not an M5 Pro gain: one machine, on battery, prompts 3-11 hand-written and not native-reviewed, ranking lists from 3-4 Wikipedia articles. Gates and prompts were fixed before the run. Output is in the research notes (`f2-results-m3pro/analyse-output.txt`), outside the repo.

### 6.2 A persistent history lane in the lattice (feasibility 0.45)

**What.** A rolling-hash pool of the user's recent text (this conversation; with consent, the workspace) looked up on the GPU inside the
lattice build. Matches become extra candidate rows in the same 8-row verify, charged by `KfCosts`.

**Measured so far** (`tools/history_lane_replay.py`, stacked PR): 16 Flash-Next/Splash sessions (`kapitsa-ksx/.auto/runs/acc-step0-20261029`), T0, count
and code prompts; lookup inside one request only (a shared corpus would find the benchmark's repeated prompts). **The replay skips 32
of the 112 logged requests (29 %) because their rowlog blocks do not add up to the request's token count; the 5,679 cycles replayed are
64 % of the 8,841 logged cycles**, so the sample is biased toward requests whose logs are clean. Logged: 2.874 tokens per cycle.

| lookup length | oracle: history branch free, verify takes the longer (upper bound) | history replaces the drafter | cost-charged branch (35B: 14.3 ms + 1.03 ms per row) | drafter misses where history knew the token |
|---|---|---|---|---|
| n ≥ 4 | 3.063 (+6.6 %) | 2.880 (+0.2 %) | 1.005x at 3 rows; 0.992x at 7 | 10.5 % |
| n ≥ 3 | 3.168 (+10.2 %) | 2.898 (+0.8 %) | 1.012x at 4 rows (best); 0.994x at 7 | 15.8 % |
| n ≥ 2 | 3.267 (+13.7 %) | 2.824 (-1.7 %) | 1.008x at 3 rows; 0.976x at 7 | 22.8 % |

A row-neutral variant (history takes the drafter's rows when the drafter's expected accepted drafts are low), with the threshold fit on half
the sessions and tested on the other half (odd/even sessions, threshold 0.75 chosen in sample), gives +0.8 to +1.0 % tokens per cycle (n ≥ 4: 1.008x both ways; n ≥ 3: 1.009-1.010x) and costs no extra row. On the 19-request opencode
agent ledger with the model's tokenizer, history alone gives 1.52 / 1.63 / 1.79 tokens per cycle at n ≥ 4 / 3 / 2 (hit rate 16 / 24 /
40 %); the drafter alone gets 4.6-4.9 there, so this is not a comparison.

**English coding-agent replay (measured 3 Nov 2026, exact replay at T0).** On the 19 requests of `opencode-ledger.json` (one English coding-agent session, M5 Pro lattices of the 35B-A3B drafter, baseline 7-node tree at 3.99 tokens per cycle), a lookup of the last 3 or more tokens (`replay_hist.py` keys n3c3, n3c4, n3c7: lookup minimum 3, then copy the next 3, 4 or 7 tokens after it into the same 8 verify rows) gains **+11.0 % [7.1, 17.9], +13.3 % [8.3, 22.3] and +12.9 % [5.8, 27.2] tokens per cycle** (95 % CI over the 19 requests; a lookup minimum of 4 with 4 copied tokens, n4c4, gains +11.5 %). Short answers that copy file paths and names gain up to about +96 % on one request; the five longest requests (by cycles) move from -7.9 % to +10.3 % (n3c7). On ordinary English chat (12 prompts, M3 Pro recordings) it is -0.2 % [-1.0, +0.5]. This is tokens per cycle before host cost, one session, so it is the upper end of what the GPU lane would deliver. The scripts and JSON stay with the task outside the repo (`kapitsa-online-drafter-notes/f3/`: `replay_hist.py`, `hist_agent.json`).

**Expected gain on M5 Pro (inference).** **0 to +1.2 % on single-request benchmark prompts** (measured replay on the 64 % of cycles whose logs add up, under the 2 % bar). Upside
only where history repeats what the drafter cannot guess: edit loops, tool-call syntax, quotes across turns. AgSpec reports 2.27-4.37x
over plain decoding on agent traces on a CUDA GPU, a different baseline, and the SE empirical study warns that agent loops inflate
model-free speed-ups. Without a drafted-35B measurement I would plan **+2 to +6 % on agent turns (guess), 0 % elsewhere.**

**Test that would reject it.**
- CPU, partly run: the table above. **Reject for benchmark-style prompts:** the cost-charged gain is ≤ +1.2 %, under the 2 % bar in
  `DESIGN-draftlen.md` (in the `kapitsa-ksx-tree` worktree).
- GPU, frozen ask F2: record the 35B on `bench/harness/agent-traces/opencode-ledger.json` with `NI_DFLASH_LAT`, then replay a joint
  drafter-plus-history policy on the CPU. **Reject if the cost-charged gain on the ledger is < +2 %.**

**Live run (F4, 5 Nov 2026, M5 Pro, AC power, same 19-request ledger and flags as §6.5, T 0.6).** A GPU lookup kernel (`dfl_hist`, `NI_DFLASH_HIST=3,8,4`: the last 3 to 8 tokens looked up in the prompt and output so far, up to 4 following tokens put into the lattice's children tables before the tree builder, same 7-node budget, no extra verify rows) in an unmerged build. Three legs in turn per run (baseline with the same build, history lane, selector update), a fresh server and the same KFlash calibration file for every leg, two runs.

| Run | Tokens per cycle: baseline → history lane | Mean per-turn write speed ratio, 11 matched turns |
|---|---|---|
| 1 | 3.91 → 4.19 (**+7 %**) | 1.22x |
| 2 | 3.89 → 4.17 (**+7 %**) | 1.11x |

The replay said +13.3 %; the live gain is about half of it, **+7 % tokens per cycle in both runs**, which clears the +2 % bar. The speed ratio moves by 0.1 between runs and is not a stable number here (11 turns that match the recording on both legs, sampled at T 0.6, so different turns match in different legs); read the tokens-per-cycle column. Answers matched the recording on 11 of 19 turns in the baseline and in both knob legs of both runs. Caveats: DSH was visible, and the harness measured other processes' GPU use during each leg (`contention` in the result files). Run 1 was clean by the 5 % bar: mean 3.0 % (baseline, max 4.8 %), 2.6 % (history lane, max 7.4 %), 0.2 % (selector update, max 0.9 %). Run 2 was not: mean 6.8 % (baseline, max 13.7 %), 5.7 % (history lane, max 8.8 %), 5.9 % (selector update, max 9.2 %), WebKit (the DSH window). The tokens-per-cycle gains do not depend on timing and agree between the two runs; the speed ratios do, so run 2's are the weaker of the two; a third run was planned and was stopped when the budget ran out; the knob lives in an unmerged build. The speed figures here are not a merge gate; they say the history lane works on the GPU and is worth about +7 % tokens per cycle on this session. Raw files: `kapitsa-316/f4/out/`.

### 6.3 Idle-time, row-sparse update of the selector from verify-pass data (feasibility 0.2)

**What.** TTS/EvoSpec/Aurora-style distillation toward the target's own logits, shrunk to fit a Mac: update the selector's
`candidate_selector.{predecessor,successor}_codebook` rows (bf16, not quantized, `dflash2.cpp:437`) of tokens the user produced, and
optionally a small LoRA on the drafter's q/v. Data: the capture ring (§5.1) and the target's top-16 at each verify row. Updates run in
slices of ≤ 5 ms under `try_lock` while idle and are swapped in at a request boundary.

**What it must deliver** (`tools/online_drafter_sim.py gain`, calibrated baseline, 8,605 recorded cycles): a uniform rise of +0.10 in the
per-position acceptance log-odds raises tokens per cycle by 3.1 %. It pays for a training share of 3 % of the GPU at +0.10, 5 % at
+0.17 and 10 % at +0.34 (odds ×1.10, ×1.18, ×1.40), at fixed verify depth. TTS reports +41 % mean acceptance length for a weaker
static drafter on 20 k-token outputs; the recorded 256-token outputs show no decay (tokens per cycle 4.9 in the first quarter of a
request, then 2.4-2.6, flat). Re-fitting KfBudget's calibration in-sample per session gains only +0.4 % tokens per cycle: the headroom
is in what the drafter proposes, not in how its confidence is read.

**English coding-agent replay (measured 3 Nov 2026, exact replay at T0).** Same 19 requests and baseline as §6.2: small gradient steps on the rows of the two selector codebooks that each verify result touches, state carried across the session, gain **+8.6 % [6.0, 12.2] tokens per cycle at learning rate 0.003** (+7.7 % [3.9, 13.4] at 0.01; `edge_agent.json` holds these two rates). Reset at every request (`replay_edge.py --cold`, `edge_agent_cold.json`) it is only +1.5 % at 0.003 (3.989 to 4.047), so it needs memory across requests. The rate was chosen after looking at this same session, so +8.6 % is optimistic. On English chat it **loses 1.2 % [-2.1, -0.3]**. Raw files: `kapitsa-online-drafter-notes/f3/` (`replay_edge.py`, `edge_agent.json`), outside the repo.

**Expected gain on M5 Pro (inference).** **+3 to +8 % tok/s on long same-user answers (thinking mode), about 0 on short ones,** with a
factor-of-three uncertainty and a real chance of a net loss if training touches decode. Nothing measures decode speed while an Apple
GPU trains (gap 5).

**Test that would reject it.**
- CPU: from a 2,000-cycle capture dump (frozen ask F3), fit the sparse update on 1,500 cycles and score the held-out 500. **Reject if
  held-out tokens per cycle rise < 5 %**, or if the update needs more than ~5 % of a cycle's GPU time per round.
- GPU, frozen ask F4: decode tok/s and p95 inter-token latency while a 5 ms training-shaped command buffer runs every 100 ms. **Reject
  if decode loses > 3 % or p95 grows > 5 %.**

**Live run (F4 of the 5 Nov 2026 run in §6.2, same legs).** The update runs on the GPU inside the verify's command buffer, after the acceptance (`dfl_sel_update`, `NI_DFLASH_SELUPD=0.02`): one threadgroup, fp32 master copies of the two codebooks (485 MB), the touched rows written back as bf16 for the next drafting pass; not idle-time CPU work as designed above. State carries across the 19 requests of one server run and is lost at restart.

| Run | Tokens per cycle: baseline → selector update | Mean per-turn write speed ratio, 11 matched turns |
|---|---|---|
| 1 | 3.91 → 4.15 (**+6 %**) | 1.13x |
| 2 | 3.89 → 4.13 (**+6 %**) | 1.16x |

The replay said +8.6 % at learning rate 0.003 (+7.7 % at 0.01); live at 0.02 it is **+6 % tokens per cycle in both runs**. The extra cost of the update (one serial threadgroup per cycle) did not show up as slower decode: the speed ratio is above 1 in both runs, with the same noise as above. The knob did not change which turns match the recording (11 of 19, as in the baseline). Not measured: English chat, where the replay lost 1.2 %; whether the update helps or hurts quality over longer sessions; learning rates other than 0.02. Caveats as in §6.2 (run 1 clean, run 2 above the 5 % foreign-GPU bar, two runs, unmerged build).

### 6.4 A wider verify tree on English code: replayed, rejected for the 35B-A3B

The 7-node tree is the engine's default on the 35B. Replaying larger node budgets with `tools/tree_sim.py` on the 19 English opencode-ledger lattices (M5 Pro, 1,015 chain cycles, exact at T0) gives, in tokens per cycle: chain 3.696, 7 nodes 3.989 (+7.9 % over the chain), 11 nodes 4.220 (+5.8 % over 7 nodes), 15 nodes 4.410 (+10.6 % over 7 nodes); the lattice oracle is 5.433. With the 35B's measured cycle cost (14.26 ms at one row, about 1.03 ms per extra row, §2.2; the 12 and 16 row costs are extrapolated, not measured) tokens per ms fall to 0.89x (11 nodes) and 0.80x (15 nodes) of the 7-node tree. The extra rows pay only if each one beyond the 8th costs under 0.31 ms (11 nodes) or 0.28 ms (15 nodes): the tokens-per-cycle gain over the 7-node tree (+5.8 %, +10.6 %) times its 21.47 ms cycle, over the 4 or 8 extra rows. That is a quarter to a third of the measured 1.03 ms. The dense 27B, where 16 rows cost 1.07x of 8 (`docs/kflash.md`), is the case where a wider tree can win, and its 16-row block is already the default. Output is in `kapitsa-online-drafter-notes/f3/tree_budget_agent.txt`, outside the repo. Verdict: not a candidate for the 35B on an English coding session.

### 6.5 F2 measured: the 35B on the English coding-agent session (M5 Pro, AC power, 4 Nov 2026)

`compare_engines.py` agent phase (`agent_replay.py`) on `opencode-ledger.json`: 19 requests of one English coding-agent session (OpenCode, up to 22.7k tokens of context, greedy, thinking off, recorded tool-call gaps of at most 5 s), `kapitsa serve --slots 1`, 35B-A3B 4-bit, High Power Mode, GPU lock held, AC checked before and after every leg, a fresh server per leg. Two runs, the second in the opposite leg order. Raw files stay with the task outside the repo (`kapitsa-316/f2-out/`, `f2-out-rep/`, `f2/f2_acceptance.json`).

| Run (leg order) | DFlash2 drafter | `--dflash none` (MTP head, depth 1) | Drafter faster by |
|---|---|---|---|
| Run 1 (drafter, then plain) | 173.5 tok/s, 30.8 s | 106.9 tok/s, 45.5 s, **7,996 pages swapped** | 62 % (not clean, see below) |
| Run 2 (plain, then drafter) | 173.2 tok/s, 30.9 s | 128.3 tok/s, 36.7 s, no swap | **35 %** |

The no-speculator baseline (`--dflash none --spec 0`: no drafter and no MTP drafting), one clean run on AC with no swap, right after run 2: **98.6 tok/s, 45.1 s, 3,420 tokens, 11 of 19 answers match the recording.** Against it the drafter is **1.76x** in writing speed (173.2 against 98.6 tok/s) and the session is 1.46x shorter (30.9 against 45.1 s); per turn the drafter is faster on 18 of 19 turns, mean ratio 1.58, bootstrap 95 % CI over turns [1.42, 1.75]; over the 11 turns whose answers match on both legs the mean per-turn ratio is 1.51. The engine's MTP head (depth 1) is itself 1.30x over no speculation (128.3 against 98.6 tok/s, faster on 19 of 19 turns), so of the drafter's 76 % over no speculation, 30 points come from MTP and the drafter adds the 35 % on top. One run for the no-speculator leg, so no interval over repeats, and run 1's swapped plain leg (106.9) was not used.

Writing speed in tok/s, then the time for the 19 turns (prefill included). **Use run 2: the drafter is about 35 % faster than the engine's own MTP head on this session, and the session finishes about 16 % sooner (30.9 against 36.7 s).** The drafter leg reads 173 tok/s in both runs; only the plain leg moved (107 against 128), and it is the one that swapped in run 1. Per turn in run 2 the drafter is faster on 13 of 19 turns, mean ratio 1.21, median 1.20, bootstrap 95 % CI over turns [1.09, 1.34]. Edits, which carry the long code answers, gain most: 4.50 tokens per cycle against 2.91 on reads and shell commands. Tokens per verify cycle: 4.01 with the drafter (code 4.06, the one prose answer 3.61; 4.01 again in run 2), against 1.83 with the MTP head. Acceptance is 4.01 on the timed drafter leg (4.006 in run 2) and 4.04 on an earlier log-only parse of the same 19 requests (`f2_acceptance.json`, no timing used); both are a little above the 3.99 of the replay in §6.2 and §6.3.

**Caveats, in order of weight.**
- **Run 1's 62 % is inflated.** The harness flagged the swap on its plain leg only; plain decode there ran about 17 % slower than in the clean run 2. Quote 35 %, not 62 %.
- **Two runs, one session.** The drafter leg agrees to 0.2 % across runs; the plain legs differ by 20 % because of the swap. No interval over repeats exists, and the interval above is over the 19 turns of one run.
- **The legs did not write identical answers.** 12 of 19 turns match the recording on the drafter leg and 11 on the plain leg (3,440 and 3,311 tokens), so the rates average over slightly different text. Over the 10 turns whose answers match on both legs, run 2's mean per-turn ratio is 1.08, the most conservative figure: about 8 % faster (`kapitsa-316/f2/matched_ratios.py`). The honest range on this session is therefore +8 % to +35 %, centered near +20 to +35 %.
- The 35 % is against the engine's MTP head, the fallback a Mac gets without the drafter; the no-speculator leg (98.6 tok/s) gives 1.76x. Both are one session of 19 turns and one run per leg after the swapped first run.
- Short answers (turns of 31 to 66 tokens) are dominated by start-up and say little on their own; 6 of 19 turns were slower with the drafter in run 2.

What it settles: the drafter works on English coding-agent traffic, 4.0 tokens per cycle, and the replay baseline the history lane and the selector update are measured against is the live number. What it did not settle at the time: the history lane and the selector update were still replays. They were run live on 5 Nov 2026 (§6.2 and §6.3): +7 % and +6 % tokens per cycle against the replays' +13.3 % and +8.6 %.

### Not in the top 3

- The draft head that follows the user's tokens (§6.1): a non-English idea with no English-code gain; optional multilingual check only.
- Offline-trained EAGLE-3 / MTP heads: the F72 estimate is -6 to +32 % (central +10-15 %), needs a training budget, and has no online part.
- Weaver itself: an offline per-target adapter, so competitor and idea source (§5.6), not an online learner. Measured in `uzu` against its own chain: about 15% faster than the chain and about 2x over no speculator on the 27B, no clear gain over the chain in Kazakh (§5.7).
- KfBudget refits, depth gates, row bandits: shipped or replay-rejected (≤ +1.2 %). On the 16 recorded sessions, leave-one-session-out
  log-loss is 0.4297 for a frozen calibration fitted on the other sessions, 0.4366 for a warm-started online refit, 0.4463 for the refit from
  scratch, and 0.4623 with an extra online drift term (`online_drafter_sim.py calib`). More online fitting does not beat a good cache.
- ANE draft or ANE training: unverified, private APIs, M4-class evidence only. Kept as research (phase 5).

## 7. Phased plan

GPU time is wall time on an idle M5 Pro under `/tmp/kapitsa-gpu.lock`, estimated. Each phase merges only under AGENTS.md: a default
earned by an interleaved A/B plus the `logit_dump` output check, or it stays an open draft PR.

| Phase | What | Owner | Gate to pass | GPU time |
|---|---|---|---|---|
| 0 | Land the simulators. Add a per-cycle trace ring at `src/engine/engine.cpp:3677-3691` (accepted, rows, `conf[7]`, ms) behind an env switch, and expose accepted length in `usage`. No behaviour change. | Kapitsa worker | A replay of a new trace reproduces the logged tok/cycle within 1 %; `api_test` unchanged | 0 on CPU; 15 min to record a trace |
| 1 | Measure what is missing: plain 35B-A3B on M5 Pro and ask F2 (the English coding-agent ledger). **Done for the baselines, 4 Nov 2026 (§6.5):** the drafter against the engine's MTP head (35 % faster) and against no speculation (1.76x, 98.6 tok/s) are measured on the English coding-agent session. The history lane cannot be measured live because the engine has no history lane (phase 3 builds it), so its number stays a replay. Ask F1 is optional. | Orchestrator grants GPU | F2: measured on the English coding-agent session, drafter 173.2 tok/s, MTP head 128.3, no speculation 98.6. Open: a live history-lane run (needs phase 3). Optional F1 (its `analyse.py` prints the verdict): full head ≥ +10 % over the default on one non-English class (F/A ≥ 1.10) with English F/A ≥ 0.97 (the full head's own cost is 2.7-4.1 %, so English may lose up to 3 %), greedy ids identical on every prompt | F2 ≈ 30 min, plain ≈ 15 min, optional F1 ≈ 50-60 min |
| 2 (optional, multilingual) | **Full head for non-English users, then a user-ranked head only if it beats the full head** (§6.1; no English-code gain, so it is not a gate for anything else): a language check from token counts turns on `NI_DFLASH_HEAD=0` (no ranking needed); a token counter and ranked file at `KAPITSA_HOME/adapt/<model>/head.ids` only if F1 shows the ranked head ahead of the full head. Kapitan: consent toggle. | Kapitsa worker + Kapitan worker | Full head default-on for detected non-English users on the 35B-A3B on M5 Pro by interleaved A/B (3 pairs per class), `logit_dump` identical, English and code within noise; ranked head additionally ≥ +2 % over the full head | ≈ 2 h |
| 3 (do first for English code) | **History lane** (§6.2) in the lattice with a GPU hash pool, cost charged through `KfCosts`; per session first; workspace pool only after the consent flow. | Kapitsa worker | ≥ +2 % tok/s on the opencode ledger (English coding agent) and agent turns, no loss > 0.5 % on English code or English prose, exact outputs, recurrent-state tree verify unchanged | ≈ 4 h (record 30 min, A/B 3 h) |
| 4 (second for English code) | **Selector update** (§6.3): CPU offline fit on dumps first; Metal idle-slice training only if that passes. Measure watchdog and foreground latency (F4). | Kapitsa worker | Held-out tokens/cycle +5 % on English code and no loss on English chat (it cost -1.2 % at lr 0.003), decode loss ≤ 3 %, p95 ≤ +5 %, no ring invalidation, outputs identical | ≈ 6 h (dump 1 h, F4 1 h, A/B 4 h) |
| 5 | Research only: an English-code fine-tune of the drafter (slot 3 of the top 3; needs a training run, not measurable on the Mac), ANE-side drafter forward or LoRA hot-swap, a Weaver-style adapter on the lattice, a per-user LoRA for the 27B drafter. | Orchestrator, after phase 4 | A measured Mac number that none of the 87 papers has | ≈ 4 h per experiment |

Phases 3 and 4 are the English-code work and phase 2 is optional. Phase 3 does not depend on phase 4. Phase 4 needs the capture ring from phase 0. Order for English code: phase 0, phase 1 (done: no-speculator baseline and F2), phase 3, phase 4. Phase 2 and ask F1 run only if a multilingual check is wanted, and their result gates nothing else.

Frozen GPU asks (F2, F3 and F4 are the English-code asks; F1 is optional and was run once, on the M3 Pro, result in §6.1; each takes `/tmp/kapitsa-gpu.lock`, 35B-A3B on the M5 Pro unless stated):

| Ask | What | Wall time | Settles |
|---|---|---|---|
| F1 (optional, multilingual) | Head comparison in the order A B F F B A per prompt (A default, B ranked via `NI_DFLASH_HEAD=98304 NI_DFLASH_HEAD_IDS=<corpus ranking>`, F full via `NI_DFLASH_HEAD=0`); 3 Russian, 3 Kazakh, 3 Chinese, 3 English prompts, 256 tokens, greedy, 2 legs per arm = 72 legs; one run, no retry. Verdict by rule: REJECT-6.1, FULL-HEAD-SUPPORT with BUILD-RANKING or SHIP-FULL-DROP-RANKING, INCONCLUSIVE, or a correctness finding (its `analyse.py` prints the verdict) | ~50-60 min | §6.1 |
| F2 (first) | One run of the 35B on `opencode-ledger.json` (English coding agent) via `bench/harness/agent_replay.py` with `NI_DFLASH_LAT` and `NI_DFLASH_STEP1=1` (see `tools/tree_sim.py`'s header) | ~30 min | §6.2, and a drafted-35B agent tau |
| F3 | Dump 2,000 cycles of capture rows and verify top-16 (needs the phase-0 ring or a `NI_DFLASH_LAT` extension) | ~1 h | §6.3 CPU fit |
| F4 | Decode while a 5 ms training-shaped Metal command buffer runs every 100 ms, display on: tok/s, p95 inter-token latency, watchdog errors | ~45 min | §6.3, gap 5 |

## 8. What Kapitan needs

- **Consent.** Nothing consents to learning from conversation text today; read-ahead has a toggle and no consent copy. Add a client
  setting beside `readAheadWhileTyping` (`packages/contracts/src/settings.ts:437`, default off), named for what it does ("Learn my
  words on this Mac"), with one sentence: it counts the tokens you use to tune speculation, stays on this Mac, and changes speed, not
  answers. Privacy mode (`agentNetworkAccess`) must keep implying that nothing leaves the Mac; this feature adds no network path.
  Telemetry stays off by default.
- **What is stored.** Phase 2 stores a token-count vector (about 1 MB) and a ranked id list, no text. Phase 3's pool holds text: keep it in
  memory per session by default, and persist it per workspace only after a second, explicit opt-in. Phase 4's capture ring holds hidden
  states derived from user text: memory only, never persisted.
- **Where.** `~/.cache/kapitsa/adapt/<model>/` beside `kflash/` (per-model, per-chip calibration, about 700 B each, `Engine::save_kflash`
  `src/engine/engine.cpp:654`), keyed by model, drafter and bits. Adapted weights stay outside the drafter folder.
- **User controls.** A "Reset what Kapitan learned" button that deletes that directory, a size readout, and a line in model settings saying
  whether adaptation is active. A bug-report export must not include token counts unless the user adds them.
- **Engine contract.** A flag on `kapitsa serve` or a request field to turn it on, a stats field for accepted length per request (today
  only `usage.kapitsa.kflash.cycles`), and a "hold" signal while the composer has a draft, so adaptation never competes with a send. There is
  no user id in `src/`, so "per user" means per install.
- **Read-ahead.** It stays first in the idle queue. Adaptation starts after a longer quiet period than history prefill's 250 ms (say 3 s)
  and not while read-ahead's checkpoint is the live state.
- **Languages.** Phase 2 (optional): its language check comes from the token counts themselves (share of ids ≥ 98,304), with no language detector.

## 9. Prototypes, how to rerun, and what was not done

| Tool | What it does | Result above |
|---|---|---|
| `tools/online_drafter_sim.py`, `tests/test_online_drafter_sim.py` (this PR) | `calib`: log-loss of static, frozen, KfBudget-style, warm-started and drift predictors on rowlogs, leave-one-session-out. `gain`: break-even acceptance shift for a training share. `shift`: leave-one-class-out calibration shift. Port of `KfBudget::fit` (`kflash_budget.cpp:153`). | §6.3 and "Not in the top 3" |
| `tools/history_lane_replay.py` + test (stacked PR) | Token-level replay of a history lane against logged accepted counts; cost-charged branch; row-neutral swap; agent-ledger mode. | §6.2 |
| `tools/draft_head_coverage.py` + test (stacked PR) | Out-of-head share and perfect-drafter ceiling of the id-order head against a user-first head, from token-id files. | §6.1 |

```
python3 tools/online_drafter_sim.py calib kapitsa-ksx/.auto/runs/acc-step0-20261029/s*.rowlog
python3 tools/online_drafter_sim.py gain  kapitsa-ksx/.auto/runs/acc-step0-20261029/s*.rowlog --train-frac 0.05
python3 tools/online_drafter_sim.py shift kapitsa-ksx-tree/.auto/runs/rowlog-all-20261029/fresh_*.rowlog
```

The rowlogs are Flash-Next on Splash, in other worktrees (`kapitsa-ksx`, `kapitsa-ksx-tree`) and not committed here. A line is
`{acc, rows, conf[7], wall_ms}`; `acc` counts accepted drafts.

**Not done, and limits.**
- The research behind the survey and the replays used no GPU. The only live 35B-A3B numbers on the M5 Pro are the F2 legs in §6.5: three timed legs per engine setting at most, 19 turns of one coding-agent session, so no interval over repeats. Every other speed figure for the 35B is an inference from other measurements.
- Replays use Flash-Next traces from Splash (T0, 256 tokens, count/code/CJK prompts), not the 35B on Kapitsa. The history-lane replay skips 29 % of the logged requests (32 of 112; rowlog blocks that do not add up), keeping 64 % of the logged cycles. I did not check which draft head those runs used. A rowlog gives accepted
  counts and confidences, not hidden states, so weight updates cannot be replayed from it.
- The Wikipedia coverage numbers come from single articles of up to 30 kB, with the first half as "history". Treat them as direction.
- The survey is from search results and fetched abstracts on 2 Nov 2026. Four numeric tokens in the compact table are not verbatim in
  the merged source table (rows 48, 61, 78, 79: rounded or recomputed from a longer cell). Many licences were not checked (GitHub API rate
  limit); where sources disagree the table says "conflict".
- The Weaver A/B (§5.7) ran in `uzu` on the 27B, not in Kapitsa and not on the 35B-A3B; no Kapitsa-vs-Weaver head-to-head was run.
- Snowflake's patent on suffix-plus-model routing needs a legal read before anyone copies its router.
- The raw research notes (about 270 kB: the four family notes, the code map, the measured baseline) stay with the task outside the repo.
