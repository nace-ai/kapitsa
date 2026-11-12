# Results, 2026-11-12: the KQuant 4B against lmstudio-community's 4B on Kapitsa 0.2.35

The [2026-11-10 record](../2026-11-10-m5-pro-kquant/README.md) timed the 4B packs on Kapitsa 0.2.33, each with the
drafting Kapitsa gave it then. This record times them on 0.2.35 with the same drafter and MTP head on both packs, so
the weights are the only difference. The KQuant 4B writes about as fast as lmstudio-community's: 223.4 against 217.6
tok/s pooled over both decode modes, 2.7% apart, inside the 3% band we count as the same.

| Pack | Repo and revision | `model.safetensors` sha256 |
| --- | --- | --- |
| A, lmstudio-community | [`lmstudio-community/Qwen3.5-4B-MLX-4bit`](https://huggingface.co/lmstudio-community/Qwen3.5-4B-MLX-4bit) @ `c43ee1d` | `52dc943eaf6093a2…` |
| B, KQuant | [`nace-ai/Qwen3.5-4B-MLX-Q4-KQuant`](https://huggingface.co/nace-ai/Qwen3.5-4B-MLX-Q4-KQuant) @ `9f7d327` | `7918aaa709154dea…` |
| Drafter, both | [`z-lab/Qwen3.5-4B-DFlash`](https://huggingface.co/z-lab/Qwen3.5-4B-DFlash) @ `9a1996c` | `1eb221d36abb13a5…` |
| MTP head, both | the `mtp.safetensors` Kapitsa installs with `4b` | `4f491dd27975520e…` |

The hashes are of the files the legs read; the first three match each revision's published hash.

Setup: Kapitsa 0.2.35 (`30b18f9`), MacBook Pro M5 Pro 48 GB, macOS 26.5.2, on AC, one GPU lock hold. The harness is
the 2026-11-10 record's [`ab_legs.sh`](../2026-11-10-m5-pro-kquant/ab_legs.sh): each leg starts `kapitsa serve` on the
arm's folder and points `kapitsa bench --quick` at it (512-token answers, three at T 0.6 and three at T 0; a leg's
figure is the median of its three). Order `A X X B B A`: the `X` legs point at an empty folder and stop at once, which
keeps the ports in step with the other sizes of the same session; the `SHORT: X has 0 valid legs` line in
[legs.log](legs/legs.log) is about them. A leg is void if the Mac leaves AC or the load average reaches 4.5. No 4B leg
was void.

| Leg | Pack | T 0.6, tok/s | T 0, tok/s |
| --- | --- | --- | --- |
| 1-A | lmstudio-community | 208.7 | 232.6 |
| 6-A | lmstudio-community | 205.7 | 223.2 |
| 4-B | KQuant | 220.4 | 226.3 |
| 5-B | KQuant | 222.6 | 224.4 |

At T 0.6 the KQuant pack writes 222 against 207 tok/s; at T 0 225 against 228. The README's decode column pools both.

What the leg files flag: every JSON carries `CONTENTION` (mean foreign GPU load 82-87%, top process `kapitsa`) and
an `invalid` note that bench did not start the server. Bench counts a server it did not start as foreign load (with
`KAPITSA_BENCH_SERVER` its own-process set is empty), so the `kapitsa` share is the leg's own `kapitsa serve`, as in the
2026-11-10 record. The files also list foreground apps above 0.5% of the GPU: `com.apple.WebKit.GPU` on every leg,
Slack on 1-A, Microsoft Edge on 5-B and 6-A. Both packs ran with them present, and the 2.7% gap is inside the 3% band,
so the result stays "about as fast". No 4B leg flags a repetition loop.

[sp35.json](sp35.json) holds the legs; [rows.py](rows.py) prints the README's 4B cells and the table above from it,
and `python3 rows.py --check` checks it against the leg files in [legs/](legs).
