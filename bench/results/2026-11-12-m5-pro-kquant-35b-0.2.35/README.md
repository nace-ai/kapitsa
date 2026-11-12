# Results, 2026-11-12: the KQuant 35B-A3B against MTPLX's on Kapitsa 0.2.35

The [2026-11-10 record](../2026-11-10-m5-pro-kquant-35b/README.md) timed the 35B-A3B packs on Kapitsa 0.2.32 against a
`kapitsa serve` that bench did not start, which bench reports as foreign GPU load. This record times them on 0.2.35 with
bench starting its own server, and with the same drafter and MTP head on both packs, so the weights are the only
difference. The KQuant 35B-A3B writes about 8-10% faster than MTPLX's: 234.5 against 213.0 tok/s pooled over both decode
modes, +10.7% at T 0.6 and +9.6% at T 0. Leg 1-A, the lowest MTPLX leg, is the only one with foreign load during
decode (Microsoft Edge, 6.2%); with 4-A alone as the MTPLX figure, T 0.6 reads 215.0 against 198.6 (+8.3%) and the
pooled gain +9.4%.

| Pack | Repo and revision | sha256 of the files the legs read |
| --- | --- | --- |
| A, MTPLX | [`Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed`](https://huggingface.co/Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed) @ `9546051` | shards 1-4: `09f3e6ecb0b7af6e…`, `31dcdb1c49eebdb1…`, `3e66de06a1f03dad…`, `a5d0cf03519c26f8…` |
| B, KQuant | [`nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant`](https://huggingface.co/nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant) @ `37e804f` | shards 1-4: `44b58d84b22116de…`, `0d87665c02cc4722…`, `b1d5c3c38d4acfc3…`, `71feb1db60a9f131…` |
| Drafter, both | [`incoai/Qwen3.6-35B-A3B-DFlash2`](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) @ `51ef7b6` | `model.safetensors` `d1f16c0ff85afbe6…` |
| MTP head, both | `mtp.safetensors`, the same file in both repos above | `6661d105178c3d7c…` |

The hashes are of the files the legs read, and each equals the LFS sha256 the Hub publishes at that revision.

Setup: Kapitsa 0.2.35 (`30b18f9`), MacBook Pro M5 Pro 48 GB, macOS 26.5.2, on AC, one GPU lock hold, 20:17 to 20:27.
The harness is [legs.sh](legs.sh): each leg runs `kapitsa bench --model <folder> --quick --json`, so bench starts and
measures its own server (512-token answers, three at T 0.6 and three at T 0; a leg's figure is the median of its
three). Order A B B A. Before a leg it waits for AC and a 1-minute load average under 4; a leg is void if it ends on
battery, at a load of 4.5 or more, or if a Kapitsa process outside the leg's own bench ran during it (a census every
5 s). No leg was void, and the census found no other Kapitsa process.

| Leg | Pack | T 0.6, tok/s | T 0, tok/s | Accepted share, T 0.6 / T 0 |
| --- | --- | --- | --- | --- |
| 1-A | MTPLX | 189.7 | 233.5 | 0.690 / 0.746 |
| 4-A | MTPLX | 198.6 | 230.3 | 0.709 / 0.744 |
| 2-B | KQuant | 212.1 | 258.3 | 0.746 / 0.783 |
| 3-B | KQuant | 218.0 | 249.8 | 0.744 / 0.775 |

Both packs read the same bytes per token: their 2,090 weight tensors have the same names, types and shapes, and their
`config.json` quantization blocks are the same (4-bit, group 64, with the same 80 modules at 8-bit). So the difference
is the drafter's acceptance: the share of written tokens that came from accepted drafts is 0.745 against 0.70 at T 0.6 and
0.78 against 0.745 at T 0.

What the leg files flag: no leg carries an `invalid` note or a repetition-loop flag. 1-A and 2-B carry `CONTENTION`
from Microsoft Edge (mean 6.2% and 4.4% of the GPU, during the 4K prefill and on 1-A also during T 0.6 decode); 3-B and
4-A list Edge or `com.apple.WebKit.GPU` under 4%. The browser ran through legs of both packs.

`python3 rows.py` prints the table above and docs/benchmarks.md's 35B-A3B figures from the leg files in [legs/](legs);
`python3 rows.py --check` checks that every leg ran on AC, with bench's own server and no loop flag.
