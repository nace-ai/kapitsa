# Runs of b4e33647, 2026-11-17 (this branch's code on a withdrawn free-memory commit)

Added beside the other READMEs. b4e33647 is this branch's code on top of b0432561, a change to the cache's save path
that was withdrawn and is not in this branch (branch `disk-copy-free-memory`): a request's own slot was copied only
when the copy fit in free memory, and was written straight from the slot first otherwise. It touches only that path
(`disk_leaving` for a busy slot), so the 16K follow-ups below are this branch's; the 9B v2's replaced-conversation row
took the written-first path and is not. b4e33647 was built on a rented M1 under its lock from main 4a9cce41's tree plus
the branch's diff (Apple clang 21); `prefix_cache_test` (180 passed), `api_test` (497 passed) and
`kflash_cache_test` passed there. Main 4a9cce41 for the A/B is the M1's own build of it (`kapitsa-main21`, the same
toolchain). Servers ran with `--host 127.0.0.1 --private`, every step under the Mac's GPU lock.

| Folder | Mac | What |
| --- | --- | --- |
| `head-b4e33647-n4b-m4/` | M4, macOS 26.6.2 (25G83), Nano 4B FP8 | `runAB.sh` (`ab_nohit2.py`, three pairs, main first); `ttft-n4b-m4.sh` |
| `head-b4e33647-n9b-m2/` | M2, macOS 27.2.1 (26B131), Nano 9B v2 NVFP4 | `runAB.sh`; `--ttft` by `runSSD.sh` (its A/B is the other-toolchain one) |
| `head-b4e33647-other-toolchain/` | both | a first A/B against a main built with Apple clang 15: not comparable, kept as run |

## 16K-token follow-up (`ttft-16000.log`, three rounds)

| | Cold | From the SSD after a restart | From RAM |
| --- | ---: | ---: | ---: |
| 4B, M4 | 37.723 / 37.757 / 37.841 s | 0.259 / 0.245 / 0.246 s | 0.138 / 0.139 / 0.150 s |
| 9B v2, M2 | 114.272 / 114.389 / 114.098 s | 1.398 / 1.402 / 1.398 s | 1.082 / 1.083 / 1.081 s |

Every check passed; the SSD's reply is the RAM path's text in all six rounds.

## A new conversation takes the slot of one the SSD does not hold yet (`ab_nohit.log`)

Request 2's first token, rounds 1 / 2 / 3:

| | Main | b4e33647 | How the branch saved request 1's conversation |
| --- | ---: | ---: | --- |
| 4B, M4 | 4.978 / 4.976 / 4.973 s | 5.026 / 5.034 / 5.035 s | copied (77-80 ms), written behind in 179-181 ms |
| 9B v2, M2 | 15.906 / 15.990 / 15.984 s | 16.088 / 16.156 / 16.125 s | 147-190 MB free: written first, 193-202 ms (b0432561's path, not this branch's) |

The 4B's copy fit in free memory, so b0432561 took the copy path this branch takes without it. Both arms wrote the
same text in every round. On main the Nano runs without the cache and saves nothing; the
branch's extra time is the save, which a later follow-up to request 1's conversation reads back instead of prefilling
it again (16K tokens: about 38 s on the 4B, 114 s on the 9B v2).
