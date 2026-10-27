| arm | decode T=0 tok/s | decode T=0.6 tok/s | TTFT 1K s | prefill 1K tok/s | TTFT 4K s | prefill 4K tok/s | TTFT 16K s | prefill 16K tok/s | turn-2 TTFT s (4K chat) | peak RSS MiB | peak footprint MiB (tree) | system used GiB at peak | min free+reclaimable GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kapitsa --engine splash @7e478053 | 56.2 (n 6) | 55.1 (n 6) | 1.40 (n 6) | 741 (n 6) | 3.44 (n 6) | 1184 (n 6) | 16.46 (n 6) | 1172 (n 6) | 0.53 (n 6) | 2229 | not measured | not measured | 5.4 |
| Kapitsa --engine splash @ff6151c8 | 55.9 (n 10) | 52.1 (n 10) | 1.45 (n 10) | 715 (n 10) | 3.45 (n 10) | 1181 (n 10) | 16.54 (n 10) | 1166 (n 10) | 0.56 (n 10) | 2385 | 25612 (2 legs) | 39.9 | 6.1 |
| plain Splash 1.4.0 | 57.7 (n 6) | 54.8 (n 6) | 1.46 (n 6) | 711 (n 6) | 3.51 (n 6) | 1161 (n 6) | 16.32 (n 6) | 1182 (n 6) | 0.53 (n 6) | 2938 | not measured | not measured | 7.5 |
| plain Splash 1.4.0 (paired with K2) | 55.5 (n 10) | 54.0 (n 10) | 1.47 (n 10) | 706 (n 10) | 3.55 (n 10) | 1149 (n 10) | 17.32 (n 10) | 1114 (n 10) | 0.57 (n 10) | 2964 | 26038 (2 legs) | 40.0 | 5.8 |
| Kapitsa native (MTP) | 53.7 (n 12) | 57.2 (n 12) | 1.41 (n 12) | 733 (n 12) | 4.81 (n 12) | 847 (n 12) | 25.48 (n 9) | 757 (n 9) | 0.26 (n 9) | 32433 | 32467 (2 legs) | 45.7 | 2.3 |
| llama.cpp | 9.0 (n 2) | 9.0 (n 2) | 28.12 (n 2) | 37 (n 2) | 110.74 (n 2) | 37 (n 2) | 634.24 (n 2) | 30 (n 2) | 3.34 (n 2) | 32984 | not measured | not measured | 11.3 |
