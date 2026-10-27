| arm | decode T=0 tok/s | decode T=0.6 tok/s | TTFT 1K s | prefill 1K tok/s | TTFT 4K s | prefill 4K tok/s | TTFT 16K s | prefill 16K tok/s | turn-2 TTFT s (4K chat) | peak RSS MiB | min free+reclaimable GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kapitsa --engine splash @7e478053 | 56.2 (n 6) | 55.1 (n 6) | 1.40 (n 6) | 741 (n 6) | 3.44 (n 6) | 1184 (n 6) | 16.46 (n 6) | 1172 (n 6) | 0.53 (n 6) | 2229 | 5.4 |
| Kapitsa --engine splash @ff6151c8 | 58.2 (n 6) | 53.5 (n 6) | 1.44 (n 6) | 722 (n 6) | 3.44 (n 6) | 1186 (n 6) | 17.16 (n 6) | 1124 (n 6) | 0.54 (n 6) | 2385 | 6.8 |
| plain Splash 1.4.0 | 57.7 (n 6) | 54.8 (n 6) | 1.46 (n 6) | 711 (n 6) | 3.51 (n 6) | 1161 (n 6) | 16.32 (n 6) | 1182 (n 6) | 0.53 (n 6) | 2938 | 7.5 |
| plain Splash 1.4.0 (paired with K2) | 56.4 (n 6) | 53.8 (n 6) | 1.45 (n 6) | 713 (n 6) | 3.62 (n 6) | 1125 (n 6) | 18.05 (n 6) | 1069 (n 6) | 0.57 (n 6) | 2756 | 5.8 |
| Kapitsa native (MTP) | 55.4 (n 8) | 56.6 (n 8) | 1.41 (n 8) | 733 (n 8) | 4.74 (n 8) | 860 (n 8) | 25.58 (n 7) | 754 (n 7) | 0.26 (n 7) | 32433 | 2.3 |
| llama.cpp | 9.0 (n 2) | 9.0 (n 2) | 28.12 (n 2) | 37 (n 2) | 110.74 (n 2) | 37 (n 2) | 634.24 (n 2) | 30 (n 2) | 3.34 (n 2) | 32984 | 11.3 |
