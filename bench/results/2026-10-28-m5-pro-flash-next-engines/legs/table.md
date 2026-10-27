| arm | decode T=0 tok/s | decode T=0.6 tok/s | TTFT 1K s | prefill 1K tok/s | TTFT 4K s | prefill 4K tok/s | TTFT 16K s | prefill 16K tok/s | turn-2 TTFT s (4K chat) | peak RSS MiB | min free+reclaimable GiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kapitsa --engine splash | 56.2 (n 6) | 55.1 (n 6) | 1.40 (n 6) | 741 (n 6) | 3.44 (n 6) | 1184 (n 6) | 16.46 (n 6) | 1172 (n 6) | 0.53 (n 6) | 2229 | 5.4 |
| plain Splash 1.4.0 | 57.7 (n 6) | 54.8 (n 6) | 1.46 (n 6) | 711 (n 6) | 3.51 (n 6) | 1161 (n 6) | 16.32 (n 6) | 1182 (n 6) | 0.53 (n 6) | 2938 | 7.5 |
| Kapitsa native (MTP) | 55.4 (n 4) | 56.2 (n 4) | 1.41 (n 4) | 733 (n 4) | 4.89 (n 4) | 834 (n 4) | 25.47 (n 4) | 757 (n 4) | 0.28 (n 4) | 29811 | 2.4 |
| llama.cpp | 9.0 (n 2) | 9.0 (n 2) | 28.12 (n 2) | 37 (n 2) | 110.74 (n 2) | 37 (n 2) | 634.24 (n 2) | 30 (n 2) | 3.34 (n 2) | 32984 | 11.3 |
