| Prefill (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kapitsa 71d6891 | 2,428 | 3,043 | 3,126 | 3,041 | 2,754 | 2,289 | 1,694 | 1,110 |
| llama.cpp b11840 | 1,893 | 2,208 | 2,172 | 2,053 | 1,766 | 1,355 | 914 | 549 |
| MLX (mlx-lm) | <sub>not supported (mlx-lm 0.32.2)</sub> | × | × | × | × | × | × | × |
| <sub>Kapitsa 71d6891: min-max (runs)</sub> | <sub>2,326-2,432 (5)</sub> | <sub>2,909-3,057 (5)</sub> | <sub>3,038-3,140 (5)</sub> | <sub>2,970-3,045 (5)</sub> | <sub>2,544-2,766 (5)</sub> | <sub>2,093-2,300 (5)</sub> | <sub>1,691-1,701 (3)</sub> | <sub>1,104-1,112 (3)</sub> |
| <sub>llama.cpp b11840: min-max (runs)</sub> | <sub>1,837-1,896 (5)</sub> | <sub>2,203-2,210 (5)</sub> | <sub>2,151-2,174 (5)</sub> | <sub>2,036-2,055 (5)</sub> | <sub>1,764-1,768 (5)</sub> | <sub>1,350-1,356 (5)</sub> | <sub>914-914 (3)</sub> | <sub>544-551 (4)</sub> |
| <sub>Kapitsa ÷ llama.cpp</sub> | <sub>1.28×</sub> | <sub>1.38×</sub> | <sub>1.44×</sub> | <sub>1.48×</sub> | <sub>1.56×</sub> | <sub>1.69×</sub> | <sub>1.85×</sub> | <sub>2.02×</sub> |

| Decode (tok/s) | 1K | 2K | 4K | 8K | 16K | 32K | 64K | 128K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kapitsa 71d6891 | 145.6 | 142.7 | 137.8 | 131.3 | 116.9 | 97.8 | 74.1 | 49.1 |
| llama.cpp b11840 | 90.7 | 89.0 | 86.5 | 83.0 | 73.5 | 61.5 | 46.7 | 30.8 |
| MLX (mlx-lm) | <sub>not supported (mlx-lm 0.32.2)</sub> | × | × | × | × | × | × | × |
| <sub>Kapitsa 71d6891: min-max (runs)</sub> | <sub>144.1-146.8 (5)</sub> | <sub>125.5-144.5 (5)</sub> | <sub>136.1-139.3 (4)</sub> | <sub>129.2-132.7 (4)</sub> | <sub>102.9-118.2 (5)</sub> | <sub>96.6-98.2 (5)</sub> | <sub>68.5-74.4 (3)</sub> | <sub>49.0-49.6 (3)</sub> |
| <sub>llama.cpp b11840: min-max (runs)</sub> | <sub>90.2-91.0 (4)</sub> | <sub>88.6-90.3 (4)</sub> | <sub>86.4-86.7 (3)</sub> | <sub>82.7-83.0 (5)</sub> | <sub>73.0-73.7 (5)</sub> | <sub>61.4-61.7 (5)</sub> | <sub>46.7-46.7 (3)</sub> | <sub>29.9-31.0 (4)</sub> |
| <sub>Kapitsa ÷ llama.cpp</sub> | <sub>1.60×</sub> | <sub>1.60×</sub> | <sub>1.59×</sub> | <sub>1.58×</sub> | <sub>1.59×</sub> | <sub>1.59×</sub> | <sub>1.59×</sub> | <sub>1.59×</sub> |
