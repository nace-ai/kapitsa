# Latest-main M4 ANE 4K rerun

Loaded-model HTTP TTFT on this M4 only. Explicit W8A8 approximation changes replies. Full HTTP time includes possibly different continuation work; no decode-speed claim.

Primary protocol confirmed: **True**. Full first-pair schedule: True.

| Workload | Eligible pairs | TTFT A/B ratio (95% interval) | TTFT time reduction | Identical replies |
| --- | ---: | ---: | ---: | ---: |
| code-scheduler-4096 | 4 | 2.771 (2.737–2.806) | 63.92% | 4/4 |
| prose-zh-4096 | 3 | 2.747 (2.731–2.764) | 63.60% | 0/3 |

Every attempt, including rejected attempts:

| Workload | Pair | Attempt | TTFT A / B (s) | HTTP A / B (s) | TTFT other GPU peak including desktop A / B (%) | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| code-scheduler-4096 | 0 | 0 | 17.7325 / 6.6526 | 27.3460 / 16.4168 | 0.000 / 16.986 | False |
| code-scheduler-4096 | 0 | 1 | 17.6951 / 6.3945 | 27.2126 / 15.9141 | 0.000 / 0.000 | True |
| code-scheduler-4096 | 1 | 0 | 17.7461 / 6.4534 | 27.0856 / 15.9565 | 33.297 / 0.000 | False |
| code-scheduler-4096 | 1 | 1 | 17.7479 / 6.4307 | 27.1664 / 15.9427 | 0.000 / 0.000 | True |
| code-scheduler-4096 | 2 | 0 | 17.7645 / 6.3373 | 27.4441 / 15.7664 | 0.000 / 0.000 | True |
| code-scheduler-4096 | 3 | 0 | 17.7474 / 6.4517 | 27.2756 / 15.9829 | 23.666 / 0.000 | False |
| code-scheduler-4096 | 3 | 1 | 17.7440 / 6.4400 | 27.1890 / 16.0553 | 0.000 / 0.000 | True |
| prose-zh-4096 | 0 | 0 | 17.7486 / 6.5122 | 36.9448 / 24.9963 | 21.947 / 1.198 | False |
| prose-zh-4096 | 0 | 1 | 17.7413 / 6.4483 | 36.7888 / 25.0178 | 0.000 / 1.052 | True |
| prose-zh-4096 | 1 | 0 | 17.7406 / 6.4676 | 36.8827 / 25.0684 | 0.000 / 19.873 | False |
| prose-zh-4096 | 1 | 1 | 17.7423 / 6.4503 | 36.9822 / 25.2260 | 0.254 / 0.000 | True |
| prose-zh-4096 | 2 | 0 | 17.7578 / 6.4575 | 36.9209 / 24.8984 | 0.000 / 14.448 | False |
| prose-zh-4096 | 2 | 1 | 17.7392 / 6.4799 | 37.4038 / 25.4501 | 17.015 / 0.000 | False |
| prose-zh-4096 | 3 | 0 | 17.7540 / 6.4709 | 37.0374 / 25.3247 | 0.000 / 13.714 | False |
| prose-zh-4096 | 3 | 1 | 17.7464 / 6.4781 | 36.9452 / 25.2436 | 0.000 / 0.000 | True |
