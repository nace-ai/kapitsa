# All 4K timing requests in execution order

A = `dbf3f3b`, default. B = `fe74a3d`, `NI_ANE_H4096=1`. Pair and attempt numbers are zero-based; attempt 0 is the initial run, attempt 1 its only allowed retry. Every rejected measurement remains visible. Times are seconds. Eligibility is based on telemetry, not speed.

| Order | Pair / attempt | Arm | Workload | TTFT | Complete HTTP | TTFT eligible | Whole HTTP eligible | Rejection reasons |
| ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | 0 / 0 | A | Python scheduler | 17.732499 | 27.346005 | True | True | — |
| 2 | 0 / 0 | A | Chinese engineering explanation | 17.748612 | 36.944755 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop peak above2% |
| 3 | 0 / 0 | B | Python scheduler | 6.652594 | 16.416848 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop mean above1%, other GPU including desktop peak above2% |
| 4 | 0 / 0 | B | Chinese engineering explanation | 6.512218 | 24.996267 | True | True | — |
| 5 | 0 / 1 | A | Python scheduler | 17.695119 | 27.212607 | True | True | — |
| 6 | 0 / 1 | A | Chinese engineering explanation | 17.741344 | 36.788821 | True | False | HTTP: other GPU including desktop peak above2% |
| 7 | 0 / 1 | B | Python scheduler | 6.394540 | 15.914104 | True | True | — |
| 8 | 0 / 1 | B | Chinese engineering explanation | 6.448297 | 25.017757 | True | False | HTTP: other GPU including desktop peak above2% |
| 9 | 1 / 0 | B | Python scheduler | 6.453369 | 15.956481 | True | True | — |
| 10 | 1 / 0 | B | Chinese engineering explanation | 6.467612 | 25.068426 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop mean above1%, other GPU including desktop peak above2% |
| 11 | 1 / 0 | A | Python scheduler | 17.746120 | 27.085570 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop mean above1%, other GPU including desktop peak above2% |
| 12 | 1 / 0 | A | Chinese engineering explanation | 17.740618 | 36.882661 | True | True | — |
| 13 | 1 / 1 | B | Python scheduler | 6.430670 | 15.942732 | True | True | — |
| 14 | 1 / 1 | B | Chinese engineering explanation | 6.450292 | 25.226007 | True | True | — |
| 15 | 1 / 1 | A | Python scheduler | 17.747937 | 27.166419 | True | True | — |
| 16 | 1 / 1 | A | Chinese engineering explanation | 17.742281 | 36.982240 | True | True | — |
| 17 | 2 / 0 | A | Python scheduler | 17.764470 | 27.444077 | True | True | — |
| 18 | 2 / 0 | A | Chinese engineering explanation | 17.757763 | 36.920945 | True | False | HTTP: other GPU including desktop peak above2% |
| 19 | 2 / 0 | B | Python scheduler | 6.337274 | 15.766426 | True | True | — |
| 20 | 2 / 0 | B | Chinese engineering explanation | 6.457525 | 24.898414 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop peak above2% |
| 21 | 2 / 1 | A | Chinese engineering explanation | 17.739172 | 37.403822 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop peak above2% |
| 22 | 2 / 1 | B | Chinese engineering explanation | 6.479945 | 25.450102 | True | False | HTTP: other GPU including desktop peak above2% |
| 23 | 3 / 0 | B | Python scheduler | 6.451723 | 15.982889 | True | True | — |
| 24 | 3 / 0 | B | Chinese engineering explanation | 6.470946 | 25.324743 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop peak above2% |
| 25 | 3 / 0 | A | Python scheduler | 17.747391 | 27.275572 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop mean above1%, other GPU including desktop peak above2% |
| 26 | 3 / 0 | A | Chinese engineering explanation | 17.753976 | 37.037371 | True | True | — |
| 27 | 3 / 1 | B | Python scheduler | 6.439986 | 16.055344 | True | True | — |
| 28 | 3 / 1 | B | Chinese engineering explanation | 6.478086 | 25.243573 | True | True | — |
| 29 | 3 / 1 | A | Python scheduler | 17.743996 | 27.189004 | True | False | HTTP: other GPU including desktop peak above2% |
| 30 | 3 / 1 | A | Chinese engineering explanation | 17.746382 | 36.945153 | True | True | — |
