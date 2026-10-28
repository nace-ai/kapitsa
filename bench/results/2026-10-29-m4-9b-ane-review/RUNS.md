# Every 4K request in execution order

A = `6d5c2aa` default; B = `82e9b9e`, `NI_ANE_H4096=1`. Pair/attempt indices start at zero. Attempt 1 is the only permitted retry. Eligibility is based on telemetry, not speed. Times are seconds.

| Order | Pair / attempt | Arm | Workload | TTFT | Complete HTTP | TTFT eligible | HTTP eligible | Rejection reasons |
| ---: | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | 0 / 0 | A | Python scheduler | 17.647582 | 27.136715 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop mean above1%, other GPU including desktop peak above2% |
| 2 | 0 / 0 | A | Chinese engineering explanation | 17.759655 | 37.475274 | True | True | — |
| 3 | 0 / 0 | B | Python scheduler | 6.498963 | 16.510349 | True | True | — |
| 4 | 0 / 0 | B | Chinese engineering explanation | 6.588674 | 25.725957 | True | False | HTTP: other GPU including desktop peak above2% |
| 5 | 0 / 1 | A | Python scheduler | 17.733990 | 27.084398 | True | True | — |
| 6 | 0 / 1 | B | Python scheduler | 6.437022 | 15.735292 | True | True | — |
| 7 | 1 / 0 | B | Python scheduler | 6.455661 | 15.873347 | True | True | — |
| 8 | 1 / 0 | B | Chinese engineering explanation | 6.438249 | 24.959550 | True | True | — |
| 9 | 1 / 0 | A | Python scheduler | 17.753747 | 27.802470 | True | True | — |
| 10 | 1 / 0 | A | Chinese engineering explanation | 18.315806 | 37.695187 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop mean above1%, other GPU including desktop peak above2% |
| 11 | 1 / 1 | B | Chinese engineering explanation | 6.390043 | 25.339595 | True | False | HTTP: other GPU including desktop peak above2% |
| 12 | 1 / 1 | A | Chinese engineering explanation | 17.589734 | 36.695331 | True | True | — |
| 13 | 2 / 0 | A | Python scheduler | 17.538488 | 27.123821 | False | False | TTFT: other GPU including desktop mean above1%, other GPU including desktop peak above2%; HTTP: other GPU including desktop mean above1%, other GPU including desktop peak above2% |
| 14 | 2 / 0 | A | Chinese engineering explanation | 17.732149 | 36.936678 | True | True | — |
| 15 | 2 / 0 | B | Python scheduler | 6.377883 | 15.783966 | True | True | — |
| 16 | 2 / 0 | B | Chinese engineering explanation | 6.443784 | 24.910916 | True | True | — |
| 17 | 2 / 1 | A | Python scheduler | 17.534944 | 26.893418 | True | True | — |
| 18 | 2 / 1 | B | Python scheduler | 6.350166 | 15.839511 | True | True | — |
| 19 | 3 / 0 | B | Python scheduler | 6.352376 | 15.759948 | True | True | — |
| 20 | 3 / 0 | B | Chinese engineering explanation | 6.462439 | 24.740687 | True | True | — |
| 21 | 3 / 0 | A | Python scheduler | 17.748409 | 27.114425 | True | True | — |
| 22 | 3 / 0 | A | Chinese engineering explanation | 17.742791 | 36.942679 | True | False | HTTP: other GPU including desktop peak above2% |
