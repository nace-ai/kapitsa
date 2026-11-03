#!/bin/zsh
export AB_DOCS='{"133K": 548000, "186K": 830000, "252K": 1040000}'
caffeinate -is lockf -k -t 14400 /tmp/kapitsa-gpu.lock python3 <run-root>/../ab_self.py <run-root>/../bin/kapitsa-p2 ~/models/flint-4bit ~/kapitsa-bench-runs/2026-11-02-m5-max-long/bench_corpus_long.txt <run-root>/../run2 '[["plain",{}],["s2",{"NI_FLINT_SELF":"2","NI_FLINT_SELF_MIN":"0"}],["s3",{"NI_FLINT_SELF":"3","NI_FLINT_SELF_MIN":"0"}],["s2ng",{"NI_FLINT_SELF":"2","NI_FLINT_SELF_MIN":"0","NI_FLINT_NGRAM":"7"}]]'
