#!/bin/zsh
# Flint self-drafts, the PR's evidence: the output check (logit_dump, the plain step against the 3-row verify the default
# cycle runs) and the interleaved A/B of the final build (NI_FLINT_SELF=0 against the default).
cd <repo>
echo "== output check" && date
NI_LOCK_HELD=1 NI_FLINT_VERIFY_ROWS=3 <run-root>/../bin/logit_dump-p3 ~/models/flint-4bit <run-root>/../oc/dump --paths step,verify 2>&1 | tail -5
python3 tools/logit_compare.py <run-root>/../oc/dump/step <run-root>/../oc/dump/verify --floor tools/logit_floors/flint-4bit.m5.json --json <run-root>/../oc/compare.json
echo "== A/B" && date
AB_DOCS='{"186K": 830000, "252K": 1040000}' python3 <run-root>/../ab_self.py <run-root>/../bin/kapitsa-p3 ~/models/flint-4bit ~/kapitsa-bench-runs/2026-11-02-m5-max-long/bench_corpus_long.txt <run-root> '[["off",{"NI_FLINT_SELF":"0"}],["default",{}],["off",{"NI_FLINT_SELF":"0"}],["default",{}]]'
date
