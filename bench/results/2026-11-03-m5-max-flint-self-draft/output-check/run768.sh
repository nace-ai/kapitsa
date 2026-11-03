#!/bin/zsh
# The output check at the floor's own size (--rows 96: 768 rows a path): the plain step against the 3-row verify the
# default self-draft cycle runs, and against the 8-row verify the floor's "step" source compared.
cd <repo>
for R in 3 8; do
  echo "== verify rows $R" && date
  NI_LOCK_HELD=1 NI_FLINT_VERIFY_ROWS=$R <run-root>/../bin/logit_dump-p3 ~/models/flint-4bit <run-root>/../oc/dump768-v$R --paths step,verify --rows 96 2>&1 | tail -4
  python3 tools/logit_compare.py <run-root>/../oc/dump768-v$R/step <run-root>/../oc/dump768-v$R/verify --floor tools/logit_floors/flint-4bit.m5.json --json <run-root>/../oc/compare768-v$R.json
done
date
