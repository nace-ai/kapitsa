#!/bin/bash
# the default 8-row tree verify: main's logit_dump against the branch's (the same model code)
set -e
SP=$1; D=$2; M=$HOME/models/qwen3.8-27b-4bit-mtp
export NI_LOCK_HELD=1
$SP/kapitsa-err/build/logit_dump $M $D/main8 --paths verify 2>&1 | tail -2
$SP/kapitsa-lw/build/logit_dump $M $D/branch8 --paths verify 2>&1 | tail -2
