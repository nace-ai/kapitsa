#!/bin/zsh
while ! grep -q '=== done' <run-root>/ab5.log 2>/dev/null; do sleep 20; done
KAPITSA_BIN=<repo>/build/kapitsa <run-root>/run_ab6.sh q35-v8t-old q35-v8t-new q35-v8t-old q35-v8t-new clover-v8t-old clover-v8t-new clover-v8t-old clover-v8t-new > <run-root>/ab6.log 2>&1
