#!/bin/zsh
while ! grep -q '^exit' <tasks>/bayxbejj4.output 2>/dev/null; do sleep 20; done
KAPITSA_BIN=<repo>/build/kapitsa <run-root>/run_ab7.sh q27-v8t-old q27-v8t-new q27-v8t-old q27-v8t-new > <run-root>/ab7.log 2>&1
