#!/bin/zsh
cd <scratch>/mdab
AB_HARNESS=<repo>/bench/harness python3 ab_dx.py <repo>/build/ni_run ~/models/qwen3.8-flash-next-4bit prompt-ids.txt run1 3
