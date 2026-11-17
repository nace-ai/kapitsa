#!/bin/zsh
# the head's 16K follow-up on the M4 (head-b4e33647-n4b-m4/ttft-16000.log), under the GPU lock taken by the caller
# (lockf ... env NI_LOCK_HELD=1 ...); tests/test_disk_cache-ssd.py is this branch's tests/test_disk_cache.py
cd ~/kapitsa-ssdcache/v3
export KAPITSA_HOME=~/kapitsa-run/home KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1
python3 tests/test_disk_cache-ssd.py ~/kapitsa-build/b4e33647/build/kapitsa ~/kapitsa-run/home/models/nemotron-3-nano-4b-fp8 \
  --ttft --tokens 16000 --rounds 3 --server-args "--host 127.0.0.1 --private" > out/SSD2-ttft-n4b-m4/ttft-16000.log 2>&1
