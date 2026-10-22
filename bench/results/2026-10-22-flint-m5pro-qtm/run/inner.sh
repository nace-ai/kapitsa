#!/bin/bash
# One leg, run under `lockf -k /tmp/kapitsa-gpu.lock`. Kapitsa only (configs/flint.json --only flint-kapitsa).
# Exit 75 = preflight refused (memory < 40% free or port 8401 busy): the driver retries.
set -u
LEG=$1; R=<scratch>; ME=$$
port_pids() { lsof -nP -t -iTCP:8401 -sTCP:LISTEN 2>/dev/null; }
descendants() {  # every process below $1, walking the tree with pgrep -P
  local c; for c in $(pgrep -P "$1"); do echo "$c"; descendants "$c"; done
}
ours() {  # port-8401 listeners that descend from this leg
  local d l; d=" $(descendants "$ME" | tr '\n' ' ') "
  for l in $(port_pids); do case "$d" in *" $l "*) echo "$l";; esac; done
}
cleanup() {  # kill, wait, verify -- only our own descendants; a foreign listener is logged and left alone
  local p mine; mine=$(ours)
  for p in $mine; do kill -TERM "$p" 2>/dev/null; done
  for i in $(seq 1 30); do [ -z "$(ours)" ] && break; sleep 1; done
  for p in $(ours); do kill -KILL "$p" 2>/dev/null; done
  sleep 1
  [ -n "$(ours)" ] && echo "$(date +%T) leg $LEG cleanup: OUR LISTENER ON 8401 SURVIVED ($(ours))"
  for p in $(port_pids); do echo "$(date +%T) leg $LEG cleanup: foreign listener on 8401 left alone: $(ps -o pid=,ppid=,command= -p $p | cut -c1-160)"; done
  [ -z "$(port_pids)" ] && echo "$(date +%T) leg $LEG cleanup: port 8401 free"
}
trap cleanup EXIT; trap 'exit 130' INT TERM
free=$(memory_pressure | awk '/free percentage/{gsub("%","",$5); print $5}')
if [ "${free:-0}" -lt 40 ]; then echo "$(date +%T) leg $LEG preflight: memory free $free% < 40%, retry"; trap - EXIT; exit 75; fi
if [ -n "$(port_pids)" ]; then echo "$(date +%T) leg $LEG preflight: port 8401 busy (foreign, left alone: $(port_pids)), retry"; trap - EXIT; exit 75; fi
echo "$(date +%T) leg $LEG start; memory free $free%; main $(git -C <repo> rev-parse --short HEAD)"
ps -axo pid,pcpu,rss,command | sort -k2 -nr | head -10 | cut -c1-200 > $R/ps-leg$LEG.txt
cd <repo>/bench/harness
NACE_GPU_LOCK_HELD=1 BENCH_MODELS=~/Documents/nace/models KAPITSA_BIN=<repo>/build/kapitsa \
  python3 compare_engines.py configs/flint.json --only flint-kapitsa --skip agent --out-dir $R/out
rc=$?; echo "$(date +%T) leg $LEG exit $rc"; exit $rc
