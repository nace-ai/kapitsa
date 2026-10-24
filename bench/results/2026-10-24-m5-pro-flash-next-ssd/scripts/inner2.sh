#!/bin/bash
# Run under: lockf -k -t 600 /tmp/kapitsa-gpu.lock bash inner2.sh <deadline_s> <cmd...>
# Preflight (>=40% memory free, >=1 GB swap free), the command as a direct child, a wall-clock deadline
# (date-based, counts system sleep) and a trap that stops it: TERM, 10 s grace, then KILL, so its own trap runs.
set -u
LIM=$1; shift
# The checks are re-run inside the lock for up to INNER_SETTLE (90) s, every 5 s, so memory freed by the previous
# holder can settle without releasing the lock (orchestrator 11:0x); the thresholds are unchanged.
vmb() { vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages active/{a=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages wired/{w=$4} /occupied by compressor/{c=$5} END{printf "vm_stat GB: free %.1f active %.1f inactive %.1f speculative %.1f wired %.1f compressor %.1f", f*ps/1e9, a*ps/1e9, i*ps/1e9, s*ps/1e9, w*ps/1e9, c*ps/1e9}'; }
t0=$(date +%s)
while :; do
  free_pct=$(memory_pressure -Q | awk '/free percentage/{gsub("%","",$NF); print $NF}')
  swap_free=$(sysctl -n vm.swapusage | sed -E 's/.*free = ([0-9.]+)M.*/\1/')
  why=""
  [ "${free_pct:-0}" -lt 40 ] && why="memory free ${free_pct}% < 40%"
  [ "$(echo "${swap_free:-0} < ${INNER_SWAPFREE:-1024}" | bc)" = 1 ] && why="${why:+$why; }swap free ${swap_free} MB < ${INNER_SWAPFREE:-1024}"
  [ -z "$why" ] && break
  w=$(( $(date +%s) - t0 ))
  if [ $w -ge ${INNER_SETTLE:-90} ]; then echo "preflight FAILED after ${w}s: $why | $(vmb)"; exit 9; fi
  echo "preflight waiting (${w}s): $why | $(vmb)"; sleep 5
done
echo "preflight: memory free ${free_pct}%, swap free ${swap_free} MB (after $(( $(date +%s) - t0 ))s)"
end=$(( $(date +%s) + LIM ))
"$@" & P=$!
stop() { [ -n "${reaped:-}" ] && return; kill -TERM $P 2>/dev/null; for i in $(seq 1 10); do kill -0 $P 2>/dev/null || return; sleep 1; done; kill -9 $P 2>/dev/null; }
trap 'stop' EXIT
trap 'stop; exit 143' INT TERM
while kill -0 $P 2>/dev/null; do
  if [ "$(date +%s)" -ge "$end" ]; then echo "DEADLINE after ${LIM}s: stopping $P (TERM, then KILL after 10 s)"; stop; break; fi
  sleep 1
done
wait $P; rc=$?; reaped=1; echo "rc=$rc"; exit $rc
