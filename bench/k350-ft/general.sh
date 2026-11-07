#!/bin/bash
# #350 follow-up: the same interleaved A/B (shipped vs fine-tuned drafter) on 36 non-coding prompts, 12 each in English,
# Russian and Chinese (f3/rec lang_* sets, 256 tokens each), to see whether the coding fine-tune is slower on other text.
S=~/Documents/nace/kapitsa/.worktrees/k350-ft/bench/k350-ft; D=/tmp/k350d; DT=~/.agents/skills/session-orchestration/scripts/detach.py
L="lockf -k /tmp/kapitsa-gpu.lock"; W=~/Documents/nace/kapitsa/.worktrees/k350-ft
wait_ac() { until pmset -g batt | grep -q "AC Power"; do echo "waiting for AC $(date +%T) $(pmset -g batt | grep -o "[0-9]*%")"; sleep 60; done; }  # outside the lock
wait_373() { while python3 $DT list 2>/dev/null | grep -E "^RUNNING +[^ ]*373" >/dev/null; do echo "waiting for #373 $(date +%T): $(python3 $DT list | awk '/^RUNNING +[^ ]*373/{printf "%s ", $2}')"; sleep 120; done; }
mem_ok() { lv=$(sysctl -n kern.memorystatus_vm_pressure_level)
  fr=$(vm_stat | awk '/page size of/{ps=$8} /Pages (free|inactive|speculative|purgeable):/{gsub("\\.","",$NF); n+=$NF} END{printf "%.1f", n*ps/2^30}')
  big=$(ps -axo rss=,pid=,comm= | awk '$1>20*2^20{printf "%s(%.0fGB) ", $3, $1/2^20}')
  [ "$lv" = 1 ] && awk -v f="$fr" 'BEGIN{exit !(f>=12)}' && [ -z "$big" ]; }
wait_mem() { until mem_ok; do echo "waiting for memory $(date +%T) pressure level $lv, free ${fr} GB, big: ${big:-none}"; sleep 60; done; }
wait_ac; wait_373; wait_mem
AG=$W/.ftmodels/general NREQ=36 O=$W/.ftmodels/eval-general $L bash $S/eval_inner.sh; rc=$?
NREQ=36 VERDICT_NAME=VERDICT-general.json python3 $S/verdict.py $W/.ftmodels/eval-general $S/general.extra.json 2>&1 | tail -3
echo "GENERAL DONE rc=$rc $(date +%T)"; exit $rc
