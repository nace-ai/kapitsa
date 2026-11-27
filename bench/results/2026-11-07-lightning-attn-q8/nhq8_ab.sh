#!/bin/bash
# k372: NI_NH_ATTN_Q8=1 vs default on Nemotron 3.5 Lightning NVFP4 (queue3_ab.sh section c), build origin/main 2fe7c3fa.
# try1 used ni_run, which runs Qwen only (every leg aborted on Nemotron's config). Each leg is now
# `kapitsa bench --quick --json` (512-token answers, sampled and greedy, 3 runs each, 4K reading) on the model folder,
# 8 interleaved legs P Q Q P Q P P Q.
# try4: leg 1 measured fine (load 553 s) but bench exits 1 on an external server's result; rc 1 with only that
# INVALID line now counts as ok.
# try3 gave the server 4 min to load; reading 18 GB from the external drive took 296 s, so it now gets 20 min.
# try2 was an A/A run: `kapitsa bench` drops every NI_ variable from the server it starts. try3 starts the server itself
# (bench's own arguments: --ctx 32768 --slots 1 --disk-cache 0) with the arm's env and points bench at it
# (KAPITSA_BENCH_SERVER=<port>). After leg 2 the P and Q server logs' load lines must differ, or the script stops (exit 7). Untimed-safe guards: AC and load < 4 before each leg (no AC -> exit 75, lock given back);
# a leg is VOID if used swap passes 8 GB or free disk drops under 40 GB while it runs. Model reads from the external
# drive (untimed rule exception noted in the report: weights are loaded before decode is timed).
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_PROFILE=precise
B=$HOME/Documents/nace/kapitsa/.worktrees/k372-nhq8/build/kapitsa
M=$HOME/Documents/nace/models/nemotron-3.5-lightning-nvfp4; IDS=$HOME/Documents/nace/k363/q3/nh-ids.txt
A=${K372_OUT:?}; mkdir -p $A
[ -d "/Volumes/Extreme Pro/models" ] || { echo "PREFLIGHT drive"; exit 5; }
[ -r "$(readlink -f $M)/config.json" ] && [ -x $B ] || { echo "PREFLIGHT model/bin/ids"; exit 5; }
env | grep -qE '^NI_NH_ATTN_Q8=' && { echo "PREFLIGHT env"; exit 3; }
[ -n "$K372_PREFLIGHT_ONLY" ] && { echo PREFLIGHT-OK; exit 0; }
HIDE=~/.agents/skills/session-orchestration/scripts/hide_dsh.sh
log(){ echo "$(date '+%H:%M:%S') $*"; }
load(){ sysctl -n vm.loadavg | awk '{print $2}'; }
swapm(){ sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); print int(v)}'; }
diskg(){ df -g / | awk 'NR==2{print $4}'; }
gate(){ while :; do pmset -g batt | grep -q "AC Power" || { log "no AC: giving the lock back (exit 75)"; exit 75; }
  awk "BEGIN{exit !($(load) < 4)}" && break; log "waiting for load $(load)"; sleep 15; done; }
bash $HIDE on >/dev/null 2>&1; trap 'bash $HIDE off >/dev/null 2>&1' EXIT
i=0
for arm in P Q Q P Q P P Q; do i=$((i+1)); gate; l0=$(load); e=NI_FOO=0; [ $arm = Q ] && e=NI_NH_ATTN_Q8=1
  out=$A/$i-nhq8-$arm.out; slog=$A/$i-nhq8-$arm.server.log; port=$((18700+i))
  env $e $B serve --model $M --host 127.0.0.1 --port $port --ctx 32768 --slots 1 --disk-cache 0 > $slog 2>&1 & sp=$!
  for w in $(seq 1 600); do curl -sf -o /dev/null http://127.0.0.1:$port/health && curl -sf -o /dev/null http://127.0.0.1:$port/v1/models && break; kill -0 $sp 2>/dev/null || break; sleep 2; done
  env KAPITSA_BENCH_SERVER=$port timeout 1800 $B bench --model $M --quick --json --out $A/$i-nhq8-$arm.json > $A/$i-nhq8-$arm.stdout.json 2> $out & lp=$!; g=""
  while kill -0 $lp 2>/dev/null; do sw=$(swapm); dk=$(diskg)
    if [ "$sw" -gt 8192 ] || [ "$dk" -lt 40 ]; then g="void-guard(swap ${sw}M disk ${dk}G)"; kill $lp; fi; sleep 3; done
  wait $lp; rc=$?; kill $sp 2>/dev/null; wait $sp 2>/dev/null
  # bench marks a result from a server it did not start INVALID and exits 1 (and flags the measured server's own GPU
  # load as CONTENTION): that is the expected rc when the JSON is saved and no other INVALID line is present
  if [ $rc = 1 ] && [ -s $A/$i-nhq8-$arm.json ] && [ "$(grep '^INVALID' $out | grep -vc 'KAPITSA_BENCH_SERVER')" = 0 ]; then rc=0; fi; l1=$(load); st=ok; [ $rc = 0 ] || st=rc$rc; [ -n "$g" ] && st=$g
  pmset -g batt | grep -q "AC Power" || st=void-battery; awk "BEGIN{exit !($l1 >= 4.5)}" && st=void-load
  grep -q "memory pressure" $out && st="$st+mempressure"
  log "$i nhq8 arm=$arm rc=$rc load $l0->$l1 $st | $(tail -c 160 $out | tr '\n' ' ')"
  [ $i = 1 ] && [ $rc != 0 ] && { log "leg 1 failed: stopping"; exit 6; }
  if [ $i = 2 ]; then grep -o "Nemotron-H: [0-9]* layers.*weights read in [0-9.]* s ([^)]*)" $A/1-nhq8-P.server.log | sed "s/read in [0-9.]* s//" > $A/load-P.txt; grep -o "Nemotron-H: [0-9]* layers.*weights read in [0-9.]* s ([^)]*)" $A/2-nhq8-Q.server.log | sed "s/read in [0-9.]* s//" > $A/load-Q.txt
    [ -s $A/load-P.txt ] && [ -s $A/load-Q.txt ] || { log "no Nemotron-H load line: stopping"; exit 7; }
    diff -q $A/load-P.txt $A/load-Q.txt >/dev/null && { log "P and Q server load lines identical: q8 not visibly on, stopping"; exit 7; }; fi
done
log NHQ8DONE; exit 0
