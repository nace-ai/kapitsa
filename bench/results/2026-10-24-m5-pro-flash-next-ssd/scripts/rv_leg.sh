#!/bin/bash
[ -e /tmp/drex-wants-gpu ] && { echo "DREX FLAG: /tmp/drex-wants-gpu present, no model loaded $(date +%T)"; exit 75; }
# rv_leg.sh <engine> <prompt> <leg>: one same-Mac rival leg on Qwen3.8-Flash-Next, run as lockf -k + inner2.sh's direct child.
#   engine: kap (Kapitsa ni_run, Q2_0 pack, auto store) | kap-all (NI_EXPERT_STORE=all) | kap-off (=off: stacked planes resident) | llama-gpu | llama-cpumoe (llama.cpp mainline, ISTA Q2_0 GGUF)
#   prompt: dec (85 tok) | pf4 (4132 tok); both decode RV_NTOK (128) tokens greedily, EOS ignored.
# Gates: nothing else model-sized, available >= RV_AVAIL (28 GB, orchestrator 10:22; rp_leg.sh keeps 35), swap free >= 1 GB, mem-guard ok); aborts on
# swap +500 MB or pressure > 1. Exit: 0 ok, 7 aborted (swap/pressure), 8 gate refused, 9x deadline, else the run's rc.
set -u
ENG=$1; PR=$2; LEG=$3; S=/tmp/kap-ssd-scratch; O=${RV_OUT:-$S/rv}; mkdir -p $O; tag=${ENG}_${PR}_$LEG
M=$HOME/Documents/nace/models/flash-next-q20; Q=/tmp/kap-ssd-handoff/handoff/splash-1.4.0/scratch/fnq/plan
G=${RV_GGUF:-$HOME/Documents/nace/models/ista-q20-gguf/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
NT=${RV_NTOK:-128}; DL=${RV_DEADLINE:-900}
case $PR in dec) IDS=$Q/dec1.ids ;; pf4) IDS=$Q/pf4_seed11.ids ;; pf16) IDS=$Q/pf16_seed33.ids ;; c1k|c8k|c32k) IDS=/tmp/kap-moebw-scratch/ids/$PR.ids ;; *) echo "unknown prompt $PR"; exit 2 ;; esac
cd /tmp/kap-ssd
NEED=${RV_AVAIL:-28}  # orchestrator 10:22: 28 GB (was 35); other gates and aborts unchanged
vmb() { vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages active/{a=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages wired/{w=$4} /occupied by compressor/{c=$5} END{printf "vm_stat GB: free %.1f active %.1f inactive %.1f speculative %.1f wired %.1f compressor %.1f", f*ps/1e9, a*ps/1e9, i*ps/1e9, s*ps/1e9, w*ps/1e9, c*ps/1e9}'; }
# The gates are re-checked inside the lock every 5 s for up to RV_SETTLE (90) s before refusing, so memory freed by the
# previous lock holder can settle without giving the lock away (orchestrator 11:0x); thresholds unchanged.
t0=$(date +%s)
while :; do
  others=$(ps -axo pid,rss,command | awk '$2>500000' | grep -E "kapitsa|logit_dump|ni_run|fn_eval|llama|mlx|compare_engines|ni_handshake|flint_bench|kap-flint" | grep -v "conv.py" | grep -v grep)
  # available = free+inactive+speculative+purgeable; with RV_AVAIL_FILEBACKED=1 (opt-in, off by default; orchestrator 05:5x,
  # for legs that re-read the same mmapped file) max(that, free+purgeable+file-backed [reclaimable page cache,
  # active or not; mmap'd GGUF/pack pages sit in "active" and never age into inactive without pressure]) (#98, 05:5x)
  avail=$(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3} /Pages purgeable/{p=$3} /File-backed pages/{fb=$3} END{o=f+i+s+p; n=(fb_ok?f+p+fb:0); printf "%.1f", (o>n?o:n)*ps/1e9}' fb_ok=${RV_AVAIL_FILEBACKED:-0})
  swapfree=$(sysctl -n vm.swapusage | sed -E 's/.*free = ([0-9.]+)M.*/\1/'); guard=$(tail -1 ~/.agents/mem-guard.log)
  why=""
  [ -n "$others" ] && why="model-sized processes: $(echo $others | cut -c1-200)"
  [ "$(echo "$avail < $NEED" | bc)" = 1 ] && why="${why:+$why; }available ${avail} GB < $NEED"
  [ "$(echo "$swapfree < ${RV_SWAPFREE:-1024}" | bc)" = 1 ] && why="${why:+$why; }swap free ${swapfree} MB < ${RV_SWAPFREE:-1024}"
  echo "$guard" | grep -q " ok " || why="${why:+$why; }mem-guard: $guard"
  w=$(( $(date +%s) - t0 ))
  [ -z "$why" ] && break
  if [ -n "${RV_DRY:-}" ]; then echo "DRY (would refuse): $why"; break; fi
  if [ $w -ge ${RV_SETTLE:-90} ]; then echo "GATE FAIL after ${w}s: $why | $(vmb)"; exit 8; fi
  echo "gate $tag waiting (${w}s): $why | $(vmb)"; sleep 5
done
echo "gate $tag: available ${avail} GB (need $NEED), swap free ${swapfree} MB, guard: $guard (after ${w}s)"
case $ENG in
  kap)          C="env -u NI_EXPERT_STORE -u NI_GPU_HEAT -u NI_EXPERT_PREFETCH ${RV_KAP_STORE:+NI_EXPERT_STORE=$RV_KAP_STORE} ${RV_BIN:-./build/ni_run} $M @$IDS $NT 0" ;;   # RV_KAP_STORE: #98 pin (GB) if the orchestrator picks it
  kap-all)      C="env -u NI_GPU_HEAT -u NI_EXPERT_PREFETCH NI_EXPERT_STORE=all ${RV_BIN:-./build/ni_run} $M @$IDS $NT 0" ;;
  kap-off)      C="env -u NI_GPU_HEAT -u NI_EXPERT_PREFETCH NI_EXPERT_STORE=off ${RV_BIN:-./build/ni_run} $M @$IDS $NT 0" ;;
  kap-base)     C="env -u NI_EXPERT_STORE -u NI_GPU_HEAT -u NI_EXPERT_PREFETCH -u NI_EXPERT_CAP $S/bin/ni_run.base $M @$IDS $NT 0" ;;
  kap-fix)      C="env -u NI_EXPERT_STORE -u NI_GPU_HEAT -u NI_EXPERT_PREFETCH -u NI_EXPERT_CAP ./build/ni_run $M @$IDS $NT 0" ;;
  kx-*)         V=$S/var/${ENG#kx-}; B=./build/ni_run; [ -x $V.bin ] && B=$V.bin
                C="env -u NI_EXPERT_STORE -u NI_GPU_HEAT -u NI_EXPERT_PREFETCH -u NI_EXPERT_CAP $(cat $V.env 2>/dev/null) $S/en/energy $B $M @$IDS $NT 0 ${RV_LOGITS:+--dump-all-logits $O/$tag.bin}" ;;
  kap-prof)     C="env -u NI_GPU_HEAT -u NI_EXPERT_PREFETCH -u NI_EXPERT_STORE NI_PROFILE=1 ${RV_BIN:-./build/ni_run} $M @$IDS $NT 0" ;;
  kap-[0-9]*)   C="env -u NI_GPU_HEAT -u NI_EXPERT_PREFETCH NI_EXPERT_STORE=${ENG#kap-} ${RV_CBLAT:+NI_CB_LAT=$O/$tag.cb} ${RV_BIN:-./build/ni_run} $M @$IDS $NT 0" ;;
  llama-gpu)    C="bash $S/rv_llama.sh $G $IDS $NT 18431 gpu" ;;
  llama-cpumoe) C="bash $S/rv_llama.sh $G $IDS $NT 18431 cpumoe" ;;
  *) echo "unknown engine $ENG"; exit 2 ;;
esac
[ -e /tmp/drex-wants-gpu ] && { echo "DREX FLAG: /tmp/drex-wants-gpu present, no model loaded $(date +%T)"; exit 75; }
echo "run $tag: $C"
swap0=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
/usr/bin/time -l $C > $O/$tag.log 2>&1 & T=$!
# the engine is time's forked child: found by parent PID (never by name); PID + lstart in a pidfile
R=""; for i in $(seq 1 50); do R=$(ps -axo pid=,ppid= | awk -v p=$T '$2==p{print $1; exit}'); [ -n "$R" ] && break; kill -0 $T 2>/dev/null || break; sleep 0.1; done
lst() { ps -o lstart= -p $1 2>/dev/null | xargs; }
GC=""; if [ -n "$R" ]; then for i in $(seq 1 30); do GC=$(ps -axo pid=,ppid= | awk -v p=$R '$2==p{print $1; exit}'); [ -n "$GC" ] && break; kill -0 $R 2>/dev/null || break; sleep 0.1; done; fi
: > $O/$tag.pid; for x in $GC $R $T; do echo "$x $(lst $x)" >> $O/$tag.pid; done
ours() { local x=${1:-}; [ -n "$x" ] && [ -n "$(lst $x)" ] && [ "$(lst $x)" = "$(awk -v p=$x '$1==p{$1=""; sub(/^ /,""); print}' $O/$tag.pid)" ]; }
stop_leg() {  # TERM the engine (rv_llama.sh's trap stops llama-server) and time, 10 s grace, then KILL
  for x in $GC $R $T; do ours $x && kill -TERM $x; done
  for i in $(seq 1 20); do ours "$GC" || ours "$R" || ours "$T" || return 0; sleep 0.5; done
  for x in $GC $R $T; do ours $x && kill -9 $x; done; sleep 0.5
}
trap 'stop_leg' EXIT
trap 'stop_leg; exit 143' INT TERM
tl=$(date +%s); end=$(( $(date +%s) + DL )); maxswap=$swap0; maxlvl=1; why=""
while kill -0 $T 2>/dev/null; do
  s=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'); l=$(sysctl -n kern.memorystatus_vm_pressure_level)
  [ "$(echo "$s > $maxswap" | bc)" = 1 ] && maxswap=$s; [ "$l" -gt "$maxlvl" ] && maxlvl=$l
  if [ "$(echo "$maxswap - $swap0 > 500" | bc)" = 1 ] || [ "$maxlvl" -gt ${RV_PLVL:-1} ]; then why=abort; echo "ABORT: swap +$(echo "$maxswap - $swap0" | bc) MB, pressure $maxlvl"; stop_leg; break; fi
  mg=$(tail -1 ~/.agents/mem-guard.log 2>/dev/null); mgt=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(echo "$mg" | cut -c1-19)" +%s 2>/dev/null || echo 0)
  if [ -n "${RV_MG_WARN_OK:-}" ] && [ "$mgt" -ge "$tl" ] && echo "$mg" | grep -q "WARN (stress building) .* pressure=1 " && [ "$(echo "$maxswap - $swap0 > 500" | bc)" = 0 ]; then
    [ "$mg" != "${mgnoted:-}" ] && { echo "NOTE mem-guard page-cache warn, not aborted (pressure 1, swap delta $(echo "$maxswap - $swap0" | bc) MB): $(echo "$mg" | grep -o "level=[^ ]* pressure=[^ ]* swap_used=[^ ]* .*compressor=[^ ]*")"; mgnoted=$mg; }
  elif [ "$mgt" -ge "$tl" ] && ! echo "$mg" | grep -q " ok "; then why=abort; echo "ABORT: mem-guard: $mg"; stop_leg; break; fi
  [ "$(date +%s)" -ge "$end" ] && { why=deadline; echo "DEADLINE ${DL} s"; stop_leg; break; }
  sleep 0.5
done
wait $T 2>/dev/null; rc=$?
sleep 1
# llama-server (a grandchild) must be gone too: its PID is logged by rv_llama.sh
L=$(awk '/^llama-server pid/{print $3; exit}' $O/$tag.log 2>/dev/null)
if [ -n "$L" ] && kill -0 $L 2>/dev/null && [ "$(lst $L)" = "$(awk '/^llama-server pid/{for(i=5;i<=9;i++) printf "%s%s",$i,(i<9?" ":""); exit}' $O/$tag.log)" ]; then
  kill -TERM $L; sleep 5; kill -0 $L 2>/dev/null && kill -9 $L; echo "stopped leftover llama-server $L"
fi
{ ours $GC || ours $R; } && echo "WARNING: engine ${GC:-$R} still alive" || echo "process gone (engine ${R:-?})"
echo "RESULT $tag rc=$rc swap_delta_mb=$(echo "$maxswap - $swap0" | bc) max_pressure=$maxlvl stop=${why:-none}"
grep -E "ENERGY|mean GPU|expert store|prompt [0-9]+ tok|decode [0-9]+ tok|^LOAD|^TIMINGS|maximum resident|peak memory footprint" $O/$tag.log | cut -c1-400
[ "$why" = abort ] && [ "$ENG" = kap-all ] && { echo "kap-all aborted: not re-run (orchestrator 10:5x: do not fight the guard)"; exit 17; }
[ "$why" = abort ] && exit 7
[ "$why" = deadline ] && exit 90
exit $rc
