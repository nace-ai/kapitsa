#!/bin/bash
# #350 drafter fine-tune, timed evaluation (caller holds /tmp/kapitsa-gpu.lock): the 19-request OpenCode session,
# per request shipped drafter (ship) vs fine-tuned drafter (ft), same binary, history lane on (the default here);
# pass 1 ship then ft, pass 2 ft then ship. Each arm keeps its own KAPITSA_HOME for the whole session (its KFlash
# budget learns across requests as a user's would). Both model folders are symlink copies with the same basename;
# only dflash2/ differs. Every leg waits for AC and 1-min load < 4; a leg that ends on battery is dropped (rerun once).
{
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
W=~/Documents/nace/kapitsa/.worktrees/k350-ft; BIN=$W/build/ni_run; AG=${AG:-~/Documents/nace/kapitsa-online-drafter-notes/f3/agent}
FT=$W/.ftmodels; O=${O:-/tmp/k350d/eval}; PASSES=${PASSES:-2}
mkdir -p $O; rm -rf $O/home-ship $O/home-ft; mkdir -p $O/home-ship $O/home-ft
[ -r $FT/ship/qwen3.6-35b-a3b-4bit/config.json ] && [ -r $FT/ft/qwen3.6-35b-a3b-4bit/dflash2/model.safetensors ] || { echo PREFLIGHT FAIL; exit 9; }
H=~/.agents/skills/session-orchestration/scripts/hide_dsh.sh; trap "bash $H off" EXIT
echo "commit $(git -C $W rev-parse HEAD) dirty: $(git -C $W status --short | tr '\n' ' ')"; shasum -a 256 $BIN
shasum -a 256 $FT/ship/qwen3.6-35b-a3b-4bit/dflash2/model.safetensors $FT/ft/qwen3.6-35b-a3b-4bit/dflash2/model.safetensors
sw_vers -productVersion; pmset -g batt | head -1; sysctl -n vm.loadavg
ac() { pmset -g batt | head -1 | grep -q "AC Power"; }
ld() { sysctl -n vm.loadavg | awk '{print $2}'; }
gate() {
  local t=0
  while ! ac && [ $t -lt 3600 ]; do echo "waiting for AC"; sleep 30; t=$((t+30)); done
  t=0; while [ "$(ld | cut -d. -f1)" -ge 4 ] && [ $t -lt 900 ]; do echo "waiting for load ($(ld))"; sleep 15; t=$((t+15)); done
}
N=($(cat $AG/n_gen.txt))
leg() {  # pass i arm
  local p=$1 i=$2 arm=$3 tries=0
  while [ $tries -lt 2 ]; do
    gate; bash $H on >/dev/null 2>&1
    echo "== p$p r$i $arm n=${N[$i]} $(date +%T) $(pmset -g batt | head -1 | grep -o "'.*'") load $(sysctl -n vm.loadavg)"
    KAPITSA_HOME=$O/home-$arm $BIN $FT/$arm/qwen3.6-35b-a3b-4bit @$AG/r$i.ids ${N[$i]} 0 --dflash auto --ignore-eos > $O/p$p.r$i.$arm.out 2> $O/p$p.r$i.$arm.err
    rc=$?
    if ac; then echo "rc=$rc end AC load $(sysctl -n vm.loadavg)"; grep -E "history lane|dflash decode" $O/p$p.r$i.$arm.err | cut -c1-260; return; fi
    echo "leg ended on battery: dropped"; mv $O/p$p.r$i.$arm.err $O/p$p.r$i.$arm.err.battery; tries=$((tries+1))
  done
}
for p in $(seq 1 $PASSES); do
  for i in $(seq 0 $((${NREQ:-19} - 1))); do
    if [ $((p % 2)) = 1 ]; then leg $p $i ship; leg $p $i ft; else leg $p $i ft; leg $p $i ship; fi
  done
done
echo ALLDONE
}
