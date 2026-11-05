#!/bin/bash
# #350 online selector, real 35B on M5: per request, control (lr 0) then learner (lr 0.003, state carried) = A B per request
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 KAPITSA_HOME=/tmp/k350os/home; mkdir -p $KAPITSA_HOME
W=~/Documents/nace/kapitsa/.worktrees/k350-weaver-online; BIN=$W/build/ni_run; AG=~/Documents/nace/kapitsa-online-drafter-notes/f3/agent
M=~/Documents/nace/models/qwen3.6-35b-a3b-4bit; O=/tmp/k350os/out; LR=${LR:-0.003}
cd /tmp/k350os; rm -f $O/state.bin
H=~/.agents/skills/session-orchestration/scripts/hide_dsh.sh; trap "bash $H off" EXIT; bash $H on || echo "hide_dsh refused (user active): legs may see foreign GPU"
echo "commit $(git -C $W rev-parse HEAD) dirty: $(git -C $W status --short | tr '\n' ' ')"; shasum -a 256 $BIN
sw_vers -productVersion; pmset -g batt | head -1; sysctl -n vm.loadavg
N=($(cat $AG/n_gen.txt)); fail=0
for i in $(seq 0 18); do
  for arm in ctl learn; do
    if [ $arm = ctl ]; then lr=0; st=$O/ctl-state.bin; else lr=$LR; st=$O/state.bin; fi
    echo "== r$i $arm n=${N[$i]} $(date +%T) $(pmset -g batt | head -1 | grep -o "'.*'") load $(sysctl -n vm.loadavg)"
    NI_SEL_ONLINE=$lr NI_SEL_STATE=$st $BIN $M @$AG/r$i.ids ${N[$i]} 0 --dflash auto --ignore-eos > $O/r$i.$arm.out 2> $O/r$i.$arm.err
    rc=$?; echo "rc=$rc"; grep -E "online selector|dflash decode" $O/r$i.$arm.err
    [ $rc -ne 0 ] && fail=$((fail+1)); [ $fail -ge 3 ] && { echo STOP; exit 1; }
  done
done
echo ALLDONE
