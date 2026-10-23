#!/bin/bash
# M3 Pro step 2, Phase A on main (orchestrator, 23:15): build origin/main once the 0.2.8 27B run ends, then Kapitsa
# on the remaining models, Flint, and the 4B and 27B again. Rivals wait for the fresh/reused list.
. ~/kenv.sh; . ~/kq/m3/cfg.sh
L=~/kq/m3/q.log
while pgrep -f compare_engines.py >/dev/null; do sleep 15; done
echo "$(date +%T) 0.2.8 runs done; building origin/main in ~/kq/head" >> $L
H=~/kq/head
( cd $H && git fetch -q origin && git checkout -q --detach origin/main && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > build-cmake.log 2>&1 \
  && nice cmake --build build -j 10 --target kapitsa logit_dump > build.log 2>&1 ) && OK=1 || OK=0
if [ $OK = 1 ]; then
  echo "$(date +%T) main build ok: $($H/build/kapitsa --version 2>&1 | head -1) $(git -C $H rev-parse HEAD)" >> $L
  sed -e "s#^KAPITSA_BIN=.*#KAPITSA_BIN=$H/build/kapitsa#" ~/kq/rel/bench/harness/local.env > $H/bench/harness/local.env
  cd $H/bench/harness
else
  echo "$(date +%T) MAIN BUILD FAILED (tail of build.log below); staying on 0.2.8" >> $L; tail -20 $H/build.log >> $L
  cd ~/kq/rel/bench/harness
fi
grep KAPITSA_BIN local.env >> $L
step() {
  pmset displaysleepnow; sleep 5
  while :; do
    fr=$(memory_pressure | tail -1 | grep -o "[0-9]*" | tail -1); dk=$(df -g ~ | tail -1 | awk '{print $4}')
    [ "$fr" -ge 40 ] && [ "$dk" -ge 30 ] && break
    echo "$(date +%T) wait: free ${fr}% disk ${dk}G" >> $L; sleep 60
  done
  vt=$(pgrep -x VTDecoderXPCService >/dev/null && ps -o %cpu= -p "$(pgrep -x VTDecoderXPCService | head -1)" || echo none)
  echo "$(date +%T) power $(pmset -g batt | head -1 | sed 's/Now drawing from //') free ${fr}% disk ${dk}G vtdecoder $vt :: $1 ($PWD)" >> $L
  python3 compare_engines.py $C --skip agent --only "$1" >> $L 2>&1; echo "$(date +%T) rc=$? :: $1" >> $L
  pgrep -fl "llama-server|mlx_lm.server|mlx_vlm|bin/mtplx|ollama serve|kapitsa serve" >> $L && echo "$(date +%T) WARNING leftover process" >> $L
}
sleep 60
for m in q36-35b q35-9b dory nano-4b nano-9b lightning bolt flint q35-4b q38-27b; do step sw-$m-kapitsa; done
echo "$(date +%T) PHASE-A-DONE" >> $L
