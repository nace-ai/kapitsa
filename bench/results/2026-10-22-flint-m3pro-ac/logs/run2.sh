#!/bin/bash
# Protocol rerun with the display asleep (the AC display never sleeps here, so the video wallpaper decoded during run.sh protocol).
. ~/kenv.sh; cd ~/kq/main/bench/harness
until grep -q ALL-DONE ~/kq/flint/run.log; do sleep 30; done
L=~/kq/flint/run2.log
step() {
  pmset displaysleepnow; sleep 5
  echo "$(date +%T) power $(pmset -g batt | head -1 | sed "s/Now drawing from //") $(pmset -g batt | grep -o "[0-9]*%") free $(memory_pressure | tail -1 | grep -o "[0-9]*%") vtdecoder $(pgrep -x VTDecoderXPCService >/dev/null && ps -o %cpu= -p $(pgrep -x VTDecoderXPCService) || echo none) :: $*" >> $L
  python3 compare_engines.py configs/flint.json --skip agent "$@" >> $L 2>&1; echo "$(date +%T) rc=$? :: $*" >> $L
}
step --only flint-kapitsa
step --only flint-llamacpp
step --only flint-kapitsa
step --only flint-llamacpp
echo "$(date +%T) PROTOCOL2-DONE" >> $L
