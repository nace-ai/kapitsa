#!/bin/bash
# rv98b.sh: item #98, the full SSD-offload speed table (user priority 24 Oct 04:10): Qwen3.8-Flash-Next on this
# M5 Pro 48 GB, every model file on the INTERNAL disk. Kapitsa (Q2_0 pack, expert store streaming, auto store, heater
# default) vs llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 -fa on; and --cpu-moe at short/4K).
#  B. cold start: before the leg, the other engine's files are read through the page cache (sudo purge is not
#     available), then launch -> first token; order kap llama llama kap; prompt dec (85 tok).
#  C. warm legs, ABBA per length: dec (85), c1k (1024), pf4 (4132), pf16 (16287), c32k (32768) prompts, 128 greedy
#     tokens: prefill tok/s at the length, decode tok/s at that context.
#  D. llama.cpp --cpu-moe: dec and pf4, 2 legs each.
# Each leg its own lockf -k turn (inner2.sh deadline 600 s), 30 s apart for other lock waiters; rv_leg.sh's gates and
# aborts; exits 75 without loading a model while /tmp/drex-wants-gpu exists (waits here, lock released).
# Launch: detach.py run --name kap-rv98b --cwd /tmp/kap-ssd -- bash /tmp/kap-ssd-scratch/rv98b.sh
set -u
trap 'rc=$?; echo "rv98b exit rc=$rc $(date +%T)"; if [ -e /tmp/kap-ssd-scratch/rv98b.relaunch ]; then rm -f /tmp/kap-ssd-scratch/rv98b.relaunch; echo "rv98b stopped for a relaunch $(date +%T) (NOT done; the #98 freeze still holds)"; else echo "rv98b done"; fi' EXIT   # always the last line: lifts the #98 GPU freeze
[ -d "/Volumes/Extreme Pro/models" ] || { echo "drive not mounted"; exit 1; }
S=/tmp/kap-ssd-scratch; EXT="/Volumes/Extreme Pro/downloads/ista-q20-gguf/Q2_0"
INT=$HOME/Documents/nace/models/ista-q20-gguf/Q2_0; F1=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
F2=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf; PACK=$HOME/Documents/nace/models/flash-next-q20
export RV_OUT=$S/rv98b RV_GGUF=$INT/$F1 RV_BIN=/tmp/kap-fn98/build/ni_run RV_LLAMA_CTX=33280
O=$RV_OUT; mkdir -p $O   # resumable: legs with rc=0 are skipped
echo "rv98b started $(date +%T) RV_KAP_STORE=${RV_KAP_STORE:-auto}; kapitsa $(git -C /tmp/kap-fn98 log --oneline -1 | cut -c1-60); llama.cpp $(git -C /tmp/llama-main log --oneline -1 | cut -c1-60)"

# A. copy to the internal disk once d8ab6c24 has made room ($S/rv98.disk_ok) and >= 100 GiB stays free after it
until [ -e $S/rv98.disk_ok ]; do sleep 60; done; echo "disk worker: $(cat $S/rv98.disk_ok) ($(date +%T))"
need=$(( ( $(stat -f %z "$EXT/$F1") + $(stat -f %z "$EXT/$F2") ) / 1073741824 + 1 ))
if [ "$(stat -f %z "$INT/$F2" 2>/dev/null)" != "$(stat -f %z "$EXT/$F2")" ] || [ "$(stat -f %z "$INT/$F1" 2>/dev/null)" != "$(stat -f %z "$EXT/$F1")" ]; then
  while :; do
    free=$(df -g / | awk 'NR==2{print $4}'); have=$(( ( $(stat -f %z "$INT/$F1" 2>/dev/null || echo 0) + $(stat -f %z "$INT/$F2" 2>/dev/null || echo 0) ) / 1073741824 ))
    [ $(( free - need + have )) -ge 100 ] && break
    echo "copy waiting: ${free} GiB free, the copy needs ${need} GiB, floor 100 ($(date +%T))"; sleep 120
  done
  mkdir -p "$INT"
  nice -n 19 taskpolicy -b rsync -a "$EXT/$F1" "$EXT/$F2" "$INT/" || { echo "copy failed"; exit 4; }
fi
echo "copied $(date +%T): $(ls -l $INT | tail -2 | awk '{print $5, $9}' | xargs); df $(df -g / | awk 'NR==2{print $4}') GiB free"

evict() {  # evict <engine>: read the OTHER engine's files through the page cache (untimed, inside the lock)
  case $1 in kap*) cat "$INT/$F1" "$INT/$F2" > /dev/null ;; *) cat $PACK/*.safetensors > /dev/null ;; esac
}
leg() {  # leg <engine> <prompt> <tag-leg> [cold]
  local e=$1 p=$2 l=$3 cold=${4:-} t=${1}_${2}_$3 rc reruns=0
  if grep -q "^RESULT $t rc=0 .*stop=none" $O/run_$t.txt 2>/dev/null; then echo "== $t done earlier"; return; fi
  for try in $(seq 1 200); do
    while [ -e /tmp/drex-wants-gpu ]; do sleep 30; done
    if [ -n "$cold" ]; then
      RV_EVICT=$e INNER_SWAPFREE=${RV_SWAPFREE:-1024} lockf -k -t 3600 /tmp/kapitsa-gpu.lock bash -c "
        [ -e /tmp/drex-wants-gpu ] && exit 75
        t0=\$(date +%s); $(declare -f evict); INT='$INT'; F1='$F1'; F2='$F2'; PACK='$PACK'; evict $e
        echo \"evicted (read the other engine's files) in \$(( \$(date +%s) - t0 )) s\"
        exec bash $S/inner2.sh 600 bash $S/rv_leg.sh $e $p $l" > $O/run_$t.txt 2>&1; rc=$?
    else
      INNER_SWAPFREE=${RV_SWAPFREE:-1024} lockf -k -t 3600 /tmp/kapitsa-gpu.lock bash $S/inner2.sh 600 bash $S/rv_leg.sh $e $p $l > $O/run_$t.txt 2>&1; rc=$?
    fi
    while read -r pid ls; do
      [ -n "$pid" ] && [ "$(ps -o lstart= -p $pid 2>/dev/null | xargs)" = "$ls" ] || continue
      echo "leftover $pid from $t: TERM"; kill -TERM $pid; for i in $(seq 1 20); do kill -0 $pid 2>/dev/null || break; sleep 0.5; done
      kill -0 $pid 2>/dev/null && { kill -9 $pid; sleep 1; }
    done < <(cat $O/$t.pid 2>/dev/null)
    if [ $rc -eq 75 ]; then echo "$t: drex flag, waiting ($(date +%T))"; sleep 30; continue; fi
    if [ $rc -eq 7 ] && [ $reruns -lt 2 ]; then reruns=$((reruns + 1)); cp $O/run_$t.txt $O/run_$t.abort$reruns.txt; echo "$t: ABORTED, re-run $reruns"; sleep 60; continue; fi
    [ $rc -ne 9 ] && [ $rc -ne 8 ] && break
    echo "$t: refused (rc $rc, try $try): $(grep -E 'FAIL' $O/run_$t.txt | tail -1 | cut -c1-150)"; sleep 60
  done
  echo "== $t rc=$rc $(date +%T) $(grep -hoE 'prompt [0-9]+ tok in [0-9]+ ms \([0-9.]+ tok/s\); decode [0-9]+ tok in [0-9]+ ms \([0-9.]+ tok/s\)|^TIMINGS .{0,160}' $O/${t}.log 2>/dev/null | head -1)"
  sleep 30
}

# B. cold start
leg kap dec c1 cold; leg llama-gpu dec c1 cold; leg llama-gpu dec c2 cold; leg kap dec c2 cold
# C. warm ABBA per length (the engine that ran last goes first, so each engine has one leg after the other)
leg kap dec 1; leg llama-gpu dec 1; leg llama-gpu dec 2; leg kap dec 2
for p in c1k pf4 pf16 c32k; do
  leg llama-gpu $p 1; leg kap $p 1; leg kap $p 2; leg llama-gpu $p 2
done
# D. llama.cpp with the routed experts on the CPU (mmap)
leg llama-cpumoe dec 1; leg llama-cpumoe pf4 1; leg llama-cpumoe dec 2; leg llama-cpumoe pf4 2
python3 $S/rv98_table.py $O | tee $O/table.md

# E. the internal copy goes (the drive keeps the original): one shard's sha256 checked against the drive's first
a=$(shasum -a 256 "$INT/$F2" | cut -d' ' -f1); b=$(shasum -a 256 "$EXT/$F2" | cut -d' ' -f1)
if [ "$a" = "$b" ] && [ "$(stat -f %z "$INT/$F1")" = "$(stat -f %z "$EXT/$F1")" ]; then
  rm -f "${INT:?}/$F1" "${INT:?}/$F2"; rmdir "$INT" "$(dirname "$INT")" 2>/dev/null
  echo "internal copy removed; drive keeps $EXT; df $(df -g / | awk 'NR==2{print $4}') GiB free"
else
  echo "sha/size mismatch: internal copy KEPT ($a vs $b)"
fi
