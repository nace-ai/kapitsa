#!/bin/bash
# Interleaved `kapitsa bench --quick` legs where bench starts its own server (no KAPITSA_BENCH_SERVER), so its GPU
# monitor counts the server as its own: with an external server, bench_cmd.cpp's own() is empty (srv.pid == 0) and the
# server's own decode load is reported as foreign "CONTENTION" (every sp35 leg, 12 Nov 2026). Run under lockf.
# env: BIN, OUT, ORDER ("A B B A"), MODEL_<arm>=folder. Gates per leg: AC, 1-min load < 4. A leg is VOID when it ends
# on battery or at load >= 4.5, or when a Kapitsa process outside this leg's bench tree ran during it (census every 5 s).
# Top-up: an arm with < 2 valid legs gets up to TOPUPS (default 2) extra legs; TOPUPS=0 for a no-rerun stop line.
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
: "${BIN:?}" "${OUT:?}" "${ORDER:?}"
mkdir -p "$OUT"
# the timed-leg marker: detach.py --cpu-heavy jobs wait while it names this script and are paused if already running
# (12 Nov 2026: without it, a CPU test suite under --cpu-heavy ran through 27B legs 5-A and 6-A)
TL=/Users/alisheramantay/.agents/skills/session-orchestration/scripts/timed_leg.sh
bash "$TL" on $$ || exit 3
trap 'bash "$TL" off $$' EXIT
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT/legs.log"; }
load(){ sysctl -n vm.loadavg | awk '{print $2}'; }
ac(){ pmset -g batt | head -1 | grep -q "AC Power"; }
for a in $ORDER; do v="MODEL_$a"; [ -r "${!v}/config.json" ] || { log "PREFLIGHT model for $a: ${!v}"; exit 5; }; done
log "bin=$BIN sha=$(shasum -a 256 "$BIN" | cut -c1-12) $("$BIN" --version 2>&1 | head -1) order=[$ORDER] macos=$(sw_vers -productVersion)"
for a in $(echo $ORDER | tr ' ' '\n' | awk '!s[$0]++'); do v="MODEL_$a"; log "arm $a = ${!v}"; done
tree(){ local out=" $1" kids k; kids=$(pgrep -P "$1"); for k in $kids; do out="$out $(tree "$k")"; done; echo "$out"; }
i=0; OKS=" "
leg(){ local arm=$1 v="MODEL_$1"; local model=${!v}; i=$((i+1)); local tag=$i-$arm
  while :; do ac || { log "no AC: giving the lock back (exit 75)"; exit 75; }
    awk "BEGIN{exit !($(load) < 4)}" && break; log "waiting for load $(load)"; sleep 15; done
  local l0; l0=$(load)
  "$BIN" bench --model "$model" --quick --json --out "$OUT/$tag.json" > "$OUT/$tag.stdout.json" 2> "$OUT/$tag.err" & local bp=$!
  local foreign=""
  while kill -0 $bp 2>/dev/null; do
    local mine; mine=$(tree $bp)
    for p in $(pgrep -x kapitsa); do case " $mine " in *" $p "*) ;; *)
      c=$(ps -o command= -p "$p" 2>/dev/null); echo "$(date +%T) $tag foreign pid=$p $c" >> "$OUT/census.log"
      case "$c" in *" serve"*|*" bench"*|*" run"*) foreign="$foreign $p";; esac;; esac; done
    sleep 5; done
  wait $bp; local rc=$? l1; l1=$(load); local st=ok
  [ $rc = 0 ] || st=rc$rc
  ac || st=void-battery
  awk "BEGIN{exit !($l1 >= 4.5)}" && st=void-load
  [ -n "$foreign" ] && st="void-foreign-kapitsa($(echo $foreign | tr ' ' '\n' | sort -u | tr '\n' ' '))"
  log "$tag rc=$rc load $l0->$l1 $st flags=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("flags"))' "$OUT/$tag.json" 2>/dev/null)"
  [ "$st" = ok ] && OKS="$OKS$arm "
}
for arm in $ORDER; do leg "$arm"; done
nok(){ echo "$OKS" | tr ' ' '\n' | grep -cx "$1"; }
for arm in $(echo $ORDER | tr ' ' '\n' | awk '!s[$0]++'); do x=0
  while [ "$(nok "$arm")" -lt 2 ] && [ $x -lt "${TOPUPS:-2}" ]; do x=$((x+1)); log "top-up: $arm has $(nok "$arm") valid, extra leg $x/2"; leg "$arm"; done
  [ "$(nok "$arm")" -lt 2 ] && log "SHORT: $arm has $(nok "$arm") valid legs"
done
log LEGSDONE
