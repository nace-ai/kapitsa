#!/bin/bash
# #396 track D: interleaved serve+bench legs in ONE lock hold. Run under:
#   detach.py run --name k396-d-<x> -- lockf -k /tmp/kapitsa-gpu.lock bash ab_legs.sh
# Inputs (env): BIN (kapitsa binary), MODEL (folder), OUT (result dir), ORDER (arm names, e.g. "K M C C M K"),
#   ARM_<name>="<env assignments>|<extra serve flags>[|<model folder, default MODEL>]" per arm, KHOME_SEED (optional
#   kflash/ dir to copy). An arm whose flags hold "--engine splash" serves the --splash-model folder: no --model/--ctx.
# kapitsa bench drops NI_* from a server it starts, so this starts the server itself with the arm's env
# (bench's own args: --ctx 32768 --slots 1 --disk-cache 0) and points bench at it (KAPITSA_BENCH_SERVER).
# Gates: AC + 1-min load < 4 before each leg; leg VOID if it ends on battery, at load >= 4.5, swap grows > 3 GB in the
# leg, memory pressure critical, or disk < 40 GB.
# Top-up (9 Nov 2026, launch rule: a published row needs >= 2 valid legs): after ORDER, an arm with fewer than
# MIN_VALID (2) ok legs gets extra legs, at most MAX_EXTRA (2) per arm; still short -> logged SHORT, exit 0.
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
: "${BIN:?}" "${MODEL:?}" "${OUT:?}" "${ORDER:?}"
mkdir -p "$OUT"
TO="timeout 1800"; command -v timeout >/dev/null || TO=""  # (the M3 Pro has no coreutils timeout)
[ -x "$BIN" ] && [ -r "$(readlink -f "$MODEL")/config.json" ] || { echo "PREFLIGHT bin/model"; exit 5; }
for a in $ORDER; do v="ARM_$a"; [ -n "${!v}" ] || { echo "PREFLIGHT arm $a undefined"; exit 5; }; done
[ -n "$PREFLIGHT_ONLY" ] && { echo PREFLIGHT-OK; exit 0; }
HOME_DIR=$OUT/home; mkdir -p "$HOME_DIR/kflash"
[ -n "$KHOME_SEED" ] && cp "$KHOME_SEED"/*.json "$HOME_DIR/kflash/" 2>/dev/null
export KAPITSA_HOME=$HOME_DIR
HIDE=${HIDE:-~/.agents/skills/session-orchestration/scripts/hide_dsh.sh}
LOAD_MAX=${LOAD_MAX:-4}; LOAD_VOID=${LOAD_VOID:-4.5}; MIN_VALID=${MIN_VALID:-2}; MAX_EXTRA=${MAX_EXTRA:-2}
log(){ echo "$(date '+%H:%M:%S') $*" | tee -a "$OUT/legs.log"; }
load(){ sysctl -n vm.loadavg | awk '{print $2}'; }
swapm(){ sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); print int(v)}'; }
diskg(){ df -g / | awk 'NR==2{print $4}'; }
gate(){ while :; do { pmset -g batt | grep -q "AC Power" || [ -e /tmp/k396-battery-ok ]; } || { log "no AC: giving the lock back (exit 75)"; exit 75; }
  awk "BEGIN{exit !($(load) < $LOAD_MAX)}" && break; log "waiting for load $(load)"; sleep 15; done; }
sp=""; trap 'bash $HIDE off >/dev/null 2>&1; [ -n "$sp" ] && kill $sp 2>/dev/null; wait $sp 2>/dev/null' EXIT
log "bin=$BIN sha=$(shasum -a 256 "$BIN" | cut -c1-12) commit=${COMMIT:-$("$BIN" --version 2>/dev/null | sed -n "s/.*(\(.*\)).*/\1/p")} model=$MODEL order=[$ORDER] macos=$(sw_vers -productVersion)"
i=0; OKS=" "
leg(){ local arm=$1; i=$((i+1)); bash $HIDE on >/dev/null 2>&1; gate; l0=$(load)
  v="ARM_$arm"; spec=${!v}; aenv=${spec%%|*}; rest=${spec#*|}; aflags=${rest%%|*}; amodel=$MODEL
  [ "$rest" != "$aflags" ] && amodel=${rest#*|}
  margs="--model $amodel --ctx 32768 --slots 1 --disk-cache 0"; case "$aflags" in *"--engine splash"*) margs="";; esac
  tag=$i-$arm; slog=$OUT/$tag.server.log; port=$((18960+i))
  # shellcheck disable=SC2086
  env $aenv "$BIN" serve $margs --host 127.0.0.1 --port $port $aflags > "$slog" 2>&1 & sp=$!
  for w in $(seq 1 600); do curl -sf -o /dev/null http://127.0.0.1:$port/health && curl -sf -o /dev/null http://127.0.0.1:$port/v1/models && break; kill -0 $sp 2>/dev/null || break; sleep 2; done
  env KAPITSA_BENCH_SERVER=$port $TO "$BIN" bench --model "$amodel" --quick --json --out "$OUT/$tag.json" > "$OUT/$tag.stdout.json" 2> "$OUT/$tag.err" & lp=$!; g=""
  sw0=$(swapm); pkmax=0
  while kill -0 $lp 2>/dev/null; do sw=$(swapm); dk=$(diskg); pl=$(sysctl -n kern.memorystatus_vm_pressure_level)
    r=$(ps -o rss= -p $sp 2>/dev/null | tr -d " "); [ -n "$r" ] && [ "$r" -gt "$pkmax" ] && pkmax=$r
    if [ $((sw - sw0)) -gt 3072 ] || [ "$dk" -lt "${DISK_MIN_GB:-40}" ] || [ "$pl" -ge 4 ]; then g="void-guard(swap ${sw0}->${sw}M disk ${dk}G pressure $pl)"; kill $lp; fi; sleep 3; done
  wait $lp; rc=$?
  pk=$(ps -o rss= -p $sp 2>/dev/null | awk '{printf "%.1f", $1/1048576}')
  kill $sp 2>/dev/null; wait $sp 2>/dev/null; sp=""
  if [ $rc = 1 ] && [ -s "$OUT/$tag.json" ] && [ "$(grep '^INVALID' "$OUT/$tag.err" | grep -vc 'KAPITSA_BENCH_SERVER')" = 0 ]; then rc=0; fi
  l1=$(load); st=ok; [ $rc = 0 ] || st=rc$rc; [ -n "$g" ] && st=$g
  { pmset -g batt | grep -q "AC Power" || [ -e /tmp/k396-battery-ok ]; } || st=void-battery; awk "BEGIN{exit !($l1 >= $LOAD_VOID)}" && [ -z "$g" ] && st=void-load  # the memory guard's label wins (9 Nov: 8-bit said void-load)
  pkg=$(echo "scale=2; $pkmax/1048576" | bc)
  pw=$(pmset -g batt | head -1 | grep -q "AC Power" && echo ac || echo battery)
  log "$tag rc=$rc load $l0->$l1 $st power=$pw serve_rss_end=${pk}G serve_peak=${pkg}G kflash_lines=$(grep -c 'KFlash' "$slog") drafter_lines=$(grep -ci 'dflash2\|drafter' "$slog")"
  [ $i = 1 ] && [ $rc != 0 ] && { log "leg 1 failed: stopping"; exit 6; }
  [ "$st" = ok ] && OKS="$OKS$arm "
}
nok(){ echo "$OKS" | tr " " "\n" | grep -cx "$1"; }
for arm in $ORDER; do leg "$arm"; done
for arm in $(echo $ORDER | tr " " "\n" | awk '!s[$0]++'); do x=0
  while [ "$(nok "$arm")" -lt "$MIN_VALID" ] && [ $x -lt "$MAX_EXTRA" ]; do x=$((x+1)); log "top-up: $arm has $(nok "$arm") valid of $MIN_VALID, extra leg $x/$MAX_EXTRA"; leg "$arm"; done
  [ "$(nok "$arm")" -lt "$MIN_VALID" ] && log "SHORT: $arm has $(nok "$arm") valid legs (< $MIN_VALID): not publishable"
done
log LEGSDONE; exit 0
