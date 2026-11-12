#!/bin/bash
set -u
export PATH=/Users/ec2-user/venv/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
unset KAPITSA_BENCH_SERVER
BIN=/Users/ec2-user/k/v0235/build/kapitsa
OUT=/Users/ec2-user/k/job1/legs
MODEL_A=/Users/ec2-user/.cache/kapitsa/models/qwen3.8-27b-4bit
MODEL_B=/Users/ec2-user/.cache/kapitsa/models/qwen3.8-27b-kquant
mkdir -p "$OUT"
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/legs.log"; }
load(){ sysctl -n vm.loadavg | awk '{print $2}'; }
tree(){ local out=" $1" kids k; kids=$(pgrep -P "$1"); for k in $kids; do out="$out $(tree "$k")"; done; echo "$out"; }
gate(){ local n=0; while [ "$n" -lt 3 ]; do
  ps -axo pid=,comm=,args= > "$OUT/preflight-current.txt"
  python3 - "$OUT/preflight-current.txt" <<'PY'
import sys
bad=[]
for l in open(sys.argv[1]):
 p=l.split(None,2)
 if len(p)<3: continue
 name=p[1].rsplit('/',1)[-1]
 if name=='kapitsa' or (name.lower().startswith('python') and any(x in p[2].lower() for x in ['model','fresh','torch','mlx'])): bad.append(l.strip())
if bad: print('FOREIGN',*bad,sep='\n');sys.exit(1)
PY
  if [ $? != 0 ]; then n=0; else if awk "BEGIN{exit !($(load)<4)}"; then n=$((n+1)); else n=0; fi; fi
  log "gate reading=$n load=$(load)"; [ "$n" -lt 3 ] && sleep 20
 done; }
i=0; OK_A=0; OK_B=0
leg(){ local arm=$1 v="MODEL_$1"; local model=${!v}; i=$((i+1)); local tag=$i-$arm
 gate
 cp "$OUT/preflight-current.txt" "$OUT/$tag.preflight.txt"
 local l0=$(load)
 NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 lockf -k /tmp/kapitsa-gpu.lock "$BIN" bench --model "$model" --quick --json --out "$OUT/$tag.json" > "$OUT/$tag.stdout.json" 2> "$OUT/$tag.err" & local bp=$!
 local foreign=""
 while kill -0 "$bp" 2>/dev/null; do
  local mine=$(tree "$bp")
  for p in $(pgrep -x kapitsa); do case " $mine " in *" $p "*) ;; *)
   c=$(ps -o command= -p "$p" 2>/dev/null); echo "$(date -u +%FT%TZ) $tag pid=$p $c" >> "$OUT/census.log"; foreign="$foreign $p";; esac; done
  sleep 5
 done
 wait "$bp"; local rc=$? l1=$(load) st=ok
 [ "$rc" = 0 ] || st=rc$rc
 awk "BEGIN{exit !($l1>=4.5)}" && st=void-load
 [ -n "$foreign" ] && st=void-foreign
 python3 - "$OUT/$tag.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1]))
def invalid(x):
 if isinstance(x,dict): return any(('invalid' in str(k).lower() and bool(v)) or invalid(v) for k,v in x.items())
 if isinstance(x,list): return any(invalid(v) for v in x)
 return isinstance(x,str) and 'invalid' in x.lower()
sys.exit(1 if invalid(j) else 0)
PY
 [ $? = 0 ] || st=void-json-invalid
 log "$tag rc=$rc load=$l0->$l1 status=$st foreign=$foreign"
 printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "$rc" "$l0" "$l1" "$st" >> "$OUT/status.tsv"
 if [ "$st" = ok ]; then if [ "$arm" = A ]; then OK_A=$((OK_A+1)); else OK_B=$((OK_B+1)); fi; fi
}
for arm in A B B A; do leg "$arm"; done
for arm in A B; do if [ "$OK_A" -lt 2 ] || [ "$OK_B" -lt 2 ]; then leg "$arm"; fi; done
log "LEGSDONE valid_A=$OK_A valid_B=$OK_B"
