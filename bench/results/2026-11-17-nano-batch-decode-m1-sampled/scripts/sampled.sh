#!/bin/zsh
# sampled.sh MODEL TAG: (under one GPU lock) builds in the order main, pr, pr, main; on each, a fresh kapitsa serve
# (CTX and SLOTS as in arm.sh) runs sampled.py's seeded plan alone then together, and its ctx6k plan alone then
# together. Texts: together against alone, alone against alone (seeded replies repeat) and pr against main.
set -u
MODEL=$1; TAG=$2
D=${0:A:h}; O=$D/out/$TAG/sampled; mkdir -p $O
PORT=8143; B=http://127.0.0.1:$PORT/v1
k=0
for b in main pr pr main; do
  k=$((k+1)); a=$k-$b
  vm_stat | egrep "Swapins|Swapouts" | tr -s ' ' | tr '\n' ' ' > $O/$a-env.txt; echo >> $O/$a-env.txt
  $D/bin-$b/kapitsa --version >> $O/$a-env.txt
  $D/bin-$b/kapitsa serve --model $MODEL --model-id x --port $PORT --ctx ${CTX:-16384} --slots ${SLOTS:-4} --disk-cache 0 > $O/$a-server.log 2>&1 &
  SP=$!
  for i in {1..2400}; do grep -q "^Ready" $O/$a-server.log && break; sleep 0.25; done
  for p in seeded ctx6k; do
    for m in alone together; do
      python3 $D/sampled.py $B $m $O/$a-$p-$m.json $p > $O/$a-$p-$m.out 2>&1
    done
  done
  kill $SP; wait $SP 2>/dev/null
  vm_stat | egrep "Swapins|Swapouts" | tr -s ' ' | tr '\n' ' ' >> $O/$a-env.txt; echo >> $O/$a-env.txt
done
echo SAMPLEDDONE
