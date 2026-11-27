#!/bin/zsh
# The 27B's 4K prompt after the decode phase (dec), the same with a 500 ms Neural Engine keepalive (keep), and alone
# (alone), interleaved, 30 s idle before each run, main 87d7dd5's tree (kapitsa-cells3), the session idle. The run wrote
# under <scratch>/hot2; its outputs were copied here (D) as they are.
S=${S:?}; W=$S/kapitsa-cells3; D=$W/bench/results/2026-11-05-m5-max-qwen-cells-87d7dd5/after-decode
cd $W/bench/harness
for rd in 1 2; do
  if (( rd % 2 == 1 )); then arms=(dec keep alone); else arms=(alone keep dec); fi
  for a in $arms; do
    sleep 30
    case $a in
      dec)   lab=q38-27b-kapitsa; sk=greedy,tools,agent,cache,sustain,sweep ;;
      keep)  lab=q38-27b-kapitsa-keep500; sk=greedy,tools,agent,cache,sustain,sweep ;;
      alone) lab=q38-27b-kapitsa; sk=decode,greedy,tools,agent,cache,sustain,sweep ;;
    esac
    lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
      python3 compare_engines.py configs/kapitsa.json $D/extra.json --only $lab --agentic --skip $sk --sizes 4000 \
      --quiet-timeout 60 --out-dir $D/$a > $D/r$rd-$a.log 2>&1
    f=$(ls -t $D/$a/logs/*.log | head -1)
    echo "round $rd $a: $(grep -o 'prompt 41[0-9]* (cached 0) in [0-9]* ms' $f) $(grep -o 'Neural Engine busy [0-9]*% of [0-9]* ms' $f | tail -1) keep_ms=$(grep -c NI_ANE_KEEP $D/r$rd-$a.log)"
  done
done
echo done
