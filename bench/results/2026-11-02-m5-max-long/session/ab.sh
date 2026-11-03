#!/bin/zsh
# ab.sh: Dory's prefill at 2K/4K/16K, --ctx 32K against 256K (and the summary question), interleaved
R=<run-root>
cd <repo>/bench/harness
for l in dory-ab-ctx256k-code dory-ab-ctx32k-code dory-ab-ctx32k-summary dory-ab-ctx256k-code dory-ab-ctx32k-code; do
  while pmset -g batt | grep -q "Battery Power"; do sleep 30; done
  echo "=== $l $(date +%H:%M:%S) $(pmset -g batt | head -1)"
  caffeinate -is python3 compare_engines.py configs/dory.json $R/ab.json --only "$l" --agentic --skip decode,greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60 --out-dir $R/ab 2>&1 | grep -E 'INVALID|invalid|rror|sweep [0-9]|quiet'
done
echo "=== done $(date +%H:%M:%S)"
