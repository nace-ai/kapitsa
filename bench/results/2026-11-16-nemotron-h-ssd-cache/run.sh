#!/bin/zsh
# run.sh MODEL [STEPS]: under the GPU lock taken once by the caller (lockf ... env NI_LOCK_HELD=1 ...).
# steps: func (tests/test_disk_cache.py), ttft (2K/8K/16K, 3 rounds), logit (logit_dump base vs new, no cache)
cd ~/kapitsa-ssdcache
export KAPITSA_HOME=~/kapitsa-run/home KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1
M=$1; shift
MD=$KAPITSA_HOME/models/$M
OUT=out/$M-$(hostname -s)
mkdir -p $OUT
steps=${@:-func ttft logit}
sysinfo() { echo "== $(date '+%F %T') $1"; vm_stat | egrep 'Swapins|Swapouts'; ps -Ao %cpu,command -r | sed -n 2,3p | cut -c1-120; }
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; ./kapitsa --version 2>&1 | head -2; ./kapitsa-base --version 2>&1 | head -2; } > $OUT/machine.txt
for st in ${=steps}; do
  case $st in
    func)
      sysinfo func >> $OUT/swap.log
      python3 tests/test_disk_cache.py ./kapitsa $MD --keep > $OUT/func.log 2>&1
      sysinfo func-end >> $OUT/swap.log ;;
    ttft)
      for n in 2000 8000 16000; do
        sysinfo ttft-$n >> $OUT/swap.log
        python3 tests/test_disk_cache.py ./kapitsa $MD --ttft --tokens $n --rounds 3 > $OUT/ttft-$n.log 2>&1
        sysinfo ttft-$n-end >> $OUT/swap.log
      done ;;
    logit)
      rm -rf /tmp/ssdc-ref /tmp/ssdc-new
      ./logit_dump-base $MD /tmp/ssdc-ref > $OUT/logit_dump-base.log 2>&1
      ./logit_dump $MD /tmp/ssdc-new --from /tmp/ssdc-ref > $OUT/logit_dump-new.log 2>&1
      python3 tools/logit_compare.py /tmp/ssdc-ref /tmp/ssdc-new --no-gate > $OUT/logit_compare.txt 2>&1
      for p in /tmp/ssdc-ref/*/; do b=$(basename $p); for f in logits.f32 records.txt sequences.txt; do
        if cmp -s /tmp/ssdc-ref/$b/$f /tmp/ssdc-new/$b/$f; then echo "$b/$f identical ($(stat -f %z /tmp/ssdc-ref/$b/$f) bytes)"; else echo "$b/$f DIFFERS"; fi
      done; done > $OUT/logit_cmp.txt
      shasum -a 256 /tmp/ssdc-ref/*/logits.f32 /tmp/ssdc-new/*/logits.f32 >> $OUT/logit_cmp.txt
      rm -rf /tmp/ssdc-ref /tmp/ssdc-new ;;
  esac
done
echo done > $OUT/DONE
