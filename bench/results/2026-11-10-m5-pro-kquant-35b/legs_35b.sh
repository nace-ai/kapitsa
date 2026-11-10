# 35B-A3B pair: the catalog's 4-bit pack (Youssofal MTPLX @ 95460516) vs the KQuant 4-bit pack (same layout, ws expert refit), each with the
# catalog's MTP head and DFlash2 drafter, default decoding, released kapitsa. Runs inside six3's lock hold (no one slips in between); B A B A B A, 1 top-up per arm.
main() {
  export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
  K=$HOME/Documents/nace/k436; CAT=$K/packs/cat-35b-4bit; KQ=$K/packs/kq-35b-4bit
  for d in $CAT $KQ; do [ -r $d/config.json ] && [ -r $d/dflash2/model.safetensors ] && [ -r $d/mtp.safetensors ] || { mkdir -p $K/mac/runs/35b-six; echo "$(date +%T) 35B PREFLIGHT exit 5: $d incomplete (k436-dl35 frozen/not finished): no 35B leg run, lock released" | tee -a $K/mac/runs/35b-six/legs.log; return 5; }; done
  export BIN=/opt/homebrew/bin/kapitsa MODEL=$CAT OUT=$K/mac/runs/35b-six ORDER="B A B A B A" MAX_EXTRA=1
  export ARM_A="KAPITSA_HOME=$K/mac/khome||$CAT" ARM_B="KAPITSA_HOME=$K/mac/khome||$KQ"
  bash $HOME/Documents/nace/k436/mac/ab_legs_k436.sh
  rc=$?
  # rows written (legs.log has LEGSDONE) -> free the re-downloadable packs (mini-orch 14:05)
  if grep -q LEGSDONE $OUT/legs.log 2>/dev/null; then
    for d in cat-35b-4bit kq-35b-4bit dflash35-src; do
      case "$K/packs/$d" in */Documents/nace/k436/packs/cat-35b-4bit|*/Documents/nace/k436/packs/kq-35b-4bit|*/Documents/nace/k436/packs/dflash35-src) rm -rf "$K/packs/$d";; esac
    done
    echo "$(date +%T) 35B packs deleted; free $(df -g / | awk 'NR==2{print $4}') GiB" | tee -a $OUT/legs.log
  fi
  return $rc
}
main "$@"
