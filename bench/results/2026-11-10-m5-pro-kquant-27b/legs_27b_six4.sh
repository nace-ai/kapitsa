# (v2, 10 Nov 12:30: ab_legs_d.sh = no end-load void, void on bench flags / new foreign process; run 1 voided on end load 3.66->5.00) SIX legs (A B A B A B), rerun of the 09:30 pair for the decode rule; one lock hold: the catalog's Qwen3.8-27B 4-bit pack (lmstudio-community @ 6067b15) vs the KQuant 4-bit pack (same layout, vision tower kept,
# ws language tensors), each with the catalog's MTP head and DFlash2 drafter, default decoding. Released kapitsa (homebrew) for both arms.
main() {
  export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
  K=$HOME/Documents/nace/k436; CAT=$HOME/.cache/kapitsa/models/qwen3.8-27b-4bit; KQ=$K/packs/kq-27b-4bit
  for d in $CAT $KQ; do [ -r $d/config.json ] && [ -d $d/dflash2 ] && [ -r $d/mtp.safetensors ] || { echo "missing $d"; exit 5; }; done
  for i in $(seq 1 40); do f=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "$f" -ge 40 ] && break; echo "$(date +%T) memory free $f% < 40%, waiting"; sleep 30; done
  [ "$f" -ge 40 ] || { echo "memory never reached 40% free"; exit 76; }
  export BIN=/opt/homebrew/bin/kapitsa MODEL=$CAT OUT=$K/mac/runs/27b-six4 ORDER="B A B A B A"
  export ARM_A="KAPITSA_HOME=$K/mac/khome||$CAT" ARM_B="KAPITSA_HOME=$K/mac/khome||$KQ"
  export MAX_EXTRA=1  # 25 min window: 6 legs ~20 min + at most 1 top-up per arm
  bash $HOME/Documents/nace/k436/mac/ab_legs_k436.sh
}
main "$@"
