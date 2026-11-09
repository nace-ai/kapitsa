# One lock hold: the catalog's Qwen3.5-9B 4-bit pack (lmstudio-community @ b455506) vs the KQuant 4-bit and 3-bit packs
# (same layout, vision tower kept, ws language tensors), each with the catalog's MTP head and DFlash2 drafter, default decoding.
main() {
  export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
  K=$HOME/Documents/nace/k421; P=$K/packs; CAT=$P/cat-9b-4bit
  for d in $CAT $P/kq-9b-4bit; do [ -r $d/config.json ] && [ -d $d/dflash2 ] || { echo "missing $d"; exit 5; }; done
  for i in $(seq 1 40); do f=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "$f" -ge 40 ] && break; echo "$(date +%T) memory free $f% < 40%, waiting"; sleep 30; done
  [ "$f" -ge 40 ] || { echo "memory never reached 40% free"; exit 76; }
  export BIN=$HOME/Documents/nace/kapitsa/.worktrees/k421-kquant/build/kapitsa MODEL=$CAT OUT=$K/mac/runs/9b-default ORDER="A B B A"
  export ARM_A="KAPITSA_HOME=$K/mac/khome||$CAT" ARM_B="KAPITSA_HOME=$K/mac/khome||$P/kq-9b-4bit"
  bash $HOME/Documents/nace/k396/support/ab_legs.sh
}
main "$@"
