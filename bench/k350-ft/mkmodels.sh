#!/bin/bash
# Symlink copies of the 35B-A3B model folder for the two eval arms: ship (dflash2/ files symlinked to the shipped
# drafter) and ft (dflash2/ = the fine-tuned drafter, copied to the internal disk). Usage: mkmodels.sh <ft_drafter_dir>
set -e
W=~/Documents/nace/kapitsa/.worktrees/k350-ft; FT=$W/.ftmodels
# 7 Nov: the Extreme Pro came back on a 480 Mb/s link and each eval leg spent 477 s reading the 35B from it; use the
# internal copy (same shipped drafter, sha256 d1f16c0f...) when present
SRC=$(readlink -f ~/Documents/nace/kapitsa-home/models-internal/qwen3.6-35b-a3b-4bit)
[ -r "$SRC/model-00001-of-00004.safetensors" ] || SRC=$(readlink -f ~/Documents/nace/models/qwen3.6-35b-a3b-4bit)
[ -r "$SRC/config.json" ] && [ -r "$1/model.safetensors" ] || { echo "missing source"; exit 1; }
for arm in ship ft; do
  D=$FT/$arm/qwen3.6-35b-a3b-4bit; rm -rf "$D"; mkdir -p "$D/dflash2"
  for f in "$SRC"/*; do b=$(basename "$f"); [ "$b" = dflash2 ] || ln -s "$f" "$D/$b"; done
done
ln -s "$SRC/dflash2/config.json" $FT/ship/qwen3.6-35b-a3b-4bit/dflash2/config.json
ln -s "$SRC/dflash2/model.safetensors" $FT/ship/qwen3.6-35b-a3b-4bit/dflash2/model.safetensors
cp "$1/config.json" "$1/model.safetensors" $FT/ft/qwen3.6-35b-a3b-4bit/dflash2/
ls -la $FT/*/qwen3.6-35b-a3b-4bit/dflash2/
