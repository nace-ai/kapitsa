#!/bin/zsh
# #923's output check on its head, the build stamped with it: reconfigure with KAPITSA_GIT set to HEAD (the cached
# stamp was d93297d), rebuild logit_dump and ni_run, logit_dump step and verify with NI_MOE4_DX=0 and the default,
# compare; then a short NI_PROFILE decode under each setting, to show which verify MoE kernels ran. All under
# caffeinate -is lockf -k /tmp/kapitsa-gpu.lock (the build too, so it does not overlap a measurement).
set -e
export NI_LOCK_HELD=1
S=${S:?}
O=$S/lc/ochead
rm -rf $O && mkdir -p $O
cd $S/kapitsa-md
H=$(git rev-parse --short HEAD)
echo "kapitsa-md $H $(git status --short --untracked-files=no | wc -l | tr -d ' ') modified" > $O/build.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT=$H > $O/cmake.out 2>&1
cmake --build build -j --target logit_dump ni_run > $O/build.out 2>&1
NI_MOE4_DX=0 build/logit_dump ~/models/qwen3.8-flash-next-4bit $O/old --paths step,verify > $O/dump-old.out 2>&1
build/logit_dump ~/models/qwen3.8-flash-next-4bit $O/new --paths step,verify > $O/dump-new.out 2>&1
for p in prefill step verify; do python3 tools/logit_compare.py $O/old/$p $O/new/$p; done > $O/compare.out 2>&1
IDS=$(cat $S/lc/ids.txt)
NI_PROFILE=1 NI_MOE4_DX=0 build/ni_run ~/models/qwen3.8-flash-next-4bit $IDS 64 0 --spec 2 --dflash none --ignore-eos > $O/profile-old.out 2>&1
NI_PROFILE=1 build/ni_run ~/models/qwen3.8-flash-next-4bit $IDS 64 0 --spec 2 --dflash none --ignore-eos > $O/profile-new.out 2>&1
echo done
