#!/bin/zsh
# moe4_unique against the serial pass (400 random sets) and the verify MoE against the decode MoE per row, with and
# without the validation layer; and NI_PROFILE's per-dispatch times of both builds, 96 tokens of the A/B's prompt
lockf -k /tmp/kapitsa-gpu.lock env MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 FN_ROWS=1 <repo>/build/fn_check kernels > fn-rows-validation.out 2>&1
lockf -k /tmp/kapitsa-gpu.lock env FN_ROWS=1 <repo>/build/fn_check kernels > fn-rows.out 2>&1
NI_PROFILE=1 <ref>/build/ni_run ~/models/qwen3.8-flash-next-4bit "$(cat ../ab/prompt-ids.txt)" 96 0 --spec 2 --dflash none > profile-old.out 2>&1
NI_PROFILE=1 <repo>/build/ni_run ~/models/qwen3.8-flash-next-4bit "$(cat ../ab/prompt-ids.txt)" 96 0 --spec 2 --dflash none > profile-new.out 2>&1
