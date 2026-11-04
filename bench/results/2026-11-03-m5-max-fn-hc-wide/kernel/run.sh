#!/bin/zsh
# the pair at Flash-Next's shape (D 2560, S 4, rank 320, 6-bit), R = 1..4 rows, 96 weight sets streamed (hc-time.out);
# and every variant bitwise against the one-row kernels at small shapes under the validation layer (fn-rows-validation.out)
lockf -k /tmp/kapitsa-gpu.lock build/fn_check hc-time > hc-time.out 2>&1
lockf -k /tmp/kapitsa-gpu.lock env MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 FN_ROWS=1 build/fn_check kernels > fn-rows-validation.out 2>&1
