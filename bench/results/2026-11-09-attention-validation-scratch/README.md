# Attention scratch selected for shader validation

Source `8be65afbcd4edc0727a8c67d6b72344b1e2b64e4` confines shared K/score scratch and its additional barrier to tensor-library compilation when `MTL_SHADER_VALIDATION` is nonzero. Normal compilation retains the original K/V/score arrays and synchronization. This follows [the maintainer's requested revision](https://github.com/nace-ai/kapitsa/pull/1057#issuecomment-6724456162) after the prior head's measured roughly 2% release-kernel regression.

`VALIDATION.json` is a privacy-filtered derived summary. The helper is the exact source used; execution paths and raw host logs are not published. Both M3 runtime library compilations passed under the exclusive GPU lock, with validation unset and with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. No kernel was dispatched. The two earlier offline commands returned 72 because the offline compiler command was unavailable; neither result is called a successful offline compile.

Clang preprocessing of `qattn_t.metal`, with the four include lines removed identically in each arm, gives the original main kernel's tokens when the new macro is zero and the preceding head's tokens when it is one. This checks source equivalence, not GPU output or speed. The original main kernel is `065c4e36`; the preceding shared-scratch head is `ff16d70c`.

The M5 measurements in the PR body belong to that preceding head and to the maintainer. The old M3 numerical error remains a failure. This revision still needs the maintainer's new-head M5 timing, small validation dispatch and full-model logit gate; runtime compilation cannot replace those checks.

The following are derived placeholder examples, not original execution commands. Generate the tensor source using this source's `cmake/embed_kernels_tensor.cmake`, extracting the `NIKERNEL` string unchanged into `$TENSOR_SOURCE`.

```sh
c++ -O3 -std=c++20 -fobjc-arc "$EVIDENCE/runtime-compile.mm" -framework Metal -framework Foundation -o "$OUT/runtime-compile"
lockf -k "$GPU_LOCK" env -u MTL_SHADER_VALIDATION -u MTL_DEBUG_LAYER NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$OUT/runtime-compile" "$TENSOR_SOURCE"
lockf -k "$GPU_LOCK" env MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 "$OUT/runtime-compile" "$TENSOR_SOURCE"
```

`CPU-CHECKS.json` records the explicit Release configure/build, all 21 `vk cpu-tests` entries, 18 supplementary native checks and six supplemental scripts. They passed at this source. The lint launcher initially could not locate the executable; only the missing lint and remaining checks were resumed, all with return code 0. The build used `-O3`, arm64, without `NDEBUG`. No CPU test result is a GPU qualification.
