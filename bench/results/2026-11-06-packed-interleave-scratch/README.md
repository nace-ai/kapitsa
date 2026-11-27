# Packed interleave scratch validation

Reference `39f20c8d`, candidate `e554264e`; the subsequent evidence commit changes no source.

`evidence.json` retains every host run, both preliminary and final campaigns, the full four-run M4 Pro `kapitsa bench --quick` outputs, the logit gate report, GPU-check logs and HTTP results. Two sampled benchmark arms flagged repetition loops; there were no hardware/contention flags. These timings do not establish a decode speedup.

The final host campaign used fresh `clang++ -O3 -std=c++20` executables, one with each revision's `src/` include path, running `interleave_bench.cpp N GROUPS`. Six alternating AB/BA pairs were run for each of `(18432,40)`, `(2560,144)`, `(248320,40)`, `(4096,192)`. It times the layout permutation only and hashes the resulting full image; peak RSS includes the fixture and runtime. All output digests agree. The earlier campaign ran during compilation and is retained separately.

For the GPU test, apply `interleaved-checker.patch` to the unchanged existing checker in a temporary checkout, link against the candidate library, then run `check-small`, `check-codec`, `check-scales`, `check-segments`, `check-shapes` under the project GPU lock with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. The inverse-codec leg uses the plain layout its inverse decoder expects; the GEMM legs exercise the interleaved images and bind argument 25.

Production check on M4 Pro: reference `logit_dump MODEL REF`, candidate `logit_dump MODEL NEW --from REF`, then `tools/logit_compare.py REF NEW`. All 584 rows are bitwise identical. The normal `tests/test_server.py URL` passed all 49 checks for each build on a two-slot server with 32K context and disk cache disabled. The local target model has MTP, not DFlash2; synthetic GPU coverage additionally includes the drafter's shapes.

Complete `vk cpu-tests`, the supplemental CPU tools listed in AGENTS.md, and `prek run --all-files` passed on M3 Ultra. No kernels or dispatch defaults changed. Base M4 hardware and other models were not tested.
