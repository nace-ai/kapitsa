# BF16 NaN preservation: validation snapshot

Source: `5e174cb89eb991a705752d033f7f9da8edfeff87`; base: `bbd84e0d53be75866087aa615bdeb6f2411f393f`. M3 Ultra, macOS 26.4.1, Release builds.

The old helper can turn `0x7f800001` into positive infinity and `0x7fffffff` into negative zero. Quieting NaNs before the finite rounding bias preserves their classification and sign, consistent with the existing GGUF conversion policy. Finite rounding remains round-to-nearest-even.

The extended existing `affine_pack_test` checks all 16,777,214 float32 NaNs, all 2,046 float16 NaNs, 391,680 finite rounding cases and the existing 63,488 finite float16 cases. Candidate: zero failures. The same test source compiled with the pre-change helper: 8,454,174 assertion failures, exit1, on both the local Mac and M3. The expected failure is retained.

M3 CPU suite:21 passes,0 failures;19 supplemental checks pass. All-files prek passed locally. The initial correct source label was seven digits while vk doctor expects eight; the explicit-label configure and updater check are separate records. No GPU, model-output or inference-speed claim is made in this snapshot.

Reproduce with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`, `cmake --build build -j --target affine_pack_test`, then `build/affine_pack_test`. The existing CI target runs this regression. Full model output/HTTP/bench checks remain pending for this Draft.

`validation.tar.gz` contains raw CPU/build/regression transcripts; its `MANIFEST.json` hashes each archived file. Future verification is appended under a new path.
