# Q6_K CPU row decoding

These are explicitly redacted, derived exports of private execution records, not raw logs.
CPU.json preserves every initial and formal timing leg, source/binary identities and limitations.
INPUT.json identifies pinned model tensor bytes; no model data is redistributed. Both driver files
are unchanged copies of the corresponding executed source. The initial pilot and formal runs
use separate protocols. The formal timer excludes the complete byte comparison after each pass.

A is the reference decoder, B the candidate. CPU component timings do not establish full import
or inference speedups. The default targets arm64 Q6_K imports; exceptional scales and overlapping
buffers retain the scalar path. Other architectures also retain the scalar implementation.

## Reproduction examples

These relative/placeholder commands are derived recipes, not original execution logs.
$MODEL is the pinned GGUF; $REF_SRC and $CAND_SRC contain the exact revisions in CPU.json.
Extract INPUT.json's sample byte range to row-input.q6 before running:

```sh
c++ -std=c++20 -O3 -I"$REF_SRC/src" -I"$REF_SRC/third_party" initial-pilot-driver.cpp "$REF_SRC/src/gguf/quant.cpp" "$REF_SRC/src/gguf/gguf.cpp" -o reference-pilot
./reference-pilot row-input.q6 expected.fp32
c++ -std=c++20 -O3 -I"$REF_SRC/src" -I"$REF_SRC/third_party" row-driver.cpp "$REF_SRC/src/gguf/quant.cpp" "$REF_SRC/src/gguf/gguf.cpp" -o reference
c++ -std=c++20 -O3 -I"$CAND_SRC/src" -I"$CAND_SRC/third_party" row-driver.cpp "$CAND_SRC/src/gguf/quant.cpp" "$CAND_SRC/src/gguf/gguf.cpp" -o candidate
./reference row-input.q6 A1.fp32 expected.fp32
./candidate row-input.q6 B1.fp32 expected.fp32
# Continue A2/B2/A3/B3/A4/B4 with new output filenames.
```

The integrated regression in tools/gguf_q6_k_test.h is called by iq_layout_test. Its integer
significand oracle uses independent logical lanes. All half bit patterns run in four rounding
modes; selected halves additionally cover every int8 scale and six-bit code combination.
Exceptional values keep scalar byte/exception behavior. Alignment, overlap and protected-page
checks exercise the existing truncation contract. Missing-mask and wrong-scale mutants fail.

## Not verified by these CPU results

Own-source full-model import, GPU output, default inference performance and HTTP checks are
being validated separately. No GPU/full-model pass or full-import/inference speedup is claimed.
