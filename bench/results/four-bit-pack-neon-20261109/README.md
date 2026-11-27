# Four-bit CPU packing evidence

This directory contains **redacted derived evidence**, not an original execution-log archive. Source is `7d1b0bbcdcfb36a495036eb0d6d00405fd5955ca`; reference is `4b9ce88708282f754b76cbd7214e188786680c89`. Raw commands and original records remain private. The two C++ files are derived copies with only one terminal blank line removed from each. The manifest records executed-original and public-copy digests separately; they are not claimed to be byte-identical source files.

`summary.json` retains every formal timing leg, complete-output byte gates, compiler/binary/source identities, model identity, and completed CPU checks. The earlier synthetic pilot is not the formal dataset or a claimed performance result. Initial pilot variation motivated the longer real-weight protocol.

The input is the first 1,024 rows of `blk.0.attn_gate.weight`, shape `[4096,2560]` in row-major notation, from the pinned Q4_0 GGUF. The unchanged reference `decode4` extracts 2,621,440 code bytes; the driver packs 1,310,720 output bytes per pass. Each timing pass makes 1,024 independent K=2560 row calls. The independent expected bytes combine adjacent low nibbles; every pass compares the entire output and checks the original input outside the timer. Raw high-bit masking, partial tails, misalignment, overlap, and protected boundaries are additionally covered by the committed CPU test.

The converter's exact four-bit path in `src/gguf/import.cpp` calls this function for these weights. Neither scales nor quantization selection changes. Other bit widths are unchanged. The optimization is enabled for nonoverlapping four-bit packing on all AArch64 builds; the scalar path remains for other platforms, overlapping consumed ranges, and the final whole words.

These are warm CPU packing measurements. They do not establish whole import or inference speed, or a GPU output check. Full-model import/output/default-path validation is tracked separately and must complete before a Ready claim.

## Reproduction examples

The following are derived command examples using placeholders, **not raw logged commands**. Use the pinned model and commits in the JSON. Build both drivers with the same compiler, `-std=c++20 -O3`, and the same `gguf.cpp`, headers, and driver. The reference replaces only `quant.cpp` with the reference commit's file.

```sh
c++ -std=c++20 -O3 -I"$SRC/src" -I"$SRC/third_party" extract-real.cpp "$REF_QUANT" "$SRC/src/gguf/gguf.cpp" -o extract-real
./extract-real "$MODEL" first1024.q4_0 first1024.codes
c++ -std=c++20 -O3 -I"$SRC/src" -I"$SRC/third_party" perrow-driver.cpp "$REF_QUANT" "$SRC/src/gguf/gguf.cpp" -o reference-pack4
c++ -std=c++20 -O3 -I"$SRC/src" -I"$SRC/third_party" perrow-driver.cpp "$SRC/src/gguf/quant.cpp" "$SRC/src/gguf/gguf.cpp" -o candidate-pack4
./reference-pack4 first1024.codes A1.packed
./candidate-pack4 first1024.codes B1.packed
# Continue A2, B2, A3, B3, A4, B4 using fresh output filenames.
```
