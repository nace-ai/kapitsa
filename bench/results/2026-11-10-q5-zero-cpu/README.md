# Q5_0 CPU decoder evidence

This folder contains redacted derived CPU evidence and the exact timing harnesses. It is not a raw execution log. Production source is d22d5e25c4185ba52ecbe7a49049ae63a2a2a2a929; reference is 20d9db6cd30d45a8ccff41bca001610a5d7e1e7e. Complete output files and original records are retained privately.

CPU-VALIDATION.json preserves every sample, both chips, binary/source hashes, oracle coverage and the harness preparation failures. The trained single-decode timings vary considerably; the first M3 pair improves only about 2.5%. No inference-speed conclusion follows.

The trained driver takes a packed-row input, a shared reference output and an output file. The reference runs first to create the expected full bytes; both arms compare every subsequent complete output. The synthetic driver uses a fixed deterministic input and the same complete-array comparison. Build each driver with stubs.cpp and the respective src/gguf/quant.cpp using -std=c++20 -O3 and the repository src/ and third_party/ include directories. All timings use an exclusive benchmark lock. Commands here describe the method, not literal raw commands.

Full-model GPU/output qualification is not part of this CPU-stage record; any subsequent model evidence is added in new files.
