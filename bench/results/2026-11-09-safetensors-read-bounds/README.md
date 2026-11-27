# Safetensors relative-read bounds

The indexed reader rejected a range with `off + n > tensor_length`. On a valid eight-byte tensor, `off=SIZE_MAX, n=1` wrapped that sum to zero and returned the final header byte (`125`, `}`), outside the tensor. The candidate uses a short-circuit subtraction check and rejects the range before I/O.

`VALIDATION.json` is a deliberately derived public report containing source, binary and method identities and actual outcomes. The two `.out` files are exact copies of the combined M4 standard-output/standard-error captures; they needed no redaction. Original configure/build commands and complete execution records remain privately verifiable.

The reference production source is `607073ad6b23d73e4faf3d0553352a803bac0118`, built with the same new test source as the candidate. The candidate production/test commit is `420eae44d8f9e251af698e9be96b05f988e7de92`. The new suite also runs in the default CPU mode before the existing cache-residency check.

The M3 supplement compiles the same common test source, then links one reference driver with the retained reference support library and one candidate driver with the exact changed loader object ahead of that same library. The support library is from `9c97df7c637cdca18b709b9ef5452efba5fc4d76`; its loader source and the fourteen compiler-reported source dependencies match current canonical/candidate bytes where unchanged. This isolates the changed loader and is not a full M3 candidate build. Both arms retain their actual binary identities.

Descriptor cleanup checks the count of descriptors in the bounded range 0 through 1,023. A valid read from a physically truncated file may write its available prefix before reporting the existing read error; the test preserves and verifies that behavior, then retries on the same index after restoring the file.

Portable reproduction examples, written with relative paths rather than original execution commands:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4 --target kapitsa load_error_test
build/kapitsa --version
build/load_error_test --safetensors-read-bounds
build/load_error_test
bash .dsh/skills/verify-kapitsa/scripts/vk cpu-tests
prek run --all-files
```

No GPU, full-model output or inference performance result is claimed for this CPU read-admission fix.

The completed M4 broad run passed all 22 vk CPU entries, 18 supplementary native programs, the candidate full loader CPU suite, and lint. Five supplemental scripts passed initially; the sixth, the output-directory check, could not start because logit_dump had not been built. It passed after that target was built under a new recovery label. The original reference CPU control failed its existing raw-U8 cache threshold assertion; that result is retained and is not a claim of a cache fix. No original machine-state values are exported.
