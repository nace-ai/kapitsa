# Safetensors shape and byte-range validation

Reference: `bf30004c6397250337e5bb3eaeabdf43ac61d5b7`.
Implementation: `0514b0967984a1a0f67d0c28d4b631fae7d6a49a`.

Mac Studio (2025), M3 Ultra (60 GPU cores, 96 GiB), macOS 26.4.1 (25E253), Release `-O3`.
The GPU checks use Qwen3.5-4B-4bit and its default DFlash2 drafter.
This is a malformed-file fix; the benchmark is a regression check, not a speedup claim.

## CPU reproduction

`tools/load_error_test.cpp` contains the fixtures and is already run by CI:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target load_error_test
build/load_error_test
```

The first malformed fixture has BF16 dtype, shape `[1]` and offsets `[0, 8]`. A reader allocating from
shape/dtype expects two bytes, but the reference's deferred read copies eight. A controlled eight-byte
allocation, with the last six bytes reserved as a guard, changes that guard from `??????` to `CDEFGH`.
The fix rejects the header before reading. The reverse mismatch, malformed dimensions, overflowing shapes
and absolute offsets, and truncated deferred payloads are checked too.

The same test source was compiled against the reference library with Release flags: it fails at the first
rejection assertion (`FAIL caught`). With the fixed library it passes. Positive cases cover all ten supported
dtypes, scalar/empty/matrix shapes, unaligned offsets, header-only indexes used by the planner, and bytes and
guards on both deferred and indexed reads. Nothing in the CPU test initializes Metal.

The pre-existing uncached-read residency assertion had one failing run: 91 -> 897 resident pages of 4,420.
A sequential reference/candidate recheck passed (3 -> 67 / 3 -> 3). Both the original result and the recheck
are retained in `evidence.json`; no product or test workaround was introduced for that sample.

## GPU and HTTP checks

Run the standard `build/logit_dump MODEL OUTPUT` in the reference, then
`build/logit_dump MODEL OUTPUT --from REFERENCE_OUTPUT` in the candidate and compare with
`python3 tools/logit_compare.py REFERENCE_OUTPUT CANDIDATE_OUTPUT --json REPORT`.
Each dump runs under `.dsh/skills/verify-kapitsa/scripts/vk gpu run`, holding the shared GPU lock.
All 584 rows are bitwise identical: prefill 8, step 192, verify 192 and batch 192.

Each arm's server runs under the same GPU lock, using the default drafter, two slots, context 32,768,
`--disk-cache 0`, an isolated `KAPITSA_HOME` and a free loopback port. The test is
`python3 tests/test_server.py http://127.0.0.1:PORT`. Both arms have 47 passes and the same two timing failures:
the long counting reply stops at 693 tokens before the delayed side request, and the slot holder ends before
a second 15-second keepalive. The raw failure text is retained; this is not an all-pass HTTP result.

The performance regression check is `kapitsa bench --model MODEL --quick --json --out RESULT` in
reference/candidate/reference/candidate order. It uses the benchmark's own GPU lock and disposable server.
All raw JSONs, including flags and any invalid status, are retained. No seed or NI setting is supplied.

No 35B or other-model GPU run was performed. Unrecognized dtypes retain their earlier behavior;
this does not add dtype support or full-format safetensors validation.
