# Additional CPU evidence

These are explicitly redacted derived summaries, not raw execution logs.
M3.json adds the second own-source CPU setup: M3 Ultra (60 GPU cores,
96 GiB), macOS 26.4.1, Apple clang 17 (clang-1700.6.4.2).
Candidate and reference were explicitly configured and built independently.
The candidate integrated oracle passed 450,762 cases and the guarded blocks.
All eight trained-row legs passed the complete-byte, finite, input and canary
checks described in README.md; every per-pass timing is retained in M3.json.
Adjacent-pair median latency decreased 54.60–54.84%. This is CPU row decoding,
not full import or inference acceleration. Cross-chip absolute times differ;
the cause of that difference was not established.

UPDATE-CONTROLS.json records an unchanged-reference updater test and a separate
candidate recheck, both passing. The original CPU suite remains 21 PASS / 1 FAIL
in CHECKS.json. Its one empty-version observation is retained and unexplained;
the entire suite was not repeated. GPU and full-model qualification remains
outstanding for this Draft.
