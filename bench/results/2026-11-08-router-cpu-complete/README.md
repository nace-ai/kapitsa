# Completed CPU validation for the router change

This additive follow-up completes the CPU job that was still running in the initial evidence snapshot.
Production source `dc82cf0f39b942157e86b0cd5e518e5caed17e61`, Apple M3 Ultra, macOS 26.4.1 (25E253),
explicitly configured Release build; the production files are unchanged in the PR head.

- `vk cpu-tests`: 21 entries passed, 0 failed.
- Fourteen supplemental native test targets: all passed.
- Six additional CPU scripts: Drex Max CLI, profile gates, comment gates, logit-dump CLI,
  expert-pack helpers, and packaging-patch application: all passed.
- `vk cleanup`: passed. The original driver and additional CPU driver have completed.

Exact commands, exit codes and full outputs are in `raw-evidence.tar.gz`; `manifest.json` gives every
uncompressed member's SHA256 and `summary.json` gives the archive SHA256. These are CPU checks.
The original driver state also contains earlier GPU attempts, whose compiler-mode limitations remain
as described in the initial evidence. No new GPU or model result is claimed by this follow-up.

The initial evidence folder remains frozen. Full-model output/HTTP/inference benchmarks and stable
performance qualification are still pending; this CPU completion does not make the PR ready to merge.
