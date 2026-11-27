# Completed CPU validation for the router change

This additive follow-up completes the CPU job that was still running in the initial evidence snapshot.
Production source `dc82cf0f39b942157e86b0cd5e518e5caed17e61`, Apple M3 Ultra, macOS 26.4.1 (25E253),
explicitly configured Release build; the production files are unchanged in the PR head.

- `vk cpu-tests`: 21 entries passed, 0 failed.
- Fourteen supplemental native test targets: all passed.
- Six additional CPU scripts: Drex Max CLI, profile gates, comment gates, logit-dump CLI,
  expert-pack helpers, and packaging-patch application: all passed.
- `vk cleanup`: passed. The original driver and additional CPU driver have completed.

Privacy-filtered command examples, unchanged exit codes and result outputs are in `raw-evidence.tar.gz`; `manifest.json` gives every
uncompressed member's SHA256 and `summary.json` gives the archive SHA256. These are CPU checks.
The original driver state also contains earlier GPU attempts, whose compiler-mode limitations remain
as described in the initial evidence. No new GPU or model result is claimed by this follow-up.

The measurement snapshot is retained; this declared privacy export is the sole replacement. Full-model output/HTTP/inference benchmarks and stable
performance qualification are still pending; this CPU completion does not make the PR ready to merge.

## Privacy export erratum

This archive and its checksum manifest replace the original public export to remove private machine paths, usernames and resource records. The replacement is explicitly a privacy-filtered derived export, not a verbatim raw log. Modified commands contain `$PRIVATE_PATH` or other explicit placeholders and are reproduction examples. Original archive bytes and their SHA256 remain privately preserved for audit. Timing values, output results, pass/fail records and tested source/binary identities are unchanged; no test was rerun for this export. Kernel static allocation measurements remain intact.
