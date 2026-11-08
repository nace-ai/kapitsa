# Nemotron-H FP8 native M4 cohort

This is a derived privacy-filtered export of four actual `kapitsa bench --quick` runs, ordered A1/B1/A2/B2, with each source's own Release CLI and default server under the exclusive GPU lock. It is not raw benchmark JSON. Source/binary identities, every per-run timing, output gate, return code, flag and invalid reason are retained in `native-summary.json`; original execution records remain privately verifiable.

B1 exited 1: its first temperature-0.6 sample stopped after 13 tokens instead of 512, leaving too few streamed events for a middle-window rate. A1/A2/B2 exited 0 with empty flags and invalid lists. Do not treat the four-run cohort as all passing or compute a qualified sampled speedup from it. All greedy rows remain in the export.

The native metadata for this unlisted folder reports `precision=bf16` and `revision=null`; these original values are retained. The separately verified pinned checkpoint and actual loader show FP8 Mamba/MLP weights. The output and HTTP checks are separate evidence.

The native CLI does not expose matched request seeds and uses random request nonces. A separate, predeclared fixed-request and nonzero-seed comparison is running to examine comparability. That method does not replace these four native results. No inference or complete model-load speedup is claimed.
