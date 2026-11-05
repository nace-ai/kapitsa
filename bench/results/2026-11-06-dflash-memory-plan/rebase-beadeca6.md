# Rebase after the 9B packed path merged

Upstream `beadeca6` merged #973 after this PR opened. The rebased candidate is `f7b1727c` (the subsequent commit only adds these records). The plan retains the new 9B down-only packing: it does not charge q/kv/o, context, fc, the reduced head or lrA packed images to the 9B. Its ordinary low-rank factors and widened preparation scratch still follow the actual loader. The earlier records in this directory remain historical measurements at their named revisions.

Full `vk cpu-tests`, supplemental CPU tools and `prek` were rerun on M3 Ultra. Against the newly built upstream baseline, the 4B logit gate remains 584/584 bitwise; the real rank-768 drafter comparison remains 270/270 cases, 35,198,376 bytes identical. Both remote rank0/rank768 allocation checks pass.

Local M4 Pro actual allocations were repeated. 4B is unchanged: 1,026,555,904 / 937,426,944 bytes at rank0 / rank768 versus the 1,082,441,728-byte bound. New 9B coverage: 1,167,245,312 actual / 1,184,599,808 planned at rank0; 1,211,940,864 actual / 1,229,279,360 planned at rank768. All four pass. The new records retain commands, source revisions, binary hashes and complete logs. No new inference-speed claim is made; the earlier 4B benchmark remains labeled with its original revisions.
