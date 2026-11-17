# The 9B v2's replaced conversation on the M2 (2026-11-17)

Added beside the other READMEs. `nohit-nano-9b-m2/` (`ab_nohit.py`, this branch's code as 0c051ab9 against main
4a9cce41, both built with Apple clang 15) showed request 2's first token 2.4-2.9 s later than
on main: 18.256-18.792 s against 15.870-15.900 s. `rerun-nano-9b-m2/` (`rerun-9b.sh`, the same binaries, the same M2)
repeats it with vm_stat sampled twice a second (`vmlog.py`) and the pressure level each second (`pressure.log`):
+2.91, +2.63 and +0.29 s in rounds 1 to 3.

The M2 serves the Nano 9B v2 with about 4,000 free pages (64 MB). Before request 2's prefill the engine copies request
1's 16,800-token conversation (422 MB, under the 512 MiB it copies at once on 16 GB) to write it behind the request
(`disk_leaving`). In rounds 1 and 2 the copy set off bursts of 466K pages compressed within a second (10:44:46) and
401K (10:49:56); round 3, +0.29 s, had 35K (10:55:05). The copy's own time shows in request 2's `chat:` line as
`template/tokenize` (1,641 ms in round 1), and the prefill's `encoded by` came later. macOS's pressure level stayed 1
(normal) throughout.

It happens only when a request replaces a long conversation the SSD does not hold yet (within 10 s of its last
request, before the write-behind) on a Mac with that little free memory: the 4B on the 16 GB M4 copied its 361 MB in
77-80 ms (`nohit-nano-4b-m4/`, `head-b4e33647-n4b-m4/`).
