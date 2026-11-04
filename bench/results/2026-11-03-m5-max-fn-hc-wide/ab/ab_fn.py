#!/usr/bin/env python3
# Flash-Next decode A/B of NI_HC_WIDE (0 = old, the default = new) on one build: ni_run, the README poster's decode prompt,
# greedy, 512 tokens (--ignore-eos), the model's MTP head at depth 2 as the server's default. Rounds alternate the arm
# order (old new, new old, ...). Each run waits for a quiet GPU (no foreign process above 5% for 15 s, at most 60 s;
# bench/harness/gpu_hogs.py) and samples the foreign GPU load every 2 s while it runs. Every arm's ids are compared.
# usage: ab_fn.py NI_RUN MODEL_DIR IDS_FILE OUT_DIR ROUNDS   (AB_HARNESS: <repo>/bench/harness)
import json, os, re, subprocess, sys, time
sys.path.insert(0, os.environ["AB_HARNESS"])
import gpu_hogs  # noqa: E402

nirun, model, idsf, out, rounds = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
os.makedirs(out, exist_ok=True)
ids = open(idsf).read().strip()
res = []
for rd in range(rounds):
    for arm in (("old", "new") if rd % 2 == 0 else ("new", "old")):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "KAPITSA_"))}
        env["NI_LOCK_HELD"] = "1"
        if arm == "old":
            env["NI_HC_WIDE"] = "0"
        ok, waited, busy = gpu_hogs.wait_quiet(5.0, 15, 60)
        log = os.path.join(out, f"r{rd}-{arm}.out")
        t0 = time.time()
        with open(log, "w") as f:
            p = subprocess.Popen([nirun, model, ids, "512", "0", "--spec", "2", "--dflash", "none", "--ignore-eos"],
                                 env=env, stdout=f, stderr=subprocess.STDOUT)
            mon = gpu_hogs.ContentionMonitor(lambda: [p.pid])  # (ni_run's own GPU time is never foreign)
            mon.start()
            if p.wait() != 0:
                sys.exit(f"ni_run failed: {log}")
        mon.stop()
        load = gpu_hogs.summarize_samples(mon.samples, t0, time.time())
        txt = open(log).read()
        m = re.search(r"spec decode D=2: (\d+) tok in (\d+) ms \(([\d.]+) tok/s\), (\d+) cycles, ([\d.]+) tok/cycle.*?([\d.]+) ms GPU/cycle", txt)
        gen = re.search(r"^ids: (.*)$", txt, re.M).group(1)
        r = dict(round=rd, arm=arm, tok_s=float(m.group(3)), tok_cycle=float(m.group(5)), gpu_ms_cycle=float(m.group(6)),
                 quiet=ok, quiet_wait_s=round(waited, 1), load=load, ids=gen)
        res.append(r)
        print(f"round {rd} {arm}: {r['tok_s']} tok/s, {r['tok_cycle']} tok/cycle, {r['gpu_ms_cycle']} ms GPU/cycle | quiet {ok} "
              f"after {waited:.1f} s, foreign GPU mean {load.get('mean_foreign_gpu_pct')}% max {load.get('max_foreign_gpu_pct')}%",
              flush=True)
json.dump(res, open(os.path.join(out, "ab_fn.json"), "w"), indent=1)
same = len({r["ids"] for r in res}) == 1
print("ids identical across every run:", same)
