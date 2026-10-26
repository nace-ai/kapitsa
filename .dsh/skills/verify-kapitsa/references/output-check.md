# Output check and speed A/B, step by step

The tier 1 and 2 GPU checks of `pr-checks.md`, for one model. Run them only when the task allows loading a model,
and repeat them per model `vk checks` names. CONTRIBUTING.md ("Output check", "Speed changes") is the authority. This
file adds the order, the locks and where the evidence goes.

## 0. Before the GPU

```bash
V=.dsh/skills/verify-kapitsa/scripts/vk
$V doctor                       # binary is HEAD's, GPU lock free, record the Mac line for Runs:
$V build kapitsa logit_dump     # the new build
REF=$($V ref)                   # origin/main's build: /tmp/verify-kapitsa/ref/<sha>/build (kapitsa, logit_dump)
M=$NACE_MODELS/qwen3.5-4b-4bit-mtp
E=/tmp/verify-kapitsa/evidence/<run>   # vk doctor prints it
```

`vk ref` builds the base once per commit, in its own worktree under `/tmp/verify-kapitsa/ref/`. CONTRIBUTING calls
that build `build-main/`, but `build-main/` is not gitignored, so it is not built inside the checkout. Every session
comparing against the same commit reuses it.

## 1. Logit gate

`logit_dump` takes the GPU lock itself and prints "waiting for the GPU lock" while another process holds it. Do not wrap
it in `lockf`.

```bash
D=/tmp/verify-kapitsa/dumps/$(basename $M)          # ~190 MB per path on Qwen; not in the evidence dir
$REF/logit_dump $M $D/ref                           # the old build
build/logit_dump $M $D/new --from $D/ref            # the new build, on the reference's prefixes
python3 tools/logit_compare.py $D/ref $D/new | tee $E/logit-$(basename $M).txt; echo "gate exit ${PIPESTATUS[0]}"
rm -rf "${D:?}"                                     # keep the report, delete the dumps
```

Read the result:

- **Exit 0 and every row bitwise identical**: the PR may say "the same outputs to the bit" for this model, on this
  Mac.
- **Exit 0 with differences**: within twice the noise floor. Say "within the noise floor", never "identical".
- **Exit 1**: past the gate. The change moves outputs. Either it is a bug, or it is tier 3 and the PR says so with
  the numbers. A pure reassociation of a K reduction is the one exception, and it takes `--reorder` plus AGENTS.md's
  evidence (per-GEMM check, temperature-0 continuations).
- On an M5, a floor file without `.m5` is an M4 Pro measurement, so the gate is an approximation here. Report which
  file it used (`vk checks` prints it).
- Never edit or re-measure a floor in the same PR. `tools/logit_floor.py` re-measures one, in a PR of its own.

## 2. Speed A/B

Interleave the old and new builds on an idle GPU: A B A B, at least two of each. `kapitsa bench` takes the GPU lock
itself. Each run is 3 to 15 minutes, so start the sequence with detach.py (~/.agents/AGENTS.md) and do not wait in
a shell:

```bash
cat > $E/ab.sh <<EOS
set -e
for i in 1 2; do
  KAPITSA_NO_AUTO_UPDATE=1 $REF/kapitsa bench --model $M --quick --json --out $E/bench-A\$i.json
  KAPITSA_NO_AUTO_UPDATE=1 $PWD/build/kapitsa bench --model $M --quick --json --out $E/bench-B\$i.json
done
EOS
python3 ~/.agents/skills/session-orchestration/scripts/detach.py run --name vk-ab-$(basename $M) --cwd $PWD -- bash $E/ab.sh
```

Report every run in order: decode tok/s, 4K prefill and follow-up TTFT for A1 B1 A2 B2. Report any run flagged for
contention (another GPU app, low power, thermal state, swap) and rerun the pair. Do not drop it silently. A
difference smaller than the spread between same-build runs is "no measurable change".

## Decode legs

Any row that decodes tokens (`ni_run`-family tools, a temperature-0 continuation, a tok/s figure, a speculative-decode
leg) follows these rules. Each one comes from a failure that was booked as a result. A row that breaks one is VOID,
not a pass.

1. **Non-empty output first.** Before any md5 or ids comparison, assert the text is non-empty and print its token
   count. Two empty outputs hash to the same md5 (`d41d8cd98f00b204e9800998ecf8427e`) and "matched".
2. **Count the end tokens.** Print how many of the first N output ids are 248044 or 248046 (`<|endoftext|>`,
   `<|im_end|>`) and how many ids come before the first one. `ni_run`-family tools do not stop at `<|im_end|>`, so the
   junk after it was counted as tok/s. Use a chat-templated prompt ending in `<|im_start|>assistant\n` that gives at
   least 256 tokens before `<|im_end|>`, and measure only those.
3. **Name the drafter.** A speculative-decode row prints the drafter load line (which drafter, its path and sha), the
   tokens per cycle and the accepted draft tokens per cycle. Flash-Next ran its MTP head instead of its DFlash2
   drafter for weeks and nobody noticed.

`vk checks` prints these rules when the diff touches `src/model/`, `src/engine/spec*` or `tools/ni_run*`.

## 3. Server end to end (tier 1 and up)

```bash
$V serve $M && python3 tests/test_server.py http://127.0.0.1:8090 | tee $E/test-server.txt; $V stop
```

## 4. Metal validation (tier 2, changed kernels)

Run the changed kernel at small sizes with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`, through the tool that
exercises it (e.g. `build/batch_check`, `tools/bench_moe.cpp`), under `lockf -k -t 0 /tmp/kapitsa-gpu.lock`. On
macOS 26.5.2 the layer counts static threadgroup memory twice. A kernel above 16,384 B static aborts under the layer
whatever the change is. The PR then says the layer cannot run it and gives the size (`tools/ni_kstats --limits`).

## Finish

`$V cleanup`. Copy the reports from `$E` into the PR (logit report per model and path, every A/B run, test_server
summary) and list everything not run under "Not verified".
