# Day-0 support playbook

Internal playbook for adding a new Qwen or Nemotron release to Kapitsa within 24 hours.

## 1. Convert the pack

### MLX affine (Qwen, standard path)

If the release ships an MLX safetensors repo (`mlx-community` or `lmstudio-community`), add it to the catalog directly (step 3). The catalog entry points at the HF repo; `kapitsa pull` downloads it.

For a GGUF (from LM Studio, Ollama or a Hugging Face GGUF repo), convert once:

```bash
kapitsa import /path/to/Model-Q4_K.gguf --name <short-name>
kapitsa import /path/to/Model-Q4_K.gguf --name <short-name> --mtp /path/to/mtp-head.gguf
kapitsa import /path/to/Model-Q4_K.gguf --dry-run   # show formats and size without writing
```

`kapitsa import` takes a `.gguf` file (not an HF repo). `--name` sets the folder name in `~/.cache/kapitsa/models/`.

### Flash-Next pack conversion

For a Flash-Next model (hyper-connections, n-gram table, sparse attention), three stages:

```bash
# 1. Stream experts from HF and quantize (the quality eval's --stage writes them)
python3 bench/quality/eval_qwen4_exp.py stream ... --stage /tmp/stage --dense /tmp/dense

# 2. Build the n-gram table
python3 tools/convert_qwen4_exp.py table --out /path/to/pack --bits 4 --group 32

# 3. Finalize the pack
python3 tools/convert_qwen4_exp.py finalize --stage /tmp/stage --dense /tmp/dense \
  --out /path/to/pack --recipe '4,router=8,indexer=8,embed=8'
```

This produces the MLX affine pack with the MTP head in `mtp.safetensors` and the n-gram table in `ple.safetensors`.

### Nemotron (NVFP4 / FP8)

NVIDIA ships NVFP4 and FP8 checkpoints as safetensors repos. Add them to the catalog (step 3) pointing at the HF repo directly; no conversion needed.

## 2. Attach or train the drafter

### DFlash2 drafter (Qwen models)

Check if `incoai` has a DFlash2 drafter for the new model on Hugging Face:

```bash
# Search for existing drafters
curl -s "https://huggingface.co/api/models?search=incoai+DFlash2+<model-name>" | python3 -m json.tool
```

If a drafter exists, add it to `src/hub/catalog.cpp` as a `Companion` entry (defined in `src/hub/hub.h`):

```cpp
static Companion dflash_new() {
    return {.kind = "dflash2",
            .repo = "incoai/<ModelName>-DFlash2",
            .revision = "<commit-sha>",
            .files = {"config.json", "model.safetensors"},
            .subdir = "dflash2"};
}
```

If no drafter exists yet, the model runs with prompt lookup (`--spec`) until one appears. File an issue to track it.

### Other drafter types

- **MTP** (multi-token prediction): built from the model's own `mtp.*` tensors. Added as a `mtp-sidecar` companion.
- **DSpark**: for Nemotron Lightning. Already in the catalog.
- **LiLiCorr**: for Nemotron Bolt and Super-VL. Already in the catalog.

## 3. Add to the catalog

Edit `src/hub/catalog.cpp`. Follow the existing pattern:

```cpp
{.name = "<short-name>",
 .aliases = {"<alias>"},
 .title = "<Full Model Name>",
 .about = "<one line: what it is and who it's for>",
 .brief = "<shorter version for tables>",
 .default_variant = "4bit",
 .fallbacks = {},
 .variants = {
     {.name = "4bit",
      .quant = "MLX affine 4-bit, group 64",
      .repo = "<hf-org>/<repo>",
      .revision = "<commit-sha>",
      .companions = {dflash_new()},
      .download_gb = <size>},
 }},
```

Build and run the CPU tests:

```bash
cmake --build build -j --target kapitsa
build/api_test
python3 tests/test_server.py --cli build/kapitsa
```

## 4. Run the benchmark harness

`kapitsa bench` starts its own server, so do not start one separately.

```bash
# Full benchmark (all prompt sizes, multiple runs)
kapitsa bench --model <short-name>

# Quick mode for a first look
kapitsa bench --model <short-name> --quick

# JSON output for the community results
kapitsa bench --model <short-name> --json
```

Record the numbers: prefill tok/s at 4K and 16K, decode tok/s (median of 5 runs), greedy decode tok/s.

## 5. Publish the speed table

Update these files with the new numbers:

- `docs/benchmarks.md`: add a section with the model's results
- `README.md`: add the model to the Models tables if it belongs in the poster
- `bench/results/`: commit the raw benchmark output

Format: `prefill / decode` in tok/s, with the Mac, macOS version, and commit.

```bash
# Generate the community results table
python3 tools/community_table.py
```

## 6. Ship the update

```bash
# One PR with the catalog entry, drafter, and benchmark results
git checkout -b add-<model-name>
git add src/hub/catalog.cpp docs/benchmarks.md bench/results/
git commit -m "Add <Model Name> to the catalog (<decode> tok/s on <Mac>)"
```

## 7. Post on r/LocalLLaMA

Within 24 hours of the model release. Template:

```
Title: Kapitsa runs <Model Name> at <N> tok/s on <Mac> (Apple silicon)

<Model Name> just dropped. Kapitsa has day-0 support:

    brew install nace-ai/tap/kapitsa
    kapitsa serve --model <short-name>

<N> tok/s decode on <Mac chip> (<RAM> GB), <M> tok/s prefill at 4K context.
[Benchmark numbers / comparison table]

Kapitsa is a native inference engine for Apple silicon.
GitHub: https://github.com/nace-ai/kapitsa
```

Include a comparison against other engines if `kapitsa bench compare` data is available.

## Timeline

| Hour | Task |
|------|------|
| 0 | Model drops. Start conversion. |
| 1 | Pack converted, import tested. |
| 2 | Catalog entry added, drafter attached (if available). |
| 3 | Benchmark run on at least one Mac. |
| 4 | Output check: `build/logit_dump` old vs new, `tools/logit_compare.py` gated at 2x the floor. |
| 5 | PR opened with catalog + benchmarks + logit evidence. |
| 6 | PR merged after review. |
| 8 | `kapitsa update` picks up the new model. |
| 24 | r/LocalLLaMA post with final numbers. |

## Rehearsal notes (2026-11-08)

Rehearsed steps 1 and 4 on `qwen3.5-4b` (already in the catalog), M5 Pro 48 GB:

- **Import:** model already converted (4.57 GB download, MLX affine 4-bit). A fresh `kapitsa import` of a GGUF takes under a minute for a 4B model.
- **Benchmark:** `kapitsa bench --model 4b --quick` (bench starts its own server) completed in 3 min 40 s. Results: 221 tok/s decode, 237 tok/s greedy, 3,733 tok/s prefill at 4K. Three flags (GPU contention from Edge, two repetition loops in the sampled decode).
- **Total wall time for steps 1+4:** under 5 min for a model already in the catalog. For a new model, add time for the catalog edit, drafter search, and build (~10 min).
