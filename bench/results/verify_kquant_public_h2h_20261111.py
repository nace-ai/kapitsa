#!/usr/bin/env python3
"""Replay task #469's statistics from saved window sums, without a GPU or model.

Run: uv run --with numpy python bench/results/verify_kquant_public_h2h_20261111.py
The JSON beside this script also contains candidate revisions and input hashes.
"""
import json
import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).with_name("kquant-public-h2h-20261111.json")


def check_value(row, key, actual):
    assert math.isclose(row[key], float(actual), rel_tol=0, abs_tol=1e-12), (
        key, row[key], actual
    )


def check_interval(row, prefix, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    check_value(row, prefix + "_lo", lo)
    check_value(row, prefix + "_hi", hi)


def bootstrap(sums, counts, indices):
    return sums[indices].sum(1) / counts[indices].sum(1)


def accuracy_pairs(candidate, reference):
    a = {item["id"]: item for item in candidate}
    b = {item["id"]: item for item in reference}
    assert a.keys() == b.keys()
    wins = losses = 0
    for ident in a:
        assert a[ident]["gold"] == b[ident]["gold"]
        wins += a[ident]["correct"] and not b[ident]["correct"]
        losses += b[ident]["correct"] and not a[ident]["correct"]
    discordant = wins + losses
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant)
    return wins, losses, p


def main():
    data = json.loads(DATA.read_text())
    groups = data["results"]
    assert set(groups) == {"q35-9b.k436", "q35-9b.chat", "q35-4b.k436", "q35-4b.chat", "mlx.k436"}
    assert data["gsm8k"]["gguf_data_sha256"] == data["gsm8k"]["mlx_data_sha256"]
    assert sum(map(len, groups.values())) == 28
    scope = json.loads(DATA.with_name("kquant-public-h2h-20261111-mlx-byte-scope.json").read_text())
    assert scope["rows"].keys() == groups["mlx.k436"].keys()
    for name, audit in scope["rows"].items():
        row = groups["mlx.k436"][name]
        assert (audit["repo"], audit["revision"]) == (row["repo"], row["repo_sha"])
        assert audit["legacy_matches"] is True
        assert audit["frozen_lm_bytes"] == row["lm_bytes"]
        assert audit["frozen_total_bytes"] == row["total_bytes"] == audit["root_whole_safetensor_bytes"]
        assert audit["root_base_payload_bytes"] + audit["root_mtp_payload_bytes"] == row["lm_bytes"]
        assert sum(audit[k] for k in ("root_base_payload_bytes", "root_mtp_payload_bytes",
                   "root_vision_payload_bytes", "root_file_overhead_bytes")) == row["total_bytes"]
        assert audit["root_whole_safetensor_bytes"] + audit["nested_whole_safetensor_bytes"] == audit["all_whole_safetensor_bytes"]
    standard = ("kquant", "AtomicChat", "lmstudio-community", "mlx-community-4bit", "mlx-community-MLX-4bit")
    assert all(scope["rows"][name]["root_base_payload_bytes"] == 5038041600 for name in standard)
    assert all(scope["rows"][name]["root_mtp_payload_bytes"] == 0 for name in standard)
    print("PASS: eight frozen MLX byte counts, component sums and five equal base-LM payload sizes")
    item_count = 0
    all_gold = None
    for group, rows in groups.items():
        assert len(rows) == (8 if group.startswith("mlx") else 5)
        first = rows["kquant"]
        windows = len(first["kl_chunk_sums"])
        per = first["chunk_tokens"]
        assert (windows, per) == ((65, 255) if group.endswith(".k436") and not group.startswith("mlx") else (64, 511))
        counts = np.full(windows, per, dtype=np.float64)
        draws = data["bootstrap"]["replicates"]
        seed = data["bootstrap"]["seed"]
        indices = np.random.default_rng(seed).integers(0, windows, size=(draws, windows))
        reference_kl = np.array(first["kl_chunk_sums"])
        reference_top = np.array(first["top1_chunk_sums"])
        for name, row in rows.items():
            kl = np.array(row["kl_chunk_sums"])
            top = np.array(row["top1_chunk_sums"])
            assert len(kl) == len(top) == windows
            assert row["chunk_tokens"] == per and row["tokens"] == windows * per
            assert np.isfinite(kl).all() and np.isfinite(top).all()
            assert ((0 <= top) & (top <= per) & (top == np.floor(top))).all()
            check_value(row, "kl", kl.sum() / counts.sum())
            check_value(row, "top1", top.sum() / counts.sum() * 100)
            check_interval(row, "kl", bootstrap(kl, counts, indices))
            check_interval(row, "top1", bootstrap(top, counts, indices) * 100)
            if name != "kquant":
                delta = kl - reference_kl
                delta_top = top - reference_top
                check_value(row, "dkl", delta.sum() / counts.sum())
                check_value(row, "dtop", delta_top.sum() / counts.sum() * 100)
                check_interval(row, "dkl", bootstrap(delta, counts, indices))
                check_interval(row, "dtop", bootstrap(delta_top, counts, indices) * 100)
                starts = np.arange(0, windows, 4)
                block_indices = np.random.default_rng(seed).integers(0, len(starts), size=(draws, len(starts)))
                check_interval(row, "dkl_block4", bootstrap(np.add.reduceat(delta, starts), np.add.reduceat(counts, starts), block_indices))
            if group.endswith(".chat"):
                tool_mask = (np.arange(windows) // 4) % 2 == 0
                for label, mask in (("tool", tool_mask), ("general_chat", ~tool_mask)):
                    check_value(row["by_source"][label], "kl", kl[mask].sum() / counts[mask].sum())
                    check_value(row["by_source"][label], "top1", top[mask].sum() / counts[mask].sum() * 100)
            else:
                items = row["gsm8k_items"]
                assert len(items) == row["gsm8k_n"] == 100
                gold = {it["id"]: it["gold"] for it in items}
                assert len(gold) == 100
                all_gold = gold if all_gold is None else all_gold
                assert gold == all_gold
                assert all(it["correct"] == (it["pred"] is not None and abs(it["pred"] - it["gold"]) < 1e-6) for it in items)
                check_value(row, "gsm8k", sum(it["correct"] for it in items) / 100)
                item_count += len(items)
                if name != "kquant":
                    wins, losses, p = accuracy_pairs(items, first["gsm8k_items"])
                    print(f"GSM8K {group} {name}: {wins} wins / {losses} losses vs KQuant, exact paired p={p:.4f}")
        if not group.startswith("mlx"):
            model, corpus = group.split(".")
            assert first["tokens"] == data["reference"][model][corpus]["scored_tokens"]
    for size, pair in data["released_blob_identity"].items():
        assert pair[0]["files"][0]["lfs_sha256"] == pair[1]["files"][0]["lfs_sha256"]
        row = groups[f"q35-{size}.k436"]["kquant"]
        assert pair[0]["repo_sha"] == row["evaluated_staging_revision"]
        assert pair[1]["repo_sha"] == row["repo_sha"]
        assert row["file_bytes"] == pair[1]["files"][0]["bytes"]
    assert item_count == 1800
    print("PASS: 28 metric rows, all window CIs and paired differences, source splits, 1,800 accuracy items and released blob identities")


if __name__ == "__main__":
    main()
