#!/usr/bin/env python3
"""End-to-end checks against a running `kapitsa serve` (stdlib only): the chat page, OpenAI chat completions, Anthropic
Messages and OpenAI Responses (tool loops, the cache across them, errors).
usage: test_server.py [base_url] [--api-key KEY] [--model-id ID]   (default http://127.0.0.1:8090)
       test_server.py --cli <path to kapitsa>    offline checks of the command line: model names and aliases,
                                                 `pull --dry-run`, `list`, `rm`, early port errors (no server)"""
import json, os, re, socket, subprocess, sys, tempfile, time, urllib.error, urllib.request

ARGS = sys.argv[1:]
def opt(name):
    if name in ARGS:
        i = ARGS.index(name)
        v = ARGS[i + 1]
        del ARGS[i:i + 2]
        return v
    return None
CLI = opt("--cli")
API_KEY = opt("--api-key")
MODEL_ID = opt("--model-id")
BASE = ARGS[0] if ARGS else "http://127.0.0.1:8090"
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from disk.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path"},
            "max_lines": {"type": "integer", "description": "Optional line limit"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

# A follow-up after a tool call reuses its whole first prompt, less at most the generation prompt's tokens: when the
# model's own tool-call text differs from the template's rendering of it (Qwen3.5-4B), a hybrid model's recurrent
# state resumes from the checkpoint at the end of the user's turn, which is just before the generation prompt.
GEN_TAIL = 8
hosts = "127.0.0.1 localhost\n255.255.255.255 broadcasthost\n::1 localhost\n"  # /etc/hosts, as a tool returns it


def hosts_call(name, args):
    """A call that reads /etc/hosts: read_file on the path, or run_shell on a command naming it (Nemotron 3.5
    Lightning counts the lines with `wc -l < /etc/hosts`; either is a tool call the server must return)."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return False
    return (name == "read_file" and args.get("path") == "/etc/hosts") or \
        (name == "run_shell" and "/etc/hosts" in str(args.get("command", "")))


hosts = "127.0.0.1 localhost\n255.255.255.255 broadcasthost\n::1 localhost\n"


def hosts_result(name, args=None):
    """What a call on /etc/hosts returns: the file for read_file, `wc -l`'s output through run_shell."""
    if name != "run_shell":
        return hosts
    cmd = str((json.loads(args) if isinstance(args, str) else (args or {})).get("command", ""))
    return "3\n" if "<" in cmd else "3 /etc/hosts\n"


def auth():
    return {"Authorization": "Bearer " + API_KEY} if API_KEY else {}

def get(path, headers=None):
    """(status, headers, body) of a GET; HTTP errors are returned, not raised."""
    req = urllib.request.Request(BASE + path, headers=headers or {})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()

def post(body, stream=False):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **auth()})
    r = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.loads(r.read())
    out = {"reasoning": "", "content": "", "tool_calls": [], "finish": None, "usage": None, "chunks": 0,
           "t_first": None}
    t0 = time.time()
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        c = json.loads(line[6:])
        out["chunks"] += 1
        if "error" in c:
            raise RuntimeError(c["error"])
        ch = c["choices"][0]
        d = ch["delta"]
        if (d.get("reasoning_content") or d.get("content")) and out["t_first"] is None:
            out["t_first"] = time.time() - t0
        out["reasoning"] += d.get("reasoning_content") or ""
        out["content"] += d.get("content") or ""
        out["tool_calls"] += d.get("tool_calls") or []
        if ch.get("finish_reason"):
            out["finish"] = ch["finish_reason"]
            out["usage"] = c.get("usage")
    return out

def chat_stream(body, out=None):
    """(status, [(t, data), ...]) of a streamed chat completion: when each data line came (s) and its JSON ("[DONE]"),
    appended to `out` as they come; an HTTP error is (status, its JSON body)"""
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Content-Type": "application/json", **auth()})
    t0, out = time.time(), [] if out is None else out
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: "):
            out.append((time.time() - t0, "[DONE]" if line == "data: [DONE]" else json.loads(line[6:])))
    return r.status, out

def choice(d):
    """a chunk's first choice ({} for an error event or [DONE])"""
    return (d.get("choices") or [{}])[0] if isinstance(d, dict) else {}

fails = 0
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + detail))
    if not ok:
        fails += 1

# The app and the CLI download the same pinned files for the 4-bit variants (src/hub/catalog.cpp), which are the
# defaults.
DEFAULTS = {"qwen3.6-35b-a3b": ("4bit", "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed", "95460516fbfe34c48f577f67c9ed191f5d61fbaf"),
            "qwen3.8-27b": ("4bit", "lmstudio-community/Qwen3.8-27B-MLX-4bit", "6067b15cf581666a4aecf6af3afaba4bb5efc20c"),
            "qwen3.5-4b": ("4bit", "lmstudio-community/Qwen3.5-4B-MLX-4bit", "c43ee1d65576a5d98de1e8405cac93c371a655c1"),
            "qwen3.5-9b": ("4bit", "lmstudio-community/Qwen3.5-9B-MLX-4bit", "b455506b0f574c74616dbcd56879bde38fafcff3")}
PINNED = {
    "qwen3.6-35b-a3b": ("Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed", "95460516fbfe34c48f577f67c9ed191f5d61fbaf",
                        [("dflash2", "incoai/Qwen3.6-35B-A3B-DFlash2", "51ef7b6923ad6c14cb1bb41c37a9041446496ab6", "dflash2")]),
    "qwen3.8-27b": ("lmstudio-community/Qwen3.8-27B-MLX-4bit", "6067b15cf581666a4aecf6af3afaba4bb5efc20c",
                    [("mtp", "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed", "1d5087d2062c02b279180a53e4016cf9cd7a3d7e", "."),
                     ("dflash2", "incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5", "dflash2")]),
    "qwen3.5-4b": ("lmstudio-community/Qwen3.5-4B-MLX-4bit", "c43ee1d65576a5d98de1e8405cac93c371a655c1",
                   [("mtp-sidecar", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "."),
                    ("dflash", "z-lab/Qwen3.5-4B-DFlash", "9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf", "dflash2")]),
    "qwen3.5-9b": ("lmstudio-community/Qwen3.5-9B-MLX-4bit", "b455506b0f574c74616dbcd56879bde38fafcff3",
                   [("mtp-sidecar", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "."),
                    ("dflash", "z-lab/Qwen3.5-9B-DFlash", "5fc3b3d474760f18c516db87d84c37edbfd3ede6", "dflash2")]),
}
# the 35B's 16gb variant: repo, revision, file, sha256, size
GGUF16 = ("unsloth/Qwen3.6-35B-A3B-GGUF", "a483e9e6cbd595906af30beda3187c2663a1118c", "Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf",
          "2e8f5f705355c56311432d0a8a5d14a696dbb7e4b197d05c75ba805fc1857bef", 10756586464)
# Flint's 16gb variant
GGUFF = ("unsloth/flint-GGUF", "20468edfc38ab43d6914cc88b08cbecb05d0fc62", "flint-UD-IQ2_XXS.gguf",
         "c1fa4a6d5ec67fbeff460c68829984e4aa3eea95dc9d2067691f19e5406deeee", 11855541696)

def cli_checks(kapitsa):
    home = tempfile.mkdtemp(prefix="kapitsa-test-")
    # a closed port as the Hub: any network access in these checks fails loudly
    env = dict(os.environ, KAPITSA_HOME=home, HF_ENDPOINT="http://127.0.0.1:9", NI_LOG="0")
    def run(*a, env_add=None):
        p = subprocess.run([kapitsa, *a], env=dict(env, **(env_add or {})), capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout, p.stderr
    def plan(name):
        rc, out, err = run("pull", "--dry-run", "--json", name)
        return json.loads(out) if rc == 0 else {"rc": rc, "err": err}
    models = os.path.join(home, "models")
    rc, out, _ = run("list", "--json")
    check("cli: list --json on an empty store", rc == 0 and json.loads(out) == [], out)
    for name, aliases in [("qwen3.6-35b-a3b", ["35b:4bit", "QWEN3.6-35B-A3B:4bit",
                                               "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"]),
                          ("qwen3.8-27b", ["27b:4bit", "27B:4BIT", "lmstudio-community/Qwen3.8-27B-MLX-4bit",
                                           "hf.co/lmstudio-community/Qwen3.8-27B-MLX-4bit"]),
                          ("qwen3.5-4b", ["4b:4bit", "qwen3.5-4b:4bit",
                                          "lmstudio-community/Qwen3.5-4B-MLX-4bit@c43ee1d65576a5d98de1e8405cac93c371a655c1"]),
                          ("qwen3.5-9b", ["9b:4bit", "QWEN3.5-9B:4bit", "lmstudio-community/Qwen3.5-9B-MLX-4bit"])]:
        repo, rev, comps = PINNED[name]
        dv = DEFAULTS[name][0]
        for a in aliases:
            j = plan(a)
            got = [(c["kind"], c["repo"], c["revision"], c["into"]) for c in j.get("companions", [])]
            check(f"cli: {a} -> {name}:4bit at the app's pinned revisions",
                  j.get("model") == name + ":4bit" and j.get("id") == (name if dv == "4bit" else name + ":4bit") and
                  j.get("kind") == "catalog" and j.get("variant") == "4bit" and
                  j.get("dir") == os.path.join(models, name + "-4bit") and j.get("repo") == repo and
                  j.get("revision") == rev and got == comps and j.get("downloaded") is False, json.dumps(j))
    # a name without a variant: the default (offline nothing can be planned, so no fallback for a smaller Mac)
    for name, aliases in [("qwen3.6-35b-a3b", ["35b", "qwen3.6-35b", "QWEN3.6-35B-A3B"]), ("qwen3.8-27b", ["27b"]),
                          ("qwen3.5-4b", ["4b", "qwen3.5-4b"]), ("qwen3.5-9b", ["9b", "qwen3.5-9b"])]:
        dv, repo, rev = DEFAULTS[name]
        for a in aliases:
            j = plan(a)
            check(f"cli: {a} -> {name}:{dv}, the default",
                  j.get("model") == name + ":" + dv and j.get("id") == name and j.get("variant") == dv and
                  j.get("repo") == repo and j.get("revision") == rev and
                  j.get("dir") == os.path.join(models, name + "-" + dv), json.dumps(j))
    # NVIDIA's Nemotron 3.5 checkpoints: one variant each, the LiLiCorr (Bolt, Super-VL) or DSpark (Lightning) drafter in dflash/
    for name, dv, repo, rev, comp, aliases in [
            ("nemotron-3.5-bolt", "24gb", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
             "c6cc2e8ffa353403213d6b9b18aaf8eb684d705a",
             ("lilicorr", "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr", "abd478065678ee83a93821c4b3f512da24e4cecb",
              "dflash"), ["bolt", "nemotron-bolt"]),
            ("nemotron-3.5-super-vl", "nvfp4", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
             "59b8721646c38111a242da18218859fb2b454322",
             ("lilicorr", "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr",
              "292bf1efc06ba39567da7351f2ad7a522afa3959", "dflash"),
             ["super", "super-vl", "nemotron-super", "NEMOTRON-3.5-SUPER-VL:nvfp4",
              "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"]),
            ("nemotron-3.5-lightning", "nvfp4", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
             "bee7596271d1495f6992ae224aefde4410e816b8",
             ("dspark", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark",
              "6d127b50ecd98562cea85d0087680b7d86091fb0", "dflash"),
             ["lightning", "nemotron-lightning", "nemotron-3.5-lightning:nvfp4",
              "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"])]:
        for a in aliases:
            j = plan(a)
            got = [(c["kind"], c["repo"], c["revision"], c["into"]) for c in j.get("companions", [])]
            check(f"cli: {a} -> {name}:{dv} with its {comp[0]} drafter",
                  j.get("model") == name + ":" + dv and j.get("id") == name and j.get("kind") == "catalog" and
                  j.get("repo") == repo and j.get("revision") == rev and got == [comp] and
                  j.get("dir") == os.path.join(models, name + "-" + dv), json.dumps(j))
    # NVIDIA's Dory: NVFP4 (default) and BF16, no drafter
    for a, dv, repo, rev in [
            ("dory", "nvfp4", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4", "41f02335e1a1ad3289b8b32485e2265cbe97cfdf"),
            ("nemotron-labs-dory1.0-4b", "nvfp4", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4",
             "41f02335e1a1ad3289b8b32485e2265cbe97cfdf"),
            ("dory:bf16", "bf16", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "b366b87f762ea334ba3eb55912031baba1515810"),
            ("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16", "bf16", "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
             "b366b87f762ea334ba3eb55912031baba1515810")]:
        name = "nemotron-labs-dory1.0-4b"
        j = plan(a)
        check(f"cli: {a} -> {name}:{dv}, no drafter",
              j.get("model") == name + ":" + dv and j.get("kind") == "catalog" and j.get("repo") == repo and
              j.get("revision") == rev and j.get("companions") == [] and
              j.get("id") == (name if dv == "nvfp4" else name + ":" + dv) and
              j.get("dir") == os.path.join(models, name + "-" + dv), json.dumps(j))
    # Recursive's Flint: its MLX 4-bit export, no drafter (plain decoding; prompt lookup with --spec); its repo by name
    # resolves to the catalog's
    for a in ["flint", "recursive-flint", "flint:4bit", "RecursiveSI/flint-int4"]:
        name = "flint"
        j = plan(a)
        check(f"cli: {a} -> {name}:4bit, no drafter",
              j.get("model") == name + ":4bit" and j.get("kind") == "catalog" and j.get("repo") == "RecursiveSI/flint-int4"
              and j.get("revision") == "3b93f9053b530488405ae9fca51d4900d6ce21f0" and j.get("companions") == [] and
              j.get("id") == name and j.get("dir") == os.path.join(models, name + "-4bit"), json.dumps(j))
    # NVIDIA's Nemotron Nano: the 4B in FP8 (default) and BF16, the 9B v2 in NVFP4; no drafter (prompt lookup)
    nano4, nano9 = "nemotron-3-nano-4b", "nemotron-nano-9b-v2"
    for a, name, dv, default, repo, rev in [
            ("nano-4b", nano4, "fp8", True, "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8", "3fe6dab75665a93884214ad4b1b95cf02717d081"),
            ("nemotron-nano-4b", nano4, "fp8", True, "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8",
             "3fe6dab75665a93884214ad4b1b95cf02717d081"),
            ("nano-4b:bf16", nano4, "bf16", False, "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"),
            ("nano-9b", nano9, "nvfp4", True, "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4", "8556c9164ddb43fe1f4f4ad730593b3c5e3f7328"),
            ("nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4", nano9, "nvfp4", True, "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4",
             "8556c9164ddb43fe1f4f4ad730593b3c5e3f7328")]:
        j = plan(a)
        check(f"cli: {a} -> {name}:{dv}, no drafter",
              j.get("model") == name + ":" + dv and j.get("kind") == "catalog" and j.get("repo") == repo and
              j.get("revision") == rev and j.get("companions") == [] and
              j.get("id") == (name if default else name + ":" + dv) and
              j.get("dir") == os.path.join(models, name + "-" + dv), json.dumps(j))
    # the 35B's 16gb: one pinned file of unsloth's GGUF repo, imported after its download; no drafter (offline here,
    # so the import is not planned: its size on disk is unknown)
    for a in ["35b:16gb", "QWEN3.6-35B-A3B:16GB", "unsloth/Qwen3.6-35B-A3B-GGUF"]:
        j = plan(a)
        check(f"cli: {a} -> qwen3.6-35b-a3b:16gb, one pinned GGUF file to import, no drafter",
              j.get("model") == "qwen3.6-35b-a3b:16gb" and j.get("id") == "qwen3.6-35b-a3b:16gb" and
              j.get("kind") == "catalog" and j.get("variant") == "16gb" and
              (j.get("repo"), j.get("revision"), j.get("file"), j.get("sha256"), j.get("download_bytes")) == GGUF16 and
              j.get("imported_bytes") is None and j.get("companions") == [] and j.get("downloaded") is False and
              j.get("dir") == os.path.join(models, "qwen3.6-35b-a3b-16gb"), json.dumps(j))
    # no file list to fetch: the disk is checked first, with 6 GB kept for the import, and nothing is left behind
    rc, _, err = run("pull", "35b:16gb", env_add={"KAPITSA_TEST_DISK_TAKEN": str(1 << 60)})
    check("cli: 35b:16gb checks the disk before its download, keeping 6 GB for the import",
          rc == 1 and "Not enough disk space for qwen3.6-35b-a3b:16gb: it needs 10.76 GB more" in err and
          "keeping 6 GB spare" in err and not os.path.exists(os.path.join(models, "qwen3.6-35b-a3b-16gb")), err)
    # `35b` takes 16gb where the 4-bit default does not fit. Both "downloaded" here, as far as a memory plan reads them:
    # the 35B's config and one safetensors header each, of their weights' sizes (19.5 and 11.0 GB); NI_MEM_BUDGET is a
    # 16 GB Mac's budget, then a 48 GB one's
    cfg35 = {"model_type": "qwen3_5_moe", "text_config": {
        "hidden_size": 2048, "num_hidden_layers": 40, "vocab_size": 248320, "linear_num_key_heads": 16,
        "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "num_attention_heads": 16,
        "num_key_value_heads": 2, "head_dim": 256, "num_experts": 256, "num_experts_per_tok": 8,
        "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512, "max_position_embeddings": 262144,
        "eos_token_id": 248046}}
    def fake_download(variant, weights, manifest):
        d = os.path.join(models, "qwen3.6-35b-a3b-" + variant)
        os.makedirs(os.path.join(d, ".kapitsa"))
        hdr = json.dumps({"model.language_model.layers.0.mlp.switch_mlp.gate_proj.weight":
                          {"dtype": "U8", "shape": [weights], "data_offsets": [0, weights]}}).encode()
        files = {"config.json": json.dumps(cfg35).encode(), "model.safetensors": len(hdr).to_bytes(8, "little") + hdr}
        for n, b in files.items():
            with open(os.path.join(d, n), "wb") as f:
                f.write(b)
        manifest.update(name="qwen3.6-35b-a3b:" + variant, variant=variant, commit=manifest["revision"],
                        files=[{"path": n, "size": len(b)} for n, b in files.items()],
                        total_bytes=sum(len(b) for b in files.values()))
        with open(os.path.join(d, ".kapitsa", "manifest.json"), "w") as f:
            json.dump(manifest, f)
    repo, rev, comps = PINNED["qwen3.6-35b-a3b"]
    fake_download("4bit", 19_500_000_000, {"repo": repo, "revision": rev, "companions": [
        {"kind": k, "repo": r, "revision": v, "files": []} for k, r, v, _ in comps]})
    fake_download("16gb", 10_990_000_000, {"repo": GGUF16[0], "revision": GGUF16[1], "companions": [],
                                           "gguf": {"path": GGUF16[2], "sha256": GGUF16[3], "size": GGUF16[4]}})
    for budget, want in [("12.2", "16gb"), ("48", "4bit")]:
        rc, out, err = run("pull", "--dry-run", "--json", "35b", env_add={"NI_MEM_BUDGET": budget})
        j = json.loads(out) if rc == 0 else {"rc": rc, "err": err}
        check(f"cli: 35b on a {budget} GB budget -> 35b:{want}",
              j.get("model") == "qwen3.6-35b-a3b:" + want and j.get("downloaded") is True, json.dumps(j))
    j = plan("35b:16gb")
    check("cli: a downloaded 16gb says its size on disk", j.get("imported_bytes", 0) > 0 and j.get("downloaded") is True,
          json.dumps(j))
    rc, out, err = run("pull", "35b:16gb")
    check("cli: a finished 16gb is reused offline", rc == 0 and "already downloaded" in out, out + err)
    rc, out, _ = run("list", "--json")
    lst = {m["name"]: m for m in json.loads(out)} if rc == 0 else {}
    check("cli: list shows 16gb complete, with its variant",
          lst.get("qwen3.6-35b-a3b:16gb", {}).get("complete") is True and lst["qwen3.6-35b-a3b:16gb"]["variant"] == "16gb", out)
    for a in ["35b:16gb", "35b:4bit"]:
        rc, out, err = run("rm", a)
        check(f"cli: rm {a} deletes it", rc == 0 and "Deleted qwen3.6-35b-a3b:" + a.split(":")[1] in out, out + err)
    # Flint's 16gb: unsloth's UD-IQ2_XXS GGUF, imported after its download; `flint` takes it where the 4-bit does not
    # fit (planned from a Flint config and a header of each variant's weights: 22.6 and 11.7 GB)
    for a in ["flint:16gb", "FLINT:16GB", "unsloth/flint-GGUF"]:
        j = plan(a)
        check(f"cli: {a} -> flint:16gb, one pinned GGUF file to import, no drafter",
              j.get("model") == "flint:16gb" and j.get("id") == "flint:16gb" and j.get("kind") == "catalog" and
              j.get("variant") == "16gb" and
              (j.get("repo"), j.get("revision"), j.get("file"), j.get("sha256"), j.get("download_bytes")) == GGUFF and
              j.get("companions") == [] and j.get("dir") == os.path.join(models, "flint-16gb"), json.dumps(j))
    full = [0, 1] + list(range(5, 48, 4)) + [48]
    cfgf = {"model_type": "flint", "hidden_size": 2048, "num_hidden_layers": 49, "vocab_size": 150144,
            "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128, "intermediate_size": 6144,
            "moe_intermediate_size": 512, "shared_expert_intermediate_size": 1024, "num_experts": 256,
            "num_experts_per_tok": 6, "sliding_window": 513, "mlp_only_layers": [0, 1], "no_rope_layer_indices": full,
            "layer_types": ["full_attention" if i in full else "sliding_attention" for i in range(49)],
            "eos_token_id": [150001], "bos_token_id": 150000, "quantization": {"bits": 4, "group_size": 64}}
    def fake_flint(variant, weights, manifest):
        d = os.path.join(models, "flint-" + variant)
        os.makedirs(os.path.join(d, ".kapitsa"))
        hdr = json.dumps({"model.layers.2.mlp.switch_mlp.gate_proj.weight":
                          {"dtype": "U8", "shape": [weights], "data_offsets": [0, weights]}}).encode()
        files = {"config.json": json.dumps(cfgf).encode(), "model.safetensors": len(hdr).to_bytes(8, "little") + hdr}
        for n, b in files.items():
            with open(os.path.join(d, n), "wb") as f:
                f.write(b)
        manifest.update(name="flint:" + variant, variant=variant, commit=manifest["revision"], companions=[],
                        files=[{"path": n, "size": len(b)} for n, b in files.items()],
                        total_bytes=sum(len(b) for b in files.values()))
        with open(os.path.join(d, ".kapitsa", "manifest.json"), "w") as f:
            json.dump(manifest, f)
    fake_flint("4bit", 22_570_000_000, {"repo": "RecursiveSI/flint-int4",
                                        "revision": "3b93f9053b530488405ae9fca51d4900d6ce21f0"})
    fake_flint("16gb", 11_690_000_000, {"repo": GGUFF[0], "revision": GGUFF[1],
                                        "gguf": {"path": GGUFF[2], "sha256": GGUFF[3], "size": GGUFF[4]}})
    for budget, want in [("12.2", "16gb"), ("48", "4bit")]:
        rc, out, err = run("pull", "--dry-run", "--json", "flint", env_add={"NI_MEM_BUDGET": budget})
        j = json.loads(out) if rc == 0 else {"rc": rc, "err": err}
        check(f"cli: flint on a {budget} GB budget -> flint:{want}",
              j.get("model") == "flint:" + want and j.get("downloaded") is True, json.dumps(j))
    # the small start (src/engine/engine.cpp, fits_small): on a 12.2 GB budget the 16gb holds an 8K conversation only
    # with one slot and prefill chunks below 1024 rows, its default context 8K, and the catalog check counts it as
    # fitting there; on 16 GB the usual rules (two slots or more, 1024 rows or more, 32K or more) are unchanged.
    # doctor plans every catalog model (about 40 s against the unreachable hub), so both run at once
    procs = {b: subprocess.Popen([kapitsa, "doctor", "--json", "--model", "flint:16gb", "--context", "8k"],
                                 env=dict(env, NI_MEM_BUDGET=b), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True) for b in ["12.2", "16"]}
    for budget, p in procs.items():
        out, err = p.communicate(timeout=300)
        try:
            j = json.loads(out)
            m, cv = j["models"][0], {c["variant"]: c for c in j["catalog_variants"] if c["model"] == "flint"}
        except (ValueError, KeyError, IndexError):
            check(f"cli: doctor on a {budget} GB budget", False, out + err)
            continue
        ch = m.get("prefill_chunk", 0)
        if budget == "12.2":
            ok = (m.get("starts") is True and m.get("slots") == 1 and 256 <= ch < 1024 and ch % 64 == 0 and
                  m.get("default_context") == 8192 and cv["16gb"]["fits"] is True and cv["16gb"]["chosen"] is True)
            what = "flint:16gb starts with 1 slot, a chunk below 1024 rows and 8K, and fits an 8K conversation"
        else:
            ok = m.get("slots", 0) >= 2 and ch >= 1024 and m.get("default_context", 0) >= 32768
            what = "flint:16gb keeps the usual start (2+ slots, 1024+ rows, 32K+)"
        check(f"cli: doctor on a {budget} GB budget: {what}", ok, json.dumps({k: m.get(k) for k in (
            "starts", "slots", "prefill_chunk", "default_context")}) + " " + json.dumps(cv.get("16gb")))
    for a in ["flint:16gb", "flint:4bit"]:
        rc, out, err = run("rm", a)
        check(f"cli: rm {a} deletes it", rc == 0 and "Deleted flint:" + a.split(":")[1] in out, out + err)
    rc, _, err = run("pull", "--dry-run", "4b:9bit")
    check("cli: an unknown variant is refused with the model's variants",
          rc == 2 and "has no variant" in err and DEFAULTS["qwen3.5-4b"][0] + " (the default)" in err, err)
    j = plan("mlx-community/Qwen3.5-9B-4bit@main")
    check("cli: org/name@rev is a Hugging Face repo",
          j.get("kind") == "repo" and j.get("id") == "mlx-community/Qwen3.5-9B-4bit" and j.get("revision") == "main" and
          j.get("dir") == os.path.join(models, "mlx-community--Qwen3.5-9B-4bit@main") and j.get("companions") == [], json.dumps(j))
    j = plan("https://huggingface.co/mlx-community/Qwen3.5-9B-4bit")
    check("cli: a huggingface.co URL is a repo", j.get("kind") == "repo" and j.get("revision") == "main", json.dumps(j))
    folder = os.path.join(home, "my-model")
    os.makedirs(folder)
    with open(os.path.join(folder, "config.json"), "w") as f:
        f.write("{}")
    j = plan(folder)
    check("cli: a folder is used as is", j.get("kind") == "local" and j.get("dir") == folder and j.get("id") == "my-model", json.dumps(j))
    for bad in ["llama-70b", "./no-such-folder", "org/"]:
        rc, _, err = run("pull", "--dry-run", bad)
        check(f"cli: {bad!r} is refused with the choices", rc == 2 and ("Unknown model" in err or "No folder" in err) and "35b" in err, err)
    rc, _, err = run("pull", "some-org/some-model")
    check("cli: a failed first pull leaves no folder behind",
          rc == 1 and not os.path.exists(os.path.join(models, "some-org--some-model")), err)
    # a finished download (manifest written last) is reused without any network access, listed, and removed
    m4 = os.path.join(models, "qwen3.5-4b-4bit")
    os.makedirs(os.path.join(m4, ".kapitsa"))
    with open(os.path.join(m4, "config.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(m4, "mtp.safetensors"), "wb") as f:
        f.write(b"x" * 10)
    with open(os.path.join(m4, ".kapitsa", "manifest.json"), "w") as f:
        repo, rev, comps = PINNED["qwen3.5-4b"]
        json.dump({"name": "qwen3.5-4b:4bit", "variant": "4bit", "repo": repo, "revision": rev, "commit": rev,
                   "files": [{"path": "config.json", "size": 2}, {"path": "mtp.safetensors", "size": 10}],
                   "companions": [{"kind": k, "repo": r, "revision": v, "files": ["mtp.safetensors"]} for k, r, v, _ in comps]}, f)
    rc, out, err = run("pull", "4b")
    check("cli: a finished download is reused offline", rc == 0 and "already downloaded" in out, out + err)
    rc, out, _ = run("list", "--json")
    lst = json.loads(out) if rc == 0 else []
    check("cli: list shows it complete, with its variant and MTP head",
          len(lst) == 1 and lst[0]["name"] == "qwen3.5-4b:4bit" and lst[0]["variant"] == "4bit" and lst[0]["complete"] and
          "MTP head" in lst[0]["includes"], out)
    os.remove(os.path.join(m4, "mtp.safetensors"))
    rc, out, _ = run("list", "--json")
    check("cli: a missing file makes it incomplete", rc == 0 and not json.loads(out)[0]["complete"], out)
    rc, out, err = run("pull", "4b")
    check("cli: completing it needs the Hub (offline here: a clear error)", rc == 1 and "Can't reach" in err, out + err)
    rc, out, _ = run("list", "--json")
    check("cli: a failed repair keeps what is there", rc == 0 and os.path.exists(os.path.join(m4, ".kapitsa", "manifest.json")), out)
    rc, out, _ = run("rm", "4b")
    check("cli: rm deletes it", rc == 0 and "Deleted qwen3.5-4b:4bit" in out and not os.path.exists(m4), out)
    rc, _, err = run("rm", folder)
    check("cli: rm leaves a folder of yours alone", rc == 2 and os.path.exists(folder), err)
    # rm says what it gives back: a model that is an APFS clone of another copy frees nothing
    a, b = os.path.join(models, "copy-a"), os.path.join(models, "copy-b")
    os.makedirs(a)
    with open(os.path.join(a, "config.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(a, "model.safetensors"), "wb") as f:
        f.write(os.urandom(20 << 20))
    subprocess.run(["cp", "-c", "-R", a, b], check=True)  # a clone: the same blocks
    rc, out, err = run("rm", "copy-b")
    check("cli: rm of a clone frees nothing, and says so", rc == 0 and "nothing freed" in out and not os.path.exists(b), out + err)
    rc, out, err = run("rm", "copy-a")
    check("cli: rm of the last copy frees its size", rc == 0 and "MB freed" in out and not os.path.exists(a), out + err)
    # the port is checked before the model loads
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy = s.getsockname()[1]
    t0 = time.time()
    rc, _, err = run("serve", "--model", folder, "--port", str(busy))
    check("cli: a busy port fails at once, before loading", rc == 1 and "already in use" in err and time.time() - t0 < 10, err)
    # --context: tokens, k or m of them, or max (--ctx too); anything else is refused before the model is looked at
    for ok in ["131072", "128k", "256K", "1m", "max"]:
        for flag in ["--context", "--ctx"]:
            rc, _, err = run("serve", "--model", folder, "--port", str(busy), flag, ok)
            check(f"cli: serve {flag} {ok} is taken", rc == 1 and "already in use" in err, err)
    for bad in ["12q", "0", "-5", "128kb", "1.5m", "99999999999", ""]:
        rc, _, err = run("serve", "--model", folder, "--context", bad)
        check(f"cli: serve --context {bad!r} is refused", rc == 2 and "--context takes tokens" in err, err)
        rc, _, err = run("claude", "--context", bad)
        check(f"cli: claude --context {bad!r} is refused", rc == 2 and "--context takes tokens" in err, err)
    # --timing: ttft, prefill or both; anything else is refused before the model is looked at
    for ok in ["ttft", "prefill", "both"]:
        rc, _, err = run("serve", "--model", folder, "--port", str(busy), "--timing", ok)
        check(f"cli: serve --timing {ok} is taken", rc == 1 and "already in use" in err, err)
    for bad in ["bogus", "", "TTFT", "prefill,ttft"]:
        rc, _, err = run("serve", "--model", folder, "--timing", bad)
        check(f"cli: serve --timing {bad!r} is refused", rc == 2 and "--timing takes ttft, prefill or both" in err, err)
    rc, _, err = run("doctor", "--context", "max")
    check("cli: doctor --context max asks for a size", rc == 2 and "--context 256k" in err, err)
    s.close()
    rc, _, err = run("serve", "--model", "nonsense")
    check("cli: serve with an unknown model", rc == 2 and "Unknown model" in err, err)
    for a in ["flash-next:4bit", "flash:4bit", "nace-ai/Qwen3.8-Flash-Next-MLX-4bit"]:
        j = plan(a)
        check(f"cli: {a} -> qwen3.8-flash-next:4bit at the app's pinned revision",
              j.get("model") == "qwen3.8-flash-next:4bit" and j.get("kind") == "catalog" and
              j.get("repo") == "nace-ai/Qwen3.8-Flash-Next-MLX-4bit" and
              j.get("revision") == "5d423f9f563c1f1b7ca7945b268e65c0b92b0eae" and j.get("downloaded") is False, json.dumps(j))
    rc, _, err = run("serve", "--port", str(busy))
    check("cli: serve with no model, offline and nothing downloaded: a clear error", rc == 1 and "Can't reach" in err, err)
    rc, out, _ = run("--help")
    check("cli: --help names the commands and models", rc == 0 and all(w in out for w in ("serve", "pull", "list", "35b", "27b", "4b")), out)

if CLI:
    cli_checks(CLI)
    print("FAILED: %d" % fails if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)

# 0. the chat page and the model list
st, hd, page = get("/")
html = page.decode("utf-8", "replace")
check("GET / serves the chat page", st == 200 and hd.get("Content-Type", "").startswith("text/html") and
      "<title>Kapitsa</title>" in html and "/v1/chat/completions" in html, f"{st} {hd.get('Content-Type')}")
check("chat page is self-contained (no external requests)",
      not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//|@import|url\(\s*["']?(https?:)?//""", html, re.I) and
      "default-src 'none'" in hd.get("Content-Security-Policy", "") and len(page) < 64 * 1024,
      hd.get("Content-Security-Policy", "") + f" ({len(page)} bytes)")
m = re.search(r'const TIMING = "(\w+)"', html)
check("chat page carries the server's --timing (ttft, prefill or both; the placeholder is filled in)",
      bool(m) and m.group(1) in ("ttft", "prefill", "both") and "__TIMING__" not in html, m and m.group(0))
st, _, body = get("/v1/models", auth())
models_ok = st == 200 and json.loads(body)["data"][0]["id"]
check("/v1/models lists the model" + (f" as {MODEL_ID}" if MODEL_ID else ""),
      bool(models_ok) and (not MODEL_ID or json.loads(body)["data"][0]["id"] == MODEL_ID), body[:200].decode())
if API_KEY:
    check("--api-key: the page loads without a key", get("/")[0] == 200)
    check("--api-key: the API needs it", get("/v1/models")[0] == 401 and get("/v1/models", auth())[0] == 200)

# 1. thinking on, streaming: reasoning separate from content
r = post({"messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}],
          "max_tokens": 2048, "stream": True}, stream=True)
check("thinking: reasoning streamed separately", len(r["reasoning"]) > 0 and "<think>" not in r["content"], repr(r)[:300])
check("thinking: correct answer", "391" in r["content"], repr(r["content"])[:200])
check("thinking: finish=stop + usage", r["finish"] == "stop" and r["usage"] and r["usage"]["completion_tokens"] > 0, str(r["finish"]))

# 2. tool call (streaming), then tool result round trip with prefix reuse
msgs = [{"role": "system", "content": "You are a coding agent. Use tools to answer."},
        {"role": "user", "content": "Read the file /etc/hosts and tell me how many lines it has."}]
r = post({"messages": msgs, "tools": TOOLS, "max_tokens": 4096, "stream": True,
          "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
tc = r["tool_calls"]
check("tool call: finish=tool_calls", r["finish"] == "tool_calls", str(r["finish"]) + " " + repr(r["content"])[:200])
ok = len(tc) == 1 and hosts_call(tc[0]["function"]["name"], tc[0]["function"]["arguments"])
check("tool call: read_file or run_shell on /etc/hosts", ok, json.dumps(tc)[:300])
check("tool call: no markup in content", "<tool_call>" not in r["content"] and "<function" not in r["content"], repr(r["content"]))
first_prompt = r["usage"]["prompt_tokens"] if r["usage"] else 0
msgs += [{"role": "assistant", "content": r["content"], "tool_calls": [
            {"id": tc[0]["id"], "type": "function", "function": tc[0]["function"]}] if ok else []},
         {"role": "tool", "tool_call_id": tc[0]["id"] if ok else "x",
          "content": hosts_result(tc[0]["function"]["name"], tc[0]["function"]["arguments"]) if ok else hosts}]
r2 = post({"messages": msgs, "tools": TOOLS, "max_tokens": 2048, "stream": True,
           "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
# (a model may check once more with another call, e.g. Qwen3.8-Flash-Next runs `wc -l` after reading the file: up to 2
#  more rounds, as in the Responses loop below)
rl, rounds = r2, 0
while rounds < 2 and rl["finish"] == "tool_calls" and rl["tool_calls"] and "3" not in rl["content"]:
    msgs += [{"role": "assistant", "content": rl["content"], "tool_calls": [
                {"id": c["id"], "type": "function", "function": c["function"]} for c in rl["tool_calls"]]}] + \
            [{"role": "tool", "tool_call_id": c["id"], "content": hosts_result(c["function"]["name"], c["function"]["arguments"])}
             for c in rl["tool_calls"]]
    rl = post({"messages": msgs, "tools": TOOLS, "max_tokens": 2048, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
    rounds += 1
check("tool result: final answer mentions 3", "3" in rl["content"], repr(rl["content"])[:200])
cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"] if r2["usage"] else 0
check("tool result: prefix reused (cached >= first prompt, less the generation prompt's tail)",
      cached >= first_prompt - GEN_TAIL, f"cached={cached} first_prompt={first_prompt} usage={r2['usage']}")

# 3. next user turn: reuse from the checkpoint
msgs += [{"role": "assistant", "content": rl["content"]},
         {"role": "user", "content": "Thanks. Now in one word: what OS uses /etc/hosts?"}]
r3 = post({"messages": msgs, "tools": TOOLS, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}})
u = r3["usage"]
check("next turn: cached prefix", u["prompt_tokens_details"]["cached_tokens"] > first_prompt // 2, json.dumps(u))
check("next turn: answer", len(r3["choices"][0]["message"]["content"]) > 0, json.dumps(r3)[:300])

# 4. non-streaming tool call with integer argument coercion
r4 = post({"messages": [{"role": "user", "content": "Use read_file to read the first 5 lines of /etc/passwd."}],
           "tools": TOOLS, "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False}})
m = r4["choices"][0]["message"]
tcs = m.get("tool_calls") or []
a = json.loads(tcs[0]["function"]["arguments"]) if tcs else {}
check("non-stream tool call + int arg", tcs and a.get("path") == "/etc/passwd" and a.get("max_lines") in (5, None),
      json.dumps(m)[:300])
if tcs and "max_lines" in a:
    check("int argument is a JSON number", isinstance(a["max_lines"], int), repr(a))

# 5. side slot: a short unrelated request must not evict the main conversation
post({"messages": [{"role": "user", "content": "Title for: hosts file question"}], "max_tokens": 20,
      "chat_template_kwargs": {"enable_thinking": False}})
msgs += [{"role": "assistant", "content": r3["choices"][0]["message"]["content"]},
         {"role": "user", "content": "And on Windows, where is it?"}]
r5 = post({"messages": msgs, "tools": TOOLS, "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}})
check("main slot survives side request", r5["usage"]["prompt_tokens_details"]["cached_tokens"] > first_prompt // 2,
      json.dumps(r5["usage"]))

# 6. thinking on + tools: the call comes back as a tool call (Qwen3.6 sometimes writes it inside its think
#    block and stops without closing it; the server returns such a call as a tool call, NI_THINK_TOOLCALLS)
r6 = post({"messages": [{"role": "system", "content": "You are a coding agent. Use tools to answer."},
                        {"role": "user", "content": "Read the file /etc/hosts and tell me how many lines it has."}],
           "tools": TOOLS, "max_tokens": 4096, "temperature": 0, "top_k": 1, "stream": True}, stream=True)
tc6 = r6["tool_calls"]
check("thinking + tools: a call on /etc/hosts returned as a tool call",
      r6["finish"] == "tool_calls" and len(tc6) >= 1 and hosts_call(tc6[0]["function"]["name"], tc6[0]["function"]["arguments"]),
      str(r6["finish"]) + " " + json.dumps(tc6)[:200] + " reasoning tail " + repr(r6["reasoning"][-200:]))

# 7. one-offs never evict the conversation in progress, whatever their length (2 slots): a long one-off,
#    then a 3-turn conversation shorter than it, a new one-off and a title request, then the conversation
#    continues. Evicting the slot with the fewest cached tokens dropped the conversation at the new one-off.
def turn(m, n):
    return post({"messages": m, "max_tokens": n, "temperature": 0, "top_k": 1,
                 "chat_template_kwargs": {"enable_thinking": False}})
turn([{"role": "user", "content": "Write a detailed 500-word essay about the history of the printing press."}], 700)
m7 = [{"role": "user", "content": "Let's plan a small Python CLI that counts words in a file. What's step one?"}]
for q in ["Okay. What's step two?", "And how do I test it?", None]:
    a7 = turn(m7, 60)
    m7 += [{"role": "assistant", "content": a7["choices"][0]["message"]["content"]}]
    if q:
        m7 += [{"role": "user", "content": q}]
held = a7["usage"]["prompt_tokens"]
turn([{"role": "user", "content": "Use one sentence: what is a hash map?"}], 40)
turn([{"role": "user", "content": "Title for: word counting CLI"}], 12)
r7 = turn(m7 + [{"role": "user", "content": "Great, summarize the plan in one line."}], 40)
check("one-offs keep the conversation in progress", r7["usage"]["prompt_tokens_details"]["cached_tokens"] >= held // 2,
      json.dumps(r7["usage"]))

# 7b. an agent's first conversation survives the side request it sends beside it (opencode sends its title request with
#     the first request), even when the other slot holds an older conversation of several turns
import threading
c7 = [{"role": "user", "content": "Name one prime number."}]
a = turn(c7, 20)
c7 += [{"role": "assistant", "content": a["choices"][0]["message"]["content"]}, {"role": "user", "content": "And another?"}]
turn(c7, 20)
b7 = [{"role": "system", "content": "You are a coding agent. Use tools to answer. " + "Be careful and precise. " * 40},
      {"role": "user", "content": "Read the file /etc/hosts and tell me how many lines it has."}]
first = {}
def first_request():
    first["r"] = post({"messages": b7, "tools": TOOLS, "max_tokens": 1024, "temperature": 0, "top_k": 1,
                       "chat_template_kwargs": {"enable_thinking": False}})
th = threading.Thread(target=first_request)
th.start()
time.sleep(0.05)
turn([{"role": "user", "content": "Title for: counting lines in /etc/hosts"}], 12)
th.join()
r = first["r"]
calls7 = r["choices"][0]["message"].get("tool_calls") or []
if calls7:
    b7 += [{"role": "assistant", "content": r["choices"][0]["message"]["content"] or "", "tool_calls": calls7[:1]},
           {"role": "tool", "tool_call_id": calls7[0]["id"], "content": "127.0.0.1 localhost\n::1 localhost\n"}]
    r2 = post({"messages": b7, "tools": TOOLS, "max_tokens": 256, "temperature": 0, "top_k": 1,
               "chat_template_kwargs": {"enable_thinking": False}})
    check("a side request sent beside a first request does not evict it",
          r2["usage"]["prompt_tokens_details"]["cached_tokens"] >= r["usage"]["prompt_tokens"], json.dumps(r2["usage"]))
else:
    check("a side request sent beside a first request does not evict it (it called a tool)", False, json.dumps(r)[:300])

# ---------------------------------------------------------------- Anthropic Messages and OpenAI Responses
def call(path, body, headers=None, stream=False):
    """(status, JSON body) of a POST, or (status, [(event, data), ...]) of a stream; HTTP errors are returned"""
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(auth() if headers is None else headers)})
    try:
        r = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    if not stream:
        return r.status, json.loads(r.read())
    events, name = [], None
    for line in r:
        line = line.decode().rstrip("\n")
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return r.status, events

def blocks_of(events):
    """the assistant message an Anthropic client assembles from a stream"""
    blocks, inputs = [], {}
    for name, d in events:
        if name == "content_block_start":
            blocks.append(dict(d["content_block"]))
        elif name == "content_block_delta":
            b, x = blocks[d["index"]], d["delta"]
            if x["type"] == "text_delta": b["text"] += x["text"]
            elif x["type"] == "thinking_delta": b["thinking"] += x["thinking"]
            elif x["type"] == "signature_delta": b["signature"] = x["signature"]
            elif x["type"] == "input_json_delta": inputs[d["index"]] = inputs.get(d["index"], "") + x["partial_json"]
        elif name == "content_block_stop" and blocks[d["index"]]["type"] == "tool_use":
            blocks[d["index"]]["input"] = json.loads(inputs.get(d["index"], "{}"))
    return blocks

A_TOOLS = [{"name": t["function"]["name"], "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"]} for t in TOOLS]
CTX = json.loads(get("/v1/models", auth())[2])["data"][0]["context_length"]

# 8. Anthropic: a Claude Code-like tool loop (adaptive thinking shown as a signature only, every turn's thinking kept)
a_body = {"model": "m", "max_tokens": 4096, "temperature": 0, "top_k": 1,
          "system": [{"type": "text", "text": "x-anthropic-billing-header: cc_version=1; cch=00000;"},
                     {"type": "text", "text": "You are a coding agent. Use tools to answer."}],
          "tools": A_TOOLS, "thinking": {"type": "adaptive", "display": "omitted"},
          "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
          "messages": [{"role": "user", "content": [{"type": "text", "text": "Read the file /etc/hosts and tell me how many lines it has."}]}]}
st, m1 = call("/v1/messages", a_body)
uses = [b for b in m1.get("content", []) if b["type"] == "tool_use"] if st == 200 else []
check("anthropic: non-streamed tool call (stop_reason tool_use, tool_use on /etc/hosts, usage)",
      st == 200 and m1["type"] == "message" and m1["stop_reason"] == "tool_use" and len(uses) >= 1 and
      hosts_call(uses[0]["name"], uses[0]["input"]) and
      m1["usage"]["input_tokens"] + m1["usage"]["cache_read_input_tokens"] > 0 and m1["usage"]["output_tokens"] > 0,
      json.dumps(m1)[:400])
thinking = [b for b in m1.get("content", []) if b["type"] == "thinking"] if st == 200 else []
check("anthropic: display omitted: the thinking block has no text and a signature",
      len(thinking) == 1 and thinking[0]["thinking"] == "" and thinking[0]["signature"], json.dumps(thinking)[:200])
first = m1["usage"]["input_tokens"] + m1["usage"]["cache_read_input_tokens"] if st == 200 else 0
st, n = call("/v1/messages/count_tokens", {k: v for k, v in a_body.items() if k not in ("max_tokens", "temperature", "top_k")})
check("anthropic: count_tokens counts the prompt", st == 200 and n.get("input_tokens") == first, f"{n} vs {first}")
a_body["messages"] += [{"role": "assistant", "content": m1.get("content", [])},
                       {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"] if uses else "x", "content": hosts_result(uses[0]["name"], uses[0]["input"]) if uses else hosts},
                                                    {"type": "text", "text": "<system-reminder>Answer briefly.</system-reminder>"}]}]
st, ev = call("/v1/messages", dict(a_body, stream=True), stream=True)
names = [e for e, _ in ev] if st == 200 else []
check("anthropic stream: message_start, ping, blocks, message_delta, message_stop",
      names[:2] == ["message_start", "ping"] and names[-2:] == ["message_delta", "message_stop"] and
      "content_block_start" in names, str(names)[:300])
usage = ev[0][1]["message"]["usage"] if names else {}
check("anthropic stream: message_start reports the tool loop's prompt as cached (the reply's tokens included)",
      usage.get("cache_read_input_tokens", 0) >= first - GEN_TAIL and usage.get("cache_creation_input_tokens") == 0,
      f"{usage} first {first}")
reply = blocks_of(ev)
check("anthropic stream: final answer mentions 3", any(b["type"] == "text" and "3" in b["text"] for b in reply) and
      ev[-2][1]["delta"]["stop_reason"] == "end_turn", json.dumps(reply)[:300])
a_body["messages"] += [{"role": "assistant", "content": reply}, {"role": "user", "content": "And how many mention localhost?"}]
st, m3 = call("/v1/messages", dict(a_body, stream=True), stream=True)
u3 = m3[0][1]["message"]["usage"] if st == 200 and m3 else {}
check("anthropic: the next question's prompt continues the conversation (every turn's thinking kept)",
      u3.get("cache_read_input_tokens", 0) > first, f"{u3} first {first}")
if API_KEY:
    st, _ = call("/v1/messages", {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]},
                 headers={"x-api-key": API_KEY})
    check("--api-key: x-api-key works too", st == 200, st)
    st, e = call("/v1/messages", {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}, headers={})
    check("--api-key: without it, 401 in Anthropic's error format", st == 401 and e.get("type") == "error", e)
long_text = "hello " * int(CTX * 1.1)
st, e = call("/v1/messages", {"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": long_text}]})
check("anthropic: a prompt over the context is 400 \"prompt is too long: N tokens > M maximum\"",
      st == 400 and e["error"]["type"] == "invalid_request_error" and
      re.match(r"prompt is too long: \d+ tokens > \d+ maximum", e["error"]["message"]), json.dumps(e)[:300])
st, e = call("/v1/messages", dict(a_body, stream=True, messages=[{"role": "user", "content": long_text}]), stream=True)
check("anthropic: streamed too, a plain 400 before any event", st == 400 and "prompt is too long" in json.dumps(e), str(e)[:200])

# 9. Responses: a Codex-like tool loop (reasoning kept through encrypted_content, store false), then previous_response_id
r_tools = [{"type": "function", "name": t["function"]["name"], "description": t["function"]["description"],
            "parameters": t["function"]["parameters"], "strict": False} for t in TOOLS]
r_input = [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "Use tools to answer."}]},
           {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Read the file /etc/hosts and tell me how many lines it has."}]}]
r_body = {"model": "m", "instructions": "You are a coding agent.", "tools": r_tools, "store": False,
          "include": ["reasoning.encrypted_content"], "temperature": 0, "top_k": 1, "input": r_input}
st, ev = call("/v1/responses", dict(r_body, stream=True), stream=True)
names = [e for e, _ in ev] if st == 200 else []
done = ev[-1][1]["response"] if names else {}
calls_ = [o for o in done.get("output", []) if o["type"] == "function_call"]
check("responses stream: created, in_progress, items, completed; sequence numbers in order",
      names[:2] == ["response.created", "response.in_progress"] and names[-1] == "response.completed" and
      [d["sequence_number"] for _, d in ev] == list(range(len(ev))), str(names)[:300])
check("responses stream: a function_call on /etc/hosts",
      len(calls_) >= 1 and hosts_call(calls_[0]["name"], calls_[0]["arguments"]),
      json.dumps(done.get("output"))[:300])
first_r = done.get("usage", {}).get("input_tokens", 0)
r_body["input"] = r_input + done.get("output", []) + [{"type": "function_call_output", "call_id": c["call_id"], "output": hosts_result(c["name"], c["arguments"])} for c in calls_]
st, r2 = call("/v1/responses", r_body)
# (a model may check once more with another call, e.g. Lightning reads the file after `wc -l`: up to 2 more rounds)
rl, rounds = r2, 0
while st == 200 and rounds < 2 and any(o["type"] == "function_call" for o in rl["output"]) and \
        not any(o["type"] == "message" for o in rl["output"]):
    more = [o for o in rl["output"] if o["type"] == "function_call"]
    r_body["input"] = r_body["input"] + rl["output"] + [{"type": "function_call_output", "call_id": c["call_id"],
                                                         "output": hosts_result(c["name"], c["arguments"])} for c in more]
    st, rl = call("/v1/responses", r_body)
    rounds += 1
check("responses: the tool loop's next request is answered (mentions 3)",
      st == 200 and rl["status"] == "completed" and any(o["type"] == "message" and "3" in o["content"][0]["text"] for o in rl["output"]),
      json.dumps(rl)[:300])
check("responses: its prompt is cached (input_tokens_details.cached_tokens >= the first prompt)",
      st == 200 and r2["usage"]["input_tokens_details"]["cached_tokens"] >= first_r, json.dumps(r2.get("usage")))
st, p1 = call("/v1/responses", {"model": "m", "input": "Say just the word: apple.", "reasoning": {"effort": "none"}, "max_output_tokens": 20})
st2, p2 = call("/v1/responses", {"model": "m", "previous_response_id": p1.get("id", ""), "input": "Now just the word: pear.",
                                 "reasoning": {"effort": "none"}, "max_output_tokens": 20})
check("responses: previous_response_id continues a stored response",
      st == 200 and st2 == 200 and p2["previous_response_id"] == p1["id"] and
      p2["usage"]["input_tokens"] > p1["usage"]["input_tokens"] and p2["usage"]["input_tokens_details"]["cached_tokens"] > 0,
      json.dumps(p2)[:300])
st, e = call("/v1/responses", {"model": "m", "previous_response_id": "resp_unknown", "input": "x"})
check("responses: an unknown previous_response_id is a 400", st == 400 and e["error"]["code"] == "previous_response_not_found", e)
st, e = call("/v1/responses", {"model": "m", "input": long_text, "stream": True}, stream=True)
check("responses: a prompt over the context is 400 context_length_exceeded",
      st == 400 and e["error"]["code"] == "context_length_exceeded", json.dumps(e)[:300])
st, r = call("/v1/responses", {"model": "m", "input": "Count from 1 to 100.", "reasoning": {"effort": "none"}, "max_output_tokens": 8})
check("responses: a max_output_tokens cut is status incomplete", st == 200 and r["status"] == "incomplete" and
      r["incomplete_details"] == {"reason": "max_output_tokens"}, json.dumps(r)[:300])

# ---------------------------------------------------------------- looped models (Dory)
# 10. A looped model reports the loop count each request ran (usage.kapitsa.loops): a request's "loops", else its
#     reasoning effort (low 1, medium 2, high 4) or Anthropic thinking (off 1, a budget by its size), else the server's
#     default. A cached prefix is reused only at the loop count that computed it.
probe = post({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4,
              "chat_template_kwargs": {"enable_thinking": False}})
if (probe.get("usage") or {}).get("kapitsa"):
    base_loops = probe["usage"]["kapitsa"]["loops"]
    lmsgs = [{"role": "system", "content": "[loops] You are terse. " + "Answer in one short sentence. " * 30},
             {"role": "user", "content": "What is the capital of France?"}]
    def lp(**kw):
        return post(dict({"messages": lmsgs, "max_tokens": 16, "temperature": 0, "top_k": 1,
                          "chat_template_kwargs": {"enable_thinking": False}}, **kw))
    la = lp(loops=1)
    lb = lp(loops=1)
    lc = lp(loops=2)
    ld = lp(loops=2)
    n_l = la["usage"]["prompt_tokens"]
    check("loops: usage.kapitsa.loops reports the request's loops",
          la["usage"]["kapitsa"]["loops"] == 1 and lc["usage"]["kapitsa"]["loops"] == 2, json.dumps(lc["usage"]))
    check("loops: the same loop count reuses the cached prompt",
          lb["usage"]["prompt_tokens_details"]["cached_tokens"] >= n_l - GEN_TAIL and
          ld["usage"]["prompt_tokens_details"]["cached_tokens"] >= n_l - GEN_TAIL, json.dumps([lb["usage"], ld["usage"]]))
    check("loops: another loop count reads the prompt again (no cached tokens)",
          lc["usage"]["prompt_tokens_details"]["cached_tokens"] == 0, json.dumps(lc["usage"]))
    check("loops: the same greedy answer at the same loop count",
          la["choices"][0]["message"]["content"] == lb["choices"][0]["message"]["content"],
          repr([la["choices"][0]["message"]["content"], lb["choices"][0]["message"]["content"]]))
    le = lp(reasoning_effort="low")
    lh = lp(reasoning_effort="high")
    check("loops: reasoning_effort low runs 1 loop, high 4",
          le["usage"]["kapitsa"]["loops"] == 1 and lh["usage"]["kapitsa"]["loops"] == 4,
          json.dumps([le["usage"], lh["usage"]]))
    try:
        lp(loops=9)
        bad_loops = False
    except urllib.error.HTTPError as e:
        bad_loops = e.code == 400
    check("loops: 9 loops is a 400", bad_loops)
    st, ma = call("/v1/messages", {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "Say hi."}]})
    st2, mb = call("/v1/messages", {"model": "m", "max_tokens": 64, "thinking": {"type": "enabled", "budget_tokens": 20000},
                                    "messages": [{"role": "user", "content": "Say hi."}]})
    st3, mc = call("/v1/messages", {"model": "m", "max_tokens": 64, "thinking": {"type": "adaptive"},
                                    "messages": [{"role": "user", "content": "Say hi."}]})
    check("loops: Anthropic thinking off 1 loop, a 20000-token budget 4, adaptive the server's default",
          st == st2 == st3 == 200 and ma["usage"]["kapitsa"]["loops"] == 1 and mb["usage"]["kapitsa"]["loops"] == 4 and
          mc["usage"]["kapitsa"]["loops"] == base_loops, json.dumps([ma.get("usage"), mb.get("usage"), mc.get("usage")]))
    st, rr = call("/v1/responses", {"model": "m", "input": "Say hi.", "reasoning": {"effort": "high"}, "max_output_tokens": 64})
    check("loops: Responses reasoning.effort high runs 4 loops",
          st == 200 and rr["usage"]["kapitsa"]["loops"] == 4, json.dumps(rr.get("usage")))

# ---------------------------------------------------------------- requests in flight at once
# Each request in flight has a slot of its own and they take turns on the GPU between prefill chunks and decode cycles;
# with every slot busy a request waits for one. /v1/models says how many slots there are.
SLOTS = json.loads(get("/v1/models", auth())[2])["data"][0].get("slots", 2)
check("/v1/models reports the requests that run at once (slots)", isinstance(SLOTS, int) and SLOTS >= 1, str(SLOTS))
def story(tag, n):
    return {"messages": [{"role": "system", "content": f"[{tag}] You are a storyteller."},
                         {"role": "user", "content": "Tell a long story about a lighthouse keeper and a storm."}],
            "max_tokens": n, "temperature": 0, "top_k": 1, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
def counting(tag, n):  # a reply that runs to max_tokens
    return {"messages": [{"role": "user", "content": f"[{tag}] Count from 1 to 2000, one number per line, nothing else."}],
            "max_tokens": n, "temperature": 0, "top_k": 1, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
def sampled(tag, n, seed):
    return {"messages": [{"role": "user", "content": f"[{tag}] Write a short poem about rain on a tin roof."}],
            "max_tokens": n, "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": seed, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}}
def together(bodies, delays=None):
    out = [None] * len(bodies)
    def go(i):
        if delays:
            time.sleep(delays[i])
        t0 = time.time()
        out[i] = post(bodies[i], stream=True)
        out[i]["t_done"] = time.time()
        out[i]["t_sent"] = t0
    th = [threading.Thread(target=go, args=(i,)) for i in range(len(bodies))]
    for t in th: t.start()
    for t in th: t.join()
    return out
def evict_slots():  # one-offs that take every slot, so the next requests prefill cold
    for i in range(SLOTS):
        post({"messages": [{"role": "user", "content": f"[evict {i} {time.time()}] " + "Say ok. " * 40}], "max_tokens": 1,
              "chat_template_kwargs": {"enable_thinking": False}})

# 10. requests sent together are answered in full (greedy, and sampled with a seed); cold prefills both times
#     (one-offs take the slots in between, so neither run reuses the other's cache). Replies decoded in one batched
#     pass may differ from the ones a request gets alone where two tokens are within the pass's numerical noise
#     (near-ties, greedy) and in their random draws (sampled): the greedy reply must start as it does alone, and a
#     request alone must give the same reply every time.
pair = [story("c1", 160), sampled("c2", 120, 11)]
evict_slots()
both = together(pair, delays=[0, 0.3])
evict_slots()
alone = [together([b])[0] for b in pair]
evict_slots()
again = together([pair[0]])[0]
check("a request alone gives the same reply every time (greedy)",
      again["content"] == alone[0]["content"] and again["usage"]["completion_tokens"] == alone[0]["usage"]["completion_tokens"],
      f"{again['usage']} vs {alone[0]['usage']}")
for i, name in enumerate(["greedy", "sampled T=0.7 seed 11"]):
    n = both[i]["usage"]["completion_tokens"] if both[i]["usage"] else 0
    check(f"in flight together: the {name} reply is answered in full",
          both[i]["finish"] in ("stop", "length") and n > 0 and (both[i]["finish"] == "stop" or n == pair[i]["max_tokens"]),
          f"{both[i]['finish']} {both[i]['usage']}")
check("in flight together: the greedy reply starts as it does alone",
      both[0]["content"][:32] == alone[0]["content"][:32], f"{both[0]['content'][:80]!r} vs {alone[0]['content'][:80]!r}")

# 10b. a request sent while a long reply streams is answered before that reply ends (no wait for the whole reply)
ctx = "\n".join(f"def helper_{i}(x):\n    return x * {i} + {i % 7}\n" for i in range(150))
side = {"messages": [{"role": "user", "content": f"[side {time.time()}]\n{ctx}\nHow many functions are defined above? "
                      "Answer with a number."}], "max_tokens": 16, "temperature": 0, "top_k": 1, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}}
if SLOTS >= 2:  # with one slot the request waits for the reply's end, by design
    long_, short = together([counting("c3", 700), side], delays=[0, 1.0])
    check("in flight together: a 2K-token request sent during a long reply finishes first",
          short["t_done"] < long_["t_done"] and short["usage"]["prompt_tokens"] > 1500 and long_["usage"]["completion_tokens"] >= 600,
          f"side done {short['t_done'] - long_['t_sent']:.1f} s, long reply done {long_['t_done'] - long_['t_sent']:.1f} s "
          f"({long_['usage']['completion_tokens']} tokens), side usage {short['usage']}")
else:
    print("SKIP in flight together: a 2K-token request sent during a long reply finishes first (--slots 1)")

# 10c. more requests than slots: they wait for one, first come first served, and all are answered
many = together([counting(f"q{i}", 40) for i in range(SLOTS + 2)], delays=[0.05 * i for i in range(SLOTS + 2)])
check("more requests than slots: every one is answered in full",
      all(m["finish"] == "length" and m["usage"]["completion_tokens"] == 40 for m in many),
      str([(m["finish"], (m["usage"] or {}).get("completion_tokens")) for m in many]))

# 10d. a client gone mid-stream: its request stops (the next request does not wait for its reply to end), and the
#      request beside it is answered in full
def abandon(body, after_chunks):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **auth()})
    r = urllib.request.urlopen(req, timeout=600)
    got = 0
    for line in r:
        if line.startswith(b"data: "):
            got += 1
            if got >= after_chunks:
                break
    r.close()
gone = threading.Thread(target=abandon, args=(counting("c4", 3000), 20))
gone.start()
time.sleep(0.2)
beside = together([counting("c5", 100)])[0]
gone.join()
t0 = time.time()
after = post(counting("c6", 8), stream=True)
check("a client gone mid-stream: the request beside it is answered, the next one starts at once",
      beside["usage"]["completion_tokens"] == 100 and after["usage"]["completion_tokens"] == 8 and time.time() - t0 < 10,
      f"beside {beside['usage']}, next request {time.time() - t0:.1f} s")

# 10e. a Chat Completions stream: a prompt over the context is a plain 400 context_length_exceeded (opencode and other
#      clients compact on it), as without "stream"; the stream starts before the request has a slot, and while it waits
#      (or a long prefill runs) it gets a chunk with an empty delta every 15 s (clients drop a stream idle for minutes)
ROLE = {"role": "assistant", "content": ""}
def is_keepalive(d):
    return choice(d).get("delta") == {} and choice(d).get("finish_reason") is None
for streamed in (False, True):
    if streamed:
        st, e = chat_stream({"messages": [{"role": "user", "content": long_text}], "max_tokens": 10})
    else:
        st, e = call("/v1/chat/completions", {"messages": [{"role": "user", "content": long_text}], "max_tokens": 10})
    check(f"chat{' stream' if streamed else ''}: a prompt over the context is a 400 context_length_exceeded",
          st == 400 and isinstance(e, dict) and e.get("error", {}).get("code") == "context_length_exceeded" and
          "maximum context length" in e["error"]["message"], str(e)[:300])
def hold(tag):
    """a stream that writes until its client goes, open once its first token is out (it has a slot)"""
    body = {"messages": [{"role": "user", "content": f"[{tag}] Count from 1 to 100000, one number per line, nothing else."}],
            "max_tokens": 20000, "temperature": 0, "top_k": 1, "stream": True, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.urlopen(urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json", **auth()}), timeout=600)
    for line in r:
        if line.startswith(b"data: {") and choice(json.loads(line[6:])).get("delta", {}).get("content"):
            break
    return r
holders = [hold(f"hold {i} {time.time()}") for i in range(SLOTS)]
ev = []
waiting = threading.Thread(target=chat_stream, args=(counting(f"wait {time.time()}", 8), ev))
waiting.start()
t0 = time.time()
while time.time() - t0 < 45 and sum(is_keepalive(d) for _, d in list(ev)) < 2:
    time.sleep(0.2)
for h in holders:  # their clients go: they stop at their next token, and the waiting request gets a slot
    h.close()
waiting.join()
keep = [t for t, d in ev if is_keepalive(d)]
first = next((t for t, d in ev if choice(d).get("delta", {}).get("content")), None)
check("chat stream: while every slot is busy the stream starts at once, gets an empty chunk every 15 s, and its first "
      "token comes once a slot frees",
      len(ev) > 2 and choice(ev[0][1]).get("delta") == ROLE and ev[0][0] < 1.0 and len(keep) == 2 and
      15 <= keep[0] < 17.5 and 14 <= keep[1] - keep[0] < 17.5 and first is not None and first > keep[1] and
      ev[-1][1] == "[DONE]", f"first chunk {ev[0] if ev else None}, keepalives at {keep}, first token at {first}")

print("FAILED: %d" % fails if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
