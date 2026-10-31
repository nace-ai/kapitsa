"""Qwen3.8-27B sampled plain decode: fixed-seed HTTP outputs and A/B/A/B timings.

Uses --spec 0 so Qwen35::enc_sample, rather than a speculative verifier, selects tokens.
The child server holds the GPU lock. Output directories must be new.
"""

import argparse, hashlib, json, os, pathlib, shutil, signal, subprocess, time, urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", required=True, type=pathlib.Path)
parser.add_argument("--candidate", required=True, type=pathlib.Path)
parser.add_argument("--model", required=True)
parser.add_argument("--out", required=True, type=pathlib.Path)
parser.add_argument("--prompt", required=True, type=pathlib.Path)
parser.add_argument("--port", type=int, default=18130)
parser.add_argument("--candidate-sample-sg", choices=["auto", "0", "1"], default="1",
                    help="candidate NI_QWEN_SAMPLE_SG; 1 opts in on M1–M4, auto uses the GPU default")
args = parser.parse_args()
E = args.out.resolve()
E.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier benchmark
M = args.model
PORT = args.port
URL = f"http://127.0.0.1:{PORT}"
prompt = args.prompt.read_text()
(E / "prompt.txt").write_text(prompt)


def thermal():
    return subprocess.check_output(
        [
            "osascript",
            "-l",
            "JavaScript",
            "-e",
            'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState',
        ],
        text=True,
    ).strip()


def chat(text, limit):
    body = {
        "model": "qwen3.8-27b-4bit",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful coding assistant. Answer with code only.",
            },
            {"role": "user", "content": text},
        ],
        "temperature": 0.7,
        "top_k": 32,
        "seed": 42,
        "max_tokens": limit,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = None
    last = None
    parts = []
    usage = {}
    finish = None
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                value = choice.get("delta", {}).get("content") or ""
                if value:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    parts.append(value)
                finish = choice.get("finish_reason") or finish
    text = "".join(parts)
    n = usage.get("completion_tokens", 0)
    assert text and n == limit and finish == "length", (n, finish, len(text))
    return {
        "tokens": n,
        "finish": finish,
        "ttft_s": first - start,
        "decode_tps": (n - 1) / (last - first),
        "wall_s": time.perf_counter() - start,
        "usage": usage,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


for pair in [1, 2]:
    for arm in ["A", "B"]:
        label = f"{arm}{pair}"
        variant = "main" if arm == "A" else "sample-sg"
        env = dict(
            os.environ,
            NI_LOCK_HELD="1",
            KAPITSA_NO_AUTO_UPDATE="1",
            KAPITSA_NO_PROMPT="1",
            KAPITSA_HOME=str(E / f"home-{label}"),
        )
        env.pop("NI_DFLASH", None)
        env.pop("NI_NO_MTP", None)
        env.pop("NI_QWEN_SAMPLE_SG", None)
        if arm == "B" and args.candidate_sample_sg != "auto":
            env["NI_QWEN_SAMPLE_SG"] = args.candidate_sample_sg
        cmd = [
            "lockf",
            "-k",
            os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock"),
            str((args.reference if arm == "A" else args.candidate).resolve()),
            "serve",
            "--model",
            M,
            "--host",
            "127.0.0.1",
            "--private",
            "--port",
            str(PORT),
            "--context",
            "8192",
            "--slots",
            "1",
            "--spec",
            "0",
        ]
        log = (E / f"{label}-server.log").open("w")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=log, start_new_session=True
        )
        try:
            ready = False
            for _ in range(180):
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited {proc.returncode}")
                try:
                    with urllib.request.urlopen(URL + "/health", timeout=1) as r:
                        ready = r.status == 200
                except OSError:
                    pass
                if ready:
                    break
                time.sleep(1)
            assert ready
            log.flush()
            assert "plain decoding" in (E / f"{label}-server.log").read_text()
            wait_start = time.monotonic()
            if (
                "compiling the Neural Engine program in the background"
                in (E / f"{label}-server.log").read_text()
            ):
                while (
                    "Neural Engine program ready in the background"
                    not in (E / f"{label}-server.log").read_text()
                ):
                    if proc.poll() is not None or time.monotonic() - wait_start > 180:
                        raise RuntimeError(
                            "ANE initialization did not finish; do not measure a partial startup"
                        )
                    time.sleep(0.2)
            ane_wait = time.monotonic() - wait_start
            chat(
                "Implement a Python HTTP server with request parsing, routing and error handling. Output only code.",
                64,
            )
            before = thermal()
            result = chat(prompt, 512)
            after = thermal()
            assert (
                result["usage"].get("kapitsa", {}).get("kflash", {}).get("source")
                == "none"
            ), "not plain decode"
            text = result.pop("text")
            (E / f"{label}-output.txt").write_text(text)
            result.update(
                sample_sg=env.get("NI_QWEN_SAMPLE_SG", "auto"),
                temperature=0.7,
                top_k=32,
                seed=42,
                label=label,
                variant=variant,
                ane_ready_wait_s=ane_wait,
                thermal_before=before,
                thermal_after=after,
                command=cmd,
                version=subprocess.check_output(
                    [cmd[3], "--version"], text=True
                ).strip(),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                power=subprocess.check_output(["pmset", "-g", "batt"], text=True),
            )
            (E / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(
                label, result["decode_tps"], result["usage"], before, after, flush=True
            )
        finally:
            children = subprocess.run(
                ["pgrep", "-P", str(proc.pid)], capture_output=True, text=True
            ).stdout.split()
            for child in children:
                try:
                    os.kill(int(child), signal.SIGINT)
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=10)
            log.close()
            shutil.rmtree(
                env["KAPITSA_HOME"], ignore_errors=True
            )  # only this run's scratch cache
results = [
    json.loads((E / f"{arm}{pair}.json").read_text())
    for pair in [1, 2]
    for arm in ["A", "B"]
]
assert len({r["sha256"] for r in results}) == 1, (
    "generated output differs; keep evidence and investigate"
)
print(
    "Four nonempty 512-token length-limited outputs match; no early stop.", flush=True
)
