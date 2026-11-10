#!/usr/bin/env python3
"""Real pull/resume against a local Hub; tiny files, no model load or external network."""
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


def download_checks(kapitsa, check):
    config = json.dumps({"model_type": "qwen3_5", "linear_num_value_heads": 1,
                         "num_hidden_layers": 1}).encode()
    files = {"config.json": config, "model.safetensors": b"new weights",
             "README.md": b"new metadata", "tokenizer.json": b"unchanged",
             "NOTICE": b"no digest"}

    def blob(data):
        return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

    listing = {"sha": "fixture-revision", "siblings": []}
    for name, data in files.items():
        entry = {"rfilename": name, "size": len(data)}
        if name.endswith(".safetensors"):
            entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest()}
        elif name != "NOTICE":
            entry["blobId"] = blob(data)
        listing["siblings"].append(entry)

    class Hub(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.server.requests.append((self.path, self.headers.get("Range")))
            if self.server.unavailable:
                self.send_error(404)
                return
            if self.path == "/api/models/test/resume/revision/main?blobs=true":
                data = json.dumps(listing).encode()
            elif self.path.startswith("/test/resume/resolve/fixture-revision/"):
                name = self.path.rsplit("/", 1)[1]
                if name not in files:
                    self.send_error(404)
                    return
                data = files[name]
                if self.server.corrupt and name == "model.safetensors":
                    data = b"bad weights"
            else:
                self.send_error(404)
                return
            status = 200
            content_range = None
            if span := self.headers.get("Range"):
                first, last = map(int, span.removeprefix("bytes=").split("-"))
                content_range = f"bytes {first}-{last}/{len(data)}"
                data = data[first:last + 1]
                status = 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            if content_range:
                self.send_header("Content-Range", content_range)
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hub)
    server.requests = []
    server.unavailable = False
    server.corrupt = False
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="kapitsa-pull-") as tmp:
            root = Path(tmp)
            # Neither the user's Hub credentials nor their file caches reach this fixture.
            env = dict(os.environ, HF_ENDPOINT=f"http://127.0.0.1:{server.server_port}",
                       HF_HOME=str(root / "hf"), HF_HUB_CACHE=str(root / "hf" / "hub"),
                       HUGGINGFACE_HUB_CACHE=str(root / "hf" / "hub"), HF_TOKEN="",
                       HUGGING_FACE_HUB_TOKEN="", HF_TOKEN_PATH=str(root / "no-token"),
                       KAPITSA_NO_AUTO_UPDATE="1", KAPITSA_NO_PROMPT="1", NI_LOG="0")

            def pull(home):
                return subprocess.run([kapitsa, "pull", "test/resume"],
                                      env=dict(env, KAPITSA_HOME=str(home)),
                                      capture_output=True, text=True, timeout=30)

            home = root / "adopt"
            model = home / "models" / "test--resume"
            model.mkdir(parents=True)
            old = dict(files, **{"model.safetensors": b"old weights", "README.md": b"old metadata",
                                 "NOTICE": b"old notes"})
            for name, data in old.items():
                (model / name).write_bytes(data)
            kept = {name: (model / name).stat().st_mtime_ns for name in ("config.json", "tokenizer.json", "NOTICE")}
            manifest = model / ".kapitsa" / "manifest.json"
            server.unavailable = True
            failed = pull(home)
            check("pull: failed adoption keeps old files and leaves no complete manifest",
                  failed.returncode != 0 and not manifest.exists() and (model / ".kapitsa").is_dir()
                  and all((model / name).read_bytes() == data for name, data in old.items()), failed.stderr)
            server.unavailable = False
            server.requests.clear()
            resumed = pull(home)
            check("pull: retry of failed adoption completes", resumed.returncode == 0 and manifest.exists(), resumed.stderr)
            for name in ("model.safetensors", "README.md"):
                check(f"pull: retry replaces same-size stale {name} using its digest",
                      (model / name).read_bytes() == files[name])
                check(f"pull: retry actually fetches stale {name}",
                      any(path.endswith("/" + name) and span for path, span in server.requests))
            check("pull: matching files and size-only files are retained without range downloads",
                  all((model / name).read_bytes() == old[name] and (model / name).stat().st_mtime_ns == stamp
                      for name, stamp in kept.items())
                  and not any(path.rsplit("/", 1)[-1] in kept and span for path, span in server.requests))
            server.unavailable = True
            server.requests.clear()
            completed = pull(home)
            check("pull: a completed manifest is still reused offline",
                  completed.returncode == 0 and not server.requests, completed.stderr)

            # A failed transfer can also leave verified final files but no manifest.
            home = root / "transfer"
            model = home / "models" / "test--resume"
            manifest = model / ".kapitsa" / "manifest.json"
            server.unavailable = False
            server.corrupt = True
            bad = pull(home)
            check("pull: a bad downloaded digest cannot publish a complete manifest",
                  bad.returncode != 0 and "checksum mismatch" in bad.stderr and not manifest.exists(), bad.stderr)
            server.corrupt = False
            retry = pull(home)
            check("pull: a failed transfer resumes to the complete requested bytes",
                  retry.returncode == 0 and manifest.exists()
                  and all((model / name).read_bytes() == data for name, data in files.items()), retry.stderr)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    failures = []

    def report(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name)
        if not ok:
            failures.append(name)
            if detail:
                print(detail)

    download_checks(str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/kapitsa").resolve()), report)
    sys.exit(bool(failures))
