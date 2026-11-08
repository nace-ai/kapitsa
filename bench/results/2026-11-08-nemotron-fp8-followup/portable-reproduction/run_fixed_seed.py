import argparse
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

parser = argparse.ArgumentParser(description="Run the portable fixed-request/seed client against an already locked server")
parser.add_argument("--server-url", required=True)
parser.add_argument("--model-id", default="fp8-review")
parser.add_argument("--driver", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
url = urlsplit(args.server_url)
if url.scheme != "http" or not url.hostname or url.port is None or url.username or url.password:
    parser.error("Use an http server URL with an explicit port and no credentials")
if url.path not in ("", "/") or url.query or url.fragment:
    parser.error("Use a server origin without a path, query or fragment")
if not any(os.environ.get(name, "") not in ("", "0") for name in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD")):
    parser.error("Hold the exclusive GPU lock across server and client and set a caller-held marker")
if args.out.exists():
    parser.error("Output exists; use a new label to preserve prior evidence")
if not args.out.parent.is_dir():
    parser.error("Create the output directory before running the client")
command = [str(args.driver.resolve()), url.hostname, str(url.port), args.model_id]
with args.out.open("x") as output:
    result = subprocess.run(command, capture_output=True, text=True)
    data = json.loads(result.stdout)
    output.write(json.dumps(data, indent=2) + chr(10))
print(result.stderr, end="")
raise SystemExit(result.returncode)
