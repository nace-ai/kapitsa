#!/bin/bash
# LM Studio's headless daemon (llmster) as a benchmark server: start a fresh daemon, load ONE model with
# the given `lms load` settings, start LM Studio's OpenAI-compatible server, then stay in the foreground
# streaming LM Studio's own logs (so compare_engines.py keeps a live child and its log gets them).
#
#   lmstudio_serve.sh serve <port> <model-key> <identifier> [lms load options...]
#   lmstudio_serve.sh stop      # unload everything, stop the server, shut the daemon down
#
# The daemon detaches (parent launchd), so the harness reads its pid from
# ~/.lmstudio/.internal/llmster-pid.lock (config "pid_file") and measures memory over its whole tree
# (llmster + the node worker that hosts the engine).
#
# GPU lock: compare_engines.py holds it for the whole server lifetime. By hand, wrap the run, e.g.
#   lockf -k /tmp/kapitsa-gpu.lock launchers/lmstudio_serve.sh serve 8197 qwen/qwen3.6-35b-a3b q36 --context-length 32768 --parallel 1
set -euo pipefail
LMS="${LMS:-$HOME/.lmstudio/bin/lms}"
cmd="${1:-}"; shift || true
case "$cmd" in
  stop)
    "$LMS" unload --all 2>&1 || true
    "$LMS" server stop 2>&1 || true
    "$LMS" daemon down 2>&1 || true
    exit 0 ;;
  serve) ;;
  *) echo "usage: $0 serve <port> <model-key> <identifier> [lms load options] | $0 stop" >&2; exit 2 ;;
esac
port="$1"; key="$2"; ident="$3"; shift 3
# a fresh daemon per run: no model, cache or server state carried over from an earlier run
"$LMS" daemon down >/dev/null 2>&1 || true
"$LMS" daemon up
"$LMS" daemon status
"$LMS" runtime ls
"$LMS" unload --all >/dev/null 2>&1 || true
# log streams first, so the load itself is logged too (children of this script: killed with its tree):
# runtime = the engine's own log, server = API requests, model = each prediction's output + stats
"$LMS" log stream --source runtime 2>&1 | sed -u "s/^/[runtime] /" &
"$LMS" log stream --source server 2>&1 | sed -u "s/^/[server] /" &
"$LMS" log stream --source model --filter output --stats 2>&1 | sed -u "s/^/[model] /" &
sleep 1
echo "### lms load $key --identifier $ident -y $*"
t0=$(date +%s)
"$LMS" load "$key" --identifier "$ident" -y "$@"
echo "### loaded in $(( $(date +%s) - t0 ))s"
"$LMS" ps
"$LMS" server start --port "$port" --bind 127.0.0.1
"$LMS" server status
wait
