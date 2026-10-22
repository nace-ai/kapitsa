#!/bin/bash
# oMLX 0.7.1rc1 serving NVIDIA's Bolt checkpoint (txgsync LiLiCorr-MLX bundle) WITH the LiLiCorr drafter,
# following the bundle README's flow, then exposing it on the harness port.
#
# Why the two loads: oMLX 0.7.1rc1 runs DFlash drafters only for model types on its list. The bundle's model file
# (nemotron_bolt.py, Trust Remote Code) adds Bolt's type to that list when it is first imported, i.e. during the
# first load, which therefore runs without the drafter ("DFlash is not supported ... loading without DFlash").
# The README's GUI flow: send one request, then enable DFlash, and oMLX reloads with "DFlash drafter attached".
# Here: settings already say dflash_enabled; load once, unload (POST /v1/models/<id>/unload), load again (the log
# shows 'DFlash drafter attached' both times in 0.7.1rc1; the reload is kept to match the README's flow). oMLX listens on an internal port during this, and a byte-level TCP forwarder opens
# the harness port only afterwards, so the harness never sees the half-ready server.
#
#   omlx_bolt_serve.sh <harness port> <internal port> <model id> <oMLX args...>
# OMLX_BIN names oMLX's command (default: omlx on PATH).
#
# GPU lock: compare_engines.py holds it for the whole server lifetime. By hand, wrap the run in
#   lockf -k /tmp/kapitsa-gpu.lock launchers/omlx_bolt_serve.sh ...
set -uo pipefail
PORT="$1"; IPORT="$2"; MID="$3"; shift 3
OMLX="${OMLX_BIN:-omlx}"
"$OMLX" serve --host 127.0.0.1 --port "$IPORT" "$@" 2>&1 | sed -u "s/^/[omlx] /" &
until curl -sf "http://127.0.0.1:$IPORT/v1/models" >/dev/null 2>&1; do sleep 1; done
req() {
  curl -sS --max-time 900 "http://127.0.0.1:$IPORT/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$MID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Answer with the number only.\"}],\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('### prime answer:', repr(d['choices'][0]['message'].get('content')))" 2>&1
}
echo "### prime 1 (imports the model file; in 0.7.1rc1 the drafter already attaches late in this first load)"; req
echo "### unload"; curl -sS --max-time 300 -X POST "http://127.0.0.1:$IPORT/v1/models/$MID/unload"; echo
for i in 1 2 3 4 5 6 7 8 9 10; do   # a request right after the unload can fail while the engine is torn down
  sleep 3
  echo "### prime 2 (reload with DFlash), try $i"
  req | tee /dev/stderr | grep -q "prime answer" && break
done
echo "### forwarding 127.0.0.1:$PORT -> $IPORT"
exec python3 -u - "$PORT" "$IPORT" <<'PY'
import asyncio, sys
port, iport = int(sys.argv[1]), int(sys.argv[2])
async def pipe(r, w):
    try:
        while (b := await r.read(65536)):
            w.write(b); await w.drain()
    except Exception:
        pass
    finally:
        try: w.close()
        except Exception: pass
async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_connection("127.0.0.1", iport)
    except Exception:
        cw.close(); return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
async def main():
    srv = await asyncio.start_server(handle, "127.0.0.1", port)
    async with srv:
        await srv.serve_forever()
asyncio.run(main())
PY
