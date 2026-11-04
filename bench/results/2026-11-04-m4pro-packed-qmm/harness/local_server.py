"""A foreground-owned local test server; caller must hold the project GPU lock."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

from record import ROOT, WORK, EVID


@contextmanager
def server(label, build, slots=1, context=8192, disk_cache='0'):
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    env = dict(os.environ, KAPITSA_HOME=str(WORK/'server-homes'/label),
               KAPITSA_NO_AUTO_UPDATE='1', KAPITSA_NO_PROMPT='1',
               KAPITSA_PARENT_PID=str(os.getpid()), NO_PROXY='127.0.0.1,localhost',
               no_proxy='127.0.0.1,localhost')
    if not (env.get('NI_LOCK_HELD') == '1' or env.get('NACE_GPU_LOCK_HELD') == '1'):
        raise RuntimeError('local_server requires the project GPU lock')
    log_path = EVID/(label+'-server.log')
    cmd = [str(Path(build)/'kapitsa'), 'serve', '--model', str(WORK/'models/qwen3.5-4b-4bit'),
           '--host', '127.0.0.1', '--port', str(port), '--private', '--slots', str(slots),
           '--context', str(context), '--disk-cache', str(disk_cache)]
    out = log_path.open('x')
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    base = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        deadline = time.monotonic()+180
        while True:
            if p.poll() is not None:
                raise RuntimeError(f'Server exited: {log_path.read_text()[-2500:]}')
            try:
                with opener.open(base+'/health', timeout=1) as r:
                    healthy = r.status == 200
                with opener.open(base+'/v1/models', timeout=2) as r:
                    ready = r.status == 200
                if healthy and ready:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Server readiness timeout')
            time.sleep(0.25)
        yield base, p, log_path
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
        out.close()
