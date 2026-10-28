"""Ensure experimental and default SSD cache namespaces survive separate real restarts."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'evidence/ane-cache-cross-modes'
BINARY = ROOT / 'candidate/kapitsa'
MODEL = Path('$MODELS/qwen3.5-9b-4bit')


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    assert os.environ.get('NI_LOCK_HELD') == '1' and os.environ.get('NACE_GPU_LOCK_HELD') == '1'
    DEST.mkdir(exist_ok=False)
    case = json.loads((ROOT / 'prompts/cache-case.json').read_text())
    save(DEST / 'plan.json', {
        'purpose': 'Functional only; preserve both SSD namespaces in one temporary task home across four real restarts.',
        'sequence': ['B cold, save', 'A cold must not import B, save', 'B warm must recover its state', 'A warm must recover its state'],
        'binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(), 'case': case,
        'required': 'Cold cached_tokens=0; warm cached_tokens>=2048; exact expected JSON each run; same-mode warm/cold text identical.',
    })
    results = []
    first_content = {}
    for label, ane, cold in [('B-cold', True, True), ('A-cold', False, True), ('B-warm', True, False), ('A-warm', False, False)]:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        env = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'KAPITSA_'))}
        env.update({'NI_LOCK_HELD': '1', 'NACE_GPU_LOCK_HELD': '1', 'KAPITSA_HOME': str(DEST / 'home'),
                    'KAPITSA_NO_AUTO_UPDATE': '1', 'KAPITSA_NO_PROMPT': '1', 'NI_DISK_CACHE_IDLE_MS': '500'})
        if ane:
            env['NI_ANE_H4096'] = '1'
        command = [str(BINARY), 'serve', '--model', str(MODEL), '--model-id', 'ane-quality', '--host', '127.0.0.1',
                   '--private', '--port', str(port), '--ctx', '32768', '--slots', '1', '--disk-cache', '2G']
        save(DEST / (label + '-command.json'), {'command': command, 'mode': label})
        log = DEST / (label + '.log')
        server = None

        def text():
            return log.read_text(errors='replace')

        def post(body):
            request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.load(response)

        try:
            with log.open('x') as output:
                server = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
                save(DEST / (label + '-pid.json'), {'pid': server.pid})
                deadline = time.monotonic() + 240
                while True:
                    assert server.poll() is None, 'server exited'
                    try:
                        with urllib.request.urlopen(base + '/health', timeout=1) as response:
                            ready = response.status == 200
                    except (OSError, urllib.error.URLError):
                        ready = False
                    current = text()
                    cache_ready = re.search(r'SSD prefix cache: .* taken back', current)
                    ane_ready = not ane or 'Neural Engine program ready in the background' in current
                    if ready and cache_ready and ane_ready:
                        break
                    assert time.monotonic() < deadline, 'readiness timeout'
                    time.sleep(0.25)
                post({'model': 'ane-quality', 'messages': [{'role': 'user', 'content': 'Hi.'}],
                      'max_tokens': 1, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
                assert ('NI_ANE_H4096=1: experimental' in text()) == ane
                mark = len(text())
                response = post(case['body'])
                content = response['choices'][0]['message']['content']
                assert json.loads(content) == case['expected']
                assert response['choices'][0]['finish_reason'] == 'stop'
                cached = response['usage']['prompt_tokens_details']['cached_tokens']
                assert cached == 0 if cold else cached >= 2048, 'wrong SSD namespace behavior'
                first_content.setdefault(ane, content)
                assert content == first_content[ane], 'same-mode cache changed the reply'
                time.sleep(0.1)
                request_log = text()[mark:]
                if ane and cold:
                    assert re.search(r'prefill: Neural Engine busy .* evaluations', request_log)
                assert 'reruns on the GPU' not in text()
                row = {'label': label, 'cached_tokens': cached, 'response': response}
                save(DEST / (label + '.json'), row)
                results.append(row)
                print(f'PASS {label}: cached_tokens={cached}, exact expected JSON, same-mode reply preserved', flush=True)
        finally:
            if server is not None and server.poll() is None:
                server.send_signal(signal.SIGTERM)
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
            if server is not None:
                save(DEST / (label + '-stopped.json'), {'pid': server.pid, 'returncode': server.returncode})
    save(DEST / 'summary.json', {'passed': len(results) == 4, 'results': results})


if __name__ == '__main__':
    main()
