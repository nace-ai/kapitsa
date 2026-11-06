import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
ROOT = P.parent.parent
TREE = ROOT / 'kflash-reduce-v8'
sys.path.insert(0, str(TREE / 'bench/harness'))
import gpu_hogs

model = P.parent / 'followup-1250/models/qwen3.5-4b-4bit'
ids = (P.parent / 'followup-1250/prompt.ids').read_text().replace(',', ' ').split()
steps = []
env = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'KAPITSA_', 'MTL_', 'REDUCE_'))}
env.update(NI_LOCK_HELD='1', KAPITSA_NO_AUTO_UPDATE='1', KAPITSA_NO_PROMPT='1')
with open('/tmp/kapitsa-gpu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for n in [64, 1536]:
        prompt = P / f'graph-{n}.ids'
        prompt.write_text(' '.join((ids * (n // len(ids) + 1))[:n]) + '\n')
        for timing in [False, True]:
            for arm, tag in enumerate(['base', 'candidate'] if not timing else ['base', 'candidate', 'base', 'candidate']):
                label = f'graph-{n}-' + ('time' if timing else 'check') + f'-{arm}-{tag}'
                out = P / label
                run_env = dict(env, KAPITSA_HOME=str(P / ('home-' + label)))
                if timing:
                    run_env['REDUCE_GRAPH_TIMING'] = '1'
                cmd = [str(P / ('graph-' + tag)), str(model), str(prompt), str(out)]
                if not timing and tag == 'candidate':
                    cmd += ['--from', str(P / f'graph-{n}-check-0-base')]
                if timing:
                    assert gpu_hogs.wait_quiet(maxpct=5, quiet_s=5, timeout_s=60, exclude_roots=[os.getpid()])
                monitor = gpu_hogs.ContentionMonitor([os.getpid()], interval=0.5)
                monitor.start()
                print('START', label, flush=True)
                start = time.time()
                try:
                    with (P / (label + '.log')).open('w') as log:
                        cp = subprocess.run(cmd, env=run_env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                finally:
                    monitor.stop()
                row = dict(label=label, command=cmd, environment=run_env, exit=cp.returncode,
                           elapsed_s=time.time()-start, source_base='2fe7c3fa',
                           binary_sha256=hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest(),
                           contention=gpu_hogs.summarize_samples(monitor.samples), samples=monitor.samples)
                # Only explicitly set variables are evidence; avoid persisting unrelated environment credentials.
                row['environment'] = {k: v for k, v in run_env.items() if k.startswith(('NI_', 'NACE_', 'KAPITSA_', 'MTL_', 'REDUCE_'))}
                if (out / 'meta.json').exists():
                    row['manifest'] = json.loads((out / 'meta.json').read_text())
                if timing:
                    times = [float(l.split('us=')[1].split()[0]) for l in (P/(label+'.log')).read_text().splitlines()
                             if l.startswith('graph sample=') and int(l.split('sample=')[1].split()[0])>=2]
                    row.update(gpu_us=times, median_gpu_us=statistics.median(times) if times else None)
                steps.append(row)
                (P/'graph-results.json').write_text(json.dumps(steps,indent=2)+'\n')
                assert cp.returncode == 0, label
                print('END',label,row.get('median_gpu_us'),row['contention'],flush=True)
(P/'graph-passed').touch()
