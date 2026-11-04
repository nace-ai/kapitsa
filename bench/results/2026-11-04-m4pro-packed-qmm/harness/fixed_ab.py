#!/usr/bin/env python3
"""A fixed-prompt A/B diagnostic using the repository's unchanged stream-rate helper."""
import hashlib
import json
import os
import sys
import time
import urllib.request

from local_server import server
from record import ROOT, WORK, EVID
from thermal import cooldown, state

sys.path.insert(0, str(ROOT/'bench/harness'))
import bench
import gpu_hogs

SAMPLED = '--sampled' in sys.argv


def body(messages, model, count=512):
    return {'model': model, 'messages': messages, 'temperature': 0.6 if SAMPLED else 0,
            'top_k': 20 if SAMPLED else 1, 'top_p': 0.95 if SAMPLED else 1,
            'min_p': 0, 'presence_penalty': 0, 'frequency_penalty': 0, 'repeat_penalty': 1,
            'seed': 42, 'max_tokens': count, 'chat_template_kwargs': {'enable_thinking': False}}


def main():
    os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'
    prefix = 'fixed-sampled' if SAMPLED else 'fixed'
    dest = EVID/(prefix+'-bench')
    dest.mkdir(exist_ok=False)
    cases = {n: json.loads((WORK/'prompts'/(n+'.messages.json')).read_text()) for n in ('short', '4k')}
    warm = [{'role': 'system', 'content': '[warm-up only] You are a helpful coding assistant.\n'+
             bench.build_context(2700, offset=100000)},
            {'role': 'user', 'content': 'Summarize the first file in the repository context.'}]
    warm_decode = [{'role':'system','content':'[separate warm-up] '+bench.DECODE_SYS},
                   {'role':'user','content':bench.DECODE_USER}]
    all_results = []
    for label in ('A1', 'B1', 'A2', 'B2'):
        build = ROOT/('build-m4pro-baseline' if label.startswith('A') else 'build')
        entry = {'label': label, 'build': str(build), 'temperature':0.6 if SAMPLED else 0,
                 'requests': {}, 'warmups': []}
        print('Starting fixed-prompt arm', label, flush=True)
        entry['cooldown'] = cooldown()
        with server(prefix+'-'+label, build) as (base, proc, log):
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(base+'/v1/models') as response:
                model = json.load(response)['data'][0]['id']
            for messages, count in ((warm, 64), (warm_decode, 512)):
                result = bench.stream_chat(base+'/v1', body(messages, model, count), timeout=300)
                entry['warmups'].append({'usage': result['usage'], 'finish': result['finish'],
                                         'nonempty': bool(result['content'].strip())})
            quiet = gpu_hogs.wait_quiet(quiet_s=15, timeout_s=120, exclude_roots=[proc.pid], own_max=1, log=print)
            entry['quiet'] = {'ok': quiet[0], 'waited_s': quiet[1], 'busy': quiet[2]}
            if not quiet[0]:
                (dest/(label+'.json')).write_text(json.dumps(entry,indent=2)+'\n')
                raise RuntimeError('GPU did not become quiet; fixed series incomplete')
            monitor = gpu_hogs.ContentionMonitor([proc.pid])
            monitor.start()
            try:
                for context, messages in cases.items():
                    time.sleep(1)
                    if state() != 0:
                        entry.setdefault('additional_cooldowns',{})[context] = cooldown(minimum=0)
                    thermal_before = state()
                    before = bench.vm_snapshot()
                    t0 = time.time()
                    result = bench.stream_chat(base+'/v1', body(messages, model), timeout=300, telemetry=True)
                    t1 = time.time()
                    after = bench.vm_snapshot()
                    thermal_after = state()
                    count = (result['usage'] or {}).get('completion_tokens')
                    mid, e2e, info = bench.decode_rates(result, count)
                    cached = bench.cached_of(result['usage'], result['timings'])
                    flags = []
                    if count != 512 or result['finish'] != 'length' or not result['content'].strip():
                        flags.append('nonempty 512-token length-finished answer required')
                    if cached is None or cached > bench.COLD_CACHE_MAX:
                        flags.append('cold-cache check failed')
                    if bench.thinking_leak(result):
                        flags.append('thinking leakage')
                    if thermal_before != 0 or thermal_after != 0:
                        flags.append('thermal state not nominal')
                    changes = {k: after[k]-before[k] for k in ('swapins','swapouts','pageouts') if k in before and k in after}
                    if any(changes.get(k,0) for k in ('swapins','swapouts')):
                        flags.append('paging during request')
                    item = {'context': context, 'completion_tokens':count, 'prompt_tokens':(result['usage'] or {}).get('prompt_tokens'),
                            'cached_tokens':cached, 'middle_window_tps':mid, 'whole_decode_tps':e2e,
                            'ttft_s':result['ttft'], 'total_s':result['total'], 'rate':info,
                            'text_sha256':hashlib.sha256(result['content'].encode()).hexdigest(),
                            'before_vm':before, 'after_vm':after, 'vm_delta':changes,
                            'thermal_before':thermal_before,'thermal_after':thermal_after,
                            'start_epoch':t0, 'end_epoch':t1, 'flags':flags, 'stream':result}
                    entry['requests'][context] = item
                    print(label,context,'tokens=',count,'cached=',cached,'middle=',mid,'whole=',e2e,'flags=',flags,flush=True)
            finally:
                monitor.stop()
                entry['gpu_samples'] = monitor.samples
            for item in entry['requests'].values():
                contention = gpu_hogs.summarize_samples(monitor.samples,item['start_epoch'],item['end_epoch'])
                item['contention'] = contention
                if contention['flagged']:
                    item['flags'].append('foreign GPU contention')
            entry['server_log'] = str(log)
        (dest/(label+'.json')).write_text(json.dumps(entry,indent=2)+'\n')
        all_results.append(entry)
        (EVID/(prefix+'-bench-runs.json')).write_text(json.dumps(all_results,indent=2)+'\n')
    comparisons = {}
    for context in cases:
        rows = [r['requests'][context] for r in all_results]
        comparisons[context] = {'all_text_identical':len({r['text_sha256'] for r in rows}) == 1,
                                'all_nonempty_512':all(r['completion_tokens']==512 and r['stream']['content'].strip() for r in rows),
                                'all_unflagged':not any(r['flags'] for r in rows)}
    (EVID/(prefix+'-bench-comparisons.json')).write_text(json.dumps(comparisons,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2),flush=True)


if __name__ == '__main__':
    main()
