#!/usr/bin/env python3
"""Check queued SSE heartbeats while actively draining the two slot holders."""
import json
import threading
import time
import urllib.request

from local_server import server
from record import ROOT, EVID


def events(base, body):
    request = urllib.request.Request(base+'/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=90) as response:
        for line in response:
            if line.startswith(b'data: '):
                value = line[6:].strip()
                yield '[DONE]' if value == b'[DONE]' else json.loads(value)


def choice(value):
    return (value.get('choices') or [{}])[0] if isinstance(value, dict) else {}


def probe(base):
    stop = threading.Event()
    ready = [threading.Event(), threading.Event()]
    ended = [threading.Event(), threading.Event()]
    counts = [0, 0]
    errors = []
    def holder(i):
        body = {'messages':[{'role':'user','content':f'[drained holder {i}] Count from 1 to 100000, one number per line, nothing else.'}],
                'max_tokens':20000, 'temperature':0, 'top_k':1, 'stream':True,
                'chat_template_kwargs':{'enable_thinking':False}}
        try:
            for event in events(base, body):
                if choice(event).get('delta', {}).get('content'):
                    counts[i] += 1
                    ready[i].set()
                if stop.is_set():
                    break
        except Exception as exc:
            if not stop.is_set():
                errors.append(repr(exc))
        finally:
            ended[i].set()
    holders = [threading.Thread(target=holder,args=(i,),daemon=True) for i in range(2)]
    for i, thread in enumerate(holders):
        thread.start()
        if not ready[i].wait(30):
            stop.set()
            raise RuntimeError('Slot holder did not start')
    observed = []
    def waiting():
        start = time.monotonic()
        body = {'messages':[{'role':'user','content':'[queued] Count from 1 to 100, one number per line.'}],
                'max_tokens':8,'temperature':0,'top_k':1,'stream':True,
                'chat_template_kwargs':{'enable_thinking':False}}
        try:
            for event in events(base, body):
                observed.append((time.monotonic()-start,event))
        except Exception as exc:
            errors.append(repr(exc))
    thread = threading.Thread(target=waiting,daemon=True)
    thread.start()
    start = time.monotonic()
    def keepalives():
        return [t for t,d in list(observed) if choice(d).get('delta') == {} and choice(d).get('finish_reason') is None]
    while len(keepalives()) < 2 and time.monotonic()-start < 45:
        time.sleep(0.1)
    held_until_release = not any(e.is_set() for e in ended)
    stop.set()
    for h in holders:
        h.join(10)
    thread.join(15)
    keep = keepalives()
    first = next((t for t,d in observed if choice(d).get('delta',{}).get('content')),None)
    passed = (held_until_release and not errors and len(observed)>2 and
              choice(observed[0][1]).get('delta') == {'role':'assistant','content':''} and observed[0][0]<1 and
              len(keep)==2 and 15<=keep[0]<17.5 and 14<=keep[1]-keep[0]<17.5 and
              first is not None and first>keep[1] and observed[-1][1]=='[DONE]')
    return {'passed':passed,'held_until_release':held_until_release,'holder_content_events':counts,
            'keepalives_s':keep,'first_token_s':first,'errors':errors,'events':observed}


def main():
    rows = []
    for arm, build in [('A',ROOT/'build-m4pro-baseline'),('B',ROOT/'build')]:
        with server('keepalive-drained-'+arm,build,slots=2,context=32768) as (base,_,_):
            result = probe(base)
            result['arm'] = arm
            rows.append(result)
            print(arm,json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)
    (EVID/'keepalive-drained.json').write_text(json.dumps(rows,indent=2)+'\n')
    raise SystemExit(not all(r['passed'] for r in rows))


if __name__ == '__main__':
    main()
