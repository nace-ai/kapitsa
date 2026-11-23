#!/usr/bin/env python3
"""Wait on filesystem events and enforce this campaign's absolute deadline."""
from datetime import datetime
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

E = Path(__file__).resolve().parent
NAME = 'k758-pr1396-ab-20261123T0934Z'
JOBS = Path.home() / '.agents/jobs'
EXIT = JOBS / (NAME + '.exit')
LOG = E / 'detached-02.log'
DETACH = Path.home() / '.agents/skills/session-orchestration/scripts/detach.py'
deadline = datetime.fromisoformat('2026-11-23T11:30:00+00:00').timestamp()
queue = select.kqueue()
fds = []
for path in (JOBS, LOG):
    fd = os.open(path, os.O_RDONLY)
    fds.append(fd)
    queue.control([select.kevent(fd, filter=select.KQ_FILTER_VNODE, flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                               fflags=select.KQ_NOTE_WRITE | select.KQ_NOTE_RENAME | select.KQ_NOTE_DELETE)], 0)
position = 0
timed_out = False
try:
    while True:
        if LOG.exists():
            with LOG.open('r', errors='replace') as f:
                f.seek(position)
                for line in f:
                    print(line, end='', flush=True)
                position = f.tell()
        if EXIT.exists():
            parts = EXIT.read_text().split()
            rc = int(parts[0])
            receipt = {'name': NAME, 'returncode': rc, 'exit_epoch': int(parts[1]),
                       'deadline_utc': '2026-11-23T11:30:00Z', 'deadline_stop': timed_out}
            (E / 'detached-terminal.json').write_text(json.dumps(receipt, indent=2) + '\n')
            print(json.dumps(receipt), flush=True)
            sys.exit(rc)
        left = deadline - time.time()
        if left <= 0:
            if timed_out:
                result = subprocess.run([sys.executable, str(DETACH), 'stop', NAME, '--force'])
                print('Forced only the owned detached campaign after its deadline grace period.', result.returncode, flush=True)
                sys.exit(124)
            result = subprocess.run([sys.executable, str(DETACH), 'stop', NAME])
            print('Absolute campaign deadline reached; requested owned-group SIGTERM.', result.returncode, flush=True)
            timed_out = True
            deadline = time.time() + 45
        queue.control([], 8, max(0.0, deadline - time.time()))
finally:
    for fd in fds:
        os.close(fd)
    queue.close()
