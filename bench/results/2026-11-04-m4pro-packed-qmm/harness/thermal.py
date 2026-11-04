"""Read the same NSProcessInfo thermal state as Kapitsa and wait without using the GPU."""
import subprocess
import time
from record import WORK


def state():
    return int(subprocess.check_output([str(WORK/'thermal')],text=True).strip())


def cooldown(minimum=45, stable=15, limit=600):
    start = time.monotonic()
    good = None
    samples = []
    last_print = -30
    while True:
        now = time.monotonic()
        value = state()
        samples.append({'elapsed_s':round(now-start,3),'state':value})
        good = (good if good is not None else now) if value == 0 else None
        if now-start >= minimum and good is not None and now-good >= stable:
            print(f'Nominal thermal state; cooled for {now-start:.1f}s',flush=True)
            return samples
        if now-start > limit:
            raise RuntimeError('Thermal state did not settle to nominal')
        if now-start-last_print >= 30:
            print(f'Cooling: {now-start:.0f}s, thermal state {value} (0=nominal)',flush=True)
            last_print = now-start
        time.sleep(2)
