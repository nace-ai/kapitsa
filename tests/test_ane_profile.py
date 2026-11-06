#!/usr/bin/env python3
"""CPU-only: profile/override policy for actual ANE input geometry, fresh process per setting."""
import os
import subprocess
import sys

binary = sys.argv[1] if len(sys.argv) > 1 else 'build/ane_segment_check'
for device in ('simd', 'matrix'):
    for profile in (None, 'precise', 'fast', 'invalid'):
        for override in (None, '', '0', '1'):
            env = {k: v for k, v in os.environ.items() if k not in ('KAPITSA_PROFILE', 'NI_ANE')}
            if profile is not None:
                env['KAPITSA_PROFILE'] = profile
            if override is not None:
                env['NI_ANE'] = override
            p = subprocess.run([binary, '--profile', device], env=env, capture_output=True, text=True, check=True)
            got = {int(h): int(on) for h, on in (line.split() for line in p.stdout.splitlines())}
            new = override == '1' or (override in (None, '') and profile == 'fast' and device == 'simd')
            old = override != '0'
            expected = {0: 0, 2048: int(new), 2560: int(old), 4096: int(new),
                        5120: int(old), 6144: int(new), 8192: int(new), 10240: 0}
            assert got == expected, (device, profile, override, got, expected)
            print(f'PASS device={device}, profile={profile!r}, NI_ANE={override!r}')
print('32 device/profile/override cases passed; no GPU or model loaded')
