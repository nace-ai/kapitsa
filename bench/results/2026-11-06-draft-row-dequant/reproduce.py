#!/usr/bin/env python3
"""Compare the exact CPU helper from two Git revisions; the requested output directory must be new."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MODE = "row"
BASE = "f92f4215"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def capture(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default=BASE)
    parser.add_argument('--candidate', default='HEAD')
    parser.add_argument('--model', type=Path, required=True, help='4B model.safetensors, with BF16 affine coefficients')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='compare all bytes without timing')
    args = parser.parse_args()
    args.model = args.model.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = []
    metadata = {'mode': MODE, 'model': str(args.model), 'check_only': args.check_only,
                'compiler': capture(['clang++', '--version']),
                'macos': capture(['sw_vers']), 'chip': capture(['sysctl', '-n', 'machdep.cpu.brand_string']),
                'memory_bytes': int(capture(['sysctl', '-n', 'hw.memsize'])), 'revisions': {}}
    for label, revision in [('reference', args.reference), ('candidate', args.candidate)]:
        commit = capture(['git', 'rev-parse', revision])
        path = 'src/model/affine_coeff_fixed.h' if MODE == 'pack' else 'src/model/dflash2.cpp'
        source = subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT, text=True)
        metadata['revisions'][label] = {'commit': commit, 'source_path': path,
                                      'source_sha256': hashlib.sha256(source.encode()).hexdigest()}
        cpp = out / (label + '.cpp')
        if MODE == 'pack':
            # Keep the identical struct and coefficient codec; give the changed inline helper a distinct name.
            # Without this rename an out-of-line weak definition could cause both wrappers to call one version.
            if label == 'candidate':
                source = source.replace('pack_affine_packed_coeff(', 'candidate_internal_pack(')
            (out / (label + '-pack.h')).write_text(source)
            function = 'candidate_internal_pack' if label == 'candidate' else 'pack_affine_packed_coeff'
            cpp.write_text(f'#include "{label}-pack.h"\n'
                           f'ni::AffinePackedCoeff {label}_pack(const uint32_t*w,const uint16_t*s,'
                           f'const uint16_t*b,uint32_t N,uint32_t G) {{return ni::{function}(w,s,b,N,G);}}\n')
        else:
            start = source.index('void dfl_row_bf16(')
            end = source.index('\nconst Tensor* DFlash2::own', start)
            function = source[start:end].replace('dfl_row_bf16(', label + '_row(')
            cpp.write_text('#include "model/dflash2.h"\n#include "util/bf16.h"\nnamespace ni {\n'
                           + function + '\n}\n')
        sources.append(str(cpp))
    exe = out / 'compare'
    command = ['clang++', '-std=c++20', '-O3', '-mmacosx-version-min=14.0',
               '-I' + str(ROOT / 'src'), '-I' + str(ROOT / 'third_party'), '-I' + str(out),
               *sources, str(HERE / 'reproduce.cpp'), '-o', str(exe)]
    with (out / 'compile.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    metadata['compile_command'] = command
    metadata['binary_sha256'] = hashlib.sha256(exe.read_bytes()).hexdigest()
    command = [str(exe), str(args.model)] + (['--check'] if args.check_only else [])
    if not args.check_only:
        command = [str(ROOT / '.dsh/skills/verify-kapitsa/scripts/vk'), 'gpu', 'run', '--eta', '3',
                   '--label', 'CPU-' + MODE + '-A-B', '--', *command]
    metadata['run_command'] = command
    with (out / 'results.jsonl').open('w') as log, (out / 'run.err').open('w') as err:
        result = subprocess.run(command, stdout=log, stderr=err)
    metadata['exit_code'] = result.returncode
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    result.check_returncode()
    rows = [json.loads(line) for line in (out / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    print(json.dumps({'output': str(out), 'matrices': len(rows), 'passed': True}))


if __name__ == '__main__':
    main()
