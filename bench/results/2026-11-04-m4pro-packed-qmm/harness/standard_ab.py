#!/usr/bin/env python3
"""Run the declared interleaved native benchmark series and keep every result."""
import json

from record import ROOT, WORK, EVID, run


def main():
    results_dir = EVID/'standard-bench'
    results_dir.mkdir(exist_ok=False)
    results = []
    for label in ('A1', 'B1', 'A2', 'B2', 'A3', 'B3'):
        build = ROOT/('build-m4pro-baseline' if label.startswith('A') else 'build')
        dst = results_dir/(label+'.json')
        home = WORK/'benchmark-homes'/label
        home.mkdir(parents=True, exist_ok=False)
        cmd = [build/'kapitsa', 'bench', '--model', WORK/'models/qwen3.5-4b-4bit',
               '--quick', '--json', '--out', dst]
        rc = run('standard-bench-'+label, cmd, {'KAPITSA_HOME': str(home)})
        results.append({'label': label, 'exit_code': rc, 'result': str(dst)})
        (EVID/'standard-bench-runs.json').write_text(json.dumps(results, indent=2)+'\n')
        if not dst.exists():
            raise SystemExit(rc or 1)
        result = json.loads(dst.read_text())
        print('RESULT', label, json.dumps(result.get('results', result.get('metrics', {}))), flush=True)
        print('FLAGS', label, json.dumps(result.get('flags', [])), flush=True)
    print('All six planned native benchmark arms completed.', flush=True)


if __name__ == '__main__':
    main()
