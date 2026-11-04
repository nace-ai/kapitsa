#!/usr/bin/env python3
"""Supplemental CLI token/cycle evidence; timing fields are not speed-acceptance data."""
import json
import re
from record import ROOT, WORK, EVID, run


def main():
    rows = []
    for temperature in (0, 0.6):
        for context in ('short','4k'):
            expected = None
            for arm, build in [('A',ROOT/'build-m4pro-baseline'),('B',ROOT/'build')]:
                label = f'tokens-{temperature}-{context}-{arm}'
                rc = run(label,[build/'ni_run',WORK/'models/qwen3.5-4b-4bit',
                         '@'+str(WORK/'prompts'/(context+'.ids')),'512',str(temperature),'--dflash','auto'],
                         {'NI_CYCLE_TRACE':'1','NI_SEED':'42'})
                if rc:
                    raise SystemExit(rc)
                text = (EVID/(label+'.log')).read_text()
                match = re.search(r'^ids:\s*(.*)$',text,re.M)
                if not match:
                    raise RuntimeError('Missing generated token ids')
                ids = [int(x) for x in match.group(1).split()]
                eos = [i for i,t in enumerate(ids) if t in (248044,248046)]
                record = {'temperature':temperature,'context':context,'arm':arm,'token_count':len(ids),
                          'end_token_count':len(eos),'tokens_before_first_end':eos[0] if eos else len(ids),
                          'same_ids_as_A':arm=='A' or ids==expected,'ids':ids}
                if len(ids)!=512 or eos or not record['same_ids_as_A']:
                    raise RuntimeError('CLI token validation failed')
                expected = ids
                rows.append(record)
                print('TOKENS',json.dumps({k:v for k,v in record.items() if k!='ids'}),flush=True)
    (EVID/'token-traces.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__ == '__main__':
    main()
