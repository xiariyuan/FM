#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
APPLY = REPO/'scripts'/'postprocess'/'apply_validator_links.py'
EVAL = REPO/'scripts'/'eval_motstyle_trackeval.py'
INPUT = REPO/'outputs'/'alink_train_inputs'/'parambest_track_results'
PRED = REPO/'outputs'/'alink_tiered_edge_validator_exp_010203'/'loso_predictions.csv'
OLD = REPO/'outputs'/'alink_train_sweep'/'recall'/'linked_results'
OUT = REPO/'outputs'/'alink_validator_topk_grid_train'
GT = Path('/gemini/code/datasets/MOT20/train')
SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
COMBOS=[]
for k2 in [40,50,60]:
    for k3 in [20,25,30,35,40]:
        COMBOS.append((f'old01_02top{k2}_03top{k3}_old05', 'old', k2, k3))
# a few variants with validator top5 for 01
for k2,k3 in [(50,25),(50,30),(60,25),(60,30)]:
    COMBOS.append((f'v01top5_02top{k2}_03top{k3}_old05', 'vtop5', k2, k3))

def parse_summary(path: Path) -> dict:
    lines=path.read_text().strip().splitlines(); h=lines[0].split(); v=lines[1].split(); d={}
    for k,val in zip(h,v):
        try: d[k]=float(val)
        except Exception: d[k]=val
    return d

def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as f:
        p=subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode!=0:
        raise RuntimeError(f'cmd failed {log}')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name, mode01, k2, k3 in COMBOS:
        rd=OUT/name; linked=rd/'linked_results'; summary=rd/'trackeval_work'/'eval'/name/'pedestrian_summary.txt'
        if not summary.is_file():
            cmd=[sys.executable,str(APPLY),'--input-dir',str(INPUT),'--pred-csv',str(PRED),'--output-dir',str(linked),'--seq-topk',f'MOT20-02:{k2}','--seq-topk',f'MOT20-03:{k3}','--copy-source',f'MOT20-05:{OLD/"MOT20-05.txt"}','--summary-json',str(rd/'apply_summary.json')]
            if mode01=='vtop5':
                cmd += ['--seq-topk','MOT20-01:5']
            else:
                cmd += ['--copy-source',f'MOT20-01:{OLD/"MOT20-01.txt"}']
            run(cmd, rd/'apply.log')
            ecmd=[sys.executable,str(EVAL),'--benchmark-name','MOT20','--split-to-eval','train','--gt-root',str(GT),'--results-dir',str(linked),'--tracker-name',name,'--work-dir',str(rd/'trackeval_work'),'--seqs',*SEQS]
            run(ecmd, rd/'eval.log')
        m=parse_summary(summary)
        row={'name':name,'mode01':mode01,'k2':k2,'k3':k3,'HOTA':m['HOTA'],'DetA':m['DetA'],'AssA':m['AssA'],'IDF1':m['IDF1'],'MOTA':m['MOTA'],'IDSW':m['IDSW'],'Frag':m['Frag'],'CLR_FN':m['CLR_FN'],'CLR_FP':m['CLR_FP']}
        rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda r:(-r['HOTA'], r['IDSW']))
    fields=list(rows[0].keys())
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    print(json.dumps(rows[:10], indent=2, sort_keys=True), flush=True)

if __name__=='__main__': main()
