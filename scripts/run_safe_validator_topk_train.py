#!/usr/bin/env python3
from __future__ import annotations

import csv, json, subprocess, sys
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
APPLY = REPO/'scripts'/'postprocess'/'apply_validator_links.py'
EVAL = REPO/'scripts'/'eval_motstyle_trackeval.py'
INPUT = REPO/'outputs'/'alink_train_inputs'/'parambest_track_results'
PRED = REPO/'outputs'/'alink_safe_link_validator_exp_all4_p085_m050'/'loso_predictions.csv'
OLD = REPO/'outputs'/'alink_train_sweep'/'recall'/'linked_results'
OUT = REPO/'outputs'/'alink_safe_validator_topk_train'
GT = Path('/gemini/code/datasets/MOT20/train')
SEQS = ['MOT20-01','MOT20-02','MOT20-03','MOT20-05']
COMBOS = []
for k2 in [40,50,60]:
    for k3 in [20,25,30]:
        COMBOS.append((f'old01_02s{k2}_03s{k3}_old05', 'old', k2, k3, 'old'))
# only test very safe 05 variants around the best old05 candidate
for k5 in [40,50]:
    COMBOS.append((f'old01_02s40_03s25_05s{k5}', 'old', 40, 25, k5))
    COMBOS.append((f'old01_02s50_03s25_05s{k5}', 'old', 50, 25, k5))

def parse_summary(path: Path) -> dict:
    lines=path.read_text().strip().splitlines(); h=lines[0].split(); v=lines[1].split()
    return {k:float(val) for k,val in zip(h,v)}

def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf-8') as f:
        p=subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f'failed: {log}')

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name, mode01, k2, k3, k5 in COMBOS:
        rd=OUT/name; linked=rd/'linked_results'; summary=rd/'trackeval_work'/'eval'/name/'pedestrian_summary.txt'
        if not summary.is_file():
            cmd=[sys.executable,str(APPLY),'--input-dir',str(INPUT),'--pred-csv',str(PRED),'--output-dir',str(linked),'--seq-topk',f'MOT20-02:{k2}','--seq-topk',f'MOT20-03:{k3}','--copy-source',f'MOT20-01:{OLD/"MOT20-01.txt"}','--summary-json',str(rd/'apply_summary.json')]
            if k5 == 'old':
                cmd += ['--copy-source',f'MOT20-05:{OLD/"MOT20-05.txt"}']
            else:
                cmd += ['--seq-topk',f'MOT20-05:{k5}']
            run(cmd, rd/'apply.log')
            ecmd=[sys.executable,str(EVAL),'--benchmark-name','MOT20','--split-to-eval','train','--gt-root',str(GT),'--results-dir',str(linked),'--tracker-name',name,'--work-dir',str(rd/'trackeval_work'),'--seqs',*SEQS]
            run(ecmd, rd/'eval.log')
        m=parse_summary(summary)
        row={'name':name,'k2':k2,'k3':k3,'k5':k5,'HOTA':m['HOTA'],'DetA':m['DetA'],'AssA':m['AssA'],'IDF1':m['IDF1'],'MOTA':m['MOTA'],'IDSW':m['IDSW'],'Frag':m['Frag'],'CLR_FN':m['CLR_FN'],'CLR_FP':m['CLR_FP']}
        rows.append(row); print(json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda r:(-r['HOTA'], r['IDSW']))
    fields=list(rows[0].keys())
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    print(json.dumps(rows[:10],indent=2,sort_keys=True), flush=True)

if __name__=='__main__': main()
