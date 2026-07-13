#!/usr/bin/env python3
from __future__ import annotations

import csv, json, subprocess, sys
from datetime import datetime
from pathlib import Path

REPO = Path('/gemini/code/FMtrack-main/FM-Track')
INPUT = REPO/'outputs'/'alink_train_inputs'/'parambest_track_results'
PHASE0 = REPO/'outputs'/'alink_train_inputs'/'phase0_root'
OUT = REPO/'outputs'/'alink_v2_select_train_sweep'
LINKER = REPO/'scripts'/'postprocess'/'safe_tracklet_linker_v2.py'
EVAL = REPO/'scripts'/'eval_motstyle_trackeval.py'
GT = Path('/gemini/code/datasets/MOT20/train')
SEQS=['MOT20-01','MOT20-02','MOT20-03','MOT20-05']

CONFIGS=[
  # name, max_links, tier2_app, tier2_score, tier2_rank, tier3_app, tier3_score, tier3_rank, margin
  ('tier1_only', 999999, 9.9, 9.9, 1, 9.9, 9.9, 1, 0.02),
  ('no_tier3', 999999, 0.64, 0.68, 2, 9.9, 9.9, 1, 0.02),
  ('cap40', 40, 0.62, 0.66, 2, 0.54, 0.62, 3, 0.01),
  ('cap60', 60, 0.62, 0.66, 2, 0.54, 0.62, 3, 0.01),
  ('margin_strict', 999999, 0.62, 0.68, 2, 0.54, 0.64, 3, 0.03),
]

def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('[run]', ' '.join(cmd[:10]), '...', flush=True)
    with log.open('w', encoding='utf-8') as f:
        f.write('[started_at] '+datetime.now().astimezone().isoformat(timespec='seconds')+'\n')
        f.write('[cmd] '+' '.join(map(str,cmd))+'\n\n'); f.flush()
        p=subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
        f.write('\n[finished_at] '+datetime.now().astimezone().isoformat(timespec='seconds')+'\n')
        f.write('[return_code] '+str(p.returncode)+'\n')
    return p.returncode

def parse_summary(path):
    lines=path.read_text().strip().splitlines(); h=lines[0].split(); v=lines[1].split(); d={}
    for k,val in zip(h,v):
        try: d[k]=float(val)
        except Exception: d[k]=val
    return d

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for name,max_links,t2a,t2s,t2r,t3a,t3s,t3r,margin in CONFIGS:
        rd=OUT/name; linked=rd/'linked_results'; sj=rd/'link_summary.json'; sc=rd/'link_summary.csv'
        if not sj.is_file():
            cmd=[sys.executable,str(LINKER),'--input-dir',str(INPUT),'--phase0-root',str(PHASE0),'--output-dir',str(linked),'--seqs',*SEQS,
                 '--max-gap','120','--min-source-len','3','--min-target-len','2','--candidate-max-center-step','180','--candidate-max-area-ratio','8','--candidate-max-height-ratio','3.5',
                 '--max-bank-feats','8','--bank-topk','3','--global-k','12','--high-k','6','--start-k','4','--end-k','4',
                 '--tier1-app','0.72','--tier1-score','0.72','--tier1-margin',str(margin),
                 '--tier2-app',str(t2a),'--tier2-score',str(t2s),'--tier2-margin',str(margin),'--tier2-max-rank',str(t2r),
                 '--tier3-app',str(t3a),'--tier3-score',str(t3s),'--tier3-margin',str(margin),'--tier3-max-rank',str(t3r),
                 '--max-links-per-seq',str(max_links),'--summary-json',str(sj),'--summary-csv',str(sc)]
            rc=run(cmd, rd/'link.log')
            if rc!=0: raise RuntimeError(f'link failed {name}')
        es=rd/'trackeval_work'/'eval'/name/'pedestrian_summary.txt'
        if not es.is_file():
            cmd=[sys.executable,str(EVAL),'--benchmark-name','MOT20','--split-to-eval','train','--gt-root',str(GT),'--results-dir',str(linked),'--tracker-name',name,'--work-dir',str(rd/'trackeval_work'),'--seqs',*SEQS]
            rc=run(cmd, rd/'eval.log')
            if rc!=0: raise RuntimeError(f'eval failed {name}')
        m=parse_summary(es)
        lrows=list(csv.DictReader(sc.open())) if sc.is_file() else []
        row={'name':name,'selected_links':sum(int(float(r.get('selected_links',0))) for r in lrows),'tiered_edges':sum(int(float(r.get('tiered_edges',0))) for r in lrows),'candidate_edges':sum(int(float(r.get('candidate_edges',0))) for r in lrows),'remapped_rows':sum(int(float(r.get('remapped_rows',0))) for r in lrows),'HOTA':m.get('HOTA'),'DetA':m.get('DetA'),'AssA':m.get('AssA'),'IDF1':m.get('IDF1'),'MOTA':m.get('MOTA'),'IDSW':m.get('IDSW'),'Frag':m.get('Frag'),'CLR_FN':m.get('CLR_FN'),'CLR_FP':m.get('CLR_FP')}
        rows.append(row); print(json.dumps(row, indent=2, sort_keys=True), flush=True)
    fields=list(rows[0].keys()) if rows else ['name']
    with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
    print(json.dumps(rows,indent=2,sort_keys=True), flush=True)

if __name__=='__main__': main()
