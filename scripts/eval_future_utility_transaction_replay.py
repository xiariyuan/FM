from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from eval_canonical_segment_transaction_replay import (
    apply_local_transactions,
    reconstruct_best_with_provenance,
    records_to_rows,
)
from eval_assa_swap_merge_fusion import write_rows


def evaluate(pdir: Path, name: str):
    cmd=[sys.executable,'scripts/eval_motstyle_trackeval.py','--benchmark-name','MOT20','--split-to-eval','train',
         '--gt-root','datasets/MOT20/train','--results-dir',str(pdir/'track_results'),
         '--tracker-name',name,'--work-dir',str(pdir/'eval_work'),'--seqs','MOT20-02']
    proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (pdir/'eval.log').write_text(proc.stdout)
    detail=pdir/'eval_work'/'eval'/name/'pedestrian_detailed.csv'
    if not detail.exists(): return {'returncode':proc.returncode}
    r=next(x for x in csv.DictReader(detail.open()) if x['seq']=='MOT20-02')
    return {'returncode':proc.returncode,'HOTA':float(r['HOTA___AUC'])*100,'AssA':float(r['AssA___AUC'])*100,
            'DetA':float(r['DetA___AUC'])*100,'IDF1':float(r['IDF1'])*100 if float(r['IDF1'])<2 else float(r['IDF1']),
            'IDSW':int(float(r['IDSW']))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-root',required=True);ap.add_argument('--best-root',required=True)
    ap.add_argument('--merge-links',required=True);ap.add_argument('--aggressive-events',required=True)
    ap.add_argument('--scores',required=True);ap.add_argument('--out-dir',required=True)
    ap.add_argument('--budgets',nargs='*',type=int,default=[5,10,20])
    args=ap.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    scores=pd.read_csv(args.scores).sort_values('oof_transaction_rank',ascending=False)
    links=pd.read_csv(args.merge_links); aggressive=pd.read_csv(args.aggressive_events)
    base,_=reconstruct_best_with_provenance(Path(args.source_root)/'MOT20-02.txt',links[links.seq=='MOT20-02'],aggressive[aggressive.seq=='MOT20-02'])
    rows=[]
    for k in args.budgets:
        events=scores.head(k).copy()
        modified,accepted,rejected,changed=apply_local_transactions(base,events.to_dict('records'),'perm')
        name=f'utility_top{k}_perm'; p=out/name
        write_rows(p/'track_results'/'MOT20-02.txt',records_to_rows(modified))
        e=evaluate(p,name)
        row={'budget':k,'requested':len(events),'accepted':len(accepted),'rejected':len(rejected),'changed_rows':changed,**e}
        rows.append(row)
        pd.DataFrame(accepted).to_csv(p/'accepted.csv',index=False)
        if rejected: pd.DataFrame(rejected).to_csv(p/'rejected.csv',index=False)
        print(json.dumps(row,indent=2),flush=True)
    pd.DataFrame(rows).to_csv(out/'summary.csv',index=False)

if __name__=='__main__':main()
